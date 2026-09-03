#!/usr/bin/env python3
from __future__ import annotations
import bpy, sys, json, traceback
from pathlib import Path
from mathutils import Vector


def parse_args():
    a = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    d = {}
    for i in range(0, len(a), 2):
        d[a[i].lstrip('-')] = a[i + 1]
    return d


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def world_bounds(objects):
    pts = []
    for o in objects:
        if o.type != 'MESH':
            continue
        pts += [o.matrix_world @ Vector(c) for c in o.bound_box]
    if not pts:
        raise RuntimeError('No mesh bounds were available for AINA camera framing')
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return mn, mx


def set_alpha_hidden(mat):
    mat.diffuse_color = (1, 1, 1, 0)
    if mat.use_nodes:
        bsdf = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf:
            if 'Alpha' in bsdf.inputs:
                bsdf.inputs['Alpha'].default_value = 0
            if 'Base Color' in bsdf.inputs:
                c = bsdf.inputs['Base Color'].default_value
                bsdf.inputs['Base Color'].default_value = (c[0], c[1], c[2], 0)
    for prop, value in (
        ('surface_render_method', 'DITHERED'),
        ('blend_method', 'BLEND'),
        ('use_transparency_overlap', False),
    ):
        if hasattr(mat, prop):
            try:
                setattr(mat, prop, value)
            except Exception:
                pass


def tune_materials():
    hidden_keys = ('FaceMouth', 'EyeIris', 'EyeHighlight', 'EyeWhite', 'FaceBrow', 'FaceEyeline')
    for m in bpy.data.materials:
        if any(k.lower() in m.name.lower() for k in hidden_keys):
            set_alpha_hidden(m)
            continue
        if not m.use_nodes:
            continue
        bsdf = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if not bsdf:
            continue
        lname = m.name.lower()
        if 'skin' in lname or 'face_00' in lname:
            if 'Roughness' in bsdf.inputs:
                bsdf.inputs['Roughness'].default_value = .48
            if 'Coat Weight' in bsdf.inputs:
                bsdf.inputs['Coat Weight'].default_value = .12
        elif 'hair' in lname or 'updo' in lname:
            if 'Roughness' in bsdf.inputs:
                bsdf.inputs['Roughness'].default_value = .33
            if 'Metallic' in bsdf.inputs:
                bsdf.inputs['Metallic'].default_value = .04
        elif 'core' in lname:
            if 'Emission Color' in bsdf.inputs:
                bsdf.inputs['Emission Color'].default_value = (.03, .35, 1, 1)
            if 'Emission Strength' in bsdf.inputs:
                bsdf.inputs['Emission Strength'].default_value = 4.0


def render_view(scene, cam, target, location, path, lens=82):
    cam.data.lens = lens
    cam.location = Vector(location)
    look_at(cam, target)
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def reset_shapes(face):
    if face and face.data.shape_keys:
        for key in face.data.shape_keys.key_blocks[1:]:
            key.value = 0


def export_sources(out):
    glb = out / 'AINA_EXPORT_V8.glb'
    fbx = out / 'AINA_EXPORT_V8.fbx'
    bpy.ops.export_scene.gltf(
        filepath=str(glb),
        export_format='GLB',
        export_animations=True,
        export_cameras=False,
        export_lights=False,
    )
    bpy.ops.export_scene.fbx(
        filepath=str(fbx),
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=False,
        path_mode='AUTO',
    )
    if not glb.is_file() or glb.stat().st_size == 0:
        raise RuntimeError('AINA GLB export was not produced')
    if not fbx.is_file() or fbx.stat().st_size == 0:
        raise RuntimeError('AINA FBX export was not produced')


def main():
    p = parse_args()
    inp = Path(p['input']).resolve()
    out = Path(p['out']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'Preview').mkdir(exist_ok=True)
    (out / 'QA').mkdir(exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(inp))
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    arms = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    if not meshes:
        raise RuntimeError('AINA V8 GLB imported without any Mesh objects')
    if not arms:
        raise RuntimeError('AINA V8 GLB imported without an Armature')

    for obj in meshes:
        for poly in obj.data.polygons:
            poly.use_smooth = True
    tune_materials()

    head_objs = [o for o in meshes if any(k in o.name.lower() for k in ('face', 'hair', 'updo'))]
    if not head_objs:
        head_objs = meshes
    mn, mx = world_bounds(head_objs)
    target = (mn + mx) * .5
    target.z = mn.z + (mx.z - mn.z) * .48

    world = bpy.data.worlds.new('AINA_WORLD')
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (.018, .025, .042, 1)
    bg.inputs['Strength'].default_value = .28

    lights = [
        ('KEY', (-.55, .65, target.z + .38), 1150, 1.35),
        ('FILL', (.55, .45, target.z + .10), 620, 1.20),
        ('RIM', (0, -.55, target.z + .35), 900, 1.0),
        ('LOW', (0, .30, target.z - .35), 250, .8),
    ]
    for name, loc, energy, size in lights:
        bpy.ops.object.light_add(type='AREA', location=loc)
        light = bpy.context.object
        light.name = 'LGT_' + name
        light.data.energy = energy
        light.data.shape = 'DISK'
        light.data.size = size
        look_at(light, target)

    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.name = 'CAM_AINA'
    cam.data.sensor_width = 36
    bpy.context.scene.camera = cam

    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = 'RGBA'
    try:
        scene.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        pass

    size = max(mx.x - mn.x, mx.z - mn.z)
    dist = max(.62, size * 2.55)
    render_view(scene, cam, target, (target.x, target.y + dist, target.z), out / 'Preview' / 'AINA_PORTRAIT_FRONT.png', 88)
    render_view(scene, cam, target, (target.x + dist * .58, target.y + dist * .82, target.z), out / 'Preview' / 'AINA_PORTRAIT_3Q.png', 88)
    render_view(scene, cam, target, (target.x + dist, target.y, target.z), out / 'Preview' / 'AINA_PORTRAIT_PROFILE.png', 88)

    face = max(
        (o for o in meshes if o.data.shape_keys),
        key=lambda o: len(o.data.shape_keys.key_blocks),
        default=None,
    )
    expressions = {'NEUTRAL': None, 'HAPPY': 3, 'BLINK': 13, 'AA': 39}
    for label, index in expressions.items():
        reset_shapes(face)
        if index is not None and face and face.data.shape_keys and index + 1 < len(face.data.shape_keys.key_blocks):
            face.data.shape_keys.key_blocks[index + 1].value = 1.0
        render_view(scene, cam, target, (target.x, target.y + dist, target.z), out / 'Preview' / f'AINA_EXPR_{label}.png', 88)
    reset_shapes(face)

    full_mn, full_mx = world_bounds(meshes)
    full_target = (full_mn + full_mx) * .5
    height = full_mx.z - full_mn.z
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = height * 1.10
    render_view(scene, cam, full_target, (full_target.x, full_target.y + 3.0, full_target.z), out / 'Preview' / 'AINA_FULLBODY_FRONT.png', 70)
    render_view(scene, cam, full_target, (full_target.x + 1.65, full_target.y + 2.35, full_target.z), out / 'Preview' / 'AINA_FULLBODY_3Q.png', 70)
    cam.data.type = 'PERSP'

    blend = out / 'AINA_MASTER_V8.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    export_sources(out)

    report = {
        'blender_version': bpy.app.version_string,
        'mesh_objects': [o.name for o in meshes],
        'armatures': [o.name for o in arms],
        'armature_bones': sum(len(a.data.bones) for a in arms),
        'shape_keys': {
            o.name: (len(o.data.shape_keys.key_blocks) - 1 if o.data.shape_keys else 0)
            for o in meshes
        },
        'materials': [m.name for m in bpy.data.materials],
        'head_bounds_min': list(mn),
        'head_bounds_max': list(mx),
        'full_bounds_min': list(full_mn),
        'full_bounds_max': list(full_mx),
        'blend_bytes': blend.stat().st_size,
        'glb_bytes': (out / 'AINA_EXPORT_V8.glb').stat().st_size,
        'fbx_bytes': (out / 'AINA_EXPORT_V8.fbx').stat().st_size,
    }
    (out / 'QA' / 'AINA_BLENDER_QA.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            p = parse_args()
            out = Path(p.get('out', '.')).resolve()
            (out / 'QA').mkdir(parents=True, exist_ok=True)
            (out / 'QA' / 'AINA_FINALIZER_ERROR.log').write_text(error, encoding='utf-8')
        except Exception:
            pass
        raise
