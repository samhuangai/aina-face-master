#!/usr/bin/env python3
from __future__ import annotations
import bpy, sys, json, traceback
from pathlib import Path
from mathutils import Vector


def args():
    raw = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    return {raw[i].lstrip('-'): raw[i + 1] for i in range(0, len(raw), 2)}


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def bounds(objects):
    pts = [o.matrix_world @ Vector(c) for o in objects if o.type == 'MESH' for c in o.bound_box]
    if not pts:
        raise RuntimeError('No AINA mesh bounds')
    return (
        Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts))),
        Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts))),
    )


def hide_anime_overlays():
    keys = ('facemouth', 'eyeiris', 'eyehighlight', 'eyewhite', 'facebrow', 'faceeyeline')
    for material in bpy.data.materials:
        if not any(key in material.name.lower() for key in keys):
            continue
        material.diffuse_color = (1, 1, 1, 0)
        if material.use_nodes:
            bsdf = next((n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if bsdf and 'Alpha' in bsdf.inputs:
                bsdf.inputs['Alpha'].default_value = 0
        for prop, value in (('surface_render_method', 'DITHERED'), ('blend_method', 'BLEND')):
            if hasattr(material, prop):
                try:
                    setattr(material, prop, value)
                except Exception:
                    pass


def reset_shapes(face):
    if face and face.data.shape_keys:
        for key in face.data.shape_keys.key_blocks[1:]:
            key.value = 0


def render(scene, camera, target, location, output, lens=82):
    camera.data.lens = lens
    camera.location = Vector(location)
    look_at(camera, target)
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def main():
    p = args()
    source = Path(p['input']).resolve()
    out = Path(p['out']).resolve()
    (out / 'Preview').mkdir(parents=True, exist_ok=True)
    (out / 'QA').mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    armatures = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    if not meshes or not armatures:
        raise RuntimeError('AINA fast finalizer requires Mesh and Armature objects')
    for obj in meshes:
        for poly in obj.data.polygons:
            poly.use_smooth = True
    hide_anime_overlays()

    # Save and export before rendering so the editable deliverables exist even if rendering fails.
    blend = out / 'AINA_MASTER_V8.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    bpy.ops.export_scene.gltf(
        filepath=str(out / 'AINA_EXPORT_V8.glb'),
        export_format='GLB',
        export_animations=True,
        export_cameras=False,
        export_lights=False,
    )
    bpy.ops.export_scene.fbx(
        filepath=str(out / 'AINA_EXPORT_V8.fbx'),
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=False,
        path_mode='AUTO',
    )

    head = [o for o in meshes if any(k in o.name.lower() for k in ('face', 'hair', 'updo'))] or meshes
    mn, mx = bounds(head)
    target = (mn + mx) * .5
    target.z = mn.z + (mx.z - mn.z) * .48
    size = max(mx.x - mn.x, mx.z - mn.z)
    distance = max(.62, size * 2.55)

    world = bpy.data.worlds.new('AINA_FAST_WORLD')
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value = (.018, .025, .042, 1)
    background.inputs['Strength'].default_value = .30

    for name, location, energy, light_size in (
        ('KEY', (-.50, .62, target.z + .32), 950, 1.25),
        ('FILL', (.48, .42, target.z + .08), 430, 1.10),
        ('RIM', (0, -.48, target.z + .28), 650, .9),
    ):
        bpy.ops.object.light_add(type='AREA', location=location)
        light = bpy.context.object
        light.name = 'LGT_FAST_' + name
        light.data.energy = energy
        light.data.shape = 'DISK'
        light.data.size = light_size
        look_at(light, target)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = 'CAM_AINA_FAST'
    camera.data.sensor_width = 36
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '8'
    scene.render.film_transparent = False
    try:
        scene.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        pass

    render(scene, camera, target, (target.x, target.y + distance, target.z), out / 'Preview' / 'AINA_PORTRAIT_FRONT.png', 86)
    render(scene, camera, target, (target.x + distance * .58, target.y + distance * .82, target.z), out / 'Preview' / 'AINA_PORTRAIT_3Q.png', 86)
    render(scene, camera, target, (target.x + distance, target.y, target.z), out / 'Preview' / 'AINA_PORTRAIT_PROFILE.png', 86)

    face = max((o for o in meshes if o.data.shape_keys), key=lambda o: len(o.data.shape_keys.key_blocks), default=None)
    for label, index in {'NEUTRAL': None, 'HAPPY': 3, 'BLINK': 13, 'AA': 39}.items():
        reset_shapes(face)
        if index is not None and face and face.data.shape_keys and index + 1 < len(face.data.shape_keys.key_blocks):
            face.data.shape_keys.key_blocks[index + 1].value = 1.0
        render(scene, camera, target, (target.x, target.y + distance, target.z), out / 'Preview' / f'AINA_EXPR_{label}.png', 86)
    reset_shapes(face)

    full_min, full_max = bounds(meshes)
    full_target = (full_min + full_max) * .5
    height = full_max.z - full_min.z
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = height * 1.10
    render(scene, camera, full_target, (full_target.x, full_target.y + 3.0, full_target.z), out / 'Preview' / 'AINA_FULLBODY_FRONT.png', 70)
    render(scene, camera, full_target, (full_target.x + 1.65, full_target.y + 2.35, full_target.z), out / 'Preview' / 'AINA_FULLBODY_3Q.png', 70)

    report = {
        'mode': 'fast_real_blender_review',
        'blender_version': bpy.app.version_string,
        'mesh_objects': [o.name for o in meshes],
        'armatures': [o.name for o in armatures],
        'armature_bones': sum(len(a.data.bones) for a in armatures),
        'max_shape_keys': max((len(o.data.shape_keys.key_blocks) - 1 for o in meshes if o.data.shape_keys), default=0),
        'blend_bytes': blend.stat().st_size,
        'glb_bytes': (out / 'AINA_EXPORT_V8.glb').stat().st_size,
        'fbx_bytes': (out / 'AINA_EXPORT_V8.fbx').stat().st_size,
    }
    (out / 'QA' / 'AINA_FAST_QA.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            p = args()
            out = Path(p.get('out', '.'))
            (out / 'QA').mkdir(parents=True, exist_ok=True)
            (out / 'QA' / 'AINA_FAST_ERROR.log').write_text(error, encoding='utf-8')
        except Exception:
            pass
        raise
