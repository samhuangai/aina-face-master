#!/usr/bin/env python3
from __future__ import annotations

import bpy
import json
import sys
import traceback
from pathlib import Path
from mathutils import Vector


def parse_args():
    raw = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    return {raw[index].lstrip('-'): raw[index + 1] for index in range(0, len(raw), 2)}


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def object_bounds(objects):
    points = [obj.matrix_world @ Vector(corner) for obj in objects if obj.type == 'MESH' for corner in obj.bound_box]
    if not points:
        raise RuntimeError('No mesh bounds available for AINA V9')
    return (
        Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
        Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
    )


def principled(material):
    if not material.use_nodes:
        material.use_nodes = True
    return next((node for node in material.node_tree.nodes if node.type == 'BSDF_PRINCIPLED'), None)


def set_input(shader, names, value):
    if shader is None:
        return
    for name in names:
        if name in shader.inputs:
            shader.inputs[name].default_value = value
            return


def tune_materials():
    report = {}
    for material in bpy.data.materials:
        shader = principled(material)
        lower = material.name.lower()
        if shader is None:
            continue

        # The V9 GLB already contains UV-correct textures. These values only control response to light.
        if 'face_00_skin' in lower or 'faceskin' in lower:
            set_input(shader, ('Roughness',), 0.52)
            set_input(shader, ('IOR Level', 'Specular IOR Level'), 0.23)
            set_input(shader, ('Coat Weight',), 0.035)
            set_input(shader, ('Subsurface Weight', 'Subsurface'), 0.035)
        elif 'body_00_skin' in lower:
            set_input(shader, ('Roughness',), 0.48)
            set_input(shader, ('IOR Level', 'Specular IOR Level'), 0.22)
        elif 'eyeiris' in lower:
            set_input(shader, ('Roughness',), 0.18)
            set_input(shader, ('Coat Weight',), 0.18)
            set_input(shader, ('IOR Level', 'Specular IOR Level'), 0.38)
        elif 'eyewhite' in lower:
            set_input(shader, ('Roughness',), 0.30)
            set_input(shader, ('Coat Weight',), 0.08)
            set_input(shader, ('IOR Level', 'Specular IOR Level'), 0.28)
        elif 'eyehighlight' in lower:
            set_input(shader, ('Roughness',), 0.12)
            set_input(shader, ('Emission Strength',), 0.25)
        elif 'facemouth' in lower:
            set_input(shader, ('Roughness',), 0.38)
            set_input(shader, ('Coat Weight',), 0.08)
        elif 'facebrow' in lower or 'faceeyeline' in lower:
            set_input(shader, ('Roughness',), 0.50)
        elif 'hairback' in lower or 'silver_updo' in lower:
            set_input(shader, ('Roughness',), 0.38)
            set_input(shader, ('Metallic',), 0.015)
            set_input(shader, ('Coat Weight',), 0.05)
        elif 'hairpins' in lower:
            set_input(shader, ('Base Color',), (0.15, 0.18, 0.27, 1.0))
            set_input(shader, ('Metallic',), 0.68)
            set_input(shader, ('Roughness',), 0.24)
        elif 'uniform' in lower:
            set_input(shader, ('Roughness',), 0.34)
            set_input(shader, ('Metallic',), 0.025)
        elif 'aina_core' in lower:
            set_input(shader, ('Emission Color',), (0.015, 0.24, 1.0, 1.0))
            set_input(shader, ('Emission Strength',), 3.2)
            set_input(shader, ('Roughness',), 0.16)

        report[material.name] = {
            'roughness': float(shader.inputs['Roughness'].default_value) if 'Roughness' in shader.inputs else None,
        }
    return report


def material_vertex_indices(obj, keywords):
    slot_names = []
    for slot in obj.material_slots:
        slot_names.append(slot.material.name.lower() if slot.material else '')
    indices = set()
    for polygon in obj.data.polygons:
        if polygon.material_index < len(slot_names) and any(key in slot_names[polygon.material_index] for key in keywords):
            indices.update(polygon.vertices)
    return sorted(indices)


def transform_mesh_indices(obj, indices, transform):
    if not indices:
        return 0
    inverse = obj.matrix_world.inverted()
    if obj.data.shape_keys:
        for key in obj.data.shape_keys.key_blocks:
            for index in indices:
                world = obj.matrix_world @ key.data[index].co
                key.data[index].co = inverse @ transform(world)
    else:
        for index in indices:
            world = obj.matrix_world @ obj.data.vertices[index].co
            obj.data.vertices[index].co = inverse @ transform(world)
    obj.data.update()
    return len(indices)


def refine_hair_geometry(meshes):
    report = {}
    body = next((obj for obj in meshes if obj.name.lower().startswith('body')), None)
    if body:
        hair_indices = material_vertex_indices(body, ('hairback', 'hair_00'))

        def compact_base_hair(point):
            p = point.copy()
            if p.z > 1.405:
                vertical = min(max((p.z - 1.405) / 0.225, 0.0), 1.0)
                p.x *= 1.0 - 0.105 * vertical
                p.z = 1.405 + (p.z - 1.405) * (1.0 - 0.070 * vertical)
                p.y *= 0.90
            return p

        report['base_hair_vertices'] = transform_mesh_indices(body, hair_indices, compact_base_hair)

    updo = next((obj for obj in meshes if 'aina_updo' in obj.name.lower()), None)
    if updo:
        all_indices = list(range(len(updo.data.vertices)))
        mn, mx = object_bounds([updo])
        center = (mn + mx) * 0.5

        def compact_updo(point):
            p = point.copy()
            p.x = center.x + (p.x - center.x) * 0.86
            p.y = center.y + (p.y - center.y) * 0.84 - 0.008
            p.z = center.z + (p.z - center.z) * 0.82 - 0.010
            return p

        report['updo_vertices'] = transform_mesh_indices(updo, all_indices, compact_updo)
    return report


def add_light(name, location, energy, size, target, shape='DISK'):
    bpy.ops.object.light_add(type='AREA', location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    light.data.shape = shape
    light.data.size = size
    look_at(light, target)
    return light


def render_view(scene, camera, target, location, output, lens=86, ortho_scale=None):
    if ortho_scale is None:
        camera.data.type = 'PERSP'
        camera.data.lens = lens
    else:
        camera.data.type = 'ORTHO'
        camera.data.ortho_scale = ortho_scale
    camera.location = Vector(location)
    look_at(camera, target)
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def reset_shapes(face):
    if face and face.data.shape_keys:
        for shape in face.data.shape_keys.key_blocks[1:]:
            shape.value = 0.0


def export_sources(output_dir):
    bpy.ops.export_scene.gltf(
        filepath=str(output_dir / 'AINA_EXPORT_V9.glb'),
        export_format='GLB',
        export_animations=True,
        export_cameras=False,
        export_lights=False,
    )
    bpy.ops.export_scene.fbx(
        filepath=str(output_dir / 'AINA_EXPORT_V9.fbx'),
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=False,
        path_mode='AUTO',
    )


def main():
    arguments = parse_args()
    source = Path(arguments['input']).resolve()
    output_dir = Path(arguments['out']).resolve()
    preview_dir = output_dir / 'Preview'
    qa_dir = output_dir / 'QA'
    preview_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
    if not meshes or not armatures:
        raise RuntimeError('AINA V9 requires imported mesh and armature objects')
    for obj in meshes:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True

    material_report = tune_materials()
    hair_report = refine_hair_geometry(meshes)

    head_objects = [obj for obj in meshes if any(key in obj.name.lower() for key in ('face', 'hair', 'updo'))]
    if not head_objects:
        head_objects = meshes
    head_min, head_max = object_bounds(head_objects)
    target = (head_min + head_max) * 0.5
    target.z = head_min.z + (head_max.z - head_min.z) * 0.475
    head_size = max(head_max.x - head_min.x, head_max.z - head_min.z)
    distance = max(0.62, head_size * 2.62)

    world = bpy.data.worlds.new('AINA_V9_WORLD')
    world.use_nodes = True
    bpy.context.scene.world = world
    background = world.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value = (0.012, 0.018, 0.030, 1.0)
    background.inputs['Strength'].default_value = 0.055

    add_light('AINA_KEY', (-0.48, 0.68, target.z + 0.30), 165, 1.15, target)
    add_light('AINA_FILL', (0.50, 0.50, target.z + 0.04), 52, 1.20, target)
    add_light('AINA_RIM', (0.0, -0.58, target.z + 0.30), 105, 0.92, target)
    add_light('AINA_EYE_FILL', (0.0, 0.44, target.z - 0.025), 18, 0.46, target)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = 'AINA_V9_CAMERA'
    camera.data.sensor_width = 36
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.image_settings.compression = 32
    scene.view_settings.exposure = -0.85
    try:
        scene.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        pass

    # Save and export before rendering: production source survives even if a render backend fails.
    blend_path = output_dir / 'AINA_MASTER_V9.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    export_sources(output_dir)

    render_view(scene, camera, target, (target.x, target.y + distance, target.z), preview_dir / 'AINA_V9_FRONT.png', lens=88)
    render_view(scene, camera, target, (target.x + distance * 0.54, target.y + distance * 0.84, target.z), preview_dir / 'AINA_V9_3Q.png', lens=88)
    render_view(scene, camera, target, (target.x + distance, target.y, target.z), preview_dir / 'AINA_V9_PROFILE.png', lens=88)

    face = max(
        (obj for obj in meshes if obj.data.shape_keys),
        key=lambda obj: len(obj.data.shape_keys.key_blocks),
        default=None,
    )
    expressions = {'NEUTRAL': None, 'HAPPY': 3, 'BLINK': 13, 'AA': 39}
    for label, index in expressions.items():
        reset_shapes(face)
        if index is not None and face and face.data.shape_keys and index + 1 < len(face.data.shape_keys.key_blocks):
            face.data.shape_keys.key_blocks[index + 1].value = 1.0
        render_view(
            scene,
            camera,
            target,
            (target.x, target.y + distance, target.z),
            preview_dir / f'AINA_V9_EXPR_{label}.png',
            lens=88,
        )
    reset_shapes(face)

    full_min, full_max = object_bounds(meshes)
    full_target = (full_min + full_max) * 0.5
    height = full_max.z - full_min.z
    render_view(
        scene,
        camera,
        full_target,
        (full_target.x, full_target.y + 3.1, full_target.z),
        preview_dir / 'AINA_V9_FULLBODY_FRONT.png',
        ortho_scale=height * 1.08,
    )

    # Persist the neutral state and final camera/light/material tuning in the editable master.
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    report = {
        'version': 'V9',
        'blender_version': bpy.app.version_string,
        'meshes': [obj.name for obj in meshes],
        'armatures': [obj.name for obj in armatures],
        'armature_bones': sum(len(obj.data.bones) for obj in armatures),
        'shape_keys': {
            obj.name: (len(obj.data.shape_keys.key_blocks) - 1 if obj.data.shape_keys else 0)
            for obj in meshes
        },
        'materials': material_report,
        'hair_refinement': hair_report,
        'head_bounds_min': list(head_min),
        'head_bounds_max': list(head_max),
        'full_bounds_min': list(full_min),
        'full_bounds_max': list(full_max),
        'files': {
            'blend': blend_path.stat().st_size,
            'glb': (output_dir / 'AINA_EXPORT_V9.glb').stat().st_size,
            'fbx': (output_dir / 'AINA_EXPORT_V9.fbx').stat().st_size,
        },
        'identity_lock': False,
        'visual_identity_lock': False,
    }
    (qa_dir / 'AINA_V9_BLENDER_QA.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            output = Path(parse_args().get('out', '.')).resolve()
            (output / 'QA').mkdir(parents=True, exist_ok=True)
            (output / 'QA' / 'AINA_V9_FINALIZER_ERROR.log').write_text(error, encoding='utf-8')
        except Exception:
            pass
        raise
