#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    raw = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    return {raw[index].lstrip('-'): raw[index + 1] for index in range(0, len(raw), 2)}


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def world_bounds(objects):
    points = [obj.matrix_world @ Vector(corner) for obj in objects if obj.type in {'MESH', 'CURVE'} for corner in obj.bound_box]
    if not points:
        raise RuntimeError('No AINA V10 bounds available')
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
        return False
    for name in names:
        if name in shader.inputs:
            shader.inputs[name].default_value = value
            return True
    return False


def disconnect_input(material, shader, names):
    if not material.use_nodes or shader is None:
        return
    sockets = [shader.inputs[name] for name in names if name in shader.inputs]
    for link in list(material.node_tree.links):
        if link.to_socket in sockets:
            material.node_tree.links.remove(link)


def force_transparent(material):
    shader = principled(material)
    disconnect_input(material, shader, ('Alpha',))
    set_input(shader, ('Alpha',), 0.0)
    material.diffuse_color = (0.0, 0.0, 0.0, 0.0)
    for prop, value in (
        ('surface_render_method', 'DITHERED'),
        ('blend_method', 'BLEND'),
        ('use_transparency_overlap', False),
    ):
        if hasattr(material, prop):
            try:
                setattr(material, prop, value)
            except Exception:
                pass


def tune_imported_materials():
    report = {}
    for material in bpy.data.materials:
        shader = principled(material)
        if shader is None:
            continue
        lower = material.name.lower()

        if 'aina_core' not in lower:
            disconnect_input(material, shader, ('Emission Color', 'Emission'))
            set_input(shader, ('Emission Color', 'Emission'), (0.0, 0.0, 0.0, 1.0))
            set_input(shader, ('Emission Strength',), 0.0)

        if 'faceeyeline' in lower:
            force_transparent(material)
        elif 'face_00_skin' in lower or 'faceskin' in lower:
            set_input(shader, ('Roughness',), 0.56)
            set_input(shader, ('IOR Level', 'Specular IOR Level'), 0.18)
            set_input(shader, ('Coat Weight',), 0.025)
            set_input(shader, ('Subsurface Weight', 'Subsurface'), 0.025)
        elif 'body_00_skin' in lower:
            set_input(shader, ('Roughness',), 0.52)
            set_input(shader, ('IOR Level', 'Specular IOR Level'), 0.18)
        elif 'eyeiris' in lower:
            set_input(shader, ('Roughness',), 0.20)
            set_input(shader, ('Coat Weight',), 0.12)
            set_input(shader, ('IOR Level', 'Specular IOR Level'), 0.34)
        elif 'eyewhite' in lower:
            set_input(shader, ('Roughness',), 0.34)
            set_input(shader, ('Coat Weight',), 0.04)
            set_input(shader, ('IOR Level', 'Specular IOR Level'), 0.22)
        elif 'eyehighlight' in lower:
            set_input(shader, ('Roughness',), 0.12)
            set_input(shader, ('Emission Strength',), 0.0)
        elif 'facemouth' in lower:
            set_input(shader, ('Roughness',), 0.42)
            set_input(shader, ('Coat Weight',), 0.05)
        elif 'facebrow' in lower:
            set_input(shader, ('Roughness',), 0.54)
        elif 'hairback' in lower:
            set_input(shader, ('Roughness',), 0.38)
            set_input(shader, ('Metallic',), 0.02)
            set_input(shader, ('Coat Weight',), 0.04)
        elif 'aina_core' in lower:
            disconnect_input(material, shader, ('Emission Color', 'Emission'))
            set_input(shader, ('Emission Color', 'Emission'), (0.01, 0.18, 1.0, 1.0))
            set_input(shader, ('Emission Strength',), 2.4)
            set_input(shader, ('Roughness',), 0.16)
        report[material.name] = {
            'roughness': float(shader.inputs['Roughness'].default_value) if 'Roughness' in shader.inputs else None,
            'transparent': 'faceeyeline' in lower,
        }
    return report


def make_material(name, base_color, metallic=0.0, roughness=0.4, emission=None, emission_strength=0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = principled(material)
    set_input(shader, ('Base Color',), base_color)
    set_input(shader, ('Metallic',), metallic)
    set_input(shader, ('Roughness',), roughness)
    set_input(shader, ('Coat Weight',), 0.05)
    if emission is not None:
        set_input(shader, ('Emission Color', 'Emission'), emission)
        set_input(shader, ('Emission Strength',), emission_strength)
    return material


def find_head_bone(armature):
    preferred = ('head', 'j_bip_c_head', 'head.x')
    lower_map = {bone.name.lower(): bone.name for bone in armature.data.bones}
    for key in preferred:
        if key in lower_map:
            return lower_map[key]
    for bone in armature.data.bones:
        if 'head' in bone.name.lower():
            return bone.name
    return ''


def parent_to_head(obj, armature, head_bone):
    if armature is None or not head_bone:
        return
    matrix_world = obj.matrix_world.copy()
    obj.parent = armature
    obj.parent_type = 'BONE'
    obj.parent_bone = head_bone
    obj.matrix_world = matrix_world


def curve_object(name, points, material, bevel_depth=0.0008, cyclic=False, resolution=3):
    curve = bpy.data.curves.new(name, type='CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = resolution
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 3
    curve.fill_mode = 'FULL'
    curve.use_fill_caps = True
    spline = curve.splines.new('BEZIER')
    spline.bezier_points.add(len(points) - 1)
    for point, coordinate in zip(spline.bezier_points, points):
        point.co = Vector(coordinate)
        point.handle_left_type = 'AUTO'
        point.handle_right_type = 'AUTO'
    spline.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def sphere_object(name, location, scale, material, rotation=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=40, ring_count=24, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def torus_object(name, location, major_radius, minor_radius, material, rotation=(math.pi / 2.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_torus_add(
        align='WORLD',
        major_segments=64,
        minor_segments=8,
        location=location,
        rotation=rotation,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def material_vertex_indices(obj, keywords):
    slot_names = [slot.material.name.lower() if slot.material else '' for slot in obj.material_slots]
    indices = set()
    for polygon in obj.data.polygons:
        if polygon.material_index < len(slot_names) and any(key in slot_names[polygon.material_index] for key in keywords):
            indices.update(polygon.vertices)
    return sorted(indices)


def world_points(obj, indices):
    return [obj.matrix_world @ obj.data.vertices[index].co for index in indices]


def group_bounds(obj, indices):
    points = world_points(obj, indices)
    if not points:
        return None
    return (
        Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
        Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
    )


def transform_indices(obj, indices, transform):
    if not indices:
        return 0
    inverse = obj.matrix_world.inverted()
    key_blocks = obj.data.shape_keys.key_blocks if obj.data.shape_keys else None
    if key_blocks:
        for key in key_blocks:
            for index in indices:
                world = obj.matrix_world @ key.data[index].co
                key.data[index].co = inverse @ transform(world)
    else:
        for index in indices:
            world = obj.matrix_world @ obj.data.vertices[index].co
            obj.data.vertices[index].co = inverse @ transform(world)
    obj.data.update()
    return len(indices)


def compact_base_hair(meshes, eye_z, head_center):
    body = next((obj for obj in meshes if obj.name.lower().startswith('body')), None)
    if body is None:
        return 0
    hair_indices = material_vertex_indices(body, ('hairback', 'hair_00'))

    def compact(point):
        p = point.copy()
        upper = min(max((p.z - (eye_z + 0.025)) / 0.150, 0.0), 1.0)
        p.x = head_center.x + (p.x - head_center.x) * (1.0 - 0.16 * upper)
        p.y = head_center.y + (p.y - head_center.y) * (0.92 - 0.08 * upper) - 0.002 * upper
        p.z = eye_z + 0.025 + (p.z - (eye_z + 0.025)) * (1.0 - 0.13 * upper)
        return p

    return transform_indices(body, hair_indices, compact)


def build_facial_overlays(face, armature, head_bone):
    lash_material = make_material('AINA_V10_LASH', (0.022, 0.014, 0.020, 1.0), roughness=0.48)
    brow_material = make_material('AINA_V10_BROW', (0.070, 0.045, 0.050, 1.0), roughness=0.52)
    lip_material = make_material('AINA_V10_LIP', (0.36, 0.075, 0.095, 1.0), roughness=0.38)

    eye_indices = material_vertex_indices(face, ('eyewhite',))
    brow_indices = material_vertex_indices(face, ('facebrow',))
    mouth_indices = material_vertex_indices(face, ('facemouth',))
    objects = []

    for sign in (-1.0, 1.0):
        side_eye = [index for index in eye_indices if (face.matrix_world @ face.data.vertices[index].co).x * sign > 0]
        bounds = group_bounds(face, side_eye)
        if bounds is None:
            continue
        minimum, maximum = bounds
        center_x = (minimum.x + maximum.x) * 0.5
        half_width = (maximum.x - minimum.x) * 0.50
        eye_top = maximum.z
        front_y = maximum.y + 0.0015
        inner_x = center_x - sign * half_width * 0.92
        outer_x = center_x + sign * half_width * 0.98
        points = [
            (inner_x, front_y, eye_top - 0.0002),
            (center_x - sign * half_width * 0.32, front_y + 0.0005, eye_top + 0.0015),
            (center_x + sign * half_width * 0.30, front_y + 0.0007, eye_top + 0.0018),
            (outer_x, front_y + 0.0002, eye_top + 0.0005),
            (outer_x + sign * 0.0034, front_y - 0.0003, eye_top + 0.0015),
        ]
        objects.append(curve_object(f'AINA_V10_LASH_{int(sign)}', points, lash_material, bevel_depth=0.00072))

        side_brow = [index for index in brow_indices if (face.matrix_world @ face.data.vertices[index].co).x * sign > 0]
        brow_bounds = group_bounds(face, side_brow)
        if brow_bounds:
            bmin, bmax = brow_bounds
            bx = (bmin.x + bmax.x) * 0.5
            bw = (bmax.x - bmin.x) * 0.45
            bz = (bmin.z + bmax.z) * 0.5
            by = bmax.y + 0.0012
            brow_points = [
                (bx - sign * bw, by, bz - 0.0008),
                (bx - sign * bw * 0.38, by + 0.0003, bz + 0.0016),
                (bx + sign * bw * 0.30, by + 0.0003, bz + 0.0022),
                (bx + sign * bw, by, bz + 0.0002),
            ]
            objects.append(curve_object(f'AINA_V10_BROW_{int(sign)}', brow_points, brow_material, bevel_depth=0.00062))

    mouth_bounds = group_bounds(face, mouth_indices)
    if mouth_bounds:
        minimum, maximum = mouth_bounds
        center_x = (minimum.x + maximum.x) * 0.5
        center_z = (minimum.z + maximum.z) * 0.5
        half_width = min((maximum.x - minimum.x) * 0.43, 0.025)
        front_y = maximum.y + 0.0013
        upper = [
            (center_x - half_width, front_y, center_z),
            (center_x - half_width * 0.48, front_y + 0.0003, center_z + 0.0011),
            (center_x, front_y + 0.0004, center_z + 0.0018),
            (center_x + half_width * 0.48, front_y + 0.0003, center_z + 0.0011),
            (center_x + half_width, front_y, center_z),
        ]
        lower = [
            (center_x - half_width * 0.90, front_y - 0.0001, center_z - 0.0004),
            (center_x, front_y + 0.0005, center_z - 0.0020),
            (center_x + half_width * 0.90, front_y - 0.0001, center_z - 0.0004),
        ]
        objects.append(curve_object('AINA_V10_LIP_UPPER', upper, lip_material, bevel_depth=0.00060))
        objects.append(curve_object('AINA_V10_LIP_LOWER', lower, lip_material, bevel_depth=0.00054))

    for obj in objects:
        parent_to_head(obj, armature, head_bone)
    return objects


def build_hair_system(head_min, head_max, eye_z, armature, head_bone):
    hair_material = make_material('AINA_V10_HAIR', (0.30, 0.32, 0.43, 1.0), metallic=0.055, roughness=0.32)
    strand_material = make_material('AINA_V10_HAIR_STRAND', (0.40, 0.42, 0.54, 1.0), metallic=0.035, roughness=0.30)
    metal_material = make_material('AINA_V10_HAIRPIN', (0.24, 0.28, 0.39, 1.0), metallic=0.78, roughness=0.20)

    center_x = (head_min.x + head_max.x) * 0.5
    center_y = (head_min.y + head_max.y) * 0.5
    top_z = head_max.z
    face_front = head_max.y
    objects = []

    bun_center = Vector((center_x, center_y - 0.020, top_z + 0.033))
    lobe_specs = [
        ((0.000, 0.000, 0.012), (0.034, 0.028, 0.045), (0.10, 0.00, 0.00)),
        ((-0.023, 0.003, 0.003), (0.026, 0.023, 0.035), (0.25, 0.20, -0.36)),
        ((0.023, 0.003, 0.003), (0.026, 0.023, 0.035), (-0.25, -0.20, 0.36)),
        ((-0.012, -0.008, 0.029), (0.022, 0.020, 0.030), (0.45, 0.10, 0.40)),
        ((0.012, -0.008, 0.029), (0.022, 0.020, 0.030), (-0.45, -0.10, -0.40)),
    ]
    for index, (offset, scale, rotation) in enumerate(lobe_specs):
        location = bun_center + Vector(offset)
        objects.append(sphere_object(f'AINA_V10_BUN_{index}', location, scale, hair_material, rotation))

    objects.append(torus_object('AINA_V10_BUN_RING', bun_center + Vector((0.0, 0.004, -0.004)), 0.044, 0.0013, metal_material))
    ring_points = []
    for index in range(13):
        angle = -1.08 + index * (2.16 / 12.0)
        ring_points.append((
            center_x + 0.070 * math.sin(angle),
            face_front * 0.18 + 0.003,
            top_z - 0.005 + 0.018 * math.cos(angle),
        ))
    objects.append(curve_object('AINA_V10_HEADBAND', ring_points, metal_material, bevel_depth=0.00105))

    fringe = [
        (-0.042, -0.030, 0.050),
        (-0.030, -0.025, 0.040),
        (-0.018, -0.015, 0.028),
        (-0.008, -0.004, 0.022),
        (0.008, 0.004, 0.022),
        (0.018, 0.015, 0.028),
        (0.030, 0.025, 0.040),
        (0.042, 0.030, 0.050),
    ]
    for index, (start_x, end_x, end_drop) in enumerate(fringe):
        start = (center_x + start_x * 0.60, center_y + 0.015, top_z - 0.010)
        middle = (center_x + start_x * 0.95, face_front * 0.72, top_z - 0.038)
        end = (center_x + end_x, face_front + 0.002, eye_z + end_drop)
        objects.append(curve_object(f'AINA_V10_FRINGE_{index}', (start, middle, end), strand_material, bevel_depth=0.0015 - abs(start_x) * 0.009))

    for sign in (-1.0, 1.0):
        side_points = [
            (center_x + sign * 0.050, center_y + 0.008, top_z - 0.035),
            (center_x + sign * 0.061, face_front * 0.78, eye_z + 0.050),
            (center_x + sign * 0.065, face_front + 0.004, eye_z - 0.010),
            (center_x + sign * 0.060, face_front + 0.002, eye_z - 0.075),
        ]
        objects.append(curve_object(f'AINA_V10_SIDELOCK_{int(sign)}', side_points, strand_material, bevel_depth=0.00135))
        wisp_points = [
            (center_x + sign * 0.057, center_y + 0.002, top_z - 0.055),
            (center_x + sign * 0.071, face_front * 0.60, eye_z + 0.020),
            (center_x + sign * 0.073, face_front + 0.001, eye_z - 0.055),
        ]
        objects.append(curve_object(f'AINA_V10_WISP_{int(sign)}', wisp_points, strand_material, bevel_depth=0.00072))

    for obj in objects:
        parent_to_head(obj, armature, head_bone)
    return objects


def hide_old_updo():
    hidden = []
    for obj in bpy.context.scene.objects:
        lower = obj.name.lower()
        if 'aina_updo' in lower or 'aina_hairpins' in lower:
            obj.hide_render = True
            obj.hide_viewport = True
            hidden.append(obj.name)
    return hidden


def add_area_light(name, location, energy, size, target):
    bpy.ops.object.light_add(type='AREA', location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    light.data.shape = 'DISK'
    light.data.size = size
    look_at(light, target)
    return light


def render_view(scene, camera, target, location, output, lens=90):
    camera.data.type = 'PERSP'
    camera.data.lens = lens
    camera.location = Vector(location)
    look_at(camera, target)
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def convert_duplicate_curves_for_export(objects):
    duplicates = []
    for obj in objects:
        if obj.type != 'CURVE':
            continue
        duplicate = obj.copy()
        duplicate.data = obj.data.copy()
        bpy.context.collection.objects.link(duplicate)
        duplicate.name = obj.name + '_EXPORT'
        matrix_world = duplicate.matrix_world.copy()
        duplicate.parent = None
        duplicate.matrix_world = matrix_world
        bpy.context.view_layer.objects.active = duplicate
        duplicate.select_set(True)
        bpy.ops.object.convert(target='MESH')
        duplicate.select_set(False)
        duplicates.append(duplicate)
    return duplicates


def export_sources(output_dir, procedural_objects):
    duplicates = convert_duplicate_curves_for_export(procedural_objects)
    for obj in procedural_objects:
        if obj.type == 'CURVE':
            obj.hide_render = True
    bpy.ops.export_scene.gltf(
        filepath=str(output_dir / 'AINA_EXPORT_V10.glb'),
        export_format='GLB',
        export_animations=True,
        export_cameras=False,
        export_lights=False,
    )
    bpy.ops.export_scene.fbx(
        filepath=str(output_dir / 'AINA_EXPORT_V10.fbx'),
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=False,
        path_mode='AUTO',
    )
    for obj in procedural_objects:
        if obj.type == 'CURVE':
            obj.hide_render = False
    for obj in duplicates:
        bpy.data.objects.remove(obj, do_unlink=True)


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
    armature = next((obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE'), None)
    if not meshes or armature is None:
        raise RuntimeError('AINA V10 import did not provide mesh and armature')
    for obj in meshes:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True

    material_report = tune_imported_materials()
    hidden_updo = hide_old_updo()
    face = next((obj for obj in meshes if obj.name.lower().startswith('face')), None)
    if face is None:
        raise RuntimeError('AINA V10 face object is missing')

    eye_indices = material_vertex_indices(face, ('eyewhite',))
    eye_bounds = group_bounds(face, eye_indices)
    if eye_bounds is None:
        raise RuntimeError('AINA V10 eye landmarks could not be resolved')
    eye_z = (eye_bounds[0].z + eye_bounds[1].z) * 0.5
    preliminary_min, preliminary_max = world_bounds([face])
    head_center = (preliminary_min + preliminary_max) * 0.5
    base_hair_vertices = compact_base_hair(meshes, eye_z, head_center)

    head_bone = find_head_bone(armature)
    overlays = build_facial_overlays(face, armature, head_bone)
    post_face_min, post_face_max = world_bounds([face])
    hair_objects = build_hair_system(post_face_min, post_face_max, eye_z, armature, head_bone)
    procedural = overlays + hair_objects

    review_objects = [face] + [obj for obj in meshes if 'body' in obj.name.lower()] + hair_objects
    head_min, head_max = world_bounds(review_objects)
    target = (head_min + head_max) * 0.5
    target.z = eye_z - 0.005
    size = max(head_max.x - head_min.x, head_max.z - head_min.z)
    distance = max(0.62, size * 2.45)

    world = bpy.data.worlds.new('AINA_V10_WORLD')
    world.use_nodes = True
    bpy.context.scene.world = world
    background = world.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value = (0.055, 0.065, 0.085, 1.0)
    background.inputs['Strength'].default_value = 0.18

    add_area_light('AINA_V10_KEY', (-0.45, 0.70, target.z + 0.26), 120, 1.15, target)
    add_area_light('AINA_V10_FILL', (0.47, 0.50, target.z + 0.03), 42, 1.25, target)
    add_area_light('AINA_V10_RIM', (0.0, -0.58, target.z + 0.28), 82, 0.95, target)
    add_area_light('AINA_V10_EYE', (0.0, 0.50, eye_z), 10, 0.40, target)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = 'AINA_V10_CAMERA'
    camera.data.sensor_width = 36
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 420
    scene.render.resolution_y = 520
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.compression = 28
    scene.render.film_transparent = False
    scene.view_settings.exposure = -0.30
    try:
        scene.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        pass

    blend_path = output_dir / 'AINA_MASTER_V10.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    export_sources(output_dir, procedural)

    render_view(scene, camera, target, (target.x, target.y + distance, target.z), preview_dir / 'AINA_V10_FRONT.png')
    render_view(scene, camera, target, (target.x + distance * 0.52, target.y + distance * 0.85, target.z), preview_dir / 'AINA_V10_3Q.png')
    render_view(scene, camera, target, (target.x + distance, target.y, target.z), preview_dir / 'AINA_V10_PROFILE.png')

    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    report = {
        'version': 'V10',
        'blender_version': bpy.app.version_string,
        'mesh_objects': [obj.name for obj in meshes],
        'armature': armature.name,
        'armature_bones': len(armature.data.bones),
        'head_bone': head_bone,
        'max_shape_keys': max((len(obj.data.shape_keys.key_blocks) - 1 for obj in meshes if obj.data.shape_keys), default=0),
        'hidden_old_updo': hidden_updo,
        'base_hair_vertices_refined': base_hair_vertices,
        'procedural_objects': [obj.name for obj in procedural],
        'materials': material_report,
        'head_bounds_min': list(head_min),
        'head_bounds_max': list(head_max),
        'files': {
            'blend': blend_path.stat().st_size,
            'glb': (output_dir / 'AINA_EXPORT_V10.glb').stat().st_size,
            'fbx': (output_dir / 'AINA_EXPORT_V10.fbx').stat().st_size,
        },
        'identity_lock': False,
        'visual_identity_lock': False,
    }
    (qa_dir / 'AINA_V10_BLENDER_QA.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
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
            (output / 'QA' / 'AINA_V10_FINALIZER_ERROR.log').write_text(error, encoding='utf-8')
        except Exception:
            pass
        raise
