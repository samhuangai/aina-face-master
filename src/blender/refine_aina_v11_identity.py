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
    return {raw[i].lstrip('-'): raw[i + 1] for i in range(0, len(raw), 2)}


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def principled(material):
    if material is None:
        return None
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


def disconnect(material, shader, names):
    if material is None or shader is None or not material.use_nodes:
        return
    sockets = [shader.inputs[name] for name in names if name in shader.inputs]
    for link in list(material.node_tree.links):
        if link.to_socket in sockets:
            material.node_tree.links.remove(link)


def make_material(name, color, metallic=0.0, roughness=0.45, emission=None, emission_strength=0.0):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    shader = principled(material)
    disconnect(material, shader, ('Base Color', 'Metallic', 'Roughness', 'Emission Color', 'Emission', 'Emission Strength', 'Alpha'))
    set_input(shader, ('Base Color',), color)
    set_input(shader, ('Metallic',), metallic)
    set_input(shader, ('Roughness',), roughness)
    set_input(shader, ('IOR Level', 'Specular IOR Level'), 0.24)
    set_input(shader, ('Coat Weight',), 0.035)
    set_input(shader, ('Alpha',), color[3])
    if emission is not None:
        set_input(shader, ('Emission Color', 'Emission'), emission)
        set_input(shader, ('Emission Strength',), emission_strength)
    material.diffuse_color = color
    material.use_backface_culling = False
    return material


def hide_material(material):
    shader = principled(material)
    disconnect(material, shader, ('Alpha',))
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


def replace_face_skin_material():
    skin = next((m for m in bpy.data.materials if 'face_00_skin' in m.name.lower() or 'faceskin' in m.name.lower()), None)
    if skin is None:
        return make_material('AINA_V11_SKIN', (0.52, 0.37, 0.34, 1.0), roughness=0.52)
    shader = principled(skin)
    disconnect(skin, shader, ('Base Color', 'Alpha', 'Emission Color', 'Emission', 'Emission Strength'))
    set_input(shader, ('Base Color',), (0.52, 0.37, 0.34, 1.0))
    set_input(shader, ('Alpha',), 1.0)
    set_input(shader, ('Roughness',), 0.52)
    set_input(shader, ('IOR Level', 'Specular IOR Level'), 0.18)
    set_input(shader, ('Coat Weight',), 0.02)
    set_input(shader, ('Subsurface Weight', 'Subsurface'), 0.035)
    set_input(shader, ('Emission Strength',), 0.0)
    skin.diffuse_color = (0.52, 0.37, 0.34, 1.0)
    return skin


def tint_existing_hair():
    changed = []
    for material in bpy.data.materials:
        lower = material.name.lower()
        if 'hairback' not in lower:
            continue
        shader = principled(material)
        if shader is None:
            continue
        socket = shader.inputs.get('Base Color')
        if socket is not None and socket.is_linked:
            old = socket.links[0]
            source = old.from_socket
            material.node_tree.links.remove(old)
            mix = material.node_tree.nodes.new('ShaderNodeMixRGB')
            mix.name = 'AINA_V11_HAIR_TINT'
            mix.blend_type = 'MULTIPLY'
            mix.inputs[0].default_value = 1.0
            mix.inputs[2].default_value = (0.34, 0.38, 0.52, 1.0)
            material.node_tree.links.new(source, mix.inputs[1])
            material.node_tree.links.new(mix.outputs[0], socket)
        else:
            set_input(shader, ('Base Color',), (0.24, 0.27, 0.38, 1.0))
        set_input(shader, ('Roughness',), 0.34)
        set_input(shader, ('Metallic',), 0.035)
        set_input(shader, ('Emission Strength',), 0.0)
        changed.append(material.name)
    return changed


def hide_old_visual_layers():
    hidden_objects = []
    for obj in bpy.data.objects:
        lower = obj.name.lower()
        if lower.startswith('aina_v10_') or 'aina_updo' in lower:
            obj.hide_render = True
            obj.hide_viewport = True
            hidden_objects.append(obj.name)
    hidden_materials = []
    for material in bpy.data.materials:
        lower = material.name.lower()
        if any(key in lower for key in ('eyewhite', 'eyeiris', 'eyehighlight', 'facebrow', 'faceeyeline', 'facemouth')):
            hide_material(material)
            hidden_materials.append(material.name)
    return hidden_objects, hidden_materials


def material_vertex_indices(obj, keywords):
    names = [slot.material.name.lower() if slot.material else '' for slot in obj.material_slots]
    indices = set()
    for polygon in obj.data.polygons:
        if polygon.material_index < len(names) and any(key in names[polygon.material_index] for key in keywords):
            indices.update(polygon.vertices)
    return sorted(indices)


def group_bounds(obj, indices):
    points = [obj.matrix_world @ obj.data.vertices[index].co for index in indices]
    if not points:
        return None
    return (
        Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
        Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
    )


def transform_face_to_adult(face, eye_z):
    inverse = face.matrix_world.inverted()
    keys = face.data.shape_keys.key_blocks if face.data.shape_keys else []
    if not keys:
        keys = [None]
    for key in keys:
        for index in range(len(face.data.vertices)):
            local = key.data[index].co if key is not None else face.data.vertices[index].co
            point = face.matrix_world @ local
            point.x *= 0.925
            delta = point.z - eye_z
            if delta < 0.0:
                point.z = eye_z + delta * 1.075
                lower = min(max((-delta) / 0.120, 0.0), 1.0)
                point.x *= 1.0 - 0.035 * lower
            else:
                point.z = eye_z + delta * 0.955
                upper = min(max(delta / 0.090, 0.0), 1.0)
                point.x *= 1.0 - 0.035 * upper
            new_local = inverse @ point
            if key is not None:
                key.data[index].co = new_local
            else:
                face.data.vertices[index].co = new_local
    face.data.update()


def mesh_bounds(obj):
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
        Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
    )


def find_bone(armature, candidates):
    lower = {bone.name.lower(): bone.name for bone in armature.data.bones}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    for bone in armature.data.bones:
        name = bone.name.lower()
        if any(candidate.lower() in name for candidate in candidates):
            return bone.name
    return ''


def parent_to_bone(obj, armature, bone_name):
    if armature is None or not bone_name:
        return
    matrix = obj.matrix_world.copy()
    obj.parent = armature
    obj.parent_type = 'BONE'
    obj.parent_bone = bone_name
    obj.matrix_world = matrix


def smooth_object(obj):
    if obj.type == 'MESH':
        for polygon in obj.data.polygons:
            polygon.use_smooth = True


def sphere_object(name, location, scale, material, segments=40, rings=24):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    smooth_object(obj)
    return obj


def curve_object(name, points, material, bevel=0.0007, cyclic=False):
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 4
    curve.bevel_depth = bevel
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


def almond_object(name, center_x, center_z, front_y, half_width, half_height, material, bulge=0.0018, segments=40):
    vertices = [(center_x, front_y + bulge, center_z)]
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        cosine = math.cos(angle)
        sine = math.sin(angle)
        x = center_x + half_width * cosine
        z = center_z + half_height * math.copysign(abs(sine) ** 1.28, sine)
        edge = max(0.0, 1.0 - cosine * cosine - sine * sine)
        vertices.append((x, front_y + bulge * edge * 0.20, z))
    faces = []
    for index in range(segments):
        a = 1 + index
        b = 1 + ((index + 1) % segments)
        faces.append((0, b, a))
    mesh = bpy.data.meshes.new(name + '_MESH')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    smooth_object(obj)
    return obj


def ribbon_object(name, points, widths, material, thickness=0.0008):
    if len(points) != len(widths) or len(points) < 2:
        raise ValueError('Ribbon needs matching point and width arrays')
    points = [Vector(point) for point in points]
    vertices = []
    for index, point in enumerate(points):
        previous = points[max(0, index - 1)]
        following = points[min(len(points) - 1, index + 1)]
        tangent = following - previous
        side = Vector((tangent.z, 0.0, -tangent.x))
        if side.length < 1e-8:
            side = Vector((1.0, 0.0, 0.0))
        side.normalize()
        vertices.append(tuple(point - side * widths[index]))
        vertices.append(tuple(point + side * widths[index]))
    faces = []
    for index in range(len(points) - 1):
        a = index * 2
        faces.append((a, a + 1, a + 3, a + 2))
    mesh = bpy.data.meshes.new(name + '_MESH')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    solidify = obj.modifiers.new('AINA_V11_RIBBON_THICKNESS', 'SOLIDIFY')
    solidify.thickness = thickness
    solidify.offset = 0.0
    bevel = obj.modifiers.new('AINA_V11_RIBBON_BEVEL', 'BEVEL')
    bevel.width = 0.0007
    bevel.segments = 2
    smooth_object(obj)
    return obj


def build_v11_eyes(face, armature, head_bone, eye_material_indices, face_front, skin_material):
    white_material = make_material('AINA_V11_EYE_WHITE', (0.62, 0.65, 0.70, 1.0), roughness=0.30)
    iris_outer = make_material('AINA_V11_IRIS_OUTER', (0.045, 0.075, 0.115, 1.0), roughness=0.20)
    iris_inner = make_material('AINA_V11_IRIS_INNER', (0.18, 0.30, 0.42, 1.0), roughness=0.18)
    pupil_material = make_material('AINA_V11_PUPIL', (0.006, 0.009, 0.015, 1.0), roughness=0.16)
    highlight_material = make_material('AINA_V11_EYE_GLINT', (0.82, 0.88, 1.0, 1.0), roughness=0.08, emission=(0.35, 0.48, 0.75, 1.0), emission_strength=0.28)
    lash_material = make_material('AINA_V11_LASH', (0.025, 0.018, 0.025, 1.0), roughness=0.46)
    lower_material = make_material('AINA_V11_LOWER_LID', (0.29, 0.16, 0.16, 1.0), roughness=0.55)
    brow_material = make_material('AINA_V11_BROW', (0.075, 0.050, 0.055, 1.0), roughness=0.54)

    objects = []
    centers = []
    for sign in (-1.0, 1.0):
        side = [index for index in eye_material_indices if (face.matrix_world @ face.data.vertices[index].co).x * sign > 0.0]
        bounds = group_bounds(face, side)
        if bounds is None:
            continue
        minimum, maximum = bounds
        center_x = (minimum.x + maximum.x) * 0.5
        center_z = (minimum.z + maximum.z) * 0.5
        width = maximum.x - minimum.x
        centers.append((center_x, center_z))

        patch = sphere_object(
            f'AINA_V11_EYE_SOCKET_PATCH_{int(sign)}',
            (center_x, face_front - 0.0045, center_z),
            (max(width * 0.92, 0.0245), 0.0070, 0.0150),
            skin_material,
            segments=48,
            rings=28,
        )
        objects.append(patch)

        eye_front = face_front + 0.0015
        white = almond_object(f'AINA_V11_EYE_WHITE_{int(sign)}', center_x, center_z, eye_front, 0.0182, 0.0047, white_material)
        objects.append(white)

        iris_y = eye_front + 0.0019
        outer = sphere_object(f'AINA_V11_IRIS_OUTER_{int(sign)}', (center_x, iris_y, center_z - 0.0001), (0.0052, 0.0012, 0.00465), iris_outer, 32, 20)
        inner = sphere_object(f'AINA_V11_IRIS_INNER_{int(sign)}', (center_x, iris_y + 0.0007, center_z - 0.0001), (0.00375, 0.00075, 0.00345), iris_inner, 32, 20)
        pupil = sphere_object(f'AINA_V11_PUPIL_{int(sign)}', (center_x, iris_y + 0.0011, center_z - 0.0001), (0.00175, 0.00048, 0.00225), pupil_material, 28, 16)
        glint = sphere_object(f'AINA_V11_GLINT_{int(sign)}', (center_x - sign * 0.0015, iris_y + 0.00155, center_z + 0.0017), (0.00078, 0.00030, 0.00078), highlight_material, 20, 12)
        objects.extend((outer, inner, pupil, glint))

        inner_x = center_x - sign * 0.0180
        outer_x = center_x + sign * 0.0180
        upper = curve_object(
            f'AINA_V11_UPPER_LID_{int(sign)}',
            (
                (inner_x, eye_front + 0.0020, center_z + 0.0000),
                (center_x - sign * 0.0060, eye_front + 0.0024, center_z + 0.0046),
                (center_x + sign * 0.0070, eye_front + 0.0024, center_z + 0.0042),
                (outer_x, eye_front + 0.0019, center_z + 0.0002),
                (outer_x + sign * 0.0032, eye_front + 0.0017, center_z + 0.0012),
            ),
            lash_material,
            bevel=0.00072,
        )
        lower = curve_object(
            f'AINA_V11_LOWER_LID_{int(sign)}',
            (
                (inner_x + sign * 0.0015, eye_front + 0.0016, center_z - 0.0002),
                (center_x, eye_front + 0.0018, center_z - 0.0044),
                (outer_x - sign * 0.0015, eye_front + 0.0016, center_z - 0.0001),
            ),
            lower_material,
            bevel=0.00030,
        )
        brow = curve_object(
            f'AINA_V11_BROW_{int(sign)}',
            (
                (center_x - sign * 0.0190, eye_front + 0.0011, center_z + 0.0200),
                (center_x - sign * 0.0070, eye_front + 0.0018, center_z + 0.0225),
                (center_x + sign * 0.0080, eye_front + 0.0018, center_z + 0.0215),
                (center_x + sign * 0.0200, eye_front + 0.0010, center_z + 0.0185),
            ),
            brow_material,
            bevel=0.00082,
        )
        objects.extend((upper, lower, brow))

    for obj in objects:
        parent_to_bone(obj, armature, head_bone)
    return objects, centers


def build_v11_mouth_and_nose(face, armature, head_bone, mouth_indices, skin_indices):
    lip_material = make_material('AINA_V11_LIP', (0.33, 0.075, 0.095, 1.0), roughness=0.36)
    nostril_material = make_material('AINA_V11_NOSTRIL', (0.18, 0.075, 0.070, 1.0), roughness=0.55)
    objects = []
    bounds = group_bounds(face, mouth_indices)
    if bounds is not None:
        minimum, maximum = bounds
        cx = (minimum.x + maximum.x) * 0.5
        cz = (minimum.z + maximum.z) * 0.5
        front = maximum.y + 0.0018
        half = min((maximum.x - minimum.x) * 0.48, 0.0275)
        upper = curve_object(
            'AINA_V11_LIP_UPPER',
            (
                (cx - half, front, cz),
                (cx - half * 0.48, front + 0.0004, cz + 0.0016),
                (cx, front + 0.0008, cz + 0.0025),
                (cx + half * 0.48, front + 0.0004, cz + 0.0016),
                (cx + half, front, cz),
            ),
            lip_material,
            bevel=0.00086,
        )
        lower = curve_object(
            'AINA_V11_LIP_LOWER',
            (
                (cx - half * 0.90, front - 0.0001, cz - 0.0003),
                (cx - half * 0.42, front + 0.0005, cz - 0.0022),
                (cx, front + 0.0008, cz - 0.0030),
                (cx + half * 0.42, front + 0.0005, cz - 0.0022),
                (cx + half * 0.90, front - 0.0001, cz - 0.0003),
            ),
            lip_material,
            bevel=0.00072,
        )
        objects.extend((upper, lower))

    skin_points = [(index, face.matrix_world @ face.data.vertices[index].co) for index in skin_indices]
    central = [item for item in skin_points if abs(item[1].x) < 0.018 and 1.365 < item[1].z < 1.415]
    if central:
        tip = max((point for _, point in central), key=lambda point: point.y)
        z = tip.z - 0.0070
        y = tip.y + 0.0012
        objects.append(curve_object('AINA_V11_NOSTRIL_L', ((-0.0105, y, z), (-0.0060, y + 0.0002, z - 0.0006), (-0.0025, y, z)), nostril_material, bevel=0.00036))
        objects.append(curve_object('AINA_V11_NOSTRIL_R', ((0.0025, y, z), (0.0060, y + 0.0002, z - 0.0006), (0.0105, y, z)), nostril_material, bevel=0.00036))

    for obj in objects:
        parent_to_bone(obj, armature, head_bone)
    return objects


def build_v11_hair(face_min, face_max, eye_z, armature, head_bone):
    hair = make_material('AINA_V11_HAIR', (0.22, 0.25, 0.35, 1.0), metallic=0.045, roughness=0.34)
    hair_light = make_material('AINA_V11_HAIR_LIGHT', (0.34, 0.37, 0.48, 1.0), metallic=0.035, roughness=0.31)
    metal = make_material('AINA_V11_HAIR_TECH', (0.10, 0.16, 0.28, 1.0), metallic=0.78, roughness=0.22)
    center_x = (face_min.x + face_max.x) * 0.5
    front = face_max.y + 0.0028
    back = face_min.y - 0.012
    top = face_max.z
    objects = []

    # Compact, low three-lobed bun behind the skull.
    bun_center = Vector((center_x, back - 0.018, top + 0.003))
    for index, (offset, scale, rotation) in enumerate((
        ((0.000, 0.000, 0.000), (0.039, 0.025, 0.031), (0.10, 0.00, 0.00)),
        ((-0.026, 0.002, -0.003), (0.027, 0.021, 0.026), (0.30, 0.12, -0.30)),
        ((0.026, 0.002, -0.003), (0.027, 0.021, 0.026), (-0.30, -0.12, 0.30)),
    )):
        obj = sphere_object(f'AINA_V11_BUN_{index}', bun_center + Vector(offset), scale, hair, 44, 28)
        obj.rotation_euler = rotation
        objects.append(obj)

    # Broad center-parted locks: ribbons, not spaghetti curves.
    left_main = (
        ((center_x - 0.003, front - 0.044, top - 0.003), 0.010),
        ((center_x - 0.018, front - 0.018, top - 0.010), 0.014),
        ((center_x - 0.035, front - 0.006, top - 0.032), 0.015),
        ((center_x - 0.052, front, eye_z + 0.050), 0.013),
        ((center_x - 0.061, front + 0.001, eye_z + 0.005), 0.009),
        ((center_x - 0.060, front - 0.001, eye_z - 0.065), 0.004),
    )
    right_main = tuple(((-x + 2.0 * center_x, y, z), width) for (x, y, z), width in left_main)
    for name, specification in (('AINA_V11_FRINGE_L', left_main), ('AINA_V11_FRINGE_R', right_main)):
        points = [point for point, _ in specification]
        widths = [width for _, width in specification]
        objects.append(ribbon_object(name, points, widths, hair_light, thickness=0.0010))

    for sign in (-1.0, 1.0):
        outer_points = (
            (center_x + sign * 0.032, front - 0.035, top - 0.012),
            (center_x + sign * 0.056, front - 0.010, top - 0.035),
            (center_x + sign * 0.070, front - 0.001, eye_z + 0.030),
            (center_x + sign * 0.071, front - 0.002, eye_z - 0.040),
            (center_x + sign * 0.064, front - 0.007, eye_z - 0.095),
        )
        objects.append(ribbon_object(f'AINA_V11_SIDE_LOCK_{int(sign)}', outer_points, (0.010, 0.013, 0.011, 0.007, 0.003), hair, thickness=0.0009))

        crown_points = (
            (center_x + sign * 0.003, back + 0.028, top + 0.004),
            (center_x + sign * 0.027, back + 0.040, top + 0.001),
            (center_x + sign * 0.052, back + 0.047, top - 0.015),
            (center_x + sign * 0.075, back + 0.030, top - 0.045),
        )
        objects.append(ribbon_object(f'AINA_V11_CROWN_PANEL_{int(sign)}', crown_points, (0.011, 0.018, 0.019, 0.010), hair_light, thickness=0.0011))

    # Two restrained technology arcs behind the hair.
    for sign in (-1.0, 1.0):
        arc = []
        for index in range(9):
            t = index / 8.0
            angle = -0.95 + 1.90 * t
            arc.append((
                center_x + sign * (0.070 + 0.030 * math.cos(angle)),
                back - 0.015,
                eye_z + 0.060 + 0.095 * math.sin(angle + 0.95),
            ))
        objects.append(curve_object(f'AINA_V11_TECH_ARC_{int(sign)}', arc, metal, bevel=0.00085))

    for obj in objects:
        parent_to_bone(obj, armature, head_bone)
    return objects


def setup_scene(target):
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 600
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.compression = 20
    scene.render.film_transparent = False
    scene.view_settings.exposure = -1.10
    try:
        scene.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        pass
    if hasattr(scene, 'eevee'):
        for attribute in ('taa_samples', 'taa_render_samples'):
            if hasattr(scene.eevee, attribute):
                try:
                    setattr(scene.eevee, attribute, 32)
                except Exception:
                    pass

    if scene.world is None:
        scene.world = bpy.data.worlds.new('AINA_V11_WORLD')
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value = (0.010, 0.014, 0.024, 1.0)
    background.inputs['Strength'].default_value = 0.055

    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    for name, location, energy, size in (
        ('AINA_V11_KEY', (-0.55, 0.63, target.z + 0.38), 112, 0.82),
        ('AINA_V11_FILL', (0.48, 0.48, target.z + 0.12), 38, 0.78),
        ('AINA_V11_RIM', (0.00, -0.48, target.z + 0.32), 72, 0.64),
        ('AINA_V11_LOW', (0.00, 0.28, target.z - 0.28), 18, 0.62),
    ):
        bpy.ops.object.light_add(type='AREA', location=location)
        light = bpy.context.object
        light.name = name
        light.data.energy = energy
        light.data.shape = 'DISK'
        light.data.size = size
        look_at(light, target)

    camera = next((obj for obj in bpy.data.objects if obj.type == 'CAMERA'), None)
    if camera is None:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
    camera.name = 'CAM_AINA_V11'
    camera.data.sensor_width = 36
    scene.camera = camera
    return scene, camera


def render(scene, camera, target, location, output, lens=88):
    camera.data.type = 'PERSP'
    camera.data.lens = lens
    camera.location = Vector(location)
    look_at(camera, target)
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def duplicate_curves_as_mesh(objects):
    duplicates = []
    for obj in objects:
        if obj.type != 'CURVE':
            continue
        duplicate = obj.copy()
        duplicate.data = obj.data.copy()
        bpy.context.collection.objects.link(duplicate)
        duplicate.name = obj.name + '_EXPORT'
        matrix = duplicate.matrix_world.copy()
        duplicate.parent = None
        duplicate.matrix_world = matrix
        bpy.context.view_layer.objects.active = duplicate
        duplicate.select_set(True)
        bpy.ops.object.convert(target='MESH')
        duplicate.select_set(False)
        duplicates.append(duplicate)
    return duplicates


def main():
    arguments = parse_args()
    output = Path(arguments['out']).resolve()
    preview = output / 'Preview'
    qa = output / 'QA'
    preview.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)

    face = next((obj for obj in bpy.data.objects if obj.type == 'MESH' and obj.name.lower().startswith('face')), None)
    armature = next((obj for obj in bpy.data.objects if obj.type == 'ARMATURE'), None)
    if face is None or armature is None:
        raise RuntimeError('AINA V11 requires the V10 Face and Armature')

    old_objects, old_materials = hide_old_visual_layers()
    skin_material = replace_face_skin_material()
    tinted_hair = tint_existing_hair()

    eye_indices = material_vertex_indices(face, ('eyewhite',))
    mouth_indices = material_vertex_indices(face, ('facemouth',))
    skin_indices = material_vertex_indices(face, ('face_00_skin', 'faceskin'))
    eye_bounds_before = group_bounds(face, eye_indices)
    if eye_bounds_before is None:
        raise RuntimeError('AINA V11 could not resolve eye landmarks')
    eye_z_before = (eye_bounds_before[0].z + eye_bounds_before[1].z) * 0.5
    transform_face_to_adult(face, eye_z_before)

    eye_bounds = group_bounds(face, eye_indices)
    eye_z = (eye_bounds[0].z + eye_bounds[1].z) * 0.5
    face_min, face_max = mesh_bounds(face)
    head_bone = find_bone(armature, ('J_Bip_C_Head', 'head'))

    eyes, eye_centers = build_v11_eyes(face, armature, head_bone, eye_indices, face_max.y, skin_material)
    mouth_nose = build_v11_mouth_and_nose(face, armature, head_bone, mouth_indices, skin_indices)
    hair = build_v11_hair(face_min, face_max, eye_z, armature, head_bone)
    procedural = eyes + mouth_nose + hair

    target = Vector((0.0, face_max.y * 0.46, eye_z - 0.018))
    scene, camera = setup_scene(target)
    close_distance = 0.64
    render(scene, camera, target, (0.0, target.y + close_distance, target.z), preview / 'AINA_V11_FRONT.png', 90)
    render(scene, camera, target, (close_distance * 0.50, target.y + close_distance * 0.86, target.z), preview / 'AINA_V11_3Q.png', 90)
    render(scene, camera, target, (close_distance, target.y, target.z), preview / 'AINA_V11_PROFILE.png', 90)

    full_objects = [obj for obj in bpy.data.objects if obj.type in {'MESH', 'CURVE'} and not obj.hide_render]
    points = [obj.matrix_world @ Vector(corner) for obj in full_objects for corner in obj.bound_box]
    full_min = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    full_max = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    full_target = (full_min + full_max) * 0.5
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = (full_max.z - full_min.z) * 1.08
    camera.location = Vector((0.0, full_target.y + 3.0, full_target.z))
    look_at(camera, full_target)
    scene.render.filepath = str(preview / 'AINA_V11_FULLBODY.png')
    bpy.ops.render.render(write_still=True)
    camera.data.type = 'PERSP'

    blend_path = output / 'AINA_MASTER_V11_REVIEW.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    curve_duplicates = duplicate_curves_as_mesh(procedural)
    for obj in procedural:
        if obj.type == 'CURVE':
            obj.hide_render = True
    bpy.ops.export_scene.gltf(
        filepath=str(output / 'AINA_EXPORT_V11_REVIEW.glb'),
        export_format='GLB',
        export_animations=True,
        export_cameras=False,
        export_lights=False,
    )
    bpy.ops.export_scene.fbx(
        filepath=str(output / 'AINA_EXPORT_V11_REVIEW.fbx'),
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=False,
        path_mode='AUTO',
    )
    for obj in procedural:
        if obj.type == 'CURVE':
            obj.hide_render = False
    for obj in curve_duplicates:
        bpy.data.objects.remove(obj, do_unlink=True)

    report = {
        'version': 'V11 visual identity review',
        'blender_version': bpy.app.version_string,
        'head_bone': head_bone,
        'face_bounds': {'min': list(face_min), 'max': list(face_max)},
        'eye_z': eye_z,
        'eye_centers': eye_centers,
        'old_objects_hidden': old_objects,
        'old_materials_hidden': old_materials,
        'existing_hair_tinted': tinted_hair,
        'procedural_objects': [obj.name for obj in procedural],
        'armature_bones': len(armature.data.bones),
        'face_shape_keys': len(face.data.shape_keys.key_blocks) - 1 if face.data.shape_keys else 0,
        'identity_lock': False,
        'visual_identity_lock': False,
    }
    (qa / 'AINA_V11_REVIEW_QA.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            output = Path(parse_args().get('out', '.'))
            (output / 'QA').mkdir(parents=True, exist_ok=True)
            (output / 'QA' / 'AINA_V11_ERROR.log').write_text(error, encoding='utf-8')
        except Exception:
            pass
        raise
