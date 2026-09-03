#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
from pathlib import Path


def load_base():
    path = Path(__file__).with_name('refine_aina_v11_identity.py')
    spec = importlib.util.spec_from_file_location('aina_v11_base', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Unable to load AINA V11 base refiner')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()
EYE_SURFACE_Y = None


def build_v11_eyes(face, armature, head_bone, eye_material_indices, _nose_front, skin_material):
    global EYE_SURFACE_Y
    white_material = base.make_material('AINA_V11_EYE_WHITE', (0.62, 0.65, 0.70, 1.0), roughness=0.30)
    iris_outer = base.make_material('AINA_V11_IRIS_OUTER', (0.045, 0.075, 0.115, 1.0), roughness=0.20)
    iris_inner = base.make_material('AINA_V11_IRIS_INNER', (0.18, 0.30, 0.42, 1.0), roughness=0.18)
    pupil_material = base.make_material('AINA_V11_PUPIL', (0.006, 0.009, 0.015, 1.0), roughness=0.16)
    highlight_material = base.make_material(
        'AINA_V11_EYE_GLINT',
        (0.82, 0.88, 1.0, 1.0),
        roughness=0.08,
        emission=(0.35, 0.48, 0.75, 1.0),
        emission_strength=0.28,
    )
    lash_material = base.make_material('AINA_V11_LASH', (0.025, 0.018, 0.025, 1.0), roughness=0.46)
    lower_material = base.make_material('AINA_V11_LOWER_LID', (0.29, 0.16, 0.16, 1.0), roughness=0.55)
    brow_material = base.make_material('AINA_V11_BROW', (0.075, 0.050, 0.055, 1.0), roughness=0.54)

    objects = []
    centers = []
    surfaces = []
    for sign in (-1.0, 1.0):
        side = [
            index
            for index in eye_material_indices
            if (face.matrix_world @ face.data.vertices[index].co).x * sign > 0.0
        ]
        bounds = base.group_bounds(face, side)
        if bounds is None:
            continue
        minimum, maximum = bounds
        center_x = (minimum.x + maximum.x) * 0.5
        center_z = (minimum.z + maximum.z) * 0.5
        width = maximum.x - minimum.x
        socket_surface = maximum.y
        surfaces.append(socket_surface)
        centers.append((center_x, socket_surface, center_z))

        patch = base.sphere_object(
            f'AINA_V11_EYE_SOCKET_PATCH_{int(sign)}',
            (center_x, socket_surface - 0.0032, center_z),
            (max(width * 0.96, 0.0255), 0.0058, 0.0158),
            skin_material,
            segments=48,
            rings=28,
        )
        objects.append(patch)

        eye_front = socket_surface + 0.0012
        white = base.almond_object(
            f'AINA_V11_EYE_WHITE_{int(sign)}',
            center_x,
            center_z,
            eye_front,
            0.0182,
            0.0047,
            white_material,
        )
        objects.append(white)

        iris_y = eye_front + 0.0019
        outer = base.sphere_object(
            f'AINA_V11_IRIS_OUTER_{int(sign)}',
            (center_x, iris_y, center_z - 0.0001),
            (0.0052, 0.0012, 0.00465),
            iris_outer,
            32,
            20,
        )
        inner = base.sphere_object(
            f'AINA_V11_IRIS_INNER_{int(sign)}',
            (center_x, iris_y + 0.0007, center_z - 0.0001),
            (0.00375, 0.00075, 0.00345),
            iris_inner,
            32,
            20,
        )
        pupil = base.sphere_object(
            f'AINA_V11_PUPIL_{int(sign)}',
            (center_x, iris_y + 0.0011, center_z - 0.0001),
            (0.00175, 0.00048, 0.00225),
            pupil_material,
            28,
            16,
        )
        glint = base.sphere_object(
            f'AINA_V11_GLINT_{int(sign)}',
            (center_x - sign * 0.0015, iris_y + 0.00155, center_z + 0.0017),
            (0.00078, 0.00030, 0.00078),
            highlight_material,
            20,
            12,
        )
        objects.extend((outer, inner, pupil, glint))

        inner_x = center_x - sign * 0.0180
        outer_x = center_x + sign * 0.0180
        upper = base.curve_object(
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
        lower = base.curve_object(
            f'AINA_V11_LOWER_LID_{int(sign)}',
            (
                (inner_x + sign * 0.0015, eye_front + 0.0016, center_z - 0.0002),
                (center_x, eye_front + 0.0018, center_z - 0.0044),
                (outer_x - sign * 0.0015, eye_front + 0.0016, center_z - 0.0001),
            ),
            lower_material,
            bevel=0.00030,
        )
        brow = base.curve_object(
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

    if surfaces:
        EYE_SURFACE_Y = sum(surfaces) / len(surfaces)
    for obj in objects:
        base.parent_to_bone(obj, armature, head_bone)
    return objects, centers


def build_v11_hair(face_min, face_max, eye_z, armature, head_bone):
    hair = base.make_material('AINA_V11_HAIR', (0.22, 0.25, 0.35, 1.0), metallic=0.045, roughness=0.34)
    hair_light = base.make_material('AINA_V11_HAIR_LIGHT', (0.34, 0.37, 0.48, 1.0), metallic=0.035, roughness=0.31)
    metal = base.make_material('AINA_V11_HAIR_TECH', (0.10, 0.16, 0.28, 1.0), metallic=0.78, roughness=0.22)
    center_x = (face_min.x + face_max.x) * 0.5
    eye_surface = EYE_SURFACE_Y if EYE_SURFACE_Y is not None else face_max.y - 0.030
    front = eye_surface + 0.0045
    back = face_min.y - 0.012
    top = face_max.z
    objects = []

    bun_center = base.Vector((center_x, back - 0.018, top + 0.003))
    for index, (offset, scale, rotation) in enumerate((
        ((0.000, 0.000, 0.000), (0.039, 0.025, 0.031), (0.10, 0.00, 0.00)),
        ((-0.026, 0.002, -0.003), (0.027, 0.021, 0.026), (0.30, 0.12, -0.30)),
        ((0.026, 0.002, -0.003), (0.027, 0.021, 0.026), (-0.30, -0.12, 0.30)),
    )):
        obj = base.sphere_object(
            f'AINA_V11_BUN_{index}',
            bun_center + base.Vector(offset),
            scale,
            hair,
            44,
            28,
        )
        obj.rotation_euler = rotation
        objects.append(obj)

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
        objects.append(base.ribbon_object(name, points, widths, hair_light, thickness=0.0010))

    for sign in (-1.0, 1.0):
        outer_points = (
            (center_x + sign * 0.032, front - 0.035, top - 0.012),
            (center_x + sign * 0.056, front - 0.010, top - 0.035),
            (center_x + sign * 0.070, front - 0.001, eye_z + 0.030),
            (center_x + sign * 0.071, front - 0.002, eye_z - 0.040),
            (center_x + sign * 0.064, front - 0.007, eye_z - 0.095),
        )
        objects.append(
            base.ribbon_object(
                f'AINA_V11_SIDE_LOCK_{int(sign)}',
                outer_points,
                (0.010, 0.013, 0.011, 0.007, 0.003),
                hair,
                thickness=0.0009,
            )
        )
        crown_points = (
            (center_x + sign * 0.003, back + 0.028, top + 0.004),
            (center_x + sign * 0.027, back + 0.040, top + 0.001),
            (center_x + sign * 0.052, back + 0.047, top - 0.015),
            (center_x + sign * 0.075, back + 0.030, top - 0.045),
        )
        objects.append(
            base.ribbon_object(
                f'AINA_V11_CROWN_PANEL_{int(sign)}',
                crown_points,
                (0.011, 0.018, 0.019, 0.010),
                hair_light,
                thickness=0.0011,
            )
        )

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
        objects.append(base.curve_object(f'AINA_V11_TECH_ARC_{int(sign)}', arc, metal, bevel=0.00085))

    for obj in objects:
        base.parent_to_bone(obj, armature, head_bone)
    return objects


base.build_v11_eyes = build_v11_eyes
base.build_v11_hair = build_v11_hair


if __name__ == '__main__':
    try:
        base.main()
    except Exception:
        error = base.traceback.format_exc()
        print(error, flush=True)
        try:
            output = Path(base.parse_args().get('out', '.'))
            (output / 'QA').mkdir(parents=True, exist_ok=True)
            (output / 'QA' / 'AINA_V11_FIXED_ERROR.log').write_text(error, encoding='utf-8')
        except Exception:
            pass
        raise
