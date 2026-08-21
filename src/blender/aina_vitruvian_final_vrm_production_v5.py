#!/usr/bin/env python3
"""Integrated final AINA VRM entry point with layered ribbon-card silver hair.

The previous technical candidate used beveled curve locks, which can read as
pipes.  This version keeps the same real head, body, 52 controls, humanoid rig,
VRM binary patch and clean reimport gates, but replaces every visible lock with
flat tapered mesh ribbons layered over a smooth scalp under-cap.  Dynamic side
and back ribbons remain parented to the real spring-bone chains.
"""
from __future__ import annotations

import math

import bpy
import numpy as np
from mathutils import Vector

import aina_vitruvian_final_vrm_production_v4 as production_v4


base = production_v4.base


def normalize(value: np.ndarray, fallback=(1.0, 0.0, 0.0)) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length < 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return value / length


def quadratic_points(p0, p1, p2, count=10):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    values = []
    for t in np.linspace(0.0, 1.0, count):
        values.append((1.0 - t) ** 2 * p0 + 2.0 * (1.0 - t) * t * p1 + t ** 2 * p2)
    return np.asarray(values, dtype=np.float64)


def create_ribbon(name, centerline, root_width, tip_width, mat, radial_center, thickness=0.0):
    points = np.asarray(centerline, dtype=np.float64)
    radial_center = np.asarray(radial_center, dtype=np.float64)
    vertices = []
    uvs = []
    previous_side = None
    for index, point in enumerate(points):
        if index == 0:
            tangent = points[1] - points[0]
        elif index == len(points) - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[index + 1] - points[index - 1]
        tangent = normalize(tangent, (0.0, 0.0, -1.0))
        radial = normalize(point - radial_center, (0.0, -1.0, 0.0))
        side = normalize(np.cross(tangent, radial), (1.0, 0.0, 0.0))
        if previous_side is not None and np.dot(side, previous_side) < 0.0:
            side *= -1.0
        previous_side = side
        t = index / max(len(points) - 1, 1)
        width = root_width * (1.0 - t) + tip_width * t
        # Slight crown across the strip catches light like a layered hair card.
        normal = normalize(np.cross(side, tangent), radial)
        crown = normal * (math.sin(math.pi * t) * thickness)
        vertices.extend((point - side * width * 0.5 + crown, point + side * width * 0.5 + crown))
        uvs.extend(((0.0, t), (1.0, t)))
    faces = []
    for index in range(len(points) - 1):
        left = index * 2
        faces.append((left, left + 1, left + 3, left + 2))
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata([tuple(value) for value in vertices], [], faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            uv_layer.data[loop_index].uv = uvs[vertex_index]
        polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(mat)
    return obj


def create_cap(setup, hair_mat):
    center = setup["center"].copy()
    size = setup["size"]
    top = float(setup["hi"][2])
    face_sign = float(setup["forward_sign"])
    hair_center = np.asarray([
        center[0],
        center[1] - face_sign * 0.045 * size[1],
        top - 0.265 * size[2],
    ])
    rx, ry, rz = 0.575 * size[0], 0.590 * size[1], 0.385 * size[2]
    vertices = []
    faces = []
    nphi, ntheta = 128, 34
    for i in range(nphi):
        phi = 2.0 * math.pi * i / nphi
        frontness = face_sign * math.sin(phi)
        side = abs(math.cos(phi))
        theta_max = 0.88 + 0.30 * side if frontness > -0.16 else 2.05
        for j in range(ntheta):
            theta = theta_max * j / (ntheta - 1)
            # A small asymmetric lift creates a natural centre part rather than
            # a perfectly spherical helmet silhouette.
            part_lift = 0.018 * size[2] * math.exp(-((math.cos(phi)) / 0.22) ** 2) * (1.0 - j / ntheta)
            vertices.append((
                hair_center[0] + rx * math.sin(theta) * math.cos(phi),
                hair_center[1] + ry * math.sin(theta) * math.sin(phi),
                hair_center[2] + rz * math.cos(theta) + part_lift,
            ))
    for i in range(nphi):
        next_i = (i + 1) % nphi
        for j in range(ntheta - 1):
            a = i * ntheta + j
            b = next_i * ntheta + j
            c = next_i * ntheta + j + 1
            d = i * ntheta + j + 1
            faces.extend(((a, b, c), (a, c, d)))
    mesh = bpy.data.meshes.new("AINA_Silver_Updo_Undercap_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    cap = bpy.data.objects.new("AINA_Silver_Updo_Undercap", mesh)
    bpy.context.collection.objects.link(cap)
    mesh.materials.append(hair_mat)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return cap, hair_center, (rx, ry, rz)


def hair_variants(source):
    colors = [
        (0.41, 0.49, 0.65, 1.0),
        (0.50, 0.58, 0.73, 1.0),
        (0.61, 0.67, 0.79, 1.0),
        (0.34, 0.42, 0.58, 1.0),
    ]
    materials = []
    for index, color in enumerate(colors):
        mat = source.copy()
        mat.name = f"AINA_Hair_Silver_Ribbon_{index + 1}"
        mat.diffuse_color = color
        mat.use_backface_culling = False
        shader = mat.node_tree.nodes.get("Principled BSDF") if mat.use_nodes and mat.node_tree else None
        if shader:
            if shader.inputs.get("Base Color"):
                shader.inputs["Base Color"].default_value = color
            if shader.inputs.get("Roughness"):
                shader.inputs["Roughness"].default_value = 0.30 + 0.035 * index
            for input_name in ("Anisotropic IOR Level", "Anisotropic"):
                if shader.inputs.get(input_name):
                    shader.inputs[input_name].default_value = 0.62
            if shader.inputs.get("Coat Weight"):
                shader.inputs["Coat Weight"].default_value = 0.12
        materials.append(mat)
    return materials


def create_silver_updo_ribbons(skin, armature, rig_info, setup, hair_mat):
    variants = hair_variants(hair_mat)
    cap, hair_center, radii = create_cap(setup, variants[0])
    rx, ry, rz = radii
    size = setup["size"]
    top = float(setup["hi"][2])
    face_sign = float(setup["forward_sign"])
    back_sign = -face_sign
    base.preserve_bone_parent(cap, armature, rig_info["head_bone"])
    objects = [cap]

    bun_location = np.asarray([
        hair_center[0],
        hair_center[1] + back_sign * 0.82 * ry,
        top + 0.010 * size[2],
    ])
    bun = base.create_uv_sphere(
        "AINA_Silver_Updo_Bun",
        tuple(bun_location),
        (0.34 * size[0], 0.31 * size[1], 0.235 * size[2]),
        variants[1],
    )
    base.preserve_bone_parent(bun, armature, rig_info["head_bone"])
    objects.append(bun)

    radial_center = hair_center
    part_front = np.asarray([
        hair_center[0],
        hair_center[1] + face_sign * 0.43 * ry,
        top + 0.030 * size[2],
    ])
    rng = np.random.default_rng(20260821)

    # 64 overlapping scalp/crown cards: broad enough to read as hair masses,
    # narrow enough to expose directional strand flow.
    for side_index, sign in enumerate((-1.0, 1.0)):
        for index in range(32):
            fraction = (index + 0.35) / 32.0
            root = part_front + np.asarray([
                sign * (0.012 + 0.035 * fraction) * rx,
                -face_sign * 0.012 * ry * fraction,
                -0.010 * size[2] * fraction,
            ])
            lateral = (0.18 + 0.75 * fraction) * rx
            back = (0.04 + 0.50 * fraction ** 1.35) * ry
            drop = (0.12 + 0.26 * fraction) * size[2]
            end = np.asarray([
                hair_center[0] + sign * lateral,
                hair_center[1] + back_sign * back,
                top - drop,
            ])
            middle = 0.5 * (root + end) + np.asarray([
                sign * (0.08 + 0.04 * fraction) * rx,
                back_sign * (0.05 + 0.10 * fraction) * ry,
                (0.075 - 0.035 * fraction) * size[2],
            ])
            middle += rng.normal(0.0, [0.004 * rx, 0.004 * ry, 0.003 * size[2]])
            points = quadratic_points(root, middle, end, count=10)
            ribbon = create_ribbon(
                f"AINA_Crown_Ribbon_{'L' if sign < 0 else 'R'}_{index + 1}",
                points,
                root_width=(0.072 - 0.020 * fraction) * rx,
                tip_width=(0.030 - 0.012 * fraction) * rx,
                mat=variants[(index + side_index) % len(variants)],
                radial_center=radial_center,
                thickness=0.0045 * size[2],
            )
            base.preserve_bone_parent(ribbon, armature, rig_info["head_bone"])
            objects.append(ribbon)

    # Additional rear cards join the crown to the bun and hide under-cap seams.
    for index, fraction in enumerate(np.linspace(-0.92, 0.92, 22)):
        root = np.asarray([
            hair_center[0] + fraction * 0.86 * rx,
            hair_center[1] + back_sign * 0.18 * ry,
            top - (0.10 + 0.05 * abs(fraction)) * size[2],
        ])
        end = bun_location + np.asarray([
            fraction * 0.42 * rx,
            -back_sign * 0.08 * ry,
            -0.10 * size[2] * abs(fraction),
        ])
        middle = 0.5 * (root + end) + np.asarray([
            -fraction * 0.06 * rx,
            back_sign * 0.14 * ry,
            0.05 * size[2],
        ])
        ribbon = create_ribbon(
            f"AINA_Bun_Join_Ribbon_{index + 1}",
            quadratic_points(root, middle, end, 9),
            0.040 * rx,
            0.018 * rx,
            variants[index % len(variants)],
            radial_center,
            0.0035 * size[2],
        )
        base.preserve_bone_parent(ribbon, armature, rig_info["head_bone"])
        objects.append(ribbon)

    # Real dynamic face-framing ribbon locks. Each lock is split across the two
    # spring bones so motion bends through the length instead of moving as one tube.
    for side_name, sign, chain in (
        ("L", 1.0, rig_info["hair_chains"][0]),
        ("R", -1.0, rig_info["hair_chains"][1]),
    ):
        for index in range(7):
            root = np.asarray([
                hair_center[0] + sign * (0.26 + 0.055 * index) * rx,
                hair_center[1] + face_sign * (0.33 + 0.018 * index) * ry,
                top - (0.13 + 0.035 * index) * size[2],
            ])
            end = np.asarray([
                hair_center[0] + sign * (0.44 + 0.040 * index) * rx,
                hair_center[1] + face_sign * (0.48 + 0.012 * index) * ry,
                top - (0.48 + 0.055 * index) * size[2],
            ])
            middle = 0.5 * (root + end) + np.asarray([
                sign * 0.10 * rx,
                face_sign * 0.09 * ry,
                0.025 * size[2],
            ])
            curve = quadratic_points(root, middle, end, count=13)
            split = 7
            first = create_ribbon(
                f"AINA_Dynamic_Side_{side_name}_{index + 1}_A",
                curve[: split + 1],
                0.045 * rx,
                0.028 * rx,
                variants[(index + 1) % len(variants)],
                radial_center,
                0.0025 * size[2],
            )
            second = create_ribbon(
                f"AINA_Dynamic_Side_{side_name}_{index + 1}_B",
                curve[split - 1 :],
                0.030 * rx,
                0.006 * rx,
                variants[(index + 2) % len(variants)],
                radial_center,
                0.0018 * size[2],
            )
            base.preserve_bone_parent(first, armature, chain[0])
            base.preserve_bone_parent(second, armature, chain[1])
            objects.extend((first, second))

    # Back spring chain carries a small fan of tapered cards under the bun.
    back_chain = rig_info["hair_chains"][2]
    for index, fraction in enumerate(np.linspace(-0.72, 0.72, 13)):
        root = np.asarray([
            hair_center[0] + fraction * 0.62 * rx,
            hair_center[1] + back_sign * 0.64 * ry,
            top - 0.24 * size[2],
        ])
        end = np.asarray([
            hair_center[0] + fraction * 0.52 * rx,
            hair_center[1] + back_sign * 0.82 * ry,
            top - (0.56 + 0.08 * abs(fraction)) * size[2],
        ])
        middle = 0.5 * (root + end) + np.asarray([0.0, back_sign * 0.08 * ry, 0.025 * size[2]])
        curve = quadratic_points(root, middle, end, 11)
        first = create_ribbon(
            f"AINA_Dynamic_Back_{index + 1}_A",
            curve[:7],
            0.035 * rx,
            0.022 * rx,
            variants[index % len(variants)],
            radial_center,
            0.0020 * size[2],
        )
        second = create_ribbon(
            f"AINA_Dynamic_Back_{index + 1}_B",
            curve[5:],
            0.024 * rx,
            0.005 * rx,
            variants[(index + 1) % len(variants)],
            radial_center,
            0.0015 * size[2],
        )
        base.preserve_bone_parent(first, armature, back_chain[0])
        base.preserve_bone_parent(second, armature, back_chain[1])
        objects.extend((first, second))

    # Six extremely narrow wisps soften the hairline; still real geometry.
    for index, (sign, offset) in enumerate(((-1, 0.18), (1, 0.18), (-1, 0.31), (1, 0.31), (-1, 0.43), (1, 0.43))):
        root = np.asarray([
            hair_center[0] + sign * offset * rx,
            hair_center[1] + face_sign * 0.45 * ry,
            top - (0.10 + 0.10 * offset) * size[2],
        ])
        end = np.asarray([
            hair_center[0] + sign * (offset + 0.10) * rx,
            hair_center[1] + face_sign * 0.54 * ry,
            top - (0.42 + 0.15 * offset) * size[2],
        ])
        middle = 0.5 * (root + end) + np.asarray([sign * 0.07 * rx, face_sign * 0.05 * ry, 0.015 * size[2]])
        ribbon = create_ribbon(
            f"AINA_Hairline_Wisp_{index + 1}",
            quadratic_points(root, middle, end, 12),
            0.012 * rx,
            0.0015 * rx,
            variants[(index + 2) % len(variants)],
            radial_center,
            0.0010 * size[2],
        )
        base.preserve_bone_parent(ribbon, armature, rig_info["head_bone"])
        objects.append(ribbon)

    return objects


base.create_silver_updo = create_silver_updo_ribbons


if __name__ == "__main__":
    base.main()
