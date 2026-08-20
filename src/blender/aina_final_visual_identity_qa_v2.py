#!/usr/bin/env python3
"""AINA final real-model visual identity QA v2.

Focused visual correction over the first full portrait assembly: shorter lower
face, wider visible nose, slightly smaller/higher lips, larger irises, darker
lashes, natural centre-part hair locks, and restrained shoulders/collar. All 52
shape controls stay on the same real mesh. No effect-art generation or VRM export.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aina_final_visual_identity_qa as vis

v2 = vis.v2
base = vis.base


def transform_neutral(coords, landmarks):
    out = np.asarray(coords, dtype=np.float64).copy()
    lm = landmarks
    mouth = lm[48:60].mean(0)
    nose = lm[31:36].mean(0)
    chin = lm[8]

    # Compress only the front lower third toward the mouth plane.
    lower = np.clip((mouth[2] + 0.005 - out[:, 2]) / 0.080, 0.0, 1.0)
    front = np.exp(-0.5 * ((out[:, 1] - mouth[1]) / 0.095) ** 4)
    w = lower * front
    out[:, 2] += 0.0050 * w
    out[:, 0] += (mouth[0] - out[:, 0]) * (0.025 * w)[:, None] if False else 0.0

    # Higher, slightly smaller lips integrated into the face.
    vis.base.scale_region(out, mouth, (0.052, 0.043, 0.036), (0.92, 1.0, 0.84), 0.05, 1.06)
    vis.base.shift_region(out, mouth, (0.052, 0.042, 0.036), (0.0, 0.0012, 0.0032), 0.04, 1.04)

    # Wider alae and a visible small tip; the old nose was mathematically narrow
    # enough to disappear under soft portrait lighting.
    vis.base.scale_region(out, nose, (0.032, 0.032, 0.032), (1.20, 1.0, 0.98), 0.05, 1.06)
    vis.base.shift_region(out, nose, (0.033, 0.035, 0.035), (0.0, -0.0018, 0.0003), 0.04, 1.05)

    # Smaller, rounder and slightly higher chin.
    vis.base.scale_region(out, chin, (0.047, 0.047, 0.044), (0.90, 0.97, 0.86), 0.03, 1.05)
    vis.base.shift_region(out, chin, (0.048, 0.048, 0.045), (0.0, 0.0008, 0.0036), 0.02, 1.04)

    # Soft apple-cheek support.
    for center in ((lm[40] + lm[31] + lm[48]) / 3.0, (lm[46] + lm[35] + lm[54]) / 3.0):
        vis.base.shift_region(out, center, (0.043, 0.040, 0.040), (0.0, -0.0016, 0.0010), 0.02, 1.04)
    return out


_original_build = v2.build_character


def build_character_v2(face_path, height):
    head, objects, mapped, stats = _original_build(face_path, height)
    old_lm = mapped[base.K].copy()
    blocks = head.data.shape_keys.key_blocks
    new_basis = None
    for key in blocks:
        coords = v2.key_array(key)
        new_coords = transform_neutral(coords, old_lm)
        v2.set_key_array(key, new_coords)
        if key.name == "Basis":
            new_basis = new_coords
    if new_basis is None:
        raise RuntimeError("AINA head Basis shape key missing")
    mapped = new_basis.copy()
    new_lm = mapped[base.K]

    # Move and enlarge visible eye geometry coherently with the updated skin.
    for side, rr in (("R", range(36, 42)), ("L", range(42, 48))):
        old_center = old_lm[list(rr)].mean(0)
        new_center = new_lm[list(rr)].mean(0)
        for prefix, factor in (("AINA_Eye_", (1.08, 1.0, 1.08)), ("AINA_Iris_", (1.16, 1.0, 1.16)), ("AINA_Pupil_", (1.10, 1.0, 1.10))):
            obj = bpy.data.objects.get(prefix + side)
            if not obj or not obj.data.shape_keys:
                continue
            for key in obj.data.shape_keys.key_blocks:
                coords = v2.key_array(key)
                coords += new_center - old_center
                center = new_center.copy()
                coords = center + (coords - center) * np.asarray(factor)
                v2.set_key_array(key, coords)

    # Reposition the real brow ribbons after the neutral face correction.
    for side, old_ids, new_ids in (("Right", range(17, 22), range(17, 22)), ("Left", range(22, 27), range(22, 27))):
        obj = bpy.data.objects.get("AINA_Brow_" + side)
        if not obj or not obj.data.shape_keys:
            continue
        delta = new_lm[list(new_ids)].mean(0) - old_lm[list(old_ids)].mean(0)
        for key in obj.data.shape_keys.key_blocks:
            coords = v2.key_array(key) + delta
            v2.set_key_array(key, coords)
    return head, objects, mapped, stats


v2.build_character = build_character_v2


def create_hair_v2(head, hair_mat):
    # Top/back cap only. The front identity is framed by a small number of broad,
    # swept locks rather than dozens of vertical pipe strands.
    verts, faces = [], []
    nphi, nt = 104, 28
    center = np.array([0.0, 0.030, 1.642])
    rx, ry, rz = 0.110, 0.101, 0.128
    for i in range(nphi):
        phi = 2.0 * math.pi * i / nphi
        sy = math.sin(phi)
        side_factor = abs(math.cos(phi))
        tmax = (0.82 + 0.34 * side_factor) if sy < 0.0 else 2.03
        for k in range(nt):
            theta = tmax * k / (nt - 1)
            p = center + np.array([
                rx * math.sin(theta) * math.cos(phi),
                ry * math.sin(theta) * math.sin(phi),
                rz * math.cos(theta),
            ])
            verts.append(p.tolist())
    for i in range(nphi):
        ni = (i + 1) % nphi
        for k in range(nt - 1):
            a = i * nt + k; b = ni * nt + k; c = ni * nt + k + 1; d = i * nt + k + 1
            faces.extend(((a, b, c), (a, c, d)))
    cap = base.mesh_object("AINA_Hair_Cap", np.asarray(verts), np.asarray(faces, dtype=np.int32))
    cap.data.materials.append(hair_mat); cap.parent = head
    for poly in cap.data.polygons: poly.use_smooth = True

    bun = vis.create_uv_sphere("AINA_Hair_Bun", (0.0, 0.112, 1.710), (0.048, 0.043, 0.051), hair_mat, head)

    # Six swept locks per side create the approved airy centre-part silhouette.
    for side, sign in (("L", -1.0), ("R", 1.0)):
        for i in range(6):
            root = np.array([sign * (0.003 + 0.0030 * i), -0.060 + 0.0008 * i, 1.738 - 0.0015 * i])
            end = np.array([sign * (0.027 + 0.0100 * i), -0.092 + 0.0020 * i, 1.665 - 0.0100 * i])
            mid = (root + end) * 0.5 + np.array([sign * 0.009, -0.012, 0.012])
            vis.make_curve(f"AINA_SweptFringe_{side}_{i+1}", [root, mid, end], 0.00115 + 0.00008 * i, hair_mat, head)

    # Fine face-framing wisps.
    wisps = [
        [(-0.010, -0.064, 1.730), (-0.021, -0.088, 1.670), (-0.025, -0.098, 1.606)],
        [(0.010, -0.064, 1.730), (0.021, -0.088, 1.670), (0.025, -0.098, 1.606)],
        [(-0.028, -0.058, 1.716), (-0.049, -0.080, 1.650), (-0.060, -0.087, 1.575)],
        [(0.028, -0.058, 1.716), (0.049, -0.080, 1.650), (0.060, -0.087, 1.575)],
    ]
    for i, points in enumerate(wisps):
        vis.make_curve(f"AINA_Wisp_{i+1}", points, 0.00052, hair_mat, head)

    # A few crown flow curves break the cap surface without making a cage.
    for i, value in enumerate(np.linspace(-1.0, 1.0, 9)):
        root = (value * 0.005, -0.053, 1.758)
        end = (value * 0.088, -0.010 + abs(value) * 0.020, 1.676 - abs(value) * 0.020)
        mid = ((root[0] + end[0]) * 0.5, -0.047, (root[2] + end[2]) * 0.5 + 0.014)
        vis.make_curve(f"AINA_CrownFlow_{i+1}", [root, mid, end], 0.00044, hair_mat, head)
    return [cap, bun]


vis.create_hair = create_hair_v2


def create_lashes_v2(head, mapped, lash_mat):
    lm = mapped[base.K]
    objects = []
    for side, indices in (("R", range(36, 42)), ("L", range(42, 48))):
        c = lm[list(indices)].mean(0)
        rx = 0.0190
        points = [
            (c[0] - rx, -0.0148, c[2] + (0.0010 if side == "R" else 0.0001)),
            (c[0] - rx * 0.52, -0.0150, c[2] + 0.0051),
            (c[0], -0.0151, c[2] + 0.0065),
            (c[0] + rx * 0.52, -0.0150, c[2] + 0.0051),
            (c[0] + rx, -0.0148, c[2] + (0.0001 if side == "R" else 0.0010)),
        ]
        objects.append(vis.make_curve(f"AINA_Lash_{side}", points, 0.00078, lash_mat, head))
        outer = np.asarray(points[0 if side == "R" else -1])
        direction = -1.0 if side == "R" else 1.0
        for i in range(3):
            start = outer + np.array([direction * 0.0010 * i, 0.0, 0.00030 * i])
            end = start + np.array([direction * (0.0034 + 0.0005 * i), -0.0001, 0.0024 + 0.00035 * i])
            objects.append(vis.make_curve(f"AINA_LashTail_{side}_{i+1}", [start, end], 0.00042, lash_mat, head))
    return objects


vis.create_lashes = create_lashes_v2


def create_neck_and_collar_v2(skin_mat, suit_mat, accent_mat):
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.042, depth=0.17, location=(0.0, 0.023, 1.455))
    neck = bpy.context.object; neck.name = "AINA_Neck"; neck.scale = (0.94, 0.78, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True); neck.data.materials.append(skin_mat)
    for poly in neck.data.polygons: poly.use_smooth = True

    bpy.ops.mesh.primitive_torus_add(major_radius=0.054, minor_radius=0.011, major_segments=64, minor_segments=16, location=(0.0, 0.020, 1.458))
    collar = bpy.context.object; collar.name = "AINA_Pearl_Collar"; collar.scale = (1.0, 0.76, 1.32)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True); collar.data.materials.append(suit_mat)
    for poly in collar.data.polygons: poly.use_smooth = True

    # Restrained shoulders rather than one oversized white sphere.
    shoulders = []
    for side in (-1.0, 1.0):
        shoulder = vis.create_uv_sphere(
            f"AINA_Shoulder_{'L' if side < 0 else 'R'}",
            (side * 0.105, 0.055, 1.350),
            (0.145, 0.085, 0.075),
            suit_mat,
        )
        shoulders.append(shoulder)
    chest = vis.create_uv_sphere("AINA_Portrait_Chest", (0.0, 0.060, 1.330), (0.145, 0.085, 0.070), suit_mat)
    accent = vis.create_uv_sphere("AINA_Collar_Accent", (0.0, -0.054, 1.455), (0.009, 0.005, 0.013), accent_mat)
    return [neck, collar, chest, accent, *shoulders]


vis.create_neck_and_collar = create_neck_and_collar_v2


# Darker silver and less exposure remove the over-white helmet/costume look.
_original_material = v2.material


def material_v2(name, color, roughness, metallic=0.0):
    overrides = {
        "AINA_Hair_Silver": ((0.42, 0.48, 0.60, 1.0), 0.34, 0.04),
        "AINA_Lash": ((0.010, 0.008, 0.012, 1.0), 0.26, 0.0),
        "AINA_Suit_Pearl": ((0.55, 0.62, 0.75, 1.0), 0.36, 0.05),
    }
    if name in overrides:
        color, roughness, metallic = overrides[name]
    return _original_material(name, color, roughness, metallic)


v2.material = material_v2


if __name__ == "__main__":
    vis.main()
