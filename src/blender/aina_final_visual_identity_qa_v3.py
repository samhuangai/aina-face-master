#!/usr/bin/env python3
"""AINA final visual identity correction v3.

Third real-model portrait convergence pass. It removes the bob/helmet hairline,
shortens the lower face and visible neck, enlarges and separates the eye system,
lowers/thickens brows, strengthens the small nose silhouette, and tightens camera
framing. The same refined topology and 52 controls remain intact. No effect-art
creation and no VRM export.
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

import aina_final_visual_identity_qa_v2 as vis2

vis = vis2.vis
v2 = vis2.v2
base = vis2.base


_original_build = v2.build_character


def additional_identity_polish(coords, landmarks):
    out = np.asarray(coords, dtype=np.float64).copy()
    lm = landmarks
    mouth = lm[48:60].mean(0)
    chin = lm[8]
    nose = lm[31:36].mean(0)

    lower = np.clip((mouth[2] + 0.006 - out[:, 2]) / 0.075, 0.0, 1.0)
    front = np.exp(-0.5 * ((out[:, 1] - mouth[1]) / 0.092) ** 4)
    out[:, 2] += 0.0065 * lower * front

    base.shift_region(out, mouth, (0.052, 0.042, 0.036), (0.0, 0.0006, 0.0020), 0.04, 1.03)
    base.scale_region(out, mouth, (0.050, 0.040, 0.034), (0.94, 1.0, 0.90), 0.04, 1.03)
    base.shift_region(out, chin, (0.046, 0.046, 0.043), (0.0, 0.0006, 0.0045), 0.02, 1.03)
    base.scale_region(out, chin, (0.046, 0.046, 0.043), (0.92, 0.98, 0.82), 0.02, 1.03)

    # Small but readable nose: broaden alae and project tip just enough for soft light.
    base.scale_region(out, nose, (0.033, 0.033, 0.034), (1.12, 1.0, 0.98), 0.04, 1.04)
    base.shift_region(out, nose, (0.033, 0.035, 0.035), (0.0, -0.0015, 0.0005), 0.03, 1.04)

    # Feminine high apple-cheek support.
    for center in ((lm[40] + lm[31] + lm[48]) / 3.0, (lm[46] + lm[35] + lm[54]) / 3.0):
        base.shift_region(out, center, (0.044, 0.041, 0.040), (0.0, -0.0016, 0.0015), 0.02, 1.04)
    return out


def build_character_v3(face_path, height):
    head, objects, mapped, stats = _original_build(face_path, height)
    old_lm = mapped[base.K].copy()
    blocks = head.data.shape_keys.key_blocks
    new_basis = None
    for key in blocks:
        coords = v2.key_array(key)
        coords = additional_identity_polish(coords, old_lm)
        v2.set_key_array(key, coords)
        if key.name == "Basis":
            new_basis = coords
    mapped = new_basis.copy()
    new_lm = mapped[base.K]

    # Larger, slightly wider eye system. Move each whole eye outward by 1.2 mm.
    for side, indices, outward in (("R", range(36, 42), -0.0012), ("L", range(42, 48), 0.0012)):
        old_center = old_lm[list(indices)].mean(0)
        new_center = new_lm[list(indices)].mean(0) + np.array([outward, 0.0, 0.0])
        for prefix, factor in (
            ("AINA_Eye_", (1.12, 1.0, 1.13)),
            ("AINA_Iris_", (1.18, 1.0, 1.18)),
            ("AINA_Pupil_", (1.12, 1.0, 1.12)),
        ):
            obj = bpy.data.objects.get(prefix + side)
            if not obj or not obj.data.shape_keys:
                continue
            for key in obj.data.shape_keys.key_blocks:
                coords = v2.key_array(key)
                coords += new_center - old_center
                coords = new_center + (coords - new_center) * np.asarray(factor)
                v2.set_key_array(key, coords)

    # Lower and thicken real brow ribbons so they frame the eyes like the target.
    for side, ids in (("Right", range(17, 22)), ("Left", range(22, 27))):
        obj = bpy.data.objects.get("AINA_Brow_" + side)
        if not obj or not obj.data.shape_keys:
            continue
        old_center = old_lm[list(ids)].mean(0)
        new_center = new_lm[list(ids)].mean(0) + np.array([0.0, -0.0008, -0.0024])
        for key in obj.data.shape_keys.key_blocks:
            coords = v2.key_array(key)
            coords += new_center - old_center
            coords[:, 2] = new_center[2] + (coords[:, 2] - new_center[2]) * 1.25
            v2.set_key_array(key, coords)
    return head, objects, mapped, stats


v2.build_character = build_character_v3


def create_hair_v3(head, hair_mat):
    # Raised top/back cap; no side bob panels. Face framing comes from swept locks.
    verts, faces = [], []
    nphi, nt = 104, 27
    center = np.array([0.0, 0.034, 1.648])
    rx, ry, rz = 0.111, 0.103, 0.130
    for i in range(nphi):
        phi = 2.0 * math.pi * i / nphi
        sy = math.sin(phi)
        side = abs(math.cos(phi))
        if sy < 0.35:
            tmax = 0.66 + 0.30 * side
        else:
            tmax = 2.02
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

    bun = vis.create_uv_sphere("AINA_Hair_Bun", (0.0, 0.116, 1.742), (0.052, 0.046, 0.055), hair_mat, head)

    for side_name, sign in (("L", -1.0), ("R", 1.0)):
        for i in range(7):
            root = np.array([sign * (0.002 + 0.0032 * i), -0.058 + 0.0007 * i, 1.748 - 0.0014 * i])
            end = np.array([sign * (0.020 + 0.0090 * i), -0.094 + 0.0017 * i, 1.684 - 0.0110 * i])
            mid = (root + end) * 0.5 + np.array([sign * 0.009, -0.012, 0.013])
            vis.make_curve(f"AINA_SweptFringe_{side_name}_{i+1}", [root, mid, end], 0.00105 + 0.00006 * i, hair_mat, head)

    # Only four long wisps, matching the airy updo reference.
    wisps = [
        [(-0.014, -0.062, 1.735), (-0.027, -0.087, 1.670), (-0.031, -0.097, 1.600)],
        [(0.014, -0.062, 1.735), (0.027, -0.087, 1.670), (0.031, -0.097, 1.600)],
        [(-0.040, -0.055, 1.716), (-0.063, -0.078, 1.640), (-0.069, -0.084, 1.560)],
        [(0.040, -0.055, 1.716), (0.063, -0.078, 1.640), (0.069, -0.084, 1.560)],
    ]
    for i, points in enumerate(wisps):
        vis.make_curve(f"AINA_Wisp_{i+1}", points, 0.00050, hair_mat, head)

    for i, value in enumerate(np.linspace(-1.0, 1.0, 9)):
        root = (value * 0.005, -0.050, 1.766)
        end = (value * 0.090, -0.002 + abs(value) * 0.020, 1.688 - abs(value) * 0.020)
        mid = ((root[0] + end[0]) * 0.5, -0.044, (root[2] + end[2]) * 0.5 + 0.014)
        vis.make_curve(f"AINA_CrownFlow_{i+1}", [root, mid, end], 0.00042, hair_mat, head)
    return [cap, bun]


vis.create_hair = create_hair_v3


def create_neck_and_collar_v3(skin_mat, suit_mat, accent_mat):
    # Collar sits directly below the jaw; only a short neck remains visible.
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.041, depth=0.095, location=(0.0, 0.023, 1.475))
    neck = bpy.context.object; neck.name = "AINA_Neck"; neck.scale = (0.94, 0.78, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True); neck.data.materials.append(skin_mat)
    for poly in neck.data.polygons: poly.use_smooth = True

    bpy.ops.mesh.primitive_torus_add(major_radius=0.054, minor_radius=0.011, major_segments=64, minor_segments=16, location=(0.0, 0.020, 1.440))
    collar = bpy.context.object; collar.name = "AINA_Pearl_Collar"; collar.scale = (1.0, 0.76, 1.32)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True); collar.data.materials.append(suit_mat)
    for poly in collar.data.polygons: poly.use_smooth = True

    chest = vis.create_uv_sphere("AINA_Portrait_Chest", (0.0, 0.060, 1.345), (0.175, 0.090, 0.060), suit_mat)
    for side in (-1.0, 1.0):
        vis.create_uv_sphere(
            f"AINA_Shoulder_{'L' if side < 0 else 'R'}",
            (side * 0.125, 0.060, 1.355),
            (0.115, 0.080, 0.060),
            suit_mat,
        )
    accent = vis.create_uv_sphere("AINA_Collar_Accent", (0.0, -0.054, 1.440), (0.009, 0.005, 0.013), accent_mat)
    return [neck, collar, chest, accent]


vis.create_neck_and_collar = create_neck_and_collar_v3


# Improve camera framing and tonal separation.
_original_setup = vis.setup_render


def setup_render_v3(out):
    scene, camera, preview = _original_setup(out)
    scene.world.color = (0.93, 0.94, 0.97)
    try:
        scene.view_settings.exposure = -0.55
    except Exception:
        pass
    camera.data.lens = 86
    return scene, camera, preview


vis.setup_render = setup_render_v3

_original_render = vis.render


def render_v3(scene, camera, preview, objects, name, values, three_q=False):
    v2.reset_shapes(objects); v2.apply_case(objects, values)
    if three_q:
        camera.location = (0.34, -0.88, 1.62); target = (0.0, 0.0, 1.610)
    else:
        camera.location = (0.0, -0.94, 1.615); target = (0.0, 0.0, 1.612)
    camera.rotation_euler = (vis.Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.resolution_x = 600; scene.render.resolution_y = 600; scene.render.resolution_percentage = 100
    path = preview / name; scene.render.filepath = str(path); bpy.ops.render.render(write_still=True)
    return path


vis.render = render_v3


_original_material = v2.material


def material_v3(name, color, roughness, metallic=0.0):
    overrides = {
        "AINA_Skin": ((0.46, 0.32, 0.30, 1.0), 0.50, 0.0),
        "AINA_Hair_Silver": ((0.34, 0.40, 0.52, 1.0), 0.38, 0.03),
        "AINA_Lash": ((0.006, 0.004, 0.007, 1.0), 0.24, 0.0),
        "AINA_Suit_Pearl": ((0.46, 0.54, 0.68, 1.0), 0.38, 0.04),
    }
    if name in overrides:
        color, roughness, metallic = overrides[name]
    return _original_material(name, color, roughness, metallic)


v2.material = material_v3


if __name__ == "__main__":
    vis.main()
