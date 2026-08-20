#!/usr/bin/env python3
"""AINA final real-model visual identity assembly QA.

Consumes the surface-refined real OBJ and the identity-preserving 52-control v3
expression system. Adds only real Blender geometry: silver hair, lashes, a neck,
and a pearl collar. Renders neutral/happy front and three-quarter views for the
visual identity gate. No replacement effect art and no VRM export.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aina_surface_expression_qa_v3 as v3

v2 = v3.v2
base = v3.base


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--height", type=float, default=1.72)
    return ap.parse_args(argv)


def make_curve(name, points, radius, material, parent=None):
    curve = bpy.data.curves.new(name + "_Curve", "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 5
    curve.bevel_depth = radius
    curve.bevel_resolution = 4
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for bp, point in zip(spline.bezier_points, points):
        bp.co = point
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    curve.materials.append(material)
    if parent:
        obj.parent = parent
    return obj


def create_uv_sphere(name, location, scale, material, parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    if parent:
        obj.parent = parent
    return obj


def create_hair(head, hair_mat):
    # Scalp cap with a centre-part opening. Fine real curves carry direction and
    # prevent the old solid-helmet look.
    verts, faces = [], []
    nphi, nt = 112, 30
    center = np.array([0.0, 0.026, 1.635])
    rx, ry, rz = 0.108, 0.098, 0.126
    for i in range(nphi):
        phi = 2.0 * math.pi * i / nphi
        sy = math.sin(phi)
        side_factor = abs(math.cos(phi))
        if sy < 0.0:
            tmax = 1.04 + 0.31 * side_factor
        else:
            tmax = 2.03
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
    cap.data.materials.append(hair_mat)
    cap.parent = head
    for poly in cap.data.polygons:
        poly.use_smooth = True

    bun = create_uv_sphere("AINA_Hair_Bun", (0.0, 0.105, 1.704), (0.047, 0.042, 0.050), hair_mat, head)

    # Centre-parted bangs, matching the approved AINA silhouette.
    for side, sign in (("L", -1.0), ("R", 1.0)):
        for i in range(13):
            root = np.array([sign * (0.003 + 0.0018 * i), -0.057 + 0.0004 * i, 1.731 - 0.0009 * i])
            end = np.array([sign * (0.014 + 0.0052 * i), -0.093 + 0.0012 * i, 1.666 - 0.0047 * i])
            mid = (root + end) * 0.5 + np.array([sign * 0.0055, -0.011, 0.010])
            make_curve(f"AINA_Fringe_{side}_{i+1}", [root, mid, end], 0.00072 + 0.000025 * i, hair_mat, head)

    wisps = [
        [(-0.006, -0.062, 1.727), (-0.014, -0.085, 1.675), (-0.017, -0.096, 1.623)],
        [(0.006, -0.062, 1.727), (0.014, -0.085, 1.675), (0.017, -0.096, 1.623)],
        [(-0.016, -0.059, 1.720), (-0.029, -0.082, 1.666), (-0.034, -0.093, 1.612)],
        [(0.016, -0.059, 1.720), (0.029, -0.082, 1.666), (0.034, -0.093, 1.612)],
    ]
    for i, points in enumerate(wisps):
        make_curve(f"AINA_Wisp_{i+1}", points, 0.00048, hair_mat, head)

    # Ear-framing side locks.
    for side, sign in (("L", 1.0), ("R", -1.0)):
        chains = [
            [(sign * 0.066, -0.052, 1.681), (sign * 0.079, -0.064, 1.625), (sign * 0.074, -0.065, 1.565)],
            [(sign * 0.074, -0.044, 1.672), (sign * 0.086, -0.055, 1.605), (sign * 0.078, -0.058, 1.540)],
            [(sign * 0.081, -0.035, 1.660), (sign * 0.090, -0.046, 1.588), (sign * 0.080, -0.051, 1.522)],
        ]
        for i, points in enumerate(chains):
            make_curve(f"AINA_SideLock_{side}_{i+1}", points, 0.00095 + 0.00010 * i, hair_mat, head)

    # Crown flow lines break the cap surface and show strand direction.
    for i, value in enumerate(np.linspace(-1.0, 1.0, 17)):
        root = (value * 0.006, -0.052, 1.751)
        end = (value * 0.087, -0.018 + abs(value) * 0.022, 1.668 - abs(value) * 0.019)
        mid = ((root[0] + end[0]) * 0.5, -0.046, (root[2] + end[2]) * 0.5 + 0.013)
        make_curve(f"AINA_CrownFlow_{i+1}", [root, mid, end], 0.00040, hair_mat, head)
    return [cap, bun]


def create_lashes(head, mapped, lash_mat):
    lm = mapped[base.K]
    objects = []
    for side, indices in (("R", range(36, 42)), ("L", range(42, 48))):
        c = lm[list(indices)].mean(0)
        rx = 0.0183
        points = [
            (c[0] - rx, -0.0142, c[2] + (0.0009 if side == "R" else 0.0002)),
            (c[0] - rx * 0.52, -0.0144, c[2] + 0.0048),
            (c[0], -0.0145, c[2] + 0.0062),
            (c[0] + rx * 0.52, -0.0144, c[2] + 0.0048),
            (c[0] + rx, -0.0142, c[2] + (0.0002 if side == "R" else 0.0009)),
        ]
        objects.append(make_curve(f"AINA_Lash_{side}", points, 0.00064, lash_mat, head))
        outer = np.asarray(points[0 if side == "R" else -1])
        direction = -1.0 if side == "R" else 1.0
        for i in range(3):
            start = outer + np.array([direction * 0.0012 * i, 0.0, 0.00035 * i])
            end = start + np.array([direction * (0.0032 + 0.0005 * i), -0.0001, 0.0021 + 0.00035 * i])
            objects.append(make_curve(f"AINA_LashTail_{side}_{i+1}", [start, end], 0.00036, lash_mat, head))
    return objects


def create_neck_and_collar(skin_mat, suit_mat, accent_mat):
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.043, depth=0.18, location=(0.0, 0.023, 1.455))
    neck = bpy.context.object; neck.name = "AINA_Neck"; neck.scale = (0.94, 0.78, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    neck.data.materials.append(skin_mat)
    for poly in neck.data.polygons: poly.use_smooth = True

    bpy.ops.mesh.primitive_torus_add(major_radius=0.055, minor_radius=0.012, major_segments=64, minor_segments=16, location=(0.0, 0.020, 1.455))
    collar = bpy.context.object; collar.name = "AINA_Pearl_Collar"; collar.scale = (1.0, 0.76, 1.35)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    collar.data.materials.append(suit_mat)
    for poly in collar.data.polygons: poly.use_smooth = True

    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, location=(0.0, 0.055, 1.375))
    bust = bpy.context.object; bust.name = "AINA_Portrait_Bust"; bust.scale = (0.24, 0.13, 0.13)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bust.data.materials.append(suit_mat)
    for poly in bust.data.polygons: poly.use_smooth = True

    accent = create_uv_sphere("AINA_Collar_Accent", (0.0, -0.055, 1.455), (0.010, 0.006, 0.014), accent_mat)
    return [neck, collar, bust, accent]


def reset_and_apply(objects, values):
    v2.reset_shapes(objects)
    v2.apply_case(objects, values)


def setup_render(out):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.91, 0.93, 0.97)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = -0.20
    except Exception:
        pass

    def area(name, location, energy, size, target=(0, 0, 1.61)):
        data = bpy.data.lights.new(name, "AREA"); data.energy = energy; data.shape = "DISK"; data.size = size
        obj = bpy.data.objects.new(name, data); bpy.context.collection.objects.link(obj); obj.location = location
        obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()

    area("AINA_Key", (1.25, -1.75, 2.25), 400, 2.6)
    area("AINA_Fill", (-1.35, -1.55, 1.95), 210, 2.8)
    area("AINA_Rim", (0.0, 1.55, 2.25), 340, 2.4)
    area("AINA_FaceSoft", (0.0, -2.10, 1.60), 60, 3.0)

    camera_data = bpy.data.cameras.new("AINA_Visual_Camera")
    camera = bpy.data.objects.new("AINA_Visual_Camera", camera_data)
    bpy.context.collection.objects.link(camera); scene.camera = camera; camera.data.lens = 82
    preview = out / "Preview"; preview.mkdir(parents=True, exist_ok=True)
    return scene, camera, preview


def render(scene, camera, preview, objects, name, values, three_q=False):
    reset_and_apply(objects, values)
    if three_q:
        camera.location = (0.38, -1.02, 1.62); target = (0.0, 0.0, 1.605)
    else:
        camera.location = (0.0, -1.08, 1.615); target = (0.0, 0.0, 1.610)
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.resolution_x = 600; scene.render.resolution_y = 600; scene.render.resolution_percentage = 100
    path = preview / name; scene.render.filepath = str(path); bpy.ops.render.render(write_still=True)
    return path


def main():
    args = parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True); (out / "QA").mkdir(exist_ok=True)
    v2.clear_scene()
    head, expression_objects, mapped, stats = v2.build_character(args.face, args.height)

    hair_mat = v2.material("AINA_Hair_Silver", (0.64, 0.69, 0.79, 1.0), 0.27, 0.05)
    lash_mat = v2.material("AINA_Lash", (0.018, 0.014, 0.020, 1.0), 0.30)
    suit_mat = v2.material("AINA_Suit_Pearl", (0.70, 0.76, 0.86, 1.0), 0.30, 0.08)
    accent_mat = v2.material("AINA_Accent", (0.20, 0.56, 0.80, 1.0), 0.18, 0.15)
    skin_mat = bpy.data.materials.get("AINA_Skin")

    create_hair(head, hair_mat)
    create_lashes(head, mapped, lash_mat)
    create_neck_and_collar(skin_mat, suit_mat, accent_mat)

    scene, camera, preview = setup_render(out)
    renders = [
        render(scene, camera, preview, expression_objects, "AINA_VISUAL_NEUTRAL_FRONT.png", v2.CASES["neutral"], False),
        render(scene, camera, preview, expression_objects, "AINA_VISUAL_NEUTRAL_3Q.png", v2.CASES["neutral"], True),
        render(scene, camera, preview, expression_objects, "AINA_VISUAL_HAPPY_FRONT.png", v2.CASES["happy"], False),
        render(scene, camera, preview, expression_objects, "AINA_VISUAL_HAPPY_3Q.png", v2.CASES["happy"], True),
    ]
    v2.reset_shapes(expression_objects)
    blend = out / "AINA_FINAL_VISUAL_IDENTITY_QA.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))

    qa = {
        "product": "AINA Final Real-Model Visual Identity QA",
        "real_mesh": True,
        "replacement_effect_art_generated": False,
        "vrm_exported": False,
        "shape_control_count": 52,
        "rendered": [p.name for p in renders],
        "renders_present": all(p.exists() and p.stat().st_size > 5000 for p in renders),
        "visual_identity_lock": False,
        "next_gate": "Compare the actual neutral/happy front and 3Q renders with approved AINA art before final VRM assembly.",
        "blend": str(blend),
        "blend_bytes": blend.stat().st_size,
    }
    (out / "QA" / "AINA_FINAL_VISUAL_IDENTITY_QA.json").write_text(__import__("json").dumps(qa, indent=2), encoding="utf-8")
    print(__import__("json").dumps(qa, indent=2))


if __name__ == "__main__":
    main()
