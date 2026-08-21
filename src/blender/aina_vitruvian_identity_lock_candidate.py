#!/usr/bin/env python3
"""Build the real AINA identity-lock candidate on the original CC0 FACS head.

The earlier v3 residual was intentionally rejected after real rendering because
it over-compressed the face and over-enlarged the eyes. This candidate restarts
from the untouched Vitruvian GLB, keeps the original eye material segmentation,
applies the proven v2 neutral sculpt, then adds one bounded residual pass with
moderate AINA proportions. Every skin displacement is added identically to Basis
and all 26 FACS/viseme shape keys, so expression deltas remain unchanged.
No replacement effect art is generated and no VRM is exported.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aina_vitruvian_head_probe as probe
import aina_vitruvian_identity_sculpt_v2 as v2


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def tight_semantic(magnitude: np.ndarray, low_quantile: float) -> np.ndarray:
    positive = magnitude[magnitude > 1e-8]
    if not len(positive):
        return np.zeros_like(magnitude)
    low = max(float(np.quantile(positive, low_quantile)), 2.0e-5)
    high = max(float(np.quantile(positive, 0.997)), low * 1.35)
    return v2.smoothstep01((magnitude - low) / max(high - low, 1e-9))


def capture_expression_deltas(skin) -> dict[str, np.ndarray]:
    keys = skin.data.shape_keys.key_blocks
    basis = v2.key_array(keys.get("Basis") or keys[0])
    return {
        key.name: v2.key_array(key) - basis
        for key in keys if key.name != "Basis"
    }


def validate_expression_deltas(skin, original: dict[str, np.ndarray]) -> dict:
    keys = skin.data.shape_keys.key_blocks
    basis = v2.key_array(keys.get("Basis") or keys[0])
    max_error = 0.0
    sum_squared = 0.0
    count = 0
    for key in keys:
        if key.name == "Basis":
            continue
        error = (v2.key_array(key) - basis) - original[key.name]
        if len(error):
            max_error = max(max_error, float(np.linalg.norm(error, axis=1).max()))
            sum_squared += float(np.sum(error * error))
            count += error.size
    return {
        "shape_delta_preservation_max_m": max_error,
        "shape_delta_preservation_rms_m": math.sqrt(sum_squared / max(count, 1)),
    }


def apply_same_delta_to_all_keys(skin, world_delta: np.ndarray) -> None:
    local_delta = v2.world_vector_to_local(skin, world_delta)
    for key in skin.data.shape_keys.key_blocks:
        v2.set_key_array(key, v2.key_array(key) + local_delta)
    skin.data.update()


def eye_centres(scene):
    eyeballs = v2.eyeball_objects(scene)
    centres = [v2.world_vertices(obj).mean(axis=0) for obj in eyeballs if len(obj.data.vertices)]
    centres.sort(key=lambda point: point[0])
    return centres, eyeballs


def residual_sculpt(skin, centres: list[np.ndarray]) -> dict:
    keys = skin.data.shape_keys.key_blocks
    basis_local = v2.key_array(keys.get("Basis") or keys[0])
    points = v2.to_world(skin, basis_local)
    lo, hi = points.min(axis=0), points.max(axis=0)
    centre = (lo + hi) * 0.5
    head_height = float(hi[2] - lo[2])
    head_width = float(hi[0] - lo[0])

    if len(centres) >= 2:
        eyes = [centres[0], centres[-1]]
        average = np.mean(eyes, axis=0)
        face_x = float(average[0])
        eye_z = float(average[2])
        forward_sign = -1.0 if average[1] < centre[1] else 1.0
    else:
        face_x = float(centre[0])
        eye_z = float(lo[2] + 0.60 * head_height)
        forward_sign = -1.0
        spacing = 0.18 * head_width
        eyes = [
            np.array([face_x - spacing, centre[1] - 0.055, eye_z]),
            np.array([face_x + spacing, centre[1] - 0.055, eye_z]),
        ]

    frontness = forward_sign * (points[:, 1] - centre[1])
    front_low = float(np.quantile(frontness, 0.38))
    front_high = float(np.quantile(frontness, 0.94))
    face_front = v2.smoothstep01((frontness - front_low) / max(front_high - front_low, 1e-9))
    face_side = v2.smoothstep01((0.150 - np.abs(points[:, 0] - face_x)) / 0.075)
    face_weight = face_front * face_side

    eye_mag = v2.max_shape_delta(skin, basis_local, ("eyes_closed", "eyes_opened", "eyes_squint"))
    brow_mag = v2.max_shape_delta(skin, basis_local, ("eyebrows_",))
    lip_mag = v2.max_shape_delta(skin, basis_local, ("lips_up", "smile_lips", "kiss"))
    eye_sem = tight_semantic(eye_mag, 0.58)
    brow_sem = tight_semantic(brow_mag, 0.56)
    lip_sem = tight_semantic(lip_mag, 0.72)

    below_eye = v2.smoothstep01((eye_z - 0.004 - points[:, 2]) / 0.030)
    fallback_mouth = np.array([face_x, centre[1] + forward_sign * 0.30 * head_height, eye_z - 0.066])
    mouth = v2.weighted_center(points, lip_sem * face_weight * below_eye, fallback_mouth)
    mouth_z = float(mouth[2])

    lower_geometry = (
        (points[:, 2] < mouth_z)
        & (points[:, 2] > lo[2] + 0.09 * head_height)
        & (np.abs(points[:, 0] - face_x) < 0.078)
        & (face_weight > 0.32)
    )
    chin_z = float(np.quantile(points[lower_geometry, 2], 0.055)) if np.any(lower_geometry) else mouth_z - 0.038

    delta = np.zeros_like(points)

    # Moderate midface/lower-face shortening. v3 used 0.805 and overshot; the
    # identity-lock candidate uses 0.90 and preserves a natural jaw-to-mouth span.
    lower_ratio = np.clip(
        (eye_z + 0.006 - points[:, 2])
        / max(eye_z + 0.006 - (chin_z - 0.018), 1e-6),
        0.0,
        1.0,
    )
    compression_weight = face_weight * v2.smoothstep01(lower_ratio)
    compressed_z = eye_z + (points[:, 2] - eye_z) * 0.90
    delta[:, 2] += (compressed_z - points[:, 2]) * compression_weight

    # Soft V lower third and small rounded chin.
    lower = np.clip(
        (mouth_z + 0.012 - points[:, 2])
        / max(mouth_z + 0.012 - (chin_z - 0.016), 1e-6),
        0.0,
        1.0,
    )
    lower_weight = face_weight * v2.smoothstep01(lower)
    taper = 1.0 - 0.10 * np.power(lower, 1.20)
    delta[:, 0] += ((points[:, 0] - face_x) * taper - (points[:, 0] - face_x)) * lower_weight
    chin = np.array([face_x, mouth[1] - forward_sign * 0.004, chin_z + 0.008])
    chin_weight = v2.add_scaled_region(
        delta, points, chin, (0.052, 0.060, 0.050),
        (0.84, 0.97, 0.90), weight=face_weight, inner=0.02, outer=1.12,
    )
    delta[:, 2] += 0.0024 * chin_weight
    delta[:, 1] += -forward_sign * 0.0008 * chin_weight

    # Slightly narrower oval cranium under the final updo.
    top = v2.smoothstep01((points[:, 2] - (eye_z + 0.032)) / 0.075)
    top_weight = top * v2.smoothstep01((0.125 - np.abs(points[:, 0] - face_x)) / 0.052)
    delta[:, 0] += -0.026 * (points[:, 0] - face_x) * top_weight
    delta[:, 2] += -0.0010 * top_weight * v2.smoothstep01((points[:, 2] - (eye_z + 0.080)) / 0.045)

    # Almond eye apertures: larger than v2, but well below the rejected v3 pass.
    eye_reports = []
    for eye in eyes:
        side = -1.0 if eye[0] < face_x else 1.0
        spatial = v2.ellipsoid_weight(points, eye, (0.052, 0.055, 0.043), 0.0, 1.18)
        weight = spatial * np.maximum(eye_sem, 0.08 * brow_sem)
        delta[:, 0] += (points[:, 0] - eye[0]) * 0.10 * weight
        delta[:, 2] += (points[:, 2] - eye[2]) * 0.14 * weight
        delta[:, 0] += side * 0.00055 * weight
        delta[:, 2] += 0.0015 * weight
        delta[:, 1] += forward_sign * 0.0008 * spatial * eye_sem
        outer = eye + np.array([side * 0.023, 0.0, 0.001])
        outer_weight = v2.ellipsoid_weight(points, outer, (0.025, 0.033, 0.021), 0.0, 1.08) * eye_sem
        delta[:, 0] += side * 0.0009 * outer_weight
        delta[:, 2] += 0.0009 * outer_weight
        under = eye + np.array([0.0, -forward_sign * 0.004, -0.019])
        under_weight = v2.add_shift(
            delta, points, under, (0.052, 0.055, 0.036),
            (0.0, forward_sign * 0.0021, 0.0008), weight=face_weight, outer=1.14,
        )
        delta[:, 0] += -side * 0.00035 * under_weight
        eye_reports.append({"center": eye.tolist(), "vertices": int(np.sum(weight > 0.08))})

    delta[:, 1] += -forward_sign * 0.0014 * brow_sem * face_weight
    delta[:, 2] += -0.0003 * brow_sem * face_weight

    # Delicate nose: shorter vertically, narrower base and a small lifted tip.
    central_nose = (
        (np.abs(points[:, 0] - face_x) < 0.027)
        & (points[:, 2] > mouth_z + 0.010)
        & (points[:, 2] < eye_z - 0.004)
        & (face_weight > 0.32)
    )
    if np.any(central_nose):
        nose_index = np.where(central_nose)[0][np.argmax(frontness[central_nose])]
        tip = points[nose_index].copy()
    else:
        tip = np.array([face_x, centre[1] + forward_sign * 0.34 * head_height, mouth_z + 0.038])
    bridge = np.array([face_x, tip[1], mouth_z + 0.60 * (eye_z - mouth_z)])
    bridge_weight = v2.add_scaled_region(
        delta, points, bridge, (0.030, 0.046, 0.060),
        (0.88, 0.97, 0.92), weight=face_weight, outer=1.13,
    )
    delta[:, 2] += 0.0022 * bridge_weight
    delta[:, 1] += -forward_sign * 0.0009 * bridge_weight
    tip_weight = v2.ellipsoid_weight(points, tip, (0.023, 0.032, 0.027), 0.02, 1.09) * face_weight
    delta[:, 2] += 0.0028 * tip_weight
    delta[:, 1] += -forward_sign * 0.0010 * tip_weight
    delta[:, 0] += -(points[:, 0] - face_x) * 0.07 * tip_weight
    nose_base = np.array([face_x, tip[1], mouth_z + 0.020])
    base_weight = v2.add_scaled_region(
        delta, points, nose_base, (0.034, 0.040, 0.031),
        (0.86, 0.98, 0.92), weight=face_weight, outer=1.10,
    )
    delta[:, 2] += 0.0015 * base_weight

    # Fuller integrated lips without the v3 duck-lip deformation.
    lip_spatial = v2.ellipsoid_weight(points, mouth, (0.052, 0.050, 0.030), 0.0, 1.14)
    lip_weight = lip_spatial * lip_sem
    delta[:, 0] += (points[:, 0] - mouth[0]) * 0.040 * lip_weight
    delta[:, 2] += (points[:, 2] - mouth[2]) * 0.18 * lip_weight
    delta[:, 2] += 0.0015 * lip_weight
    delta[:, 1] += forward_sign * 0.0013 * lip_weight
    corner_ratio = np.clip(np.abs(points[:, 0] - mouth[0]) / 0.042, 0.0, 1.0)
    delta[:, 2] += 0.00065 * np.power(corner_ratio, 1.8) * lip_weight

    # High youthful apple-cheek support.
    for eye in eyes:
        side = -1.0 if eye[0] < face_x else 1.0
        cheek = np.array([
            eye[0] + side * 0.002,
            tip[1] - forward_sign * 0.010,
            mouth_z + 0.54 * (eye_z - mouth_z),
        ])
        cheek_weight = v2.add_shift(
            delta, points, cheek, (0.060, 0.058, 0.055),
            (0.0, forward_sign * 0.0032, 0.0012), weight=face_weight, outer=1.15,
        )
        delta[:, 0] += -side * 0.00065 * cheek_weight

    # Smaller ears and a portrait-ready narrow neck.
    ear_band = v2.smoothstep01((np.abs(points[:, 0] - face_x) - head_width * 0.34) / (head_width * 0.12))
    ear_vertical = v2.smoothstep01((points[:, 2] - (mouth_z - 0.025)) / 0.032) * v2.smoothstep01(((eye_z + 0.030) - points[:, 2]) / 0.032)
    ear_depth = v2.smoothstep01((front_high - frontness) / max(front_high - front_low, 1e-9))
    ear_weight = ear_band * ear_vertical * ear_depth
    delta[:, 0] += -(points[:, 0] - face_x) * 0.085 * ear_weight
    delta[:, 1] += forward_sign * 0.0016 * ear_weight
    delta[:, 2] += 0.0020 * ear_weight
    neck = v2.smoothstep01(((chin_z - 0.008) - points[:, 2]) / 0.040) * v2.smoothstep01((0.092 - np.abs(points[:, 0] - face_x)) / 0.042)
    delta[:, 0] += -0.10 * (points[:, 0] - face_x) * neck
    delta[:, 2] += 0.0015 * neck

    lengths = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1.0, 0.0115 / np.maximum(lengths, 1e-9))[:, None]
    apply_same_delta_to_all_keys(skin, delta)
    return {
        "forward_sign_y": forward_sign,
        "bounds_min": lo.tolist(),
        "bounds_max": hi.tolist(),
        "eye_z_m": eye_z,
        "mouth_center_world": mouth.tolist(),
        "chin_z_m": chin_z,
        "nose_tip_world": tip.tolist(),
        "eye_regions": eye_reports,
        "semantic_vertex_counts": {
            "eye": int(np.sum(eye_sem > 0.08)),
            "brow": int(np.sum(brow_sem > 0.08)),
            "lip_tight": int(np.sum(lip_sem > 0.08)),
        },
        "max_residual_displacement_m": float(np.linalg.norm(delta, axis=1).max()),
        "rms_residual_displacement_m": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
        "moved_vertices_over_0_5mm": int(np.sum(np.linalg.norm(delta, axis=1) > 0.0005)),
    }


def refine_eye_anatomy(scene, centres: list[np.ndarray], face_x: float) -> dict:
    objects = v2.eye_related_objects(scene)
    if len(centres) < 2:
        return {"objects": [], "max_displacement_m": 0.0, "iris_vertex_count": 0}
    left, right = centres[0], centres[-1]
    max_displacement = 0.0
    iris_vertex_count = 0
    changed = []
    for obj in objects:
        local = v2.mesh_local_array(obj)
        world = v2.to_world(obj, local)
        result = world.copy()
        original = world.copy()
        iris_slots = {
            index for index, material in enumerate(obj.data.materials)
            if material and ("iris" in material.name.lower() or "eyeback" in material.name.lower())
        }
        iris_ids = set()
        if iris_slots:
            for polygon in obj.data.polygons:
                if polygon.material_index in iris_slots:
                    iris_ids.update(polygon.vertices)
        iris_vertex_count += len(iris_ids)
        for index, point in enumerate(world):
            centre = left if point[0] < face_x else right
            side = -1.0 if centre[0] < face_x else 1.0
            relative = point - centre
            relative[0] *= 1.055
            relative[2] *= 1.055
            relative[1] *= 1.010
            result[index] = centre + relative + np.array([side * 0.00065, 0.0, 0.0012])
        for index in iris_ids:
            centre = left if result[index, 0] < face_x else right
            relative = result[index] - centre
            relative[0] *= 1.10
            relative[2] *= 1.10
            result[index] = centre + relative
        v2.set_mesh_local_array(obj, v2.to_local(obj, result))
        displacement = np.linalg.norm(result - original, axis=1)
        if len(displacement):
            max_displacement = max(max_displacement, float(displacement.max()))
        changed.append(obj.name)
    return {"objects": changed, "max_displacement_m": max_displacement, "iris_vertex_count": iris_vertex_count}


def make_material(name, color, roughness=0.45, metallic=0.0):
    return v2.make_material(name, color, roughness, metallic)


def install_candidate_materials(scene, skin):
    skin_mat = make_material("AINA_Candidate_Skin", (0.19, 0.13, 0.14), 0.50)
    mouth_mat = make_material("AINA_Candidate_Mouth", (0.055, 0.006, 0.014), 0.40)
    sclera = make_material("AINA_Candidate_Sclera", (0.48, 0.55, 0.64), 0.24)
    iris = make_material("AINA_Candidate_Iris", (0.025, 0.14, 0.17), 0.24)
    dark = make_material("AINA_Candidate_EyeBack", (0.002, 0.003, 0.006), 0.18)
    cornea = make_material("AINA_Candidate_Cornea", (0.18, 0.26, 0.32), 0.07)
    shadow = make_material("AINA_Candidate_LashShadow", (0.012, 0.008, 0.014), 0.38)
    tear = make_material("AINA_Candidate_Tear", (0.14, 0.030, 0.045), 0.22)
    brow = make_material("AINA_Candidate_Brow", (0.020, 0.022, 0.030), 0.45)

    original_skin_slots = [material.name.lower() if material else "" for material in skin.data.materials]
    slots = []
    for name in original_skin_slots:
        slots.append(mouth_mat if "mouth" in name else skin_mat)
    v2.assign_material_slots(skin, slots or [skin_mat, mouth_mat])

    for obj in scene.objects:
        if obj.type != "MESH" or obj == skin:
            continue
        name = obj.name.lower()
        if "eyeball" in name:
            mapped = []
            for old in list(obj.data.materials):
                old_name = old.name.lower() if old else ""
                if "sclera" in old_name:
                    mapped.append(sclera)
                elif "iris" in old_name:
                    mapped.append(iris)
                elif "cornea" in old_name:
                    mapped.append(cornea)
                else:
                    mapped.append(dark)
            v2.assign_material_slots(obj, mapped or [sclera, iris, dark, cornea])
        elif "eyeshadow" in name:
            v2.assign_material_slots(obj, [shadow])
        elif "tear" in name or "caruncle" in name:
            v2.assign_material_slots(obj, [tear])
        else:
            v2.assign_material_slots(obj, [skin_mat])
    return brow


def create_clean_brows(centres: list[np.ndarray], forward_sign: float, material) -> list[str]:
    for obj in list(bpy.data.objects):
        if obj.name.startswith("AINA_Candidate_Brow_"):
            bpy.data.objects.remove(obj, do_unlink=True)
    if len(centres) < 2:
        return []
    created = []
    for side_name, eye in (("L", centres[0]), ("R", centres[-1])):
        sign = -1.0 if eye[0] < 0 else 1.0
        offsets = np.linspace(-0.027, 0.027, 9)
        points = []
        for index, offset in enumerate(offsets):
            normalized = offset / 0.027
            arch = 0.020 + 0.0045 * (1.0 - normalized * normalized)
            tail = 0.0015 * max(0.0, sign * normalized)
            points.append((
                float(eye[0] + offset),
                float(eye[1] + forward_sign * 0.024),
                float(eye[2] + arch - tail),
            ))
        curve = bpy.data.curves.new(f"AINA_Candidate_Brow_{side_name}_Curve", "CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 4
        curve.bevel_depth = 0.00058
        curve.bevel_resolution = 3
        spline = curve.splines.new("BEZIER")
        spline.bezier_points.add(len(points) - 1)
        for point, coordinate in zip(spline.bezier_points, points):
            point.co = coordinate
            point.handle_left_type = "AUTO"
            point.handle_right_type = "AUTO"
        obj = bpy.data.objects.new(f"AINA_Candidate_Brow_{side_name}", curve)
        bpy.context.collection.objects.link(obj)
        curve.materials.append(material)
        created.append(obj.name)
    return created


def render_setup(scene, meshes, centres):
    points = np.concatenate([v2.world_vertices(obj) for obj in meshes], axis=0)
    lo, hi = points.min(axis=0), points.max(axis=0)
    centre = (lo + hi) * 0.5
    size = hi - lo
    eyes = np.mean(centres, axis=0) if centres else centre
    forward_sign = -1.0 if eyes[1] < centre[1] else 1.0
    target = np.array([eyes[0], centre[1], eyes[2] - 0.020])
    distance = max(float(size[2]) * 2.72, float(size[0]) * 2.95, 0.86)
    front = np.array([target[0], centre[1] + forward_sign * distance, target[2]])

    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.world.color = (0.020, 0.024, 0.036)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = -0.70
    except Exception:
        pass
    v2.clear_render_objects(scene)
    v2.create_light("AINA_Candidate_Key", tuple(front + np.array([0.45 * size[0], 0.0, 0.48 * size[2]])), 300, 2.2, target)
    v2.create_light("AINA_Candidate_Fill", tuple(front + np.array([-0.65 * size[0], 0.18 * distance, 0.10 * size[2]])), 150, 2.7, target)
    v2.create_light("AINA_Candidate_Rim", tuple(centre + np.array([0.0, -forward_sign * distance * 0.72, 0.42 * size[2]])), 235, 2.0, target)
    camera_data = bpy.data.cameras.new("AINA_Candidate_Camera")
    camera = bpy.data.objects.new("AINA_Candidate_Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.data.lens = 90
    scene.camera = camera
    locations = {
        "FRONT": front,
        "THREE_QUARTER": front + np.array([0.43 * distance, -forward_sign * 0.10 * distance, 0.0]),
        "SIDE": centre + np.array([distance, 0.0, target[2] - centre[2]]),
        "LEFT_45": front + np.array([-0.48 * distance, -forward_sign * 0.12 * distance, 0.0]),
        "RIGHT_45": front + np.array([0.48 * distance, -forward_sign * 0.12 * distance, 0.0]),
    }
    return camera, target, locations


def render_suite(scene, skin, meshes, centres, out: Path):
    preview = out / "Preview"
    preview.mkdir(exist_ok=True)
    camera, target, locations = render_setup(scene, meshes, centres)
    outputs = {}
    v2.reset_shapes(skin)
    for name in ("FRONT", "THREE_QUARTER", "SIDE", "LEFT_45", "RIGHT_45"):
        path = preview / f"AINA_IDENTITY_LOCK_CANDIDATE_{name}.png"
        v2.render_view(scene, camera, target, locations[name], path)
        outputs[name.lower()] = str(path)
    cases = {
        "HAPPY": {"Happy": 1.0},
        "SAD": {"Sad": 1.0},
        "ANGRY": {"Angry": 1.0},
        "BLINK": {"Eyes_Closed_Max": 1.0},
        "AA": {"aa_02": 1.0, "Jaw_Lower": 0.20},
        "OU": {"ow_08": 1.0},
    }
    for name, values in cases.items():
        v2.reset_shapes(skin)
        for key_name, value in values.items():
            key = skin.data.shape_keys.key_blocks.get(key_name) if skin.data.shape_keys else None
            if key:
                key.value = value
        path = preview / f"AINA_IDENTITY_LOCK_CANDIDATE_{name}.png"
        v2.render_view(scene, camera, target, locations["FRONT"], path)
        outputs[name.lower()] = str(path)
    v2.reset_shapes(skin)
    return outputs


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)

    probe.clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(args.head))
    bpy.context.view_layer.update()
    scene = bpy.context.scene
    skin = v2.identify_skin(scene)
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    for obj in meshes:
        if obj.data.shape_keys:
            for key in obj.data.shape_keys.key_blocks:
                key.value = 0.0

    original_expression_deltas = capture_expression_deltas(skin)
    centres_initial, eyeballs = eye_centres(scene)
    face_x = float(np.mean(centres_initial, axis=0)[0]) if centres_initial else float(v2.world_vertices(skin)[:, 0].mean())

    baseline_report = v2.sculpt_skin(skin, centres_initial)
    baseline_eye_report = v2.sculpt_eye_anatomy(scene, centres_initial, face_x)
    bpy.context.view_layer.update()
    centres_after_baseline = [v2.world_vertices(obj).mean(axis=0) for obj in eyeballs if len(obj.data.vertices)]
    centres_after_baseline.sort(key=lambda point: point[0])

    residual_report = residual_sculpt(skin, centres_after_baseline)
    residual_eye_report = refine_eye_anatomy(scene, centres_after_baseline, face_x)
    bpy.context.view_layer.update()
    centres_final = [v2.world_vertices(obj).mean(axis=0) for obj in eyeballs if len(obj.data.vertices)]
    centres_final.sort(key=lambda point: point[0])

    expression_preservation = validate_expression_deltas(skin, original_expression_deltas)
    brow_material = install_candidate_materials(scene, skin)
    brow_objects = create_clean_brows(centres_final, residual_report["forward_sign_y"], brow_material)
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    renders = render_suite(scene, skin, meshes, centres_final, args.out)

    blend_path = args.out / "AINA_IDENTITY_LOCK_CANDIDATE.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_IDENTITY_LOCK_CANDIDATE.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    shape_keys = [key.name for key in skin.data.shape_keys.key_blocks]
    report = {
        "product": "AINA Real Identity Lock Candidate",
        "real_3d_model": True,
        "source_topology": "CC0 Vitruvian/Antonia FACS head imported from original GLB",
        "replacement_effect_art_generated": False,
        "skin_topology_changed": False,
        "vertices": len(skin.data.vertices),
        "triangles": len(skin.data.loop_triangles),
        "shape_key_count": max(0, len(shape_keys) - 1),
        "shape_keys": shape_keys,
        "baseline_v2_sculpt": baseline_report,
        "baseline_eye_anatomy": baseline_eye_report,
        "candidate_residual": residual_report,
        "candidate_eye_anatomy": residual_eye_report,
        "expression_preservation": expression_preservation,
        "real_brow_geometry": brow_objects,
        "identity_lock": False,
        "visual_identity_lock": False,
        "candidate": True,
        "vrm_exported": False,
        "files": {"blend": str(blend_path), "glb": str(glb_path), "renders": renders},
        "next_gate": "Inspect approved AINA front/3Q/side and all real expression renders. Set visual_identity_lock only after direct human visual acceptance; otherwise continue one bounded feature-only correction.",
    }
    (qa / "AINA_IDENTITY_LOCK_CANDIDATE_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
