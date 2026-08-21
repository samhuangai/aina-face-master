#!/usr/bin/env python3
"""AINA Vitruvian Identity Sculpt v3 — facial proportion and feature lock pass.

This pass continues directly from the real v2 BLEND. It does not replace the
model, generate effect art, or export VRM. It applies one additional neutral
identity displacement identically to Basis and every existing FACS/viseme shape
key, preserving all expression deltas while correcting the remaining AINA
identity residuals: long midface, small eye apertures, broad lower face, weak
apple cheeks, thin lips, low nose tip, large ears and wide neck.
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

import aina_vitruvian_identity_sculpt_v2 as v2


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def tight_semantic(magnitude: np.ndarray, low_quantile: float = 0.72) -> np.ndarray:
    positive = magnitude[magnitude > 1e-8]
    if not len(positive):
        return np.zeros_like(magnitude)
    low = max(float(np.quantile(positive, low_quantile)), 2.0e-5)
    high = max(float(np.quantile(positive, 0.997)), low * 1.4)
    return v2.smoothstep01((magnitude - low) / max(high - low, 1e-9))


def basis_and_original_deltas(skin):
    keys = skin.data.shape_keys.key_blocks
    basis_key = keys.get("Basis") or keys[0]
    basis_local = v2.key_array(basis_key)
    original = {
        key.name: v2.key_array(key) - basis_local
        for key in keys if key.name != "Basis"
    }
    return keys, basis_local, original


def preserve_apply(skin, local_delta: np.ndarray, original: dict[str, np.ndarray]) -> dict:
    keys = skin.data.shape_keys.key_blocks
    for key in keys:
        v2.set_key_array(key, v2.key_array(key) + local_delta)
    skin.data.update()
    new_basis = v2.key_array(keys.get("Basis") or keys[0])
    max_error = 0.0
    sum_squared = 0.0
    count = 0
    for key in keys:
        if key.name == "Basis":
            continue
        error = (v2.key_array(key) - new_basis) - original[key.name]
        if len(error):
            max_error = max(max_error, float(np.linalg.norm(error, axis=1).max()))
            sum_squared += float(np.sum(error * error))
            count += error.size
    return {
        "shape_delta_preservation_max_m": max_error,
        "shape_delta_preservation_rms_m": math.sqrt(sum_squared / max(count, 1)),
    }


def material_vertex_indices(obj, token: str) -> np.ndarray:
    wanted = {
        index for index, material in enumerate(obj.data.materials)
        if material and token in material.name.lower()
    }
    if not wanted:
        return np.zeros(0, dtype=np.int64)
    ids = set()
    for polygon in obj.data.polygons:
        if polygon.material_index in wanted:
            ids.update(polygon.vertices)
    return np.asarray(sorted(ids), dtype=np.int64)


def eye_centres(scene) -> tuple[list[np.ndarray], list]:
    eyeballs = v2.eyeball_objects(scene)
    centres = [v2.world_vertices(obj).mean(axis=0) for obj in eyeballs if len(obj.data.vertices)]
    centres.sort(key=lambda point: point[0])
    return centres, eyeballs


def art_directed_v3(skin, centres: list[np.ndarray]) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    keys, basis_local, original = basis_and_original_deltas(skin)
    points = v2.to_world(skin, basis_local)
    lo, hi = points.min(axis=0), points.max(axis=0)
    centre = (lo + hi) * 0.5
    head_height = float(hi[2] - lo[2])
    head_width = float(hi[0] - lo[0])

    if len(centres) >= 2:
        eyes = [centres[0], centres[-1]]
        eye_average = np.mean(eyes, axis=0)
        eye_z = float(eye_average[2])
        face_x = float(eye_average[0])
        forward_sign = -1.0 if eye_average[1] < centre[1] else 1.0
    else:
        eye_z = float(lo[2] + 0.60 * head_height)
        face_x = float(centre[0])
        forward_sign = -1.0
        spacing = 0.18 * head_width
        eyes = [
            np.array([face_x - spacing, centre[1] - 0.06, eye_z]),
            np.array([face_x + spacing, centre[1] - 0.06, eye_z]),
        ]

    frontness = forward_sign * (points[:, 1] - centre[1])
    front_low = float(np.quantile(frontness, 0.36))
    front_high = float(np.quantile(frontness, 0.94))
    face_front = v2.smoothstep01((frontness - front_low) / max(front_high - front_low, 1e-9))
    face_side = v2.smoothstep01((0.155 - np.abs(points[:, 0] - face_x)) / 0.075)
    face_weight = face_front * face_side

    eye_mag = v2.max_shape_delta(skin, basis_local, ("eyes_closed", "eyes_opened", "eyes_squint"))
    brow_mag = v2.max_shape_delta(skin, basis_local, ("eyebrows_",))
    lip_mag = v2.max_shape_delta(skin, basis_local, ("lips_up", "smile_lips", "kiss"))
    jaw_mag = v2.max_shape_delta(skin, basis_local, ("jaw_lower", "mouth_large"))
    eye_sem = tight_semantic(eye_mag, 0.58)
    brow_sem = tight_semantic(brow_mag, 0.54)
    lip_sem = tight_semantic(lip_mag, 0.72)
    jaw_sem = tight_semantic(jaw_mag, 0.66)

    below_eye = v2.smoothstep01((eye_z - 0.005 - points[:, 2]) / 0.030)
    lip_candidate = lip_sem * face_weight * below_eye
    fallback_mouth = np.array([face_x, centre[1] + forward_sign * 0.30 * head_height, eye_z - 0.070])
    mouth = v2.weighted_center(points, lip_candidate, fallback_mouth)
    mouth_z = float(mouth[2])

    lower_geometry = (
        (points[:, 2] < mouth_z)
        & (points[:, 2] > lo[2] + 0.09 * head_height)
        & (np.abs(points[:, 0] - face_x) < 0.075)
        & (face_weight > 0.34)
    )
    if np.any(lower_geometry):
        chin_z = float(np.quantile(points[lower_geometry, 2], 0.055))
    else:
        chin_z = mouth_z - 0.040

    delta = np.zeros_like(points)

    # The dominant residual was an adult-long midface. Compress the visible face
    # around the eye line, with a broad smooth mask that includes jaw continuity
    # but excludes the back of the skull and most of the neck.
    lower_ratio = np.clip(
        (eye_z + 0.008 - points[:, 2])
        / max(eye_z + 0.008 - (chin_z - 0.022), 1e-6),
        0.0,
        1.0,
    )
    compression_weight = face_weight * v2.smoothstep01(lower_ratio)
    compressed_z = eye_z + (points[:, 2] - eye_z) * 0.805
    delta[:, 2] += (compressed_z - points[:, 2]) * compression_weight

    # Narrow the whole lower third and round the chin. This is intentionally
    # stronger than v2 because the real comparison remained broad and generic.
    mouth_lower = np.clip(
        (mouth_z + 0.012 - points[:, 2])
        / max(mouth_z + 0.012 - (chin_z - 0.018), 1e-6),
        0.0,
        1.0,
    )
    lower_weight = face_weight * v2.smoothstep01(mouth_lower)
    taper = 1.0 - 0.19 * np.power(mouth_lower, 1.18)
    delta[:, 0] += ((points[:, 0] - face_x) * taper - (points[:, 0] - face_x)) * lower_weight

    chin_center = np.array([face_x, mouth[1] - forward_sign * 0.004, chin_z + 0.010])
    chin_weight = v2.add_scaled_region(
        delta, points, chin_center, (0.052, 0.060, 0.052),
        (0.73, 0.93, 0.84), weight=face_weight, inner=0.02, outer=1.12,
    )
    delta[:, 2] += 0.0042 * chin_weight
    delta[:, 1] += -forward_sign * 0.0018 * chin_weight

    # AINA's cranium is narrower and more oval under the updo, not a broad round
    # helmet. Reduce width above the brow and lower the very top slightly.
    top = v2.smoothstep01((points[:, 2] - (eye_z + 0.030)) / 0.075)
    top_side = v2.smoothstep01((0.125 - np.abs(points[:, 0] - face_x)) / 0.050)
    top_weight = top * top_side
    delta[:, 0] += -0.065 * (points[:, 0] - face_x) * top_weight
    delta[:, 2] += -0.0035 * top_weight * v2.smoothstep01((points[:, 2] - (eye_z + 0.075)) / 0.045)

    # Larger almond apertures, raised outer tails and a slightly higher eye line.
    eye_reports = []
    for eye in eyes:
        side = -1.0 if eye[0] < face_x else 1.0
        spatial = v2.ellipsoid_weight(points, eye, (0.054, 0.055, 0.044), 0.0, 1.18)
        weight = spatial * np.maximum(eye_sem, 0.10 * brow_sem)
        delta[:, 0] += (points[:, 0] - eye[0]) * 0.22 * weight
        delta[:, 2] += (points[:, 2] - eye[2]) * 0.34 * weight
        delta[:, 0] += side * 0.0014 * weight
        delta[:, 2] += 0.0030 * weight
        delta[:, 1] += forward_sign * 0.0015 * spatial * eye_sem

        outer = eye + np.array([side * 0.024, 0.0, 0.001])
        outer_weight = v2.ellipsoid_weight(points, outer, (0.026, 0.034, 0.022), 0.0, 1.08) * eye_sem
        delta[:, 0] += side * 0.0022 * outer_weight
        delta[:, 2] += 0.0018 * outer_weight

        under = eye + np.array([0.0, -forward_sign * 0.004, -0.020])
        under_weight = v2.add_shift(
            delta, points, under, (0.052, 0.055, 0.036),
            (0.0, forward_sign * 0.0030, 0.0013), weight=face_weight, outer=1.15,
        )
        delta[:, 0] += -side * 0.0006 * under_weight
        eye_reports.append({"center": eye.tolist(), "vertices": int(np.sum(weight > 0.08))})

    # Keep the brow frame close to the enlarged eyes, removing the heavy adult
    # ridge while raising the inner half only slightly.
    delta[:, 1] += -forward_sign * 0.0026 * brow_sem * face_weight
    delta[:, 2] += -0.0010 * brow_sem * face_weight

    # Derive a real nose tip from the central front surface, then move the bridge
    # and tip upward with a delicate but readable forward projection.
    central_nose = (
        (np.abs(points[:, 0] - face_x) < 0.026)
        & (points[:, 2] > mouth_z + 0.010)
        & (points[:, 2] < eye_z - 0.004)
        & (face_weight > 0.34)
    )
    if np.any(central_nose):
        nose_index = np.where(central_nose)[0][np.argmax(frontness[central_nose])]
        tip = points[nose_index].copy()
    else:
        tip = np.array([face_x, centre[1] + forward_sign * 0.35 * head_height, mouth_z + 0.037])
    bridge = np.array([face_x, tip[1], mouth_z + 0.60 * (eye_z - mouth_z)])
    bridge_weight = v2.add_scaled_region(
        delta, points, bridge, (0.030, 0.046, 0.060),
        (0.76, 0.95, 0.86), weight=face_weight, outer=1.13,
    )
    delta[:, 2] += 0.0040 * bridge_weight
    delta[:, 1] += -forward_sign * 0.0018 * bridge_weight
    tip_weight = v2.ellipsoid_weight(points, tip, (0.023, 0.032, 0.027), 0.02, 1.09) * face_weight
    delta[:, 2] += 0.0060 * tip_weight
    delta[:, 1] += -forward_sign * 0.0025 * tip_weight
    delta[:, 0] += -(points[:, 0] - face_x) * 0.13 * tip_weight

    base_center = np.array([face_x, tip[1], mouth_z + 0.020])
    base_weight = v2.add_scaled_region(
        delta, points, base_center, (0.034, 0.040, 0.031),
        (0.76, 0.95, 0.86), weight=face_weight, outer=1.10,
    )
    delta[:, 2] += 0.0033 * base_weight

    # Full but compact lips. The tight semantic mask uses only the true lip and
    # smile keys, avoiding the v2 viseme spill into the lower cheeks.
    lip_spatial = v2.ellipsoid_weight(points, mouth, (0.052, 0.050, 0.030), 0.0, 1.14)
    lip_weight = lip_spatial * lip_sem
    delta[:, 0] += (points[:, 0] - mouth[0]) * 0.075 * lip_weight
    delta[:, 2] += (points[:, 2] - mouth[2]) * 0.48 * lip_weight
    delta[:, 2] += 0.0035 * lip_weight
    delta[:, 1] += forward_sign * 0.0030 * lip_weight
    corner_ratio = np.clip(np.abs(points[:, 0] - mouth[0]) / 0.042, 0.0, 1.0)
    delta[:, 2] += 0.0016 * np.power(corner_ratio, 1.8) * lip_weight

    # High apple cheeks and a softer eye-to-mouth plane.
    for eye in eyes:
        side = -1.0 if eye[0] < face_x else 1.0
        cheek = np.array([
            eye[0] + side * 0.002,
            tip[1] - forward_sign * 0.010,
            mouth_z + 0.54 * (eye_z - mouth_z),
        ])
        cheek_weight = v2.add_shift(
            delta, points, cheek, (0.060, 0.058, 0.055),
            (0.0, forward_sign * 0.0046, 0.0020), weight=face_weight, outer=1.15,
        )
        delta[:, 0] += -side * 0.0011 * cheek_weight

    # Smaller ears, moved slightly up; a narrower, shorter visible neck prepares
    # the final high-collar silhouette without changing head topology.
    ear_band = v2.smoothstep01((np.abs(points[:, 0] - face_x) - head_width * 0.34) / (head_width * 0.12))
    ear_vertical = v2.smoothstep01((points[:, 2] - (mouth_z - 0.025)) / 0.032) * v2.smoothstep01(((eye_z + 0.030) - points[:, 2]) / 0.032)
    ear_depth = v2.smoothstep01((front_high - frontness) / max(front_high - front_low, 1e-9))
    ear_weight = ear_band * ear_vertical * ear_depth
    delta[:, 0] += -(points[:, 0] - face_x) * 0.12 * ear_weight
    delta[:, 1] += forward_sign * 0.0030 * ear_weight
    delta[:, 2] += 0.0040 * ear_weight

    neck = v2.smoothstep01(((chin_z - 0.008) - points[:, 2]) / 0.040) * v2.smoothstep01((0.090 - np.abs(points[:, 0] - face_x)) / 0.040)
    delta[:, 0] += -0.13 * (points[:, 0] - face_x) * neck
    delta[:, 2] += 0.0030 * neck

    # Final safety cap.
    lengths = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1.0, 0.020 / np.maximum(lengths, 1e-9))[:, None]
    local_delta = v2.world_vector_to_local(skin, delta)
    preservation = preserve_apply(skin, local_delta, original)

    report = {
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
            "jaw": int(np.sum(jaw_sem > 0.08)),
        },
        "max_additional_displacement_m": float(np.linalg.norm(delta, axis=1).max()),
        "rms_additional_displacement_m": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
        "moved_vertices_over_0_5mm": int(np.sum(np.linalg.norm(delta, axis=1) > 0.0005)),
        **preservation,
    }
    return report, points, brow_sem, eye_sem


def refine_eye_objects(scene, centres: list[np.ndarray], face_x: float) -> dict:
    objects = v2.eye_related_objects(scene)
    if len(centres) < 2:
        return {"objects": [], "max_displacement_m": 0.0, "iris_vertices": 0}
    left, right = centres[0], centres[-1]
    changed = []
    max_displacement = 0.0
    iris_count = 0
    for obj in objects:
        local = v2.mesh_local_array(obj)
        world = v2.to_world(obj, local)
        original = world.copy()
        result = world.copy()
        is_eyeball = "eyeball" in obj.name.lower()
        iris_ids = material_vertex_indices(obj, "iris") if is_eyeball else np.zeros(0, dtype=np.int64)
        pupil_ids = material_vertex_indices(obj, "eyeback") if is_eyeball else np.zeros(0, dtype=np.int64)
        iris_vertices = np.unique(np.r_[iris_ids, pupil_ids]) if len(iris_ids) or len(pupil_ids) else np.zeros(0, dtype=np.int64)
        iris_count += len(iris_vertices)
        for index, point in enumerate(world):
            centre = left if point[0] < face_x else right
            side = -1.0 if centre[0] < face_x else 1.0
            relative = point - centre
            relative[0] *= 1.105
            relative[2] *= 1.105
            relative[1] *= 1.015
            result[index] = centre + relative + np.array([side * 0.0015, 0.0, 0.0018])
        if len(iris_vertices):
            for index in iris_vertices:
                centre = left if result[index, 0] < face_x else right
                relative = result[index] - centre
                relative[0] *= 1.16
                relative[2] *= 1.16
                result[index] = centre + relative
        v2.set_mesh_local_array(obj, v2.to_local(obj, result))
        displacement = np.linalg.norm(result - original, axis=1)
        if len(displacement):
            max_displacement = max(max_displacement, float(displacement.max()))
        changed.append(obj.name)
    return {"objects": changed, "max_displacement_m": max_displacement, "iris_vertices": iris_count}


def create_brows(skin, basis_world: np.ndarray, brow_sem: np.ndarray, eye_z: float, face_x: float, forward_sign: float, material):
    # Remove any previous QA brows.
    for obj in list(bpy.data.objects):
        if obj.name.startswith("AINA_V3_Brow_"):
            bpy.data.objects.remove(obj, do_unlink=True)
    created = []
    for side_name, sign in (("L", -1.0), ("R", 1.0)):
        mask = (
            (sign * (basis_world[:, 0] - face_x) > 0.006)
            & (sign * (basis_world[:, 0] - face_x) < 0.065)
            & (basis_world[:, 2] > eye_z + 0.010)
            & (basis_world[:, 2] < eye_z + 0.055)
            & (brow_sem > 0.12)
        )
        candidates = basis_world[mask]
        candidate_weights = brow_sem[mask]
        if len(candidates) < 20:
            continue
        order = np.argsort(candidates[:, 0])
        candidates = candidates[order]
        candidate_weights = candidate_weights[order]
        groups = np.array_split(np.arange(len(candidates)), 8)
        path = []
        for group in groups:
            if not len(group):
                continue
            weights = candidate_weights[group]
            point = np.sum(candidates[group] * weights[:, None], axis=0) / max(float(weights.sum()), 1e-9)
            point[1] += forward_sign * 0.0017
            point[2] += 0.0004
            path.append(point)
        if len(path) < 4:
            continue
        curve = bpy.data.curves.new(f"AINA_V3_Brow_{side_name}_Curve", "CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 4
        curve.bevel_depth = 0.00072
        curve.bevel_resolution = 3
        spline = curve.splines.new("BEZIER")
        spline.bezier_points.add(len(path) - 1)
        for point, coordinate in zip(spline.bezier_points, path):
            point.co = tuple(coordinate)
            point.handle_left_type = "AUTO"
            point.handle_right_type = "AUTO"
        obj = bpy.data.objects.new(f"AINA_V3_Brow_{side_name}", curve)
        bpy.context.collection.objects.link(obj)
        curve.materials.append(material)
        created.append(obj.name)
    return created


def install_v3_materials(scene, skin):
    skin_mat = v2.make_material("AINA_V3_ClaySkin", (0.18, 0.21, 0.28), 0.58)
    mouth_mat = v2.make_material("AINA_V3_Lips", (0.10, 0.012, 0.022), 0.42)
    sclera = v2.make_material("AINA_V3_Sclera", (0.44, 0.50, 0.58), 0.27)
    iris = v2.make_material("AINA_V3_Iris", (0.035, 0.16, 0.18), 0.25)
    dark = v2.make_material("AINA_V3_Dark", (0.002, 0.004, 0.007), 0.20)
    cornea = v2.make_material("AINA_V3_Cornea", (0.16, 0.24, 0.30), 0.08)
    shadow = v2.make_material("AINA_V3_Shadow", (0.055, 0.020, 0.032), 0.48)
    tear = v2.make_material("AINA_V3_Tear", (0.15, 0.040, 0.055), 0.24)
    brow = v2.make_material("AINA_V3_Brow", (0.018, 0.022, 0.032), 0.48)

    v2.assign_material_slots(skin, [skin_mat, mouth_mat])
    for obj in scene.objects:
        if obj.type != "MESH" or obj == skin:
            continue
        name = obj.name.lower()
        if "eyeball" in name:
            slots = []
            for old in list(obj.data.materials):
                old_name = old.name.lower() if old else ""
                if "sclera" in old_name:
                    slots.append(sclera)
                elif "iris" in old_name:
                    slots.append(iris)
                elif "cornea" in old_name:
                    slots.append(cornea)
                else:
                    slots.append(dark)
            v2.assign_material_slots(obj, slots or [sclera, iris, dark, cornea])
        elif "eyeshadow" in name:
            v2.assign_material_slots(obj, [shadow])
        elif "tear" in name or "caruncle" in name:
            v2.assign_material_slots(obj, [tear])
        else:
            v2.assign_material_slots(obj, [skin_mat])
    return brow


def render_v3_suite(scene, skin, meshes, centres, out: Path):
    preview = out / "Preview"
    preview.mkdir(exist_ok=True)
    camera, target, locations = v2.render_setup(scene, meshes, centres, out)
    scene.world.color = (0.018, 0.022, 0.035)
    try:
        scene.view_settings.exposure = -0.90
    except Exception:
        pass
    outputs = {}
    v2.reset_shapes(skin)
    for name in ("FRONT", "THREE_QUARTER", "SIDE", "LEFT_45", "RIGHT_45"):
        path = preview / f"AINA_VITRUVIAN_SCULPT_V3_{name}.png"
        v2.render_view(scene, camera, target, locations[name], path)
        outputs[name.lower()] = str(path)
    cases = {
        "HAPPY": {"Happy": 1.0},
        "BLINK": {"Eyes_Closed_Max": 1.0},
        "AA": {"aa_02": 1.0, "Jaw_Lower": 0.22},
        "OU": {"ow_08": 1.0},
    }
    for name, values in cases.items():
        v2.reset_shapes(skin)
        for key_name, value in values.items():
            key = skin.data.shape_keys.key_blocks.get(key_name) if skin.data.shape_keys else None
            if key:
                key.value = value
        path = preview / f"AINA_VITRUVIAN_SCULPT_V3_{name}.png"
        v2.render_view(scene, camera, target, locations["FRONT"], path)
        outputs[name.lower()] = str(path)
    v2.reset_shapes(skin)
    return outputs


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)
    scene = bpy.context.scene
    skin = v2.identify_skin(scene)
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    for obj in meshes:
        if obj.data.shape_keys:
            for key in obj.data.shape_keys.key_blocks:
                key.value = 0.0

    centres_before, eyeballs = eye_centres(scene)
    face_x = float(np.mean(centres_before, axis=0)[0]) if centres_before else float(v2.world_vertices(skin)[:, 0].mean())
    sculpt_report, basis_before, brow_sem, eye_sem = art_directed_v3(skin, centres_before)
    eye_report = refine_eye_objects(scene, centres_before, face_x)
    bpy.context.view_layer.update()
    centres_after = [v2.world_vertices(obj).mean(axis=0) for obj in eyeballs if len(obj.data.vertices)]
    centres_after.sort(key=lambda point: point[0])

    brow_material = install_v3_materials(scene, skin)
    basis_after = v2.to_world(skin, v2.key_array(skin.data.shape_keys.key_blocks.get("Basis") or skin.data.shape_keys.key_blocks[0]))
    brow_objects = create_brows(
        skin,
        basis_after,
        brow_sem,
        sculpt_report["eye_z_m"] + 0.003,
        face_x,
        sculpt_report["forward_sign_y"],
        brow_material,
    )
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    renders = render_v3_suite(scene, skin, meshes, centres_after, args.out)

    blend_path = args.out / "AINA_VITRUVIAN_IDENTITY_SCULPT_V3.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_VITRUVIAN_IDENTITY_SCULPT_V3.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    shape_keys = [key.name for key in skin.data.shape_keys.key_blocks]
    report = {
        "product": "AINA Vitruvian Identity Sculpt v3",
        "real_3d_model": True,
        "source_topology": "CC0 Vitruvian/Antonia FACS head",
        "replacement_effect_art_generated": False,
        "topology_changed": False,
        "vertices": len(skin.data.vertices),
        "triangles": len(skin.data.loop_triangles),
        "shape_key_count": max(0, len(shape_keys) - 1),
        "shape_keys": shape_keys,
        "sculpt": sculpt_report,
        "eye_anatomy": eye_report,
        "real_brow_geometry": brow_objects,
        "identity_lock": False,
        "visual_identity_lock": False,
        "candidate": True,
        "vrm_exported": False,
        "files": {"blend": str(blend_path), "glb": str(glb_path), "renders": renders},
        "next_gate": "Inspect approved front/3Q/side against the real v3 model. Lock only if the same AINA identity is visually present; otherwise make one final bounded residual pass.",
    }
    (qa / "AINA_VITRUVIAN_IDENTITY_SCULPT_V3_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
