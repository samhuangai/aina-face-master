#!/usr/bin/env python3
"""AINA Rain Identity Reconstruction v3 on the same verified production Mesh.

The source is the successful approved-appearance Rain candidate. This pass uses
approved versus actual front/3Q/profile landmarks, then performs a dedicated
maturity reconstruction: smaller cranium and eyes, narrower eyelid apertures,
a delicate nose, compact lips, a softer V jaw, a smaller chin and reduced ears.
The identical skin displacement is added to Basis and every source shape key.
Eye, mouth and hair geometry are moved coherently. No replacement effect art is
generated and no VRM is exported before direct visual acceptance.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aina_rain_identity_master as base
import aina_rain_appearance_candidate as appearance


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def smoothstep01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def ellipsoid(points: np.ndarray, centre, radii, outer: float = 1.0) -> np.ndarray:
    centre = np.asarray(centre, dtype=np.float64)
    radii = np.maximum(np.asarray(radii, dtype=np.float64), 1.0e-7)
    q = np.sqrt(np.sum(((points - centre) / radii) ** 2, axis=1))
    weight = np.zeros(len(points), dtype=np.float64)
    mask = q < outer
    if np.any(mask):
        t = q[mask] / outer
        weight[mask] = 0.5 * (1.0 + np.cos(np.pi * t))
    return weight


def normalize_item(item: dict) -> np.ndarray:
    width, height = item["image_size"]
    points = np.asarray(item["landmarks_xy"], dtype=np.float64)
    return points / np.array([width, height], dtype=np.float64)


def distance(points: np.ndarray, a: int, b: int) -> float:
    return max(float(np.linalg.norm(points[a] - points[b])), 1.0e-7)


def eye_height(points: np.ndarray, left: bool) -> float:
    if left:
        return 0.5 * (distance(points, 43, 47) + distance(points, 44, 46))
    return 0.5 * (distance(points, 37, 41) + distance(points, 38, 40))


def facial_ratios(points: np.ndarray) -> dict[str, float]:
    face_width = distance(points, 0, 16)
    return {
        "face_width": face_width,
        "jaw_over_face": distance(points, 4, 12) / face_width,
        "chin_over_face": distance(points, 6, 10) / face_width,
        "eye_width_over_face": 0.5 * (
            distance(points, 36, 39) + distance(points, 42, 45)
        ) / face_width,
        "eye_height_over_face": 0.5 * (
            eye_height(points, False) + eye_height(points, True)
        ) / face_width,
        "nose_over_face": distance(points, 31, 35) / face_width,
        "mouth_over_face": distance(points, 48, 54) / face_width,
        "lower_height_over_face": abs(float(points[8, 1] - points[33, 1])) / face_width,
        "eye_centre_spacing_over_face": distance(
            np.array([
                points[36:42].mean(axis=0),
                points[42:48].mean(axis=0),
            ]),
            0,
            1,
        ) / face_width,
    }


def ratio_plan(data: dict) -> dict:
    approved = normalize_item(data["approved"]["front"])
    model = normalize_item(data["model"]["front"])
    target = facial_ratios(approved)
    current = facial_ratios(model)

    def scale(name: str, low: float, high: float) -> float:
        return float(np.clip(target[name] / max(current[name], 1.0e-7), low, high))

    # Direct target/model ratios are blended with an art-directed maturity prior.
    eye_width = min(scale("eye_width_over_face", 0.52, 1.05), 0.78)
    eye_height = min(scale("eye_height_over_face", 0.42, 1.05), 0.70)
    return {
        "approved": target,
        "model": current,
        "eye_width_scale": eye_width,
        "eye_height_scale": eye_height,
        "eye_spacing_scale": float(np.clip(
            scale("eye_centre_spacing_over_face", 0.78, 1.08), 0.84, 1.02
        )),
        "nose_width_scale": min(scale("nose_over_face", 0.62, 1.02), 0.86),
        "mouth_width_scale": min(scale("mouth_over_face", 0.68, 1.02), 0.88),
        "jaw_width_scale": min(scale("jaw_over_face", 0.72, 1.02), 0.88),
        "chin_width_scale": min(scale("chin_over_face", 0.62, 1.02), 0.82),
        "lower_height_scale": float(np.clip(
            scale("lower_height_over_face", 0.90, 1.15), 0.96, 1.10
        )),
        "cranium_width_scale": 0.82,
        "cranium_height_scale": 0.83,
    }


def semantic_magnitude(skin, basis: np.ndarray, tokens: tuple[str, ...]) -> np.ndarray:
    result = np.zeros(len(basis), dtype=np.float64)
    if not skin.data.shape_keys:
        return result
    for key in skin.data.shape_keys.key_blocks:
        lower = key.name.lower().replace("_", "")
        if key.name == "Basis" or not any(token.replace("_", "") in lower for token in tokens):
            continue
        result = np.maximum(result, np.linalg.norm(base.key_array(key) - basis, axis=1))
    return result


def normalized_semantic(magnitude: np.ndarray, quantile: float) -> np.ndarray:
    positive = magnitude[magnitude > 1.0e-8]
    if not len(positive):
        return np.zeros_like(magnitude)
    low = max(float(np.quantile(positive, quantile)), 1.0e-5)
    high = max(float(np.quantile(positive, 0.987)), low * 1.40)
    return smoothstep01((magnitude - low) / max(high - low, 1.0e-9))


def weighted_centre(points: np.ndarray, weight: np.ndarray, fallback) -> np.ndarray:
    total = float(weight.sum())
    if total < 1.0e-9:
        return np.asarray(fallback, dtype=np.float64)
    return np.sum(points * weight[:, None], axis=0) / total


def true_eye_centres(scene, face_x: float) -> list[np.ndarray]:
    centres = appearance.actual_eye_centres(scene, face_x)
    if len(centres) == 2:
        centres.sort(key=lambda point: point[0])
        return centres
    return []


def maturity_reconstruction(
    skin,
    head_ids: np.ndarray,
    eyes: list[np.ndarray],
    forward_sign: float,
    plan: dict,
) -> dict:
    basis_key = skin.data.shape_keys.key_blocks.get("Basis") if skin.data.shape_keys else None
    basis_local = base.key_array(basis_key) if basis_key else base.mesh_local_array(skin)
    full_points = base.to_world(skin, basis_local)
    points = full_points[head_ids]
    lo, hi = points.min(axis=0), points.max(axis=0)
    centre = (lo + hi) * 0.5
    face_x = float(np.mean(eyes, axis=0)[0])
    eye_z = float(np.mean(eyes, axis=0)[2])
    head_height = float(hi[2] - lo[2])

    frontness = forward_sign * (points[:, 1] - centre[1])
    front_low = float(np.quantile(frontness, 0.32))
    front_high = float(np.quantile(frontness, 0.95))
    face_front = smoothstep01((frontness - front_low) / max(front_high - front_low, 1.0e-9))
    face_side = smoothstep01((0.150 - np.abs(points[:, 0] - face_x)) / 0.060)
    face_weight = face_front * face_side

    eye_sem_full = normalized_semantic(
        semantic_magnitude(skin, basis_local, ("eyelidsclose", "eyebrows")), 0.32
    )
    lip_sem_full = normalized_semantic(
        semantic_magnitude(skin, basis_local, ("lips", "smile", "cheekpuff")), 0.36
    )
    eye_sem = eye_sem_full[head_ids]
    lip_sem = lip_sem_full[head_ids]

    below_eye = smoothstep01((eye_z - 0.010 - points[:, 2]) / 0.042)
    mouth_fallback = np.array([
        face_x,
        centre[1] + forward_sign * 0.30 * head_height,
        eye_z - 0.22 * head_height,
    ])
    mouth = weighted_centre(points, lip_sem * face_weight * below_eye, mouth_fallback)
    mouth_z = float(mouth[2])

    lower_mask = (
        (points[:, 2] < mouth_z)
        & (np.abs(points[:, 0] - face_x) < 0.060)
        & (face_weight > 0.35)
    )
    chin_z = float(np.quantile(points[lower_mask, 2], 0.045)) if np.any(lower_mask) else mouth_z - 0.052

    central_nose = (
        (np.abs(points[:, 0] - face_x) < 0.028)
        & (points[:, 2] > mouth_z + 0.010)
        & (points[:, 2] < eye_z - 0.004)
        & (face_weight > 0.34)
    )
    if np.any(central_nose):
        tip_index = np.where(central_nose)[0][np.argmax(frontness[central_nose])]
        nose_tip = points[tip_index].copy()
    else:
        nose_tip = np.array([face_x, centre[1] + forward_sign * 0.10, mouth_z + 0.035])

    delta = np.zeros_like(points)
    preserve = np.zeros(len(points), dtype=np.float64)

    # Mature oval skull: reduce top width/height while preserving brow and face.
    top_origin = eye_z + 0.020
    top = np.clip((points[:, 2] - top_origin) / max(hi[2] - top_origin, 1.0e-6), 0.0, 1.0)
    top_weight = smoothstep01(top) * smoothstep01((0.160 - np.abs(points[:, 0] - face_x)) / 0.055)
    top_x_factor = 1.0 + (plan["cranium_width_scale"] - 1.0) * top
    delta[:, 0] += (
        (points[:, 0] - face_x) * top_x_factor - (points[:, 0] - face_x)
    ) * top_weight
    top_target_z = top_origin + (points[:, 2] - top_origin) * plan["cranium_height_scale"]
    delta[:, 2] += (top_target_z - points[:, 2]) * top_weight

    # Reposition eye centres and reduce actual eyelid apertures to approved ratios.
    spacing_scale = plan["eye_spacing_scale"]
    eye_reports = []
    for eye in eyes:
        side = -1.0 if eye[0] < face_x else 1.0
        desired_eye_x = face_x + (eye[0] - face_x) * spacing_scale
        spatial = ellipsoid(points, eye, (0.054, 0.054, 0.040), 1.20)
        weight = spatial * np.maximum(eye_sem, 0.18 * face_weight)
        target_x = desired_eye_x + (points[:, 0] - eye[0]) * plan["eye_width_scale"]
        target_z = eye[2] + (points[:, 2] - eye[2]) * plan["eye_height_scale"]
        delta[:, 0] += (target_x - points[:, 0]) * weight
        delta[:, 2] += (target_z - points[:, 2]) * weight
        delta[:, 1] += -forward_sign * 0.0008 * weight
        outer = eye + np.array([side * 0.024, 0.0, 0.001])
        outer_weight = ellipsoid(points, outer, (0.023, 0.030, 0.018), 1.08) * weight
        delta[:, 2] += 0.0007 * outer_weight
        preserve = np.maximum(preserve, np.clip(weight + outer_weight, 0.0, 1.0))
        eye_reports.append({
            "source_centre": eye.tolist(),
            "desired_x": desired_eye_x,
            "active_vertices": int(np.sum(weight > 0.06)),
        })

    # Delicate nose: narrow bridge/base, reduce bulbous projection, keep a readable tip.
    bridge = np.array([face_x, nose_tip[1], mouth_z + 0.62 * (eye_z - mouth_z)])
    bridge_weight = ellipsoid(points, bridge, (0.032, 0.045, 0.060), 1.14) * face_weight
    bridge_target_x = face_x + (points[:, 0] - face_x) * plan["nose_width_scale"]
    delta[:, 0] += (bridge_target_x - points[:, 0]) * bridge_weight
    delta[:, 1] += -forward_sign * 0.0020 * bridge_weight
    tip_weight = ellipsoid(points, nose_tip, (0.025, 0.034, 0.027), 1.10) * face_weight
    delta[:, 0] += -(points[:, 0] - face_x) * 0.16 * tip_weight
    delta[:, 1] += -forward_sign * 0.0030 * tip_weight
    delta[:, 2] += 0.0004 * tip_weight
    base_centre = np.array([face_x, nose_tip[1], mouth_z + 0.020])
    base_weight = ellipsoid(points, base_centre, (0.039, 0.042, 0.031), 1.10) * face_weight
    delta[:, 0] += -(points[:, 0] - face_x) * (1.0 - plan["nose_width_scale"]) * base_weight
    preserve = np.maximum(preserve, np.clip(bridge_weight + tip_weight + base_weight, 0.0, 1.0))

    # Compact integrated lips: smaller width and much less vertical thickness.
    lip_spatial = ellipsoid(points, mouth, (0.058, 0.048, 0.031), 1.15)
    lip_weight = lip_spatial * np.maximum(lip_sem, 0.20 * face_weight)
    target_x = mouth[0] + (points[:, 0] - mouth[0]) * plan["mouth_width_scale"]
    target_z = mouth[2] + (points[:, 2] - mouth[2]) * 0.68
    delta[:, 0] += (target_x - points[:, 0]) * lip_weight
    delta[:, 2] += (target_z - points[:, 2]) * lip_weight
    delta[:, 1] += -forward_sign * 0.0020 * lip_weight
    preserve = np.maximum(preserve, np.clip(lip_weight, 0.0, 1.0))

    # Lower-third length and V contour from approved ratios.
    nose_z = float(base_centre[2])
    lower = np.clip((nose_z - points[:, 2]) / max(nose_z - chin_z, 1.0e-6), 0.0, 1.0)
    lower_weight = face_weight * smoothstep01(lower)
    target_lower_z = nose_z + (points[:, 2] - nose_z) * plan["lower_height_scale"]
    delta[:, 2] += (target_lower_z - points[:, 2]) * lower_weight * 0.72
    jaw_factor = 1.0 + (plan["jaw_width_scale"] - 1.0) * np.power(lower, 1.18)
    delta[:, 0] += (
        (points[:, 0] - face_x) * jaw_factor - (points[:, 0] - face_x)
    ) * lower_weight

    chin = np.array([face_x, mouth[1] - forward_sign * 0.004, chin_z])
    chin_weight = ellipsoid(points, chin, (0.054, 0.058, 0.050), 1.12) * face_weight
    chin_target_x = face_x + (points[:, 0] - face_x) * plan["chin_width_scale"]
    delta[:, 0] += (chin_target_x - points[:, 0]) * chin_weight
    delta[:, 2] += 0.0015 * chin_weight
    delta[:, 1] += -forward_sign * 0.0008 * chin_weight
    preserve = np.maximum(preserve, np.clip(chin_weight, 0.0, 1.0))

    # Reduce broad mid-cheeks but retain high apple-cheek support.
    mid_z = mouth_z + 0.55 * (eye_z - mouth_z)
    for eye in eyes:
        side = -1.0 if eye[0] < face_x else 1.0
        cheek = np.array([eye[0], nose_tip[1] - forward_sign * 0.010, mid_z])
        cheek_weight = ellipsoid(points, cheek, (0.064, 0.060, 0.057), 1.16) * face_weight
        delta[:, 0] += -side * 0.0016 * cheek_weight
        delta[:, 1] += forward_sign * 0.0014 * cheek_weight
        delta[:, 2] += 0.0008 * cheek_weight

    # Smaller ears and narrower neck column.
    ear_weight = (
        smoothstep01((np.abs(points[:, 0] - face_x) - 0.100) / 0.042)
        * smoothstep01((points[:, 2] - (mouth_z - 0.030)) / 0.035)
        * smoothstep01(((eye_z + 0.040) - points[:, 2]) / 0.035)
        * smoothstep01((front_high - frontness) / max(front_high - front_low, 1.0e-9))
    )
    delta[:, 0] += -(points[:, 0] - face_x) * 0.18 * ear_weight
    ear_mid_z = 0.5 * (eye_z + mouth_z)
    delta[:, 2] += (ear_mid_z - points[:, 2]) * 0.16 * ear_weight
    delta[:, 1] += -forward_sign * 0.0015 * ear_weight
    neck_weight = (
        smoothstep01(((chin_z + 0.004) - points[:, 2]) / 0.042)
        * smoothstep01((0.105 - np.abs(points[:, 0] - face_x)) / 0.043)
    )
    delta[:, 0] += -(points[:, 0] - face_x) * 0.17 * neck_weight

    adjacency = base.adjacency_for_region(skin, head_ids)
    smoothed = base.smooth_region_delta(delta, adjacency, preserve, passes=2)
    lengths = np.linalg.norm(smoothed, axis=1)
    smoothed *= np.minimum(1.0, 0.024 / np.maximum(lengths, 1.0e-9))[:, None]
    full_delta = np.zeros_like(full_points)
    full_delta[head_ids] = smoothed
    base.apply_world_delta(skin, full_delta)

    return {
        "plan": plan,
        "face_x": face_x,
        "eye_z": eye_z,
        "mouth_centre": mouth.tolist(),
        "nose_tip": nose_tip.tolist(),
        "chin_z": chin_z,
        "eye_regions": eye_reports,
        "max_displacement_m": float(np.linalg.norm(smoothed, axis=1).max()),
        "rms_displacement_m": float(np.sqrt(np.mean(np.sum(smoothed * smoothed, axis=1)))),
        "moved_vertices_over_0_5mm": int(np.sum(np.linalg.norm(smoothed, axis=1) > 0.0005)),
    }


def transform_eye_geometry(scene, eyes_before: list[np.ndarray], eyes_after: list[np.ndarray], plan: dict) -> dict:
    if len(eyes_before) != 2 or len(eyes_after) != 2:
        return {"objects": [], "max_displacement_m": 0.0}
    moved = []
    max_displacement = 0.0
    for obj in scene.objects:
        if obj.type != "MESH" or not len(obj.data.vertices):
            continue
        text = (obj.name + " " + " ".join(mat.name for mat in obj.data.materials if mat)).lower()
        if not any(token in text for token in ("eye", "iris", "pupil", "cornea", "lash", "brow")):
            continue
        points = base.world_vertices(obj)
        result = points.copy()
        original = points.copy()
        for index, point in enumerate(points):
            side_index = 0 if point[0] < 0.5 * (eyes_before[0][0] + eyes_before[1][0]) else 1
            source = eyes_before[side_index]
            target = eyes_after[side_index]
            relative = point - source
            relative[0] *= plan["eye_width_scale"]
            relative[2] *= plan["eye_height_scale"]
            relative[1] *= 0.94
            result[index] = target + relative
        base.set_mesh_local_array(obj, base.to_local(obj, result))
        displacement = np.linalg.norm(result - original, axis=1)
        if len(displacement):
            max_displacement = max(max_displacement, float(displacement.max()))
        moved.append(obj.name)
    return {"objects": moved, "max_displacement_m": max_displacement}


def transform_hair(scene, face_x: float, eye_z: float, plan: dict) -> dict:
    moved = []
    max_displacement = 0.0
    for obj in scene.objects:
        if obj.type != "MESH" or not len(obj.data.vertices) or not base.is_hair(obj):
            continue
        points = base.world_vertices(obj)
        original = points.copy()
        pivot = np.array([face_x, float(np.median(points[:, 1])), eye_z + 0.020])
        local = points - pivot
        top = smoothstep01((points[:, 2] - eye_z) / 0.12)
        local[:, 0] *= 1.0 + (plan["cranium_width_scale"] - 1.0) * top
        local[:, 2] *= 1.0 + (plan["cranium_height_scale"] - 1.0) * top
        local[:, 1] *= 0.94
        result = pivot + local
        base.set_mesh_local_array(obj, base.to_local(obj, result))
        displacement = np.linalg.norm(result - original, axis=1)
        if len(displacement):
            max_displacement = max(max_displacement, float(displacement.max()))
        moved.append(obj.name)
    return {"objects": moved, "max_displacement_m": max_displacement}


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)
    data = json.loads(args.landmarks.read_text(encoding="utf-8"))
    plan = ratio_plan(data)

    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH" and len(obj.data.vertices)]
    base.reset_shape_keys(meshes)
    armature = base.find_armature(scene)
    head_bone = base.find_head_bone(armature)
    head_point = base.bone_world_point(armature, head_bone)
    skin, skin_report = base.identify_skin(scene, head_point)
    original_deltas = base.capture_shape_deltas(skin)
    original_vertices = len(skin.data.vertices)
    original_triangles = sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons)

    skin_world_initial = base.world_vertices(skin)
    face_x_initial = float(0.5 * (skin_world_initial[:, 0].min() + skin_world_initial[:, 0].max()))
    eyes_initial = true_eye_centres(scene, face_x_initial)
    if len(eyes_initial) != 2:
        raise RuntimeError(f"Expected two real eyes, got {len(eyes_initial)}")
    character_height = skin_report["character_height_m"]
    head_ids, _, _, _ = base.head_region(skin, head_point, eyes_initial, character_height)

    cameras = {
        view: bpy.data.objects.get(f"AINA_Rain_Camera_{view}")
        for view in ("front", "three_quarter", "side", "left_45", "right_45")
    }
    if not all(cameras.values()):
        cameras, _ = base.setup_cameras(
            scene, skin, head_ids, eyes_initial, head_point, character_height
        )
    skin_centre = skin_world_initial.mean(axis=0)
    forward_sign = -1.0 if np.mean(eyes_initial, axis=0)[1] < skin_centre[1] else 1.0

    # First converge approved front/3Q/profile landmarks on the actual topology.
    fitting = base.fit_identity(scene, skin, head_ids, cameras, data, forward_sign)
    fit_initial = fitting.pop("initial_world")
    fit_final = fitting.pop("final_world")
    first_related = base.move_related_objects(
        scene, skin, fit_initial, fit_final, eyes_initial
    )
    bpy.context.view_layer.update()

    fit_world = base.world_vertices(skin)
    face_x_fit = float(0.5 * (fit_world[:, 0].min() + fit_world[:, 0].max()))
    eyes_fit = true_eye_centres(scene, face_x_fit)
    if len(eyes_fit) != 2:
        eyes_fit = eyes_initial

    # Then apply the explicit mature AINA proportion reconstruction.
    maturity_before = base.world_vertices(skin)
    maturity = maturity_reconstruction(skin, head_ids, eyes_fit, forward_sign, plan)
    maturity_after = base.world_vertices(skin)
    second_related = base.move_related_objects(
        scene, skin, maturity_before, maturity_after, eyes_fit
    )
    bpy.context.view_layer.update()

    final_face_x = float(0.5 * (maturity_after[:, 0].min() + maturity_after[:, 0].max()))
    eyes_after = true_eye_centres(scene, final_face_x)
    if len(eyes_after) != 2:
        eyes_after = eyes_fit
    eye_geometry = transform_eye_geometry(scene, eyes_fit, eyes_after, plan)
    hair = transform_hair(scene, final_face_x, float(np.mean(eyes_after, axis=0)[2]), plan)
    bpy.context.view_layer.update()

    preservation = base.validate_shape_deltas(skin, original_deltas)
    cameras, camera_report = base.setup_cameras(
        scene, skin, head_ids, eyes_after, head_point, character_height
    )
    appearance.soften_lighting(scene)
    renders = appearance.render_full_suite(scene, cameras, skin, args.out)

    blend_path = args.out / "AINA_RAIN_IDENTITY_RECONSTRUCTION_V3.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_RAIN_IDENTITY_RECONSTRUCTION_V3.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    final_world = base.world_vertices(skin)
    report = {
        "product": "AINA Rain Identity Reconstruction v3",
        "source": "AINA Rain Approved-Appearance Candidate",
        "source_character": "Blender Studio Rain v3",
        "source_license": "CC BY 4.0",
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "skin_topology_changed": False,
        "armature_preserved": True,
        "skin_weights_preserved": True,
        "uvs_preserved": True,
        "skin_object": skin.name,
        "vertices": len(skin.data.vertices),
        "triangles": sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons),
        "source_vertices": original_vertices,
        "source_triangles": original_triangles,
        "head_region_vertices": len(head_ids),
        "landmark_views": data["available_views"],
        "ratio_plan": plan,
        "multi_view_fitting": fitting,
        "first_related_anatomy": first_related,
        "maturity_reconstruction": maturity,
        "second_related_anatomy": second_related,
        "eye_geometry_transform": eye_geometry,
        "hair_transform": hair,
        "shape_key_preservation": preservation,
        "eye_centres_before": [point.tolist() for point in eyes_initial],
        "eye_centres_after": [point.tolist() for point in eyes_after],
        "camera": camera_report,
        "total_skin_displacement_max_m": float(np.linalg.norm(final_world - skin_world_initial, axis=1).max()),
        "total_skin_displacement_rms_m": float(np.sqrt(np.mean(np.sum((final_world - skin_world_initial) ** 2, axis=1)))),
        "renders": renders,
        "identity_lock": False,
        "visual_identity_lock": False,
        "candidate": True,
        "vrm_exported": False,
        "next_gate": "Inspect actual mature front, 3Q, profile, expressions and clay. Lock only if the same AINA identity is visually stable; otherwise continue on this exact topology.",
        "files": {"blend": str(blend_path), "glb": str(glb_path)},
    }
    (qa / "AINA_RAIN_IDENTITY_RECONSTRUCTION_V3_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
