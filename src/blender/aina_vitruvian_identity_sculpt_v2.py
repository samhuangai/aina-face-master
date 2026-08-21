#!/usr/bin/env python3
"""AINA Vitruvian Identity Sculpt v2.

Art-directed, topology-preserving sculpt on the real CC0 Vitruvian FACS head.
The script starts from the untouched probe BLEND, derives semantic regions from
existing FACS/viseme shape-key deltas and the separate eye anatomy, applies one
smooth neutral-identity displacement to every shape key, and therefore preserves
all existing expression deltas exactly. It creates no replacement effect art and
does not export VRM.
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


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def key_array(key) -> np.ndarray:
    values = np.empty(len(key.data) * 3, dtype=np.float64)
    key.data.foreach_get("co", values)
    return values.reshape(-1, 3)


def set_key_array(key, values: np.ndarray) -> None:
    key.data.foreach_set("co", np.asarray(values, dtype=np.float32).ravel())


def mesh_local_array(obj) -> np.ndarray:
    values = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", values)
    return values.reshape(-1, 3)


def set_mesh_local_array(obj, values: np.ndarray) -> None:
    obj.data.vertices.foreach_set("co", np.asarray(values, dtype=np.float32).ravel())
    obj.data.update()


def to_world(obj, local: np.ndarray) -> np.ndarray:
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    homogeneous = np.c_[local, np.ones(len(local))]
    return (homogeneous @ matrix.T)[:, :3]


def to_local(obj, world: np.ndarray) -> np.ndarray:
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    inverse = np.linalg.inv(matrix)
    homogeneous = np.c_[world, np.ones(len(world))]
    return (homogeneous @ inverse.T)[:, :3]


def world_vector_to_local(obj, delta: np.ndarray) -> np.ndarray:
    rotation = np.asarray(obj.matrix_world, dtype=np.float64)[:3, :3]
    return delta @ np.linalg.inv(rotation).T


def smoothstep01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def ellipsoid_weight(points: np.ndarray, center, radii, inner: float = 0.0, outer: float = 1.0) -> np.ndarray:
    center = np.asarray(center, dtype=np.float64)
    radii = np.maximum(np.asarray(radii, dtype=np.float64), 1e-7)
    q = np.sqrt(np.sum(((points - center) / radii) ** 2, axis=1))
    weight = np.zeros(len(points), dtype=np.float64)
    weight[q <= inner] = 1.0
    mask = (q > inner) & (q < outer)
    if np.any(mask):
        t = (q[mask] - inner) / max(outer - inner, 1e-9)
        weight[mask] = 0.5 * (1.0 + np.cos(np.pi * t))
    return weight


def normalized_semantic(magnitude: np.ndarray, lower_quantile: float = 0.45) -> np.ndarray:
    positive = magnitude[magnitude > 1e-8]
    if not len(positive):
        return np.zeros_like(magnitude)
    low = max(float(np.quantile(positive, lower_quantile)), 2.0e-5)
    high = max(float(np.quantile(positive, 0.985)), low * 1.5)
    return smoothstep01((magnitude - low) / max(high - low, 1e-9))


def max_shape_delta(skin, basis: np.ndarray, tokens: tuple[str, ...]) -> np.ndarray:
    result = np.zeros(len(basis), dtype=np.float64)
    for key in skin.data.shape_keys.key_blocks:
        lower = key.name.lower()
        if key.name == "Basis" or not any(token in lower for token in tokens):
            continue
        result = np.maximum(result, np.linalg.norm(key_array(key) - basis, axis=1))
    return result


def world_vertices(obj) -> np.ndarray:
    return to_world(obj, mesh_local_array(obj))


def identify_skin(scene):
    skin = bpy.data.objects.get("cm_vitruvian")
    if skin and skin.type == "MESH" and skin.data.shape_keys:
        return skin
    candidates = [
        obj for obj in scene.objects
        if obj.type == "MESH" and obj.data.shape_keys and len(obj.data.shape_keys.key_blocks) > 8
    ]
    if not candidates:
        raise RuntimeError("Could not identify the Vitruvian FACS skin mesh")
    return max(candidates, key=lambda obj: len(obj.data.vertices))


def eye_related_objects(scene):
    result = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        name = obj.name.lower()
        if any(token in name for token in ("eye", "sclera", "iris", "pupil", "tear", "caruncle")):
            result.append(obj)
    return result


def eyeball_objects(scene):
    return [
        obj for obj in scene.objects
        if obj.type == "MESH" and "eyeball" in obj.name.lower()
    ]


def weighted_center(points: np.ndarray, weight: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    total = float(weight.sum())
    if total < 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return np.sum(points * weight[:, None], axis=0) / total


def add_scaled_region(delta: np.ndarray, points: np.ndarray, center, radii, factors, weight=None, inner=0.0, outer=1.0):
    center = np.asarray(center, dtype=np.float64)
    local_weight = ellipsoid_weight(points, center, radii, inner, outer)
    if weight is not None:
        local_weight *= np.asarray(weight, dtype=np.float64)
    target = center + (points - center) * np.asarray(factors, dtype=np.float64)
    delta += local_weight[:, None] * (target - points)
    return local_weight


def add_shift(delta: np.ndarray, points: np.ndarray, center, radii, shift, weight=None, inner=0.0, outer=1.0):
    local_weight = ellipsoid_weight(points, center, radii, inner, outer)
    if weight is not None:
        local_weight *= np.asarray(weight, dtype=np.float64)
    delta += local_weight[:, None] * np.asarray(shift, dtype=np.float64)
    return local_weight


def sculpt_skin(skin, eye_centers: list[np.ndarray]) -> dict:
    keys = skin.data.shape_keys.key_blocks
    basis_key = keys.get("Basis") or keys[0]
    basis_local = key_array(basis_key)
    basis_world = to_world(skin, basis_local)
    original_deltas = {
        key.name: key_array(key) - basis_local
        for key in keys if key.name != "Basis"
    }

    lo = basis_world.min(axis=0)
    hi = basis_world.max(axis=0)
    center = (lo + hi) * 0.5
    head_height = float(hi[2] - lo[2])
    head_width = float(hi[0] - lo[0])

    if eye_centers:
        eyes_average = np.mean(eye_centers, axis=0)
        forward_sign = -1.0 if eyes_average[1] < center[1] else 1.0
        eye_z = float(eyes_average[2])
        face_x = float(eyes_average[0])
    else:
        forward_sign = -1.0
        eye_z = float(lo[2] + head_height * 0.60)
        face_x = float(center[0])

    frontness = forward_sign * (basis_world[:, 1] - center[1])
    front_low = float(np.quantile(frontness, 0.54))
    front_high = float(np.quantile(frontness, 0.93))
    face_front = smoothstep01((frontness - front_low) / max(front_high - front_low, 1e-8))
    side_falloff = smoothstep01((0.125 - np.abs(basis_world[:, 0] - face_x)) / 0.060)
    face_weight = face_front * side_falloff

    eye_mag = max_shape_delta(skin, basis_local, ("eyes_closed", "eyes_opened", "eyes_squint"))
    brow_mag = max_shape_delta(skin, basis_local, ("eyebrows_",))
    lip_mag = max_shape_delta(
        skin,
        basis_local,
        ("lips_", "kiss", "smile_lips", "aa_", "ow_", "p_b_m", "f_v", "ey_eh_uh"),
    )
    mouth_mag = max_shape_delta(
        skin,
        basis_local,
        ("jaw_lower", "mouth_large", "lips_", "happy", "sad", "kiss", "aa_", "ow_", "p_b_m", "f_v", "ey_eh_uh"),
    )
    eye_semantic = normalized_semantic(eye_mag, 0.34)
    brow_semantic = normalized_semantic(brow_mag, 0.30)
    lip_semantic = normalized_semantic(lip_mag, 0.30)
    mouth_semantic = normalized_semantic(mouth_mag, 0.42)

    mouth_fallback = np.array([face_x, center[1] + forward_sign * head_height * 0.28, eye_z - head_height * 0.26])
    mouth_filter = smoothstep01((eye_z - basis_world[:, 2]) / (head_height * 0.10)) * face_weight
    mouth_center = weighted_center(basis_world, lip_semantic * mouth_filter, mouth_fallback)
    mouth_z = float(mouth_center[2])

    chin_guess = mouth_z - head_height * 0.235
    central_lower = (
        (np.abs(basis_world[:, 0] - face_x) < head_width * 0.28)
        & (basis_world[:, 2] < mouth_z)
        & (basis_world[:, 2] > lo[2] + head_height * 0.08)
        & (face_weight > 0.46)
    )
    if np.any(central_lower):
        chin_z = float(np.quantile(basis_world[central_lower, 2], 0.06))
        chin_z = max(chin_z, chin_guess - 0.010)
    else:
        chin_z = chin_guess

    delta = np.zeros_like(basis_world)

    # Youthful global proportions: shorter lower third, softer V jaw, modestly
    # narrower temples. Restricted to the visible face so the cranium and neck
    # do not collapse.
    lower = np.clip(
        (mouth_z + 0.014 - basis_world[:, 2])
        / max(mouth_z + 0.014 - chin_z, 1e-6),
        0.0,
        1.0,
    )
    lower_weight = face_weight * smoothstep01(lower)
    jaw_factor = 1.0 - 0.165 * np.power(lower, 1.22)
    delta[:, 0] += ((basis_world[:, 0] - face_x) * jaw_factor - (basis_world[:, 0] - face_x)) * lower_weight
    delta[:, 2] += 0.0105 * np.power(lower, 1.20) * lower_weight

    upper = np.clip(
        (basis_world[:, 2] - (eye_z + 0.018))
        / max(hi[2] - (eye_z + 0.018), 1e-6),
        0.0,
        1.0,
    )
    upper_weight = face_weight * smoothstep01(upper)
    delta[:, 0] += -0.034 * (basis_world[:, 0] - face_x) * upper_weight

    # Real eyelid/orbit shaping. Existing blink/open deltas identify the actual
    # eyelid topology, so the eye fissures are enlarged without deforming random
    # forehead or cheek vertices.
    sorted_eyes = sorted(eye_centers, key=lambda point: point[0]) if eye_centers else []
    if len(sorted_eyes) >= 2:
        active_eye_centers = [sorted_eyes[0], sorted_eyes[-1]]
    else:
        spacing = head_width * 0.19
        active_eye_centers = [
            np.array([face_x - spacing, center[1] + forward_sign * head_height * 0.29, eye_z]),
            np.array([face_x + spacing, center[1] + forward_sign * head_height * 0.29, eye_z]),
        ]
    eye_stats = []
    for eye_center in active_eye_centers:
        side = -1.0 if eye_center[0] < face_x else 1.0
        spatial = ellipsoid_weight(basis_world, eye_center, (0.050, 0.052, 0.041), 0.0, 1.18)
        semantic = np.maximum(eye_semantic, 0.18 * brow_semantic)
        weight = spatial * semantic
        delta[:, 0] += (basis_world[:, 0] - eye_center[0]) * 0.145 * weight
        delta[:, 2] += (basis_world[:, 2] - eye_center[2]) * 0.235 * weight
        delta[:, 1] += forward_sign * 0.0018 * spatial * eye_semantic

        outer_center = eye_center + np.array([side * 0.021, 0.0, 0.001])
        outer_weight = ellipsoid_weight(basis_world, outer_center, (0.025, 0.032, 0.021), 0.0, 1.08) * eye_semantic
        delta[:, 0] += side * 0.0022 * outer_weight
        delta[:, 2] += 0.00145 * outer_weight

        under_center = eye_center + np.array([0.0, -forward_sign * 0.003, -0.018])
        add_shift(
            delta,
            basis_world,
            under_center,
            (0.050, 0.050, 0.034),
            (0.0, forward_sign * 0.0021, 0.0010),
            weight=face_weight,
            outer=1.12,
        )
        eye_stats.append({"center": eye_center.tolist(), "semantic_vertices": int(np.sum(weight > 0.08))})

    # Reduce the adult brow ridge while retaining the real brow deformation area.
    delta[:, 1] += -forward_sign * 0.0021 * brow_semantic * face_weight
    delta[:, 2] += 0.0008 * brow_semantic * face_weight

    # Delicate readable nose. The center is derived from the frontmost central
    # surface between eyes and lips; the bridge, base and tip receive distinct
    # bounded changes so the side profile remains coherent.
    nose_z = mouth_z + 0.46 * (eye_z - mouth_z)
    central_nose = (
        (np.abs(basis_world[:, 0] - face_x) < 0.029)
        & (basis_world[:, 2] > mouth_z + 0.010)
        & (basis_world[:, 2] < eye_z - 0.004)
        & (face_weight > 0.34)
    )
    if np.any(central_nose):
        nose_index = np.where(central_nose)[0][np.argmax(frontness[central_nose])]
        nose_tip = basis_world[nose_index].copy()
    else:
        nose_tip = np.array([face_x, center[1] + forward_sign * head_height * 0.35, nose_z])
    nose_center = np.array([face_x, nose_tip[1], nose_z])
    nose_weight = add_scaled_region(
        delta,
        basis_world,
        nose_center,
        (0.034, 0.048, 0.060),
        (0.79, 0.93, 0.91),
        weight=face_weight,
        inner=0.03,
        outer=1.12,
    )
    delta[:, 1] += -forward_sign * 0.0022 * nose_weight
    delta[:, 2] += 0.0008 * nose_weight
    tip_weight = ellipsoid_weight(basis_world, nose_tip, (0.023, 0.031, 0.026), 0.02, 1.08) * face_weight
    delta[:, 1] += -forward_sign * 0.0013 * tip_weight
    delta[:, 2] += 0.0021 * tip_weight
    nose_base = np.array([face_x, nose_tip[1] - forward_sign * 0.001, mouth_z + 0.020])
    base_weight = add_scaled_region(
        delta,
        basis_world,
        nose_base,
        (0.034, 0.038, 0.030),
        (0.80, 0.95, 0.90),
        weight=face_weight,
        outer=1.10,
    )
    delta[:, 2] += 0.0015 * base_weight

    # Compact integrated lips and a shorter philtrum. The source FACS/viseme
    # deltas locate the lip surface exactly; the same neutral correction is later
    # applied to every expression key.
    lip_spatial = ellipsoid_weight(basis_world, mouth_center, (0.055, 0.052, 0.034), 0.0, 1.16)
    lip_weight = lip_spatial * np.maximum(lip_semantic, 0.18 * mouth_semantic)
    delta[:, 0] += (basis_world[:, 0] - mouth_center[0]) * 0.035 * lip_weight
    delta[:, 2] += (mouth_center[2] + (basis_world[:, 2] - mouth_center[2]) * 0.79 - basis_world[:, 2]) * lip_weight
    delta[:, 2] += 0.0042 * lip_weight
    delta[:, 1] += forward_sign * 0.0027 * lip_weight
    corner = np.clip(np.abs(basis_world[:, 0] - mouth_center[0]) / 0.045, 0.0, 1.0)
    delta[:, 2] += 0.0010 * np.power(corner, 1.6) * lip_weight

    philtrum_center = mouth_center + np.array([0.0, -forward_sign * 0.002, 0.019])
    add_shift(
        delta,
        basis_world,
        philtrum_center,
        (0.028, 0.034, 0.028),
        (0.0, 0.0, -0.0026),
        weight=face_weight,
        outer=1.10,
    )

    # High apple-cheek support and a smooth under-eye to cheek transition.
    for eye_center in active_eye_centers:
        side = -1.0 if eye_center[0] < face_x else 1.0
        cheek_center = np.array([
            eye_center[0] + side * 0.004,
            nose_tip[1] - forward_sign * 0.010,
            mouth_z + 0.48 * (eye_z - mouth_z),
        ])
        cheek_weight = add_shift(
            delta,
            basis_world,
            cheek_center,
            (0.057, 0.055, 0.052),
            (0.0, forward_sign * 0.0034, 0.0012),
            weight=face_weight,
            outer=1.15,
        )
        delta[:, 0] += -side * 0.0007 * cheek_weight

    # Small rounded chin and softened jaw angles.
    chin_front_y = nose_tip[1] - forward_sign * 0.010
    chin_center = np.array([face_x, chin_front_y, chin_z + 0.008])
    chin_weight = add_scaled_region(
        delta,
        basis_world,
        chin_center,
        (0.052, 0.058, 0.050),
        (0.78, 0.94, 0.86),
        weight=face_weight,
        inner=0.02,
        outer=1.12,
    )
    delta[:, 2] += 0.0040 * chin_weight
    delta[:, 1] += -forward_sign * 0.0015 * chin_weight
    for side in (-1.0, 1.0):
        jaw_center = np.array([face_x + side * head_width * 0.31, chin_front_y - forward_sign * 0.012, mouth_z - 0.032])
        jaw_weight = ellipsoid_weight(basis_world, jaw_center, (0.052, 0.060, 0.062), 0.0, 1.12) * face_weight
        delta[:, 0] += -side * 0.0060 * jaw_weight
        delta[:, 1] += -forward_sign * 0.0016 * jaw_weight

    # Tuck and slightly reduce ears without touching the face centre.
    ear_band = smoothstep01((np.abs(basis_world[:, 0] - face_x) - head_width * 0.34) / (head_width * 0.13))
    ear_vertical = smoothstep01((basis_world[:, 2] - (mouth_z - 0.020)) / 0.035) * smoothstep01(((eye_z + 0.035) - basis_world[:, 2]) / 0.035)
    ear_depth = smoothstep01((front_high - frontness) / max(front_high - front_low, 1e-8))
    ear_weight = ear_band * ear_vertical * ear_depth
    delta[:, 0] += -(basis_world[:, 0] - face_x) * 0.075 * ear_weight
    delta[:, 1] += forward_sign * 0.0020 * ear_weight

    # Bound the accumulated neutral sculpt. The smooth masks already provide
    # continuity; this final cap prevents any pathological local jump.
    length = np.linalg.norm(delta, axis=1)
    cap = np.minimum(1.0, 0.018 / np.maximum(length, 1e-9))
    delta *= cap[:, None]
    local_delta = world_vector_to_local(skin, delta)

    # Apply the exact same local displacement to Basis and every FACS key.
    for key in keys:
        values = key_array(key)
        set_key_array(key, values + local_delta)
    skin.data.update()

    new_basis = key_array(keys.get("Basis") or keys[0])
    preservation_max = 0.0
    preservation_rms = 0.0
    preservation_count = 0
    for key in keys:
        if key.name == "Basis":
            continue
        after_delta = key_array(key) - new_basis
        error = after_delta - original_deltas[key.name]
        preservation_max = max(preservation_max, float(np.linalg.norm(error, axis=1).max()))
        preservation_rms += float(np.sum(error * error))
        preservation_count += error.size
    preservation_rms = math.sqrt(preservation_rms / max(preservation_count, 1))

    return {
        "forward_sign_y": forward_sign,
        "bounds_min": lo.tolist(),
        "bounds_max": hi.tolist(),
        "head_height_m": head_height,
        "head_width_m": head_width,
        "eye_z_m": eye_z,
        "mouth_center_world": mouth_center.tolist(),
        "chin_z_m": chin_z,
        "nose_tip_world": nose_tip.tolist(),
        "eye_regions": eye_stats,
        "max_neutral_displacement_m": float(np.linalg.norm(delta, axis=1).max()),
        "rms_neutral_displacement_m": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
        "moved_vertices_over_0_5mm": int(np.sum(np.linalg.norm(delta, axis=1) > 0.0005)),
        "shape_delta_preservation_max_m": preservation_max,
        "shape_delta_preservation_rms_m": preservation_rms,
        "semantic_vertex_counts": {
            "eye": int(np.sum(eye_semantic > 0.08)),
            "brow": int(np.sum(brow_semantic > 0.08)),
            "lip": int(np.sum(lip_semantic > 0.08)),
            "mouth": int(np.sum(mouth_semantic > 0.08)),
        },
    }


def sculpt_eye_anatomy(scene, eye_centers: list[np.ndarray], face_x: float) -> dict:
    objects = eye_related_objects(scene)
    if not objects or len(eye_centers) < 2:
        return {"objects": [], "max_displacement_m": 0.0}
    centers = sorted(eye_centers, key=lambda point: point[0])
    left_center, right_center = centers[0], centers[-1]
    max_displacement = 0.0
    changed = []
    for obj in objects:
        local = mesh_local_array(obj)
        world = to_world(obj, local)
        original = world.copy()
        result = world.copy()
        for index, point in enumerate(world):
            center = left_center if point[0] < face_x else right_center
            side = -1.0 if center[0] < face_x else 1.0
            relative = point - center
            scale_xz = 1.075 if "eyeball" in obj.name.lower() else 1.105
            relative[0] *= scale_xz
            relative[2] *= scale_xz
            relative[1] *= 1.018
            result[index] = center + relative + np.array([side * 0.0012, 0.0, 0.0006])
        set_mesh_local_array(obj, to_local(obj, result))
        displacement = np.linalg.norm(result - original, axis=1)
        max_displacement = max(max_displacement, float(displacement.max()) if len(displacement) else 0.0)
        changed.append(obj.name)
    return {"objects": changed, "max_displacement_m": max_displacement}


def make_material(name, color, roughness=0.45, metallic=0.0, alpha=1.0):
    old = bpy.data.materials.get(name)
    mat = old or bpy.data.materials.new(name)
    mat.diffuse_color = tuple(color[:3]) + (alpha,)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
    if shader:
        shader.inputs["Base Color"].default_value = tuple(color[:3]) + (alpha,)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metallic
        if "Alpha" in shader.inputs:
            shader.inputs["Alpha"].default_value = alpha
    return mat


def assign_material_slots(obj, materials):
    obj.data.materials.clear()
    for material in materials:
        obj.data.materials.append(material)
    if not materials:
        return
    for polygon in obj.data.polygons:
        polygon.material_index = min(polygon.material_index, len(materials) - 1)
        polygon.use_smooth = True


def install_beauty_materials(scene, skin):
    skin_mat = make_material("AINA_V2_Skin", (0.58, 0.39, 0.36), 0.46)
    mouth_mat = make_material("AINA_V2_Mouth", (0.20, 0.035, 0.055), 0.38)
    sclera = make_material("AINA_V2_Sclera", (0.78, 0.82, 0.88), 0.24)
    iris = make_material("AINA_V2_Iris", (0.16, 0.34, 0.40), 0.27)
    dark = make_material("AINA_V2_EyeDark", (0.008, 0.012, 0.018), 0.20)
    cornea = make_material("AINA_V2_Cornea", (0.48, 0.58, 0.66), 0.08)
    shadow = make_material("AINA_V2_Eyeshadow", (0.22, 0.12, 0.15), 0.55)
    tear = make_material("AINA_V2_Tearline", (0.42, 0.18, 0.20), 0.24)

    assign_material_slots(skin, [skin_mat, mouth_mat])
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
            assign_material_slots(obj, slots or [sclera, iris, dark, cornea])
        elif "eyeshadow" in name:
            assign_material_slots(obj, [shadow])
        elif "tear" in name or "caruncle" in name:
            assign_material_slots(obj, [tear])
        else:
            assign_material_slots(obj, [skin_mat])
    return {"skin": skin_mat, "mouth": mouth_mat, "sclera": sclera, "iris": iris, "dark": dark, "cornea": cornea, "shadow": shadow, "tear": tear}


def clear_render_objects(scene):
    for obj in list(scene.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def create_light(name, location, energy, size, target):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
    return obj


def render_setup(scene, meshes, eye_centers, out: Path):
    points = np.concatenate([world_vertices(obj) for obj in meshes], axis=0)
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) * 0.5
    size = hi - lo
    if eye_centers:
        eyes = np.mean(eye_centers, axis=0)
        forward_sign = -1.0 if eyes[1] < center[1] else 1.0
        target = np.array([eyes[0], center[1], eyes[2] - 0.018])
    else:
        forward_sign = -1.0
        target = center.copy()
        target[2] += 0.08 * size[2]
    distance = max(float(size[2]) * 2.72, float(size[0]) * 2.95, 0.86)
    front = np.array([target[0], center[1] + forward_sign * distance, target[2]])

    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.world.color = (0.025, 0.030, 0.045)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = -0.30
    except Exception:
        pass

    clear_render_objects(scene)
    create_light("AINA_V2_Key", tuple(front + np.array([0.45 * size[0], 0.0, 0.48 * size[2]])), 310, 2.2, target)
    create_light("AINA_V2_Fill", tuple(front + np.array([-0.65 * size[0], 0.18 * distance, 0.10 * size[2]])), 165, 2.7, target)
    create_light("AINA_V2_Rim", tuple(center + np.array([0.0, -forward_sign * distance * 0.72, 0.42 * size[2]])), 250, 2.0, target)

    camera_data = bpy.data.cameras.new("AINA_V2_Camera")
    camera = bpy.data.objects.new("AINA_V2_Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.data.lens = 88
    scene.camera = camera
    locations = {
        "FRONT": front,
        "THREE_QUARTER": front + np.array([0.43 * distance, -forward_sign * 0.10 * distance, 0.0]),
        "SIDE": center + np.array([distance, 0.0, target[2] - center[2]]),
        "LEFT_45": front + np.array([-0.48 * distance, -forward_sign * 0.12 * distance, 0.0]),
        "RIGHT_45": front + np.array([0.48 * distance, -forward_sign * 0.12 * distance, 0.0]),
    }
    return camera, target, locations


def render_view(scene, camera, target, location, path: Path):
    camera.location = tuple(location)
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def reset_shapes(skin):
    if not skin.data.shape_keys:
        return
    for key in skin.data.shape_keys.key_blocks:
        key.value = 0.0


def render_suite(scene, skin, meshes, eye_centers, out: Path):
    preview = out / "Preview"
    preview.mkdir(exist_ok=True)
    camera, target, locations = render_setup(scene, meshes, eye_centers, out)
    outputs = {}

    reset_shapes(skin)
    for name in ("FRONT", "THREE_QUARTER", "SIDE", "LEFT_45", "RIGHT_45"):
        path = preview / f"AINA_VITRUVIAN_SCULPT_V2_{name}.png"
        render_view(scene, camera, target, locations[name], path)
        outputs[name.lower()] = str(path)

    expression_cases = {
        "HAPPY": {"Happy": 1.0},
        "BLINK": {"Eyes_Closed_Max": 1.0},
        "AA": {"aa_02": 1.0, "Jaw_Lower": 0.25},
        "OU": {"ow_08": 1.0},
    }
    for name, values in expression_cases.items():
        reset_shapes(skin)
        for key_name, value in values.items():
            key = skin.data.shape_keys.key_blocks.get(key_name) if skin.data.shape_keys else None
            if key:
                key.value = value
        path = preview / f"AINA_VITRUVIAN_SCULPT_V2_{name}.png"
        render_view(scene, camera, target, locations["FRONT"], path)
        outputs[name.lower()] = str(path)
    reset_shapes(skin)
    return outputs


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)

    scene = bpy.context.scene
    skin = identify_skin(scene)
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    for obj in meshes:
        if obj.data.shape_keys:
            for key in obj.data.shape_keys.key_blocks:
                key.value = 0.0

    eyeballs = eyeball_objects(scene)
    eye_centers = [world_vertices(obj).mean(axis=0) for obj in eyeballs if len(obj.data.vertices)]
    face_x = float(np.mean(eye_centers, axis=0)[0]) if eye_centers else float(world_vertices(skin)[:, 0].mean())

    skin_report = sculpt_skin(skin, eye_centers)
    eye_report = sculpt_eye_anatomy(scene, eye_centers, face_x)
    bpy.context.view_layer.update()

    # Recompute eye centres after anatomy fitting for framing.
    eye_centers_after = [world_vertices(obj).mean(axis=0) for obj in eyeballs if len(obj.data.vertices)]
    install_beauty_materials(scene, skin)
    renders = render_suite(scene, skin, meshes, eye_centers_after, args.out)

    blend_path = args.out / "AINA_VITRUVIAN_IDENTITY_SCULPT_V2.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_VITRUVIAN_IDENTITY_SCULPT_V2.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    shape_keys = [key.name for key in skin.data.shape_keys.key_blocks] if skin.data.shape_keys else []
    report = {
        "product": "AINA Vitruvian Identity Sculpt v2",
        "real_3d_model": True,
        "source_topology": "CC0 Vitruvian/Antonia FACS head",
        "replacement_effect_art_generated": False,
        "topology_changed": False,
        "vertices": len(skin.data.vertices),
        "triangles": len(skin.data.loop_triangles),
        "shape_key_count": max(0, len(shape_keys) - 1),
        "shape_keys": shape_keys,
        "skin_sculpt": skin_report,
        "eye_anatomy": eye_report,
        "identity_lock": False,
        "visual_identity_lock": False,
        "candidate": True,
        "vrm_exported": False,
        "files": {
            "blend": str(blend_path),
            "glb": str(glb_path),
            "renders": renders,
        },
        "next_gate": "Inspect approved front/3Q/side against the real v2 model. Continue only the remaining visual residuals; do not export VRM until identity is accepted.",
    }
    (qa / "AINA_VITRUVIAN_IDENTITY_SCULPT_V2_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
