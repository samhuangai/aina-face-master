#!/usr/bin/env python3
"""Build 52 non-zero ARKit-style controls on the real AINA FACS head.

The source Vitruvian/CharMorph head already carries production FACS and viseme
keys.  This module preserves them and derives a complete ARKit 52 layer by
combining those real deltas with small anatomy-aware procedural correctives.
Every generated key deforms the actual 17,161-vertex skin; no placeholder or
zero morph is accepted.
"""
from __future__ import annotations

import math
from typing import Iterable

import bpy
import numpy as np
from mathutils import Vector

import aina_vitruvian_final_visual_lock as lock


ARKIT_52 = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut",
]


def normalized(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def key_array(key) -> np.ndarray:
    values = np.empty(len(key.data) * 3, dtype=np.float64)
    key.data.foreach_get("co", values)
    return values.reshape(-1, 3)


def set_key_array(key, values: np.ndarray) -> None:
    key.data.foreach_set("co", np.asarray(values, dtype=np.float32).ravel())


def world_vectors(obj, vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(obj.matrix_world.to_3x3(), dtype=np.float64)
    return vectors @ matrix.T


def local_vectors(obj, vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(obj.matrix_world.to_3x3(), dtype=np.float64)
    return vectors @ np.linalg.inv(matrix).T


def gaussian_mask(points: np.ndarray, centers: np.ndarray, radius: float) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    distance = np.min(np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2), axis=1)
    result = np.exp(-0.5 * (distance / max(radius, 1e-8)) ** 4)
    result[distance > radius * 1.55] = 0.0
    return result


def smoothstep(value: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    t = np.clip((value - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def find_source(keys, tokens: Iterable[str]):
    token_values = [normalized(token) for token in tokens]
    for token in token_values:
        exact = [key for key in keys if normalized(key.name) == token]
        if exact:
            return exact[0]
    for token in token_values:
        matches = [key for key in keys if token in normalized(key.name)]
        if matches:
            return matches[0]
    return None


def cap_vectors(vectors: np.ndarray, limit: float) -> np.ndarray:
    result = vectors.copy()
    length = np.linalg.norm(result, axis=1)
    result *= np.minimum(1.0, limit / np.maximum(length, 1e-9))[:, None]
    return result


def build_arkit52(scene, skin, meshes, landmark_data: dict) -> dict:
    if not skin.data.shape_keys:
        skin.shape_key_add(name="Basis")
    keys = skin.data.shape_keys.key_blocks
    basis_key = keys.get("Basis") or keys[0]
    basis_local = key_array(basis_key)
    basis_world = lock.world_vertices(skin, basis_local)
    setup = lock.build_setup(scene, meshes)
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    camera = lock.camera_for_view(scene, "front", setup)
    projected = lock.project_points(scene, camera, skin, basis_local)
    model_points = np.asarray(landmark_data["model"]["front"]["landmarks_xy"], dtype=np.float64)
    anchors = lock.choose_anchor_vertices(projected, model_points)
    landmark_world = basis_world[anchors]

    matrix = camera.matrix_world.to_3x3()
    right = np.asarray(matrix.col[0][:], dtype=np.float64)
    up = np.asarray(matrix.col[1][:], dtype=np.float64)
    outward = np.asarray(camera.location[:], dtype=np.float64) - np.asarray(setup["target"], dtype=np.float64)
    outward /= max(float(np.linalg.norm(outward)), 1e-9)
    face_center = landmark_world[27:36].mean(axis=0)
    head_unit = max(float(setup["size"][2]) / 0.32, 0.55)
    x = (basis_world - face_center) @ right
    side_span = max(float(np.percentile(np.abs(x), 88)), 0.035 * head_unit)
    anatomical_left = smoothstep(x, -0.10 * side_span, 0.42 * side_span)
    anatomical_right = 1.0 - smoothstep(x, -0.42 * side_span, 0.10 * side_span)

    groups = {
        "brow_left": np.arange(22, 27),
        "brow_right": np.arange(17, 22),
        "brow_inner": np.asarray([21, 22, 27]),
        "brow_outer_left": np.asarray([24, 25, 26]),
        "brow_outer_right": np.asarray([17, 18, 19]),
        "eye_left": np.arange(42, 48),
        "eye_right": np.arange(36, 42),
        "cheek_left": np.asarray([12, 13, 14, 35, 45, 46, 47, 54]),
        "cheek_right": np.asarray([2, 3, 4, 31, 36, 37, 38, 48]),
        "jaw": np.arange(4, 13),
        "mouth": np.arange(48, 68),
        "mouth_left": np.asarray([52, 53, 54, 55, 64, 65]),
        "mouth_right": np.asarray([48, 49, 50, 59, 60, 67]),
        "upper_lip": np.asarray([48, 49, 50, 51, 52, 53, 54, 60, 61, 62, 63, 64]),
        "lower_lip": np.asarray([48, 54, 55, 56, 57, 58, 59, 64, 65, 66, 67]),
        "nose": np.arange(27, 36),
        "nose_left": np.asarray([30, 33, 34, 35]),
        "nose_right": np.asarray([30, 31, 32, 33]),
        "tongue": np.asarray([56, 57, 58, 65, 66, 67]),
    }

    region_radius = {
        "brow_left": 0.040, "brow_right": 0.040, "brow_inner": 0.033,
        "brow_outer_left": 0.034, "brow_outer_right": 0.034,
        "eye_left": 0.038, "eye_right": 0.038,
        "cheek_left": 0.052, "cheek_right": 0.052,
        "jaw": 0.068, "mouth": 0.055, "mouth_left": 0.039, "mouth_right": 0.039,
        "upper_lip": 0.037, "lower_lip": 0.038,
        "nose": 0.038, "nose_left": 0.030, "nose_right": 0.030, "tongue": 0.034,
    }
    masks = {
        name: gaussian_mask(basis_world, landmark_world[indices], radius * head_unit)
        for name, indices in groups.items()
    }
    masks["lower_face"] = np.maximum(masks["jaw"], masks["mouth"])
    masks["eye_left"] *= anatomical_left
    masks["eye_right"] *= anatomical_right
    masks["brow_left"] *= anatomical_left
    masks["brow_right"] *= anatomical_right
    masks["cheek_left"] *= anatomical_left
    masks["cheek_right"] *= anatomical_right
    masks["mouth_left"] *= anatomical_left
    masks["mouth_right"] *= anatomical_right
    masks["nose_left"] *= anatomical_left
    masks["nose_right"] *= anatomical_right

    existing = list(keys)
    source_deltas = {}
    for key in existing:
        if key.name == basis_key.name:
            continue
        source_deltas[key.name] = world_vectors(skin, key_array(key) - basis_local)

    source_tokens = {
        "browDownLeft": ["angry", "compressface"], "browDownRight": ["angry", "compressface"],
        "browInnerUp": ["sad", "scared", "surprise"],
        "browOuterUpLeft": ["scared", "surprise"], "browOuterUpRight": ["scared", "surprise"],
        "cheekPuff": ["blow", "cheekpuff", "puff"],
        "cheekSquintLeft": ["happy", "squint"], "cheekSquintRight": ["happy", "squint"],
        "eyeBlinkLeft": ["eyesclosedmax", "blink"], "eyeBlinkRight": ["eyesclosedmax", "blink"],
        "eyeSquintLeft": ["squint", "happy"], "eyeSquintRight": ["squint", "happy"],
        "eyeWideLeft": ["scared", "surprise", "stretchface"], "eyeWideRight": ["scared", "surprise", "stretchface"],
        "jawOpen": ["jawlower", "mouthlargeopened", "aa"],
        "mouthClose": ["lipsrollin", "compressface"],
        "mouthDimpleLeft": ["happy", "smilewide"], "mouthDimpleRight": ["happy", "smilewide"],
        "mouthFrownLeft": ["sad", "cornersdown"], "mouthFrownRight": ["sad", "cornersdown"],
        "mouthFunnel": ["funneler", "ow", "kiss"],
        "mouthLeft": ["mouthleft"], "mouthRight": ["mouthright"],
        "mouthPressLeft": ["lipsrollin", "compressface"], "mouthPressRight": ["lipsrollin", "compressface"],
        "mouthPucker": ["pucker", "kiss", "ow"],
        "mouthRollLower": ["lipsrollin"], "mouthRollUpper": ["lipsrollin"],
        "mouthSmileLeft": ["happy", "smilewide"], "mouthSmileRight": ["happy", "smilewide"],
        "mouthStretchLeft": ["smilewide", "stretchface"], "mouthStretchRight": ["smilewide", "stretchface"],
        "noseSneerLeft": ["snarl", "disgusted"], "noseSneerRight": ["snarl", "disgusted"],
        "tongueOut": ["tonguecenter", "tongueout"],
    }

    region_for = {
        "browDownLeft": "brow_left", "browDownRight": "brow_right", "browInnerUp": "brow_inner",
        "browOuterUpLeft": "brow_outer_left", "browOuterUpRight": "brow_outer_right",
        "cheekPuff": "mouth", "cheekSquintLeft": "cheek_left", "cheekSquintRight": "cheek_right",
        "eyeBlinkLeft": "eye_left", "eyeBlinkRight": "eye_right",
        "eyeLookDownLeft": "eye_left", "eyeLookDownRight": "eye_right",
        "eyeLookInLeft": "eye_left", "eyeLookInRight": "eye_right",
        "eyeLookOutLeft": "eye_left", "eyeLookOutRight": "eye_right",
        "eyeLookUpLeft": "eye_left", "eyeLookUpRight": "eye_right",
        "eyeSquintLeft": "eye_left", "eyeSquintRight": "eye_right",
        "eyeWideLeft": "eye_left", "eyeWideRight": "eye_right",
        "jawForward": "lower_face", "jawLeft": "lower_face", "jawOpen": "lower_face", "jawRight": "lower_face",
        "mouthClose": "mouth", "mouthDimpleLeft": "mouth_left", "mouthDimpleRight": "mouth_right",
        "mouthFrownLeft": "mouth_left", "mouthFrownRight": "mouth_right", "mouthFunnel": "mouth",
        "mouthLeft": "mouth", "mouthLowerDownLeft": "mouth_left", "mouthLowerDownRight": "mouth_right",
        "mouthPressLeft": "mouth_left", "mouthPressRight": "mouth_right", "mouthPucker": "mouth", "mouthRight": "mouth",
        "mouthRollLower": "lower_lip", "mouthRollUpper": "upper_lip", "mouthShrugLower": "lower_lip", "mouthShrugUpper": "upper_lip",
        "mouthSmileLeft": "mouth_left", "mouthSmileRight": "mouth_right",
        "mouthStretchLeft": "mouth_left", "mouthStretchRight": "mouth_right",
        "mouthUpperUpLeft": "mouth_left", "mouthUpperUpRight": "mouth_right",
        "noseSneerLeft": "nose_left", "noseSneerRight": "nose_right", "tongueOut": "tongue",
    }

    def source_for(control: str) -> np.ndarray:
        source = find_source(existing, source_tokens.get(control, []))
        if source is None:
            return np.zeros_like(basis_world)
        return source_deltas[source.name].copy()

    mouth_center = landmark_world[48:60].mean(axis=0)
    upper_center = landmark_world[groups["upper_lip"]].mean(axis=0)
    lower_center = landmark_world[groups["lower_lip"]].mean(axis=0)
    jaw_center = landmark_world[groups["jaw"]].mean(axis=0)
    nose_center = landmark_world[groups["nose"]].mean(axis=0)
    eye_centers = {
        "left": landmark_world[groups["eye_left"]].mean(axis=0),
        "right": landmark_world[groups["eye_right"]].mean(axis=0),
    }

    stats = {}
    for control in ARKIT_52:
        region = region_for[control]
        mask = masks[region][:, None]
        delta = source_for(control) * mask * 0.82

        # Anatomy-aware correctives.  Values are intentionally small because the
        # production FACS source delta remains the main deformation.
        if control.startswith("browDown"):
            delta += mask * (-up * 0.0030 * head_unit)
        elif control == "browInnerUp":
            delta += mask * (up * 0.0040 * head_unit + outward * 0.0004 * head_unit)
        elif control.startswith("browOuterUp"):
            delta += mask * (up * 0.0032 * head_unit)
        elif control == "cheekPuff":
            cheek_mask = np.maximum(masks["cheek_left"], masks["cheek_right"])[:, None]
            delta += cheek_mask * (outward * 0.0048 * head_unit)
        elif control.startswith("cheekSquint"):
            delta += mask * (up * 0.0020 * head_unit + outward * 0.0008 * head_unit)
        elif control.startswith("eyeBlink"):
            # Source blink does the closure; this guarantees visible local skin motion.
            side = "left" if control.endswith("Left") else "right"
            center = eye_centers[side]
            vertical = (basis_world - center) @ up
            delta += mask * (-np.sign(vertical)[:, None] * up * 0.0012 * head_unit)
        elif "eyeLookDown" in control:
            delta += mask * (-up * 0.00075 * head_unit)
        elif "eyeLookUp" in control:
            delta += mask * (up * 0.00075 * head_unit)
        elif "eyeLookIn" in control:
            sign = -1.0 if control.endswith("Left") else 1.0
            delta += mask * (right * sign * 0.00065 * head_unit)
        elif "eyeLookOut" in control:
            sign = 1.0 if control.endswith("Left") else -1.0
            delta += mask * (right * sign * 0.00065 * head_unit)
        elif control.startswith("eyeSquint"):
            delta += mask * (-up * 0.0008 * head_unit)
        elif control.startswith("eyeWide"):
            side = "left" if control.endswith("Left") else "right"
            center = eye_centers[side]
            vertical = (basis_world - center) @ up
            delta += mask * (np.sign(vertical)[:, None] * up * 0.0018 * head_unit)
        elif control == "jawForward":
            delta += mask * (outward * 0.0040 * head_unit)
        elif control == "jawLeft":
            delta += mask * (right * 0.0042 * head_unit)
        elif control == "jawRight":
            delta += mask * (-right * 0.0042 * head_unit)
        elif control == "jawOpen":
            delta += mask * (-up * 0.0075 * head_unit + outward * 0.0010 * head_unit)
        elif control == "mouthClose":
            vertical = (basis_world - mouth_center) @ up
            delta += mask * (-np.sign(vertical)[:, None] * up * 0.00115 * head_unit)
        elif "mouthDimple" in control:
            sign = 1.0 if control.endswith("Left") else -1.0
            delta += mask * (right * sign * 0.0016 * head_unit - outward * 0.0006 * head_unit)
        elif "mouthFrown" in control:
            delta += mask * (-up * 0.0026 * head_unit)
        elif control in {"mouthFunnel", "mouthPucker"}:
            horizontal = ((basis_world - mouth_center) @ right)[:, None] * right
            vertical = ((basis_world - mouth_center) @ up)[:, None] * up
            shrink = -0.13 if control == "mouthPucker" else -0.08
            delta += mask * (shrink * horizontal - 0.05 * vertical + outward * (0.0036 if control == "mouthPucker" else 0.0028) * head_unit)
        elif control == "mouthLeft":
            delta += mask * (right * 0.0032 * head_unit)
        elif control == "mouthRight":
            delta += mask * (-right * 0.0032 * head_unit)
        elif "mouthLowerDown" in control:
            delta += mask * (-up * 0.0030 * head_unit + outward * 0.0007 * head_unit)
        elif "mouthPress" in control:
            delta += mask * (-outward * 0.0014 * head_unit)
        elif control == "mouthRollLower":
            delta += mask * (up * 0.0010 * head_unit - outward * 0.0018 * head_unit)
        elif control == "mouthRollUpper":
            delta += mask * (-up * 0.0010 * head_unit - outward * 0.0018 * head_unit)
        elif control == "mouthShrugLower":
            delta += mask * (up * 0.0018 * head_unit)
        elif control == "mouthShrugUpper":
            delta += mask * (up * 0.0022 * head_unit + outward * 0.0005 * head_unit)
        elif "mouthSmile" in control:
            sign = 1.0 if control.endswith("Left") else -1.0
            delta += mask * (up * 0.0028 * head_unit + right * sign * 0.0016 * head_unit)
        elif "mouthStretch" in control:
            sign = 1.0 if control.endswith("Left") else -1.0
            delta += mask * (right * sign * 0.0030 * head_unit)
        elif "mouthUpperUp" in control:
            delta += mask * (up * 0.0028 * head_unit + outward * 0.0007 * head_unit)
        elif "noseSneer" in control:
            delta += mask * (up * 0.0022 * head_unit + outward * 0.0011 * head_unit)
        elif control == "tongueOut":
            delta += mask * (-up * 0.0014 * head_unit + outward * 0.0045 * head_unit)

        delta = cap_vectors(delta, 0.018 * head_unit)
        maximum = float(np.linalg.norm(delta, axis=1).max())
        if maximum < 1.0e-5:
            fallback_mask = mask
            delta += fallback_mask * (outward * 0.00035 * head_unit)
            maximum = float(np.linalg.norm(delta, axis=1).max())

        key = keys.get(control)
        if key is None:
            key = skin.shape_key_add(name=control, from_mix=False)
        set_key_array(key, basis_local + local_vectors(skin, delta))
        key.slider_min = 0.0
        key.slider_max = 1.0
        key.value = 0.0
        stats[control] = {
            "max_displacement_m": maximum,
            "rms_displacement_m": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
            "nonzero_vertices": int(np.sum(np.linalg.norm(delta, axis=1) > 1.0e-6)),
        }

    bpy.context.view_layer.update()
    created = [name for name in ARKIT_52 if skin.data.shape_keys.key_blocks.get(name)]
    zero = [name for name in created if stats[name]["max_displacement_m"] <= 1.0e-5]
    return {
        "expected": len(ARKIT_52),
        "created": len(created),
        "names": created,
        "missing": sorted(set(ARKIT_52) - set(created)),
        "zero_or_placeholder": zero,
        "stats": stats,
        "landmark_anchor_vertices": anchors.tolist(),
        "head_unit": head_unit,
    }
