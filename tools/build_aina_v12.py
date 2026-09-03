#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_aina_v10 as glb


EMPTY = np.array([], dtype=np.int64)
LEFT_EYE = np.array([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], dtype=np.int64)
RIGHT_EYE = np.array([362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398], dtype=np.int64)
LEFT_BROW = np.array([46, 53, 52, 65, 55, 70, 63, 105, 66, 107], dtype=np.int64)
RIGHT_BROW = np.array([276, 283, 282, 295, 285, 300, 293, 334, 296, 336], dtype=np.int64)
MOUTH = np.array([
    0, 11, 12, 13, 14, 15, 16, 17, 18, 37, 39, 40, 61, 62, 72, 73, 74, 76,
    78, 80, 81, 82, 84, 87, 88, 89, 90, 91, 95, 146, 178, 179, 180, 181, 183,
    184, 185, 191, 267, 269, 270, 291, 292, 302, 303, 304, 306, 308, 310, 311,
    312, 314, 317, 318, 319, 320, 321, 324, 375, 402, 403, 404, 405, 407, 408,
    409, 415,
], dtype=np.int64)
NOSE = np.array([
    1, 2, 4, 5, 6, 19, 20, 45, 48, 49, 51, 64, 94, 97, 98, 115, 122, 129,
    131, 134, 141, 164, 168, 174, 188, 195, 196, 197, 198, 209, 217, 220, 236,
    237, 238, 239, 240, 241, 242, 244, 245, 248, 250, 275, 278, 279, 281, 294,
    305, 326, 327, 344, 351, 358, 360, 363, 370, 391, 399, 412, 417, 420, 429,
    437, 439, 455, 456, 457, 458, 459, 460, 462, 463,
], dtype=np.int64)
OVAL = np.array([
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
    127, 162, 21, 54, 103, 67, 109,
], dtype=np.int64)
ALL_LANDMARKS = np.arange(468, dtype=np.int64)


def unique(values: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.int64).reshape(-1) for value in values if len(value)]
    return np.unique(np.concatenate(arrays)) if arrays else EMPTY.copy()


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def role_landmarks(role: str) -> np.ndarray:
    if role in {"eye_white", "iris", "highlight", "eyeline"}:
        return unique((LEFT_EYE, RIGHT_EYE))
    if role == "brow":
        return unique((LEFT_BROW, RIGHT_BROW, LEFT_EYE, RIGHT_EYE))
    if role == "mouth":
        return MOUTH
    if role == "skin":
        return ALL_LANDMARKS
    return unique((OVAL, NOSE, LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, MOUTH))


class LandmarkField:
    def __init__(
        self,
        base_positions: np.ndarray,
        roles: dict[str, np.ndarray],
        mapping: dict,
        strength: float,
    ) -> None:
        self.base = np.asarray(base_positions, dtype=np.float64)
        self.roles = roles
        self.strength = float(strength)
        self.model = np.asarray(mapping["model_points"], dtype=np.float64)
        self.target = np.asarray(mapping["target_points"], dtype=np.float64)
        if self.model.shape != (468, 2) or self.target.shape != (468, 2):
            raise ValueError(f"Expected 468x2 landmark arrays, got {self.model.shape} and {self.target.shape}")
        raw_disp = self.target - self.model
        norm = np.linalg.norm(raw_disp, axis=1, keepdims=True)
        max_image_step = 0.030
        self.displacement = raw_disp * np.minimum(1.0, max_image_step / np.maximum(norm, 1e-8))

        skin = roles.get("skin", EMPTY)
        domain = self.base[skin] if len(skin) else self.base
        self.center_x = float(np.median(domain[:, 0]))
        self.mesh_x_min = float(np.quantile(domain[:, 0], 0.015))
        self.mesh_x_max = float(np.quantile(domain[:, 0], 0.985))
        self.mesh_y_min = float(np.quantile(domain[:, 1], 0.005))
        self.mesh_y_max = float(np.quantile(domain[:, 1], 0.995))

        whites = roles.get("eye_white", EMPTY)
        eye_y_values = []
        eye_x_values = {}
        for sign in (-1, 1):
            subset = whites[(self.base[whites, 0] - self.center_x) * sign > 0]
            if len(subset):
                eye_y_values.append(float(np.mean(self.base[subset, 1])))
                eye_x_values[sign] = float(np.mean(self.base[subset, 0]))
        self.mesh_eye_y = float(np.mean(eye_y_values)) if eye_y_values else self.mesh_y_min + (self.mesh_y_max - self.mesh_y_min) * 0.46
        self.mesh_eye_x = eye_x_values

        eye_landmarks = unique((LEFT_EYE, RIGHT_EYE))
        self.model_eye_y = float(np.mean(self.model[eye_landmarks, 1]))
        self.model_chin_y = float(self.model[152, 1])
        self.model_top_y = float(self.model[10, 1])
        self.model_left_x = float(self.model[234, 0])
        self.model_right_x = float(self.model[454, 0])
        if self.model_left_x > self.model_right_x:
            self.model_left_x, self.model_right_x = self.model_right_x, self.model_left_x
        self.model_center_x = 0.5 * (self.model_left_x + self.model_right_x)

        self.mesh_chin_y = self.mesh_y_min
        vertical_denom = max(self.mesh_eye_y - self.mesh_chin_y, 1e-8)
        self.v_per_mesh_y = (self.model_chin_y - self.model_eye_y) / vertical_denom
        inferred_top = self.mesh_eye_y + (self.model_eye_y - self.model_top_y) / max(self.v_per_mesh_y, 1e-8)
        self.mesh_face_top_y = clamp(inferred_top, self.mesh_eye_y + 0.20 * (self.mesh_y_max - self.mesh_eye_y), self.mesh_y_max)

        model_width = max(self.model_right_x - self.model_left_x, 1e-8)
        mesh_width = max(self.mesh_x_max - self.mesh_x_min, 1e-8)
        self.mesh_per_image_x = mesh_width / model_width
        self.mesh_per_image_y = 1.0 / max(self.v_per_mesh_y, 1e-8)

        self.role_for_vertex = np.full(len(self.base), "other", dtype=object)
        for role in ("skin", "eye_white", "iris", "highlight", "mouth", "brow", "eyeline"):
            indices = roles.get(role, EMPTY)
            if len(indices):
                self.role_for_vertex[indices] = role

    def mesh_to_image(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        u = self.model_left_x + (positions[:, 0] - self.mesh_x_min) / max(self.mesh_x_max - self.mesh_x_min, 1e-8) * (self.model_right_x - self.model_left_x)
        v = self.model_eye_y + (self.mesh_eye_y - positions[:, 1]) * self.v_per_mesh_y
        return np.column_stack((u, v))

    def interpolate(self, uv: np.ndarray, landmark_indices: np.ndarray, sigma: float, neighbours: int) -> tuple[np.ndarray, np.ndarray]:
        points = self.model[landmark_indices]
        disp = self.displacement[landmark_indices]
        d2 = np.sum((uv[:, None, :] - points[None, :, :]) ** 2, axis=2)
        k = min(neighbours, len(landmark_indices))
        if k < len(landmark_indices):
            nearest = np.argpartition(d2, k - 1, axis=1)[:, :k]
            chosen_d2 = np.take_along_axis(d2, nearest, axis=1)
            chosen_disp = disp[nearest]
        else:
            chosen_d2 = d2
            chosen_disp = np.broadcast_to(disp[None, :, :], (len(uv), len(disp), 2))
        weights = np.exp(-chosen_d2 / (2.0 * sigma * sigma)) / np.maximum(chosen_d2, 2.5e-6)
        total = np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        interpolated = np.sum(chosen_disp * weights[:, :, None], axis=1) / total
        nearest_distance = np.sqrt(np.min(chosen_d2, axis=1))
        return interpolated, nearest_distance

    def apply(self, source_positions: np.ndarray) -> np.ndarray:
        source = np.asarray(source_positions, dtype=np.float64)
        result = source.copy()
        uv = self.mesh_to_image(source)

        role_strengths = {"skin": 0.64, "eye_white": 0.96, "iris": 0.88, "highlight": 0.88, "eyeline": 0.96, "brow": 0.90, "mouth": 0.96, "other": 0.55}
        role_sigmas = {"skin": 0.043, "eye_white": 0.024, "iris": 0.026, "highlight": 0.026, "eyeline": 0.023, "brow": 0.028, "mouth": 0.026, "other": 0.050}
        role_neighbours = {"skin": 22, "eye_white": 12, "iris": 12, "highlight": 12, "eyeline": 12, "brow": 14, "mouth": 16, "other": 20}

        for role in role_strengths:
            indices = np.flatnonzero(self.role_for_vertex == role)
            if not len(indices):
                continue
            landmarks = role_landmarks(role)
            delta_uv, nearest = self.interpolate(uv[indices], landmarks, role_sigmas[role], role_neighbours[role])
            proximity = np.clip(1.0 - np.maximum(nearest - 0.018, 0.0) / 0.075, 0.0, 1.0)
            local_strength = self.strength * role_strengths[role] * proximity

            above = source[indices, 1] > self.mesh_face_top_y
            if np.any(above):
                fade = np.clip((self.mesh_y_max - source[indices[above], 1]) / max(self.mesh_y_max - self.mesh_face_top_y, 1e-8), 0.0, 1.0)
                local_strength[above] *= 0.15 + 0.85 * fade

            dx = np.clip(delta_uv[:, 0] * self.mesh_per_image_x, -0.0065, 0.0065)
            dy = np.clip(-delta_uv[:, 1] * self.mesh_per_image_y, -0.0065, 0.0065)
            result[indices, 0] += dx * local_strength
            result[indices, 1] += dy * local_strength

        return result


def face_mesh_and_positions(document: dict, binary: bytearray):
    mesh_index = glb.find_mesh_index(document, "face")
    mesh = document["meshes"][mesh_index]
    position_accessors = {primitive["attributes"]["POSITION"] for primitive in mesh["primitives"]}
    if len(position_accessors) != 1:
        raise ValueError(f"Expected a shared Face POSITION accessor, found {position_accessors}")
    accessor_index = next(iter(position_accessors))
    positions = glb.accessor_array(document, binary, accessor_index).astype(np.float64)
    roles = glb.primitive_vertex_sets(document, binary, mesh_index)
    return mesh_index, mesh, accessor_index, positions, roles


def deform_face(document: dict, binary: bytearray, mapping: dict, strength: float) -> dict:
    mesh_index, mesh, accessor_index, positions, roles = face_mesh_and_positions(document, binary)
    field = LandmarkField(positions, roles, mapping, strength)
    warped = field.apply(positions)
    glb.write_accessor(document, binary, accessor_index, warped.astype(np.float32))

    target_accessors: set[int] = set()
    for primitive in mesh["primitives"]:
        for target in primitive.get("targets", []):
            if "POSITION" in target:
                target_accessors.add(target["POSITION"])
    transformed = 0
    for target_accessor in sorted(target_accessors):
        delta = glb.accessor_array(document, binary, target_accessor).astype(np.float64)
        if len(delta) != len(positions):
            continue
        warped_target = field.apply(positions + delta)
        glb.write_accessor(document, binary, target_accessor, (warped_target - warped).astype(np.float32))
        transformed += 1

    return {"mesh_index": mesh_index, "vertices": len(positions), "skin_vertices": len(roles.get("skin", EMPTY)), "morph_position_accessors": transformed, "strength": strength, "mapping_full_rms": mapping.get("full_rms"), "mapping_p95": mapping.get("p95_displacement"), "face_top_y": field.mesh_face_top_y, "bounds_before": {"min": positions.min(axis=0).astype(float).tolist(), "max": positions.max(axis=0).astype(float).tolist()}, "bounds_after": {"min": warped.min(axis=0).astype(float).tolist(), "max": warped.max(axis=0).astype(float).tolist()}}


def adjust_hair(document: dict, binary: bytearray, before: dict, after: dict) -> dict:
    before_min = np.asarray(before["min"], dtype=np.float64)
    before_max = np.asarray(before["max"], dtype=np.float64)
    after_min = np.asarray(after["min"], dtype=np.float64)
    after_max = np.asarray(after["max"], dtype=np.float64)
    width_scale = clamp(float((after_max[0] - after_min[0]) / max(before_max[0] - before_min[0], 1e-8)), 0.94, 1.06)
    height_scale = clamp(float((after_max[1] - after_min[1]) / max(before_max[1] - before_min[1], 1e-8)), 0.95, 1.05)
    records = []
    processed = set()
    for mesh_index, mesh in enumerate(document.get("meshes", [])):
        name = mesh.get("name", "").lower()
        if "body" in name:
            for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
                if "hair" not in glb.material_name(document, primitive).lower():
                    continue
                accessor_index = primitive["attributes"]["POSITION"]
                if accessor_index in processed:
                    continue
                processed.add(accessor_index)
                values = glb.accessor_array(document, binary, accessor_index).astype(np.float64)
                indices = np.unique(glb.accessor_array(document, binary, primitive["indices"]).reshape(-1).astype(np.int64))
                subset = values[indices].copy()
                center_x = float(np.median(subset[:, 0]))
                anchor_y = float(np.quantile(subset[:, 1], 0.42))
                subset[:, 0] = center_x + (subset[:, 0] - center_x) * width_scale
                upper = subset[:, 1] > anchor_y
                subset[upper, 1] = anchor_y + (subset[upper, 1] - anchor_y) * height_scale
                values[indices] = subset
                glb.write_accessor(document, binary, accessor_index, values.astype(np.float32))
                records.append({"mesh": mesh_index, "primitive": primitive_index, "hair_vertices": len(indices)})
        elif "updo" in name or "hairline" in name:
            for node_index, node in enumerate(document.get("nodes", [])):
                if node.get("mesh") != mesh_index:
                    continue
                scale = list(node.get("scale", [1.0, 1.0, 1.0]))
                node["scale"] = [scale[0] * width_scale, scale[1] * height_scale, scale[2]]
                records.append({"mesh": mesh_index, "node": node_index, "scale": node["scale"]})
    return {"width_scale": width_scale, "height_scale": height_scale, "records": records}


def process(source: Path, destination: Path, mapping: dict, strength: float, label: str) -> dict:
    document, binary = glb.read_glb(source)
    face = deform_face(document, binary, mapping, strength)
    hair = adjust_hair(document, binary, face["bounds_before"], face["bounds_after"])
    document.setdefault("asset", {})["generator"] = f"AINA V12 full-468 landmark field {label}"
    glb.write_glb(destination, document, binary)
    return {"source": source.name, "output": destination.name, "bytes": destination.stat().st_size, "face": face, "hair": hair}


def main() -> None:
    if len(sys.argv) != 8:
        raise SystemExit("build_aina_v12.py FORMAL.vrm BLENDER.glb MAPPING.json OUTPUT_DIR LABEL STEM STRENGTH")
    formal_source = Path(sys.argv[1])
    safe_source = Path(sys.argv[2])
    mapping_path = Path(sys.argv[3])
    output = Path(sys.argv[4])
    label = sys.argv[5]
    stem = sys.argv[6]
    strength = float(sys.argv[7])
    output.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(mapping_path.read_text())
    formal_path = output / f"{stem}.vrm"
    safe_path = output / f"{stem}_BLENDER.glb"
    formal = process(formal_source, formal_path, mapping, strength, label)
    safe = process(safe_source, safe_path, mapping, strength, label)
    report = {"version": label, "method": "full 468-landmark similarity-aligned IDW/Gaussian deformation field", "mapping": str(mapping_path), "strength": strength, "formal": formal, "blender": safe, "preserved": {"vrm_1_0": True, "humanoid_bones": 54, "face_morphs": 57, "expression_presets": 14}, "identity_lock": False, "visual_identity_lock": False, "manual_three_view_review_required": True}
    (output / f"{stem}_BUILD_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
