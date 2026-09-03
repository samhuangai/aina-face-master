#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_aina_v12 as face2d


glb = face2d.glb
EMPTY = face2d.EMPTY


class DepthField:
    def __init__(self, positions: np.ndarray, roles: dict[str, np.ndarray], mapping: dict, strength: float, orientation: str) -> None:
        self.base = np.asarray(positions, dtype=np.float64)
        self.roles = roles
        self.strength = float(strength)
        self.orientation = orientation
        if orientation not in {"normal", "flipped"}:
            raise ValueError(f"Unsupported profile-axis orientation: {orientation}")
        self.model = np.asarray(mapping["model_points"], dtype=np.float64)
        self.target = np.asarray(mapping["target_points"], dtype=np.float64)
        if self.model.shape != (468, 2) or self.target.shape != (468, 2):
            raise ValueError(f"Expected 468x2 profile landmark arrays, got {self.model.shape} and {self.target.shape}")
        displacement = self.target - self.model
        norm = np.linalg.norm(displacement, axis=1, keepdims=True)
        self.displacement = displacement * np.minimum(1.0, 0.032 / np.maximum(norm, 1e-8))

        skin = roles.get("skin", EMPTY)
        domain = self.base[skin] if len(skin) else self.base
        self.mesh_z_min = float(np.quantile(domain[:, 2], 0.010))
        self.mesh_z_max = float(np.quantile(domain[:, 2], 0.990))
        self.mesh_y_min = float(np.quantile(domain[:, 1], 0.005))
        self.mesh_y_max = float(np.quantile(domain[:, 1], 0.995))
        self.center_z = 0.5 * (self.mesh_z_min + self.mesh_z_max)

        oval_u = self.model[face2d.OVAL, 0]
        self.model_u_min = float(np.quantile(oval_u, 0.01))
        self.model_u_max = float(np.quantile(oval_u, 0.99))
        eye_indices = face2d.unique((face2d.LEFT_EYE, face2d.RIGHT_EYE))
        self.model_eye_y = float(np.mean(self.model[eye_indices, 1]))
        self.model_chin_y = float(self.model[152, 1])
        self.model_top_y = float(self.model[10, 1])

        whites = roles.get("eye_white", EMPTY)
        self.mesh_eye_y = float(np.mean(self.base[whites, 1])) if len(whites) else self.mesh_y_min + (self.mesh_y_max - self.mesh_y_min) * 0.46
        self.mesh_chin_y = self.mesh_y_min
        self.v_per_mesh_y = (self.model_chin_y - self.model_eye_y) / max(self.mesh_eye_y - self.mesh_chin_y, 1e-8)
        inferred_top = self.mesh_eye_y + (self.model_eye_y - self.model_top_y) / max(self.v_per_mesh_y, 1e-8)
        self.mesh_face_top_y = face2d.clamp(inferred_top, self.mesh_eye_y + 0.20 * (self.mesh_y_max - self.mesh_eye_y), self.mesh_y_max)
        self.mesh_per_image_z = (self.mesh_z_max - self.mesh_z_min) / max(self.model_u_max - self.model_u_min, 1e-8)
        self.mesh_per_image_y = 1.0 / max(self.v_per_mesh_y, 1e-8)

        self.role_for_vertex = np.full(len(self.base), "other", dtype=object)
        for role in ("skin", "eye_white", "iris", "highlight", "mouth", "brow", "eyeline"):
            indices = roles.get(role, EMPTY)
            if len(indices):
                self.role_for_vertex[indices] = role

    def mesh_to_image(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        normalized = (positions[:, 2] - self.mesh_z_min) / max(self.mesh_z_max - self.mesh_z_min, 1e-8)
        if self.orientation == "normal":
            u = self.model_u_min + normalized * (self.model_u_max - self.model_u_min)
        else:
            u = self.model_u_max - normalized * (self.model_u_max - self.model_u_min)
        v = self.model_eye_y + (self.mesh_eye_y - positions[:, 1]) * self.v_per_mesh_y
        return np.column_stack((u, v))

    def interpolate(self, uv: np.ndarray, landmarks: np.ndarray, sigma: float, neighbours: int):
        points = self.model[landmarks]
        displacement = self.displacement[landmarks]
        d2 = np.sum((uv[:, None, :] - points[None, :, :]) ** 2, axis=2)
        k = min(neighbours, len(landmarks))
        if k < len(landmarks):
            nearest = np.argpartition(d2, k - 1, axis=1)[:, :k]
            chosen_d2 = np.take_along_axis(d2, nearest, axis=1)
            chosen_displacement = displacement[nearest]
        else:
            chosen_d2 = d2
            chosen_displacement = np.broadcast_to(displacement[None, :, :], (len(uv), len(displacement), 2))
        weights = np.exp(-chosen_d2 / (2.0 * sigma * sigma)) / np.maximum(chosen_d2, 2.5e-6)
        total = np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        delta = np.sum(chosen_displacement * weights[:, :, None], axis=1) / total
        nearest_distance = np.sqrt(np.min(chosen_d2, axis=1))
        return delta, nearest_distance

    def apply(self, source_positions: np.ndarray) -> np.ndarray:
        source = np.asarray(source_positions, dtype=np.float64)
        result = source.copy()
        uv = self.mesh_to_image(source)
        strengths = {"skin": 0.62, "eye_white": 0.84, "iris": 0.76, "highlight": 0.76, "eyeline": 0.86, "brow": 0.72, "mouth": 0.90, "other": 0.46}
        sigmas = {"skin": 0.046, "eye_white": 0.027, "iris": 0.029, "highlight": 0.029, "eyeline": 0.026, "brow": 0.031, "mouth": 0.028, "other": 0.052}
        neighbours = {"skin": 24, "eye_white": 14, "iris": 14, "highlight": 14, "eyeline": 14, "brow": 16, "mouth": 18, "other": 22}
        for role, role_strength in strengths.items():
            indices = np.flatnonzero(self.role_for_vertex == role)
            if not len(indices):
                continue
            landmark_indices = face2d.role_landmarks(role)
            delta_uv, nearest = self.interpolate(uv[indices], landmark_indices, sigmas[role], neighbours[role])
            proximity = np.clip(1.0 - np.maximum(nearest - 0.020, 0.0) / 0.082, 0.0, 1.0)
            local_strength = self.strength * role_strength * proximity
            above = source[indices, 1] > self.mesh_face_top_y
            if np.any(above):
                fade = np.clip((self.mesh_y_max - source[indices[above], 1]) / max(self.mesh_y_max - self.mesh_face_top_y, 1e-8), 0.0, 1.0)
                local_strength[above] *= 0.12 + 0.88 * fade
            dz = delta_uv[:, 0] * self.mesh_per_image_z
            if self.orientation == "flipped":
                dz = -dz
            dy = -delta_uv[:, 1] * self.mesh_per_image_y
            dz = np.clip(dz, -0.0065, 0.0065)
            dy = np.clip(dy, -0.0040, 0.0040)
            result[indices, 2] += dz * local_strength
            result[indices, 1] += dy * local_strength * 0.28
        return result


def deform_face(document: dict, binary: bytearray, mapping: dict, strength: float, orientation: str) -> dict:
    mesh_index, mesh, accessor_index, positions, roles = face2d.face_mesh_and_positions(document, binary)
    field = DepthField(positions, roles, mapping, strength, orientation)
    warped = field.apply(positions)
    glb.write_accessor(document, binary, accessor_index, warped.astype(np.float32))
    target_accessors = set()
    for primitive in mesh["primitives"]:
        for target in primitive.get("targets", []):
            if "POSITION" in target:
                target_accessors.add(target["POSITION"])
    transformed = 0
    for target_accessor in sorted(target_accessors):
        delta = glb.accessor_array(document, binary, target_accessor).astype(np.float64)
        if len(delta) != len(positions):
            continue
        target = field.apply(positions + delta)
        glb.write_accessor(document, binary, target_accessor, (target - warped).astype(np.float32))
        transformed += 1
    return {"mesh_index": mesh_index, "vertices": len(positions), "morph_position_accessors": transformed, "strength": strength, "orientation": orientation, "mapping_full_rms": mapping.get("full_rms"), "bounds_before": {"min": positions.min(axis=0).astype(float).tolist(), "max": positions.max(axis=0).astype(float).tolist()}, "bounds_after": {"min": warped.min(axis=0).astype(float).tolist(), "max": warped.max(axis=0).astype(float).tolist()}}


def adjust_hair_depth(document: dict, binary: bytearray, before: dict, after: dict) -> dict:
    before_min = np.asarray(before["min"], dtype=np.float64)
    before_max = np.asarray(before["max"], dtype=np.float64)
    after_min = np.asarray(after["min"], dtype=np.float64)
    after_max = np.asarray(after["max"], dtype=np.float64)
    depth_scale = face2d.clamp(float((after_max[2] - after_min[2]) / max(before_max[2] - before_min[2], 1e-8)), 0.96, 1.04)
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
                center_z = float(np.median(subset[:, 2]))
                subset[:, 2] = center_z + (subset[:, 2] - center_z) * depth_scale
                values[indices] = subset
                glb.write_accessor(document, binary, accessor_index, values.astype(np.float32))
                records.append({"mesh": mesh_index, "primitive": primitive_index, "hair_vertices": len(indices)})
        elif "updo" in name or "hairline" in name:
            for node_index, node in enumerate(document.get("nodes", [])):
                if node.get("mesh") == mesh_index:
                    scale = list(node.get("scale", [1.0, 1.0, 1.0]))
                    node["scale"] = [scale[0], scale[1], scale[2] * depth_scale]
                    records.append({"mesh": mesh_index, "node": node_index, "scale": node["scale"]})
    return {"depth_scale": depth_scale, "records": records}


def process(source: Path, destination: Path, mapping: dict, strength: float, orientation: str, label: str) -> dict:
    document, binary = glb.read_glb(source)
    face = deform_face(document, binary, mapping, strength, orientation)
    hair = adjust_hair_depth(document, binary, face["bounds_before"], face["bounds_after"])
    document.setdefault("asset", {})["generator"] = f"AINA V13 profile-depth landmark field {label}"
    glb.write_glb(destination, document, binary)
    return {"source": source.name, "output": destination.name, "bytes": destination.stat().st_size, "face": face, "hair": hair}


def main() -> None:
    if len(sys.argv) != 9:
        raise SystemExit("build_aina_v13_depth.py FORMAL.vrm BLENDER.glb PROFILE_MAPPING.json OUTPUT_DIR LABEL STEM STRENGTH ORIENTATION")
    formal_source = Path(sys.argv[1])
    safe_source = Path(sys.argv[2])
    mapping_path = Path(sys.argv[3])
    output = Path(sys.argv[4])
    label = sys.argv[5]
    stem = sys.argv[6]
    strength = float(sys.argv[7])
    orientation = sys.argv[8].lower()
    output.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(mapping_path.read_text())
    formal_path = output / f"{stem}.vrm"
    safe_path = output / f"{stem}_BLENDER.glb"
    formal = process(formal_source, formal_path, mapping, strength, orientation, label)
    safe = process(safe_source, safe_path, mapping, strength, orientation, label)
    report = {"version": label, "method": "profile 468-landmark depth field with orientation and strength search", "mapping": str(mapping_path), "strength": strength, "orientation": orientation, "formal": formal, "blender": safe, "preserved": {"vrm_1_0": True, "humanoid_bones": 54, "face_morphs": 57, "expression_presets": 14}, "identity_lock": False, "visual_identity_lock": False, "manual_three_view_review_required": True}
    (output / f"{stem}_BUILD_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
