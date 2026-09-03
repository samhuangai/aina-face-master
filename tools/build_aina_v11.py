#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_aina_v10 as base


def empty_indices() -> np.ndarray:
    return np.array([], dtype=np.int64)


class IncrementalWarp:
    def __init__(self, positions: np.ndarray, roles: dict[str, np.ndarray], parameters: dict[str, float]) -> None:
        self.base = np.asarray(positions, dtype=np.float64)
        self.roles = roles
        self.p = {key: float(value) for key, value in parameters.items()}
        self.all = np.unique(np.concatenate([value for value in roles.values() if len(value)]))
        face = self.base[self.all]
        self.center_x = float(np.median(face[:, 0]))
        self.y_min = float(face[:, 1].min())
        self.y_max = float(face[:, 1].max())
        self.height = self.y_max - self.y_min
        self.half_width = max(abs(float(face[:, 0].min()) - self.center_x), abs(float(face[:, 0].max()) - self.center_x))
        whites = roles.get("eye_white", empty_indices())
        self.eye_centers = {}
        for sign in (-1, 1):
            subset = whites[(self.base[whites, 0] - self.center_x) * sign > 0]
            if len(subset):
                self.eye_centers[sign] = (float(self.base[subset, 0].mean()), float(self.base[subset, 1].mean()))
        mouth = roles.get("mouth", empty_indices())
        self.mouth_center = (float(self.base[mouth, 0].mean()), float(self.base[mouth, 1].mean())) if len(mouth) else (self.center_x, self.y_min + self.height * 0.18)
        self.eye_y = float(np.mean([value[1] for value in self.eye_centers.values()])) if self.eye_centers else self.y_min + self.height * 0.45
        self.nose_tip_y = self.y_min + self.height * 0.245

    def apply(self, source: np.ndarray) -> np.ndarray:
        result = np.asarray(source, dtype=np.float64).copy()
        x, y, z = result[:, 0], result[:, 1], result[:, 2]
        ids = self.all

        face_width_scale = self.p.get("face_width_scale", 1.0)
        x[ids] = self.center_x + (x[ids] - self.center_x) * face_width_scale

        upper_scale = self.p.get("upper_face_scale", 1.0)
        upper = ids[y[ids] > self.eye_y]
        y[upper] = self.eye_y + (y[upper] - self.eye_y) * upper_scale

        lower_scale = self.p.get("lower_third_scale", 1.0)
        lower = ids[y[ids] < self.mouth_center[1]]
        y[lower] = self.mouth_center[1] + (y[lower] - self.mouth_center[1]) * lower_scale

        eye_union = np.unique(np.concatenate([
            self.roles.get("eye_white", empty_indices()),
            self.roles.get("iris", empty_indices()),
            self.roles.get("highlight", empty_indices()),
            self.roles.get("eyeline", empty_indices()),
        ]))
        eye_width_scale = self.p.get("eye_width_scale", 1.0)
        eye_height_scale = self.p.get("eye_height_scale", 1.0)
        eye_spacing_scale = self.p.get("eye_spacing_scale", 1.0)
        for sign, (cx, cy) in self.eye_centers.items():
            side = eye_union[(self.base[eye_union, 0] - self.center_x) * sign > 0]
            if not len(side):
                continue
            desired_cx = self.center_x + (cx - self.center_x) * eye_spacing_scale * face_width_scale
            x[side] = desired_cx + (x[side] - (self.center_x + (cx - self.center_x) * face_width_scale)) * eye_width_scale
            y[side] = cy + (y[side] - cy) * eye_height_scale

        brows = self.roles.get("brow", empty_indices())
        brow_gap = self.p.get("brow_eye_gap_scale", 1.0)
        for sign, (cx, cy) in self.eye_centers.items():
            subset = brows[(self.base[brows, 0] - self.center_x) * sign > 0]
            if len(subset):
                desired_cx = self.center_x + (cx - self.center_x) * eye_spacing_scale * face_width_scale
                current_scaled_cx = self.center_x + (cx - self.center_x) * face_width_scale
                x[subset] += desired_cx - current_scaled_cx
                y[subset] = cy + (y[subset] - cy) * brow_gap

        skin = self.roles.get("skin", empty_indices())
        if len(skin):
            sx = (x[skin] - self.center_x) / max(self.half_width * face_width_scale, 1e-8)
            sy = (y[skin] - self.y_min) / max(self.height, 1e-8)
            nose_length_delta = self.p.get("nose_length_scale", 1.0) - 1.0
            tip = base.gaussian(sx, 0.0, 0.17) * base.gaussian(sy, 0.245, 0.065)
            bridge = base.gaussian(sx, 0.0, 0.14) * base.gaussian(sy, 0.34, 0.16)
            y[skin] -= 0.020 * nose_length_delta * (0.30 * bridge + tip)
            nose_width_scale = self.p.get("nose_width_scale", 1.0)
            alae = base.gaussian(np.abs(sx), 0.16, 0.09) * base.gaussian(sy, 0.235, 0.060)
            x[skin] = self.center_x + (x[skin] - self.center_x) * (1.0 + (nose_width_scale - 1.0) * alae)

        mouth = self.roles.get("mouth", empty_indices())
        if len(mouth):
            mx, my = self.mouth_center
            mouth_width = self.p.get("mouth_width_scale", 1.0)
            x[mouth] = self.center_x + ((x[mouth] - self.center_x) * mouth_width)
            nose_mouth = self.p.get("nose_mouth_scale", 1.0)
            desired_my = self.nose_tip_y + (my - self.nose_tip_y) * nose_mouth
            y[mouth] += desired_my - my

        result[:, 0], result[:, 1], result[:, 2] = x, y, z
        return result


def deform_face(document: dict, binary: bytearray, parameters: dict[str, float]) -> dict:
    mesh_index = base.find_mesh_index(document, "face")
    mesh = document["meshes"][mesh_index]
    position_accessors = {primitive["attributes"]["POSITION"] for primitive in mesh["primitives"]}
    if len(position_accessors) != 1:
        raise ValueError(f"Expected one shared face POSITION accessor, got {position_accessors}")
    position_accessor = next(iter(position_accessors))
    positions = base.accessor_array(document, binary, position_accessor).astype(np.float64)
    roles = base.primitive_vertex_sets(document, binary, mesh_index)
    warp = IncrementalWarp(positions, roles, parameters)
    warped = warp.apply(positions)
    base.write_accessor(document, binary, position_accessor, warped.astype(np.float32))
    target_accessors = set()
    for primitive in mesh["primitives"]:
        for target in primitive.get("targets", []):
            if "POSITION" in target:
                target_accessors.add(target["POSITION"])
    transformed = 0
    for accessor_index in sorted(target_accessors):
        delta = base.accessor_array(document, binary, accessor_index).astype(np.float64)
        if len(delta) != len(positions):
            continue
        target = warp.apply(positions + delta)
        base.write_accessor(document, binary, accessor_index, (target - warped).astype(np.float32))
        transformed += 1
    return {
        "mesh_index": mesh_index,
        "vertices": len(positions),
        "morph_position_accessors": transformed,
        "bounds_before": {"min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
        "bounds_after": {"min": warped.min(axis=0).tolist(), "max": warped.max(axis=0).tolist()},
    }


def adjust_hair(document: dict, binary: bytearray, parameters: dict[str, float]) -> dict:
    face_scale = float(parameters.get("face_width_scale", 1.0))
    upper_scale = float(parameters.get("upper_face_scale", 1.0))
    records = []
    processed = set()
    for mesh_index, mesh in enumerate(document.get("meshes", [])):
        lower_name = mesh.get("name", "").lower()
        if "body" in lower_name:
            for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
                if "hair" not in base.material_name(document, primitive).lower():
                    continue
                accessor_index = primitive["attributes"]["POSITION"]
                if accessor_index in processed:
                    continue
                processed.add(accessor_index)
                values = base.accessor_array(document, binary, accessor_index).astype(np.float64)
                indices = np.unique(base.accessor_array(document, binary, primitive["indices"]).reshape(-1).astype(np.int64))
                subset = values[indices].copy()
                cx = float(np.median(subset[:, 0]))
                y_anchor = float(np.quantile(subset[:, 1], 0.45))
                subset[:, 0] = cx + (subset[:, 0] - cx) * face_scale
                upper = subset[:, 1] > y_anchor
                subset[upper, 1] = y_anchor + (subset[upper, 1] - y_anchor) * upper_scale
                values[indices] = subset
                base.write_accessor(document, binary, accessor_index, values.astype(np.float32))
                records.append({"mesh": mesh_index, "primitive": primitive_index, "vertices": len(indices)})
        elif "updo" in lower_name or "hairline" in lower_name:
            node_indices = [index for index, node in enumerate(document.get("nodes", [])) if node.get("mesh") == mesh_index]
            for node_index in node_indices:
                node = document["nodes"][node_index]
                scale = list(node.get("scale", [1.0, 1.0, 1.0]))
                node["scale"] = [scale[0] * face_scale, scale[1] * upper_scale, scale[2]]
                records.append({"mesh": mesh_index, "node": node_index, "node_scale": node["scale"]})
    return {"records": records}


def process(source: Path, destination: Path, parameters: dict[str, float], label: str) -> dict:
    document, binary = base.read_glb(source)
    face = deform_face(document, binary, parameters)
    hair = adjust_hair(document, binary, parameters)
    document.setdefault("asset", {})["generator"] = f"AINA V11 closed-loop landmark refinement {label}"
    base.write_glb(destination, document, binary)
    return {"source": source.name, "output": destination.name, "bytes": destination.stat().st_size, "face": face, "hair": hair}


def main() -> None:
    if len(sys.argv) != 7:
        raise SystemExit("build_aina_v11.py FORMAL.vrm BLENDER.glb PARAMS.json OUTPUT_DIR VERSION_LABEL OUTPUT_STEM")
    formal_source = Path(sys.argv[1])
    blender_source = Path(sys.argv[2])
    parameters_report = json.loads(Path(sys.argv[3]).read_text())
    parameters = parameters_report.get("parameters", parameters_report)
    out = Path(sys.argv[4])
    label = sys.argv[5]
    stem = sys.argv[6]
    out.mkdir(parents=True, exist_ok=True)
    formal_path = out / f"{stem}.vrm"
    safe_path = out / f"{stem}_BLENDER.glb"
    formal = process(formal_source, formal_path, parameters, label)
    safe = process(blender_source, safe_path, parameters, label)
    report = {
        "version": label,
        "source_measurement": str(sys.argv[3]),
        "parameters": parameters,
        "formal": formal,
        "blender": safe,
        "identity_lock": False,
        "visual_identity_lock": False,
    }
    (out / f"{stem}_BUILD_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
