#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def smoothstep(value: float) -> float:
    value = clamp(value)
    return value * value * (3.0 - 2.0 * value)


def gaussian(*terms: float) -> float:
    return math.exp(-0.5 * sum(term * term for term in terms))


def face_vertex_index(token: str, vertex_count: int) -> int:
    raw = int(token.split("/", 1)[0])
    return raw - 1 if raw > 0 else vertex_count + raw


def read_obj(path: Path) -> tuple[list[str], list[list[float]], dict[str, set[int]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    vertices: list[list[float]] = []
    for line in lines:
        if line.startswith("v "):
            fields = line.split()
            vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if not vertices:
        raise RuntimeError(f"No OBJ vertices in {path}")

    groups: dict[str, set[int]] = {}
    group = "__ungrouped__"
    for line in lines:
        if line.startswith("g "):
            group = line[2:].strip() or "__ungrouped__"
            groups.setdefault(group, set())
        elif line.startswith("f "):
            target = groups.setdefault(group, set())
            for token in line.split()[1:]:
                target.add(face_vertex_index(token, len(vertices)))
    return lines, vertices, groups


def write_obj(lines: list[str], vertices: list[list[float]], output: Path) -> None:
    rendered: list[str] = []
    cursor = 0
    for line in lines:
        if line.startswith("v "):
            x, y, z = vertices[cursor]
            rendered.append(f"v {x:.9f} {y:.9f} {z:.9f}")
            cursor += 1
        else:
            rendered.append(line)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rendered) + "\n", encoding="utf-8")


def group_center(vertices: list[list[float]], indices: set[int]) -> list[float]:
    if not indices:
        raise RuntimeError("Cannot calculate empty group center")
    return [
        0.5 * (min(vertices[index][axis] for index in indices) + max(vertices[index][axis] for index in indices))
        for axis in range(3)
    ]


def group_span(vertices: list[list[float]], indices: set[int]) -> list[float]:
    return [
        max(vertices[index][axis] for index in indices) - min(vertices[index][axis] for index in indices)
        for axis in range(3)
    ]


def body_metrics(vertices: list[list[float]], groups: dict[str, set[int]]) -> dict[str, object]:
    body = groups["body"]
    eye_groups = [groups["helper-l-eye"], groups["helper-r-eye"]]
    eye_centers = sorted((group_center(vertices, indices) for indices in eye_groups), key=lambda point: point[0])
    eye_spans = [group_span(vertices, indices) for indices in eye_groups]
    eye_y = sum(point[1] for point in eye_centers) * 0.5
    face_band = [
        vertices[index]
        for index in body
        if abs(vertices[index][1] - eye_y) < 0.16 and vertices[index][2] > 0.15
    ]
    head = [vertices[index] for index in body if vertices[index][1] > 4.70 and abs(vertices[index][0]) < 1.05]
    nose_tip = max(
        (vertices[index] for index in body if abs(vertices[index][0]) < 0.16 and eye_y - 0.72 < vertices[index][1] < eye_y - 0.18),
        key=lambda point: point[2],
    )
    return {
        "eye_centers": eye_centers,
        "eye_spans": eye_spans,
        "eye_center_distance": eye_centers[1][0] - eye_centers[0][0],
        "face_width_at_eye_band": max(point[0] for point in face_band) - min(point[0] for point in face_band),
        "head_bounds": {
            "min": [min(point[axis] for point in head) for axis in range(3)],
            "max": [max(point[axis] for point in head) for axis in range(3)],
        },
        "nose_tip": nose_tip,
    }


def sculpt(
    vertices: list[list[float]],
    groups: dict[str, set[int]],
    settings: dict[str, float],
) -> None:
    body = groups["body"]
    left_eye = groups["helper-l-eye"]
    right_eye = groups["helper-r-eye"]
    initial_eye_centers = sorted(
        (group_center(vertices, left_eye), group_center(vertices, right_eye)),
        key=lambda point: point[0],
    )
    eye_y = 0.5 * (initial_eye_centers[0][1] + initial_eye_centers[1][1])
    lower_anchor = eye_y - 0.05

    # AINA's lower third is intentionally shorter than a generic adult head.
    lower_scale = settings["lower_face_scale"]
    for index, point in enumerate(vertices):
        x, y, z = point
        if y >= lower_anchor or y <= 4.62 or abs(x) >= 1.02 or z <= -0.45:
            continue
        front_weight = 0.35 + 0.65 * smoothstep((z + 0.25) / 1.25)
        vertical_weight = smoothstep((lower_anchor - y) / 0.95)
        blend = front_weight * vertical_weight
        target_y = lower_anchor + (y - lower_anchor) * lower_scale
        point[1] = y + (target_y - y) * blend

    # Soft V jaw with a small round chin, while keeping the cheek width.
    jaw_amount = settings["jaw_taper"]
    for index in body:
        x, y, z = vertices[index]
        if not (4.66 < y < eye_y - 0.18 and abs(x) < 0.92 and z > -0.18):
            continue
        lower = smoothstep((eye_y - 0.18 - y) / 0.96)
        side = smoothstep(abs(x) / 0.72)
        front = 0.55 + 0.45 * smoothstep((z + 0.05) / 1.05)
        vertices[index][0] = x * (1.0 - jaw_amount * lower * (0.45 + 0.55 * side) * front)

    chin_y = eye_y - 0.91
    for index in body:
        x, y, z = vertices[index]
        weight = gaussian(x / 0.34, (y - chin_y) / 0.20) * smoothstep((z + 0.05) / 0.95)
        vertices[index][2] += settings["chin_forward"] * weight
        vertices[index][1] += settings["chin_up"] * weight

    # High, soft apple cheeks.
    for sign in (-1.0, 1.0):
        cheek_x = sign * 0.43
        cheek_y = eye_y - 0.30
        cheek_z = 0.91
        for index in body:
            x, y, z = vertices[index]
            weight = gaussian((x - cheek_x) / 0.28, (y - cheek_y) / 0.28, (z - cheek_z) / 0.38)
            vertices[index][0] += sign * settings["cheek_width"] * weight
            vertices[index][2] += settings["cheek_forward"] * weight
            vertices[index][1] += settings["cheek_up"] * weight

    # Narrow the bridge and alae without forcing the nose farther forward.
    nose_center_y = eye_y - 0.38
    for index in body:
        x, y, z = vertices[index]
        if abs(x) > 0.38 or not (eye_y - 0.78 < y < eye_y - 0.10) or z < 0.68:
            continue
        front = smoothstep((z - 0.68) / 0.66)
        bridge = gaussian(x / 0.23, (y - nose_center_y) / 0.34) * front
        alar = gaussian(x / 0.28, (y - (eye_y - 0.50)) / 0.18) * front
        local_scale = 1.0 - (1.0 - settings["nose_width_scale"]) * bridge
        local_scale *= 1.0 - (1.0 - settings["alar_width_scale"]) * alar
        vertices[index][0] = x * local_scale
        ridge = gaussian(x / 0.11, (y - (eye_y - 0.26)) / 0.28) * front
        vertices[index][2] += settings["bridge_forward"] * ridge

    nose_candidates = [
        index
        for index in body
        if abs(vertices[index][0]) < 0.15 and eye_y - 0.72 < vertices[index][1] < eye_y - 0.22
    ]
    nose_tip_index = max(nose_candidates, key=lambda index: vertices[index][2])
    tip = vertices[nose_tip_index][:]
    for index in body:
        x, y, z = vertices[index]
        weight = gaussian((x - tip[0]) / 0.20, (y - tip[1]) / 0.18, (z - tip[2]) / 0.22)
        vertices[index][1] += settings["nose_tip_up"] * weight
        vertices[index][2] += settings["nose_tip_forward"] * weight

    # A natural M-shaped mouth: slightly fuller, not the protruding 'pout' from extreme sliders.
    mouth_y = eye_y - 0.625
    for index in body:
        x, y, z = vertices[index]
        if abs(x) > 0.54 or abs(y - mouth_y) > 0.19 or z < 0.78:
            continue
        weight = gaussian(x / 0.42, (y - mouth_y) / 0.12) * smoothstep((z - 0.78) / 0.55)
        vertices[index][0] += x * (settings["mouth_width_scale"] - 1.0) * weight
        vertices[index][1] += (y - mouth_y) * (settings["lip_height_scale"] - 1.0) * weight
        vertices[index][1] += settings["mouth_up"] * weight
        vertices[index][2] += settings["lip_forward"] * weight

    for group_name in ("helper-upper-teeth", "helper-lower-teeth", "helper-tongue"):
        for index in groups.get(group_name, set()):
            vertices[index][0] *= settings["mouth_width_scale"]
            vertices[index][1] += settings["mouth_up"]

    # Enlarge the eye opening and eyeball together, preserve one-eye spacing, and lift the outer corner.
    eye_group_names = {
        -1.0: [
            "helper-l-eye",
            "helper-l-eyelashes-1",
            "helper-l-eyelashes-2",
        ],
        1.0: [
            "helper-r-eye",
            "helper-r-eyelashes-1",
            "helper-r-eyelashes-2",
        ],
    }
    current_centers = sorted(
        (group_center(vertices, left_eye), group_center(vertices, right_eye)),
        key=lambda point: point[0],
    )
    centers_by_sign = {-1.0: current_centers[0], 1.0: current_centers[1]}
    for sign, names in eye_group_names.items():
        center = centers_by_sign[sign]
        shift_x = sign * settings["eye_outward"]
        eye_indices: set[int] = set()
        for name in names:
            eye_indices.update(groups.get(name, set()))
        for index in eye_indices:
            x, y, z = vertices[index]
            vertices[index][0] = center[0] + (x - center[0]) * settings["eye_x_scale"] + shift_x
            vertices[index][1] = center[1] + (y - center[1]) * settings["eye_y_scale"]
            vertices[index][2] = center[2] + (z - center[2]) * settings["eye_depth_scale"]

        for index in body:
            x, y, z = vertices[index]
            dx = (x - center[0]) / 0.39
            dy = (y - center[1]) / 0.255
            front = smoothstep((z - 0.48) / 0.72)
            weight = gaussian(dx, dy) * front
            if weight < 0.002:
                continue
            vertices[index][0] += ((x - center[0]) * (settings["eye_x_scale"] - 1.0) + shift_x) * weight
            vertices[index][1] += (y - center[1]) * (settings["eye_y_scale"] - 1.0) * weight
            outward = clamp(sign * (x - center[0]) / 0.30)
            vertices[index][1] += settings["outer_corner_up"] * outward * weight


def settings_grid() -> dict[str, dict[str, float]]:
    common = {
        "eye_depth_scale": 1.03,
        "bridge_forward": 0.010,
        "nose_tip_forward": 0.004,
        "chin_up": 0.006,
        "cheek_up": 0.008,
    }
    return {
        "10_CUSTOM_BALANCED": {
            **common,
            "eye_x_scale": 1.22,
            "eye_y_scale": 1.13,
            "eye_outward": 0.040,
            "outer_corner_up": 0.018,
            "lower_face_scale": 0.945,
            "jaw_taper": 0.085,
            "chin_forward": 0.014,
            "cheek_width": 0.030,
            "cheek_forward": 0.022,
            "nose_width_scale": 0.86,
            "alar_width_scale": 0.89,
            "nose_tip_up": 0.020,
            "mouth_width_scale": 1.00,
            "lip_height_scale": 1.10,
            "mouth_up": 0.015,
            "lip_forward": 0.010,
        },
        "11_CUSTOM_AINA": {
            **common,
            "eye_x_scale": 1.30,
            "eye_y_scale": 1.17,
            "eye_outward": 0.058,
            "outer_corner_up": 0.026,
            "lower_face_scale": 0.920,
            "jaw_taper": 0.115,
            "chin_forward": 0.022,
            "cheek_width": 0.044,
            "cheek_forward": 0.032,
            "nose_width_scale": 0.80,
            "alar_width_scale": 0.83,
            "nose_tip_up": 0.030,
            "mouth_width_scale": 1.01,
            "lip_height_scale": 1.15,
            "mouth_up": 0.022,
            "lip_forward": 0.014,
        },
        "12_CUSTOM_WIDE_EYE": {
            **common,
            "eye_x_scale": 1.38,
            "eye_y_scale": 1.14,
            "eye_outward": 0.075,
            "outer_corner_up": 0.032,
            "lower_face_scale": 0.910,
            "jaw_taper": 0.125,
            "chin_forward": 0.024,
            "cheek_width": 0.050,
            "cheek_forward": 0.034,
            "nose_width_scale": 0.78,
            "alar_width_scale": 0.81,
            "nose_tip_up": 0.032,
            "mouth_width_scale": 1.01,
            "lip_height_scale": 1.14,
            "mouth_up": 0.024,
            "lip_forward": 0.013,
        },
        "13_CUSTOM_SOFT_JAW": {
            **common,
            "eye_x_scale": 1.30,
            "eye_y_scale": 1.17,
            "eye_outward": 0.058,
            "outer_corner_up": 0.025,
            "lower_face_scale": 0.930,
            "jaw_taper": 0.070,
            "chin_forward": 0.019,
            "cheek_width": 0.060,
            "cheek_forward": 0.040,
            "nose_width_scale": 0.82,
            "alar_width_scale": 0.84,
            "nose_tip_up": 0.027,
            "mouth_width_scale": 1.01,
            "lip_height_scale": 1.13,
            "mouth_up": 0.020,
            "lip_forward": 0.012,
        },
        "14_CUSTOM_SHORT_FACE": {
            **common,
            "eye_x_scale": 1.29,
            "eye_y_scale": 1.18,
            "eye_outward": 0.056,
            "outer_corner_up": 0.026,
            "lower_face_scale": 0.875,
            "jaw_taper": 0.105,
            "chin_forward": 0.025,
            "cheek_width": 0.046,
            "cheek_forward": 0.034,
            "nose_width_scale": 0.80,
            "alar_width_scale": 0.83,
            "nose_tip_up": 0.033,
            "mouth_width_scale": 1.01,
            "lip_height_scale": 1.16,
            "mouth_up": 0.028,
            "lip_forward": 0.014,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    lines, source_vertices, groups = read_obj(args.input)
    required = {
        "body",
        "helper-l-eye",
        "helper-r-eye",
        "helper-upper-teeth",
        "helper-lower-teeth",
        "helper-tongue",
    }
    missing = sorted(required.difference(groups))
    if missing:
        raise RuntimeError(f"Missing required OBJ groups: {missing}")

    args.out.mkdir(parents=True, exist_ok=True)
    source_label = "09_SOFT_SOURCE"
    write_obj(lines, source_vertices, args.out / f"{source_label}.obj")
    manifest: dict[str, object] = {
        "input": str(args.input),
        "source_metrics": body_metrics(source_vertices, groups),
        "variants": {},
    }
    for label, settings in settings_grid().items():
        vertices = [point[:] for point in source_vertices]
        sculpt(vertices, groups, settings)
        output = args.out / f"{label}.obj"
        write_obj(lines, vertices, output)
        report = {
            "label": label,
            "output": output.name,
            "settings": settings,
            "metrics": body_metrics(vertices, groups),
            "vertex_count": len(vertices),
        }
        (args.out / f"{label}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        manifest["variants"][label] = report
        print(f"sculpted {label}")

    (args.out / "AINA_CUSTOM_SCULPT_VARIANTS.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"variants": list(manifest["variants"])}, indent=2))


if __name__ == "__main__":
    main()
