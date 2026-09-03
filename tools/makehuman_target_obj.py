#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def read_obj_vertices(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    vertices: list[list[float]] = []
    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise RuntimeError(f"No vertices found in {path}")
    return lines, vertices


def apply_target(vertices: list[list[float]], target_path: Path, weight: float) -> int:
    changed = 0
    with target_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            index = int(parts[0])
            if not 0 <= index < len(vertices):
                raise IndexError(f"Target index {index} outside 0..{len(vertices)-1}")
            vertices[index][0] += float(parts[1]) * weight
            vertices[index][1] += float(parts[2]) * weight
            vertices[index][2] += float(parts[3]) * weight
            changed += 1
    return changed


def write_obj(lines: Iterable[str], vertices: list[list[float]], output: Path) -> None:
    vertex_index = 0
    rendered: list[str] = []
    for line in lines:
        if line.startswith("v "):
            x, y, z = vertices[vertex_index]
            rendered.append(f"v {x:.9f} {y:.9f} {z:.9f}")
            vertex_index += 1
        else:
            rendered.append(line)
    output.write_text("\n".join(rendered) + "\n", encoding="utf-8")


def bounds(vertices: list[list[float]]) -> dict[str, object]:
    axes = list(zip(*vertices))
    mins = [min(axis) for axis in axes]
    maxs = [max(axis) for axis in axes]
    spans = [hi - lo for lo, hi in zip(mins, maxs)]
    vertical_axis = max(range(3), key=lambda idx: spans[idx])
    return {
        "min": mins,
        "max": maxs,
        "span": spans,
        "largest_axis": "XYZ"[vertical_axis],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--target", action="append", default=[], help="PATH:WEIGHT")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    lines, vertices = read_obj_vertices(args.base)
    original = [row[:] for row in vertices]
    applied: list[dict[str, object]] = []
    for item in args.target:
        target_text, weight_text = item.rsplit(":", 1)
        target = Path(target_text)
        weight = float(weight_text)
        changed = apply_target(vertices, target, weight)
        applied.append({"path": str(target), "weight": weight, "changed_vertices": changed})

    write_obj(lines, vertices, args.output)
    max_delta = max(
        ((vertices[i][0] - original[i][0]) ** 2 + (vertices[i][1] - original[i][1]) ** 2 + (vertices[i][2] - original[i][2]) ** 2) ** 0.5
        for i in range(len(vertices))
    )
    report = {
        "base": str(args.base),
        "output": str(args.output),
        "vertex_count": len(vertices),
        "face_line_count": sum(1 for line in lines if line.startswith("f ")),
        "group_lines": [line for line in lines if line.startswith(("g ", "o "))][:80],
        "base_bounds": bounds(original),
        "output_bounds": bounds(vertices),
        "targets": applied,
        "max_vertex_delta": max_delta,
        "license_basis": "MakeHuman default basemesh and targets are CC0 per LICENSE.ASSETS.md",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
