#!/usr/bin/env python3
"""Direct AINA production-mesh refinement: orbit, midface, jaw, ears and neck.

This script edits the existing locked FaceVerse OBJ in vertex space. It keeps the
same vertex order, faces, connected components and topology so the downstream
52 controls, humanoid rig and VRM bindings remain compatible. No replacement
reference image or new face identity version is generated.

The pass focuses on dense-surface differences that sparse landmarks do not fix:
- youthful eyelid/orbit continuity and softer brow shelf;
- smooth under-eye to apple-cheek transition;
- delicate but visible nose depth;
- integrated, less protruding lip volume;
- rounded chin, soft jaw corners, tucked ears and a clean under-chin/neck line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

K = np.array([
    1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,
    3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,
    6511,5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,
    11325,12452,2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,
    6079,6817,6550,13695,12973,13422,6543,6537,
], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--height", type=float, default=1.72)
    return parser.parse_args(argv)


def read_obj(path: Path):
    lines = path.read_text(errors="ignore").splitlines()
    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            q = line.split()
            vertices.append((float(q[1]), float(q[2]), float(q[3])))
        elif line.startswith("f "):
            ids = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
            for i in range(1, len(ids) - 1):
                faces.append((ids[0], ids[i], ids[i + 1]))
    return lines, np.asarray(vertices, np.float64), np.asarray(faces, np.int64)


def write_obj(lines, vertices: np.ndarray, path: Path) -> None:
    output = []
    index = 0
    for line in lines:
        if line.startswith("v "):
            x, y, z = vertices[index]
            output.append(f"v {x:.9f} {y:.9f} {z:.9f}")
            index += 1
        else:
            output.append(line)
    if index != len(vertices):
        raise RuntimeError(f"OBJ vertex replacement mismatch: {index} != {len(vertices)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def component_roots(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    parent = np.arange(vertex_count, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, c in faces:
        union(int(a), int(b)); union(int(b), int(c)); union(int(c), int(a))
    return np.asarray([find(i) for i in range(vertex_count)], dtype=np.int32)


def map_to_blender(raw: np.ndarray, height: float):
    scale = 1.08
    mapped = np.empty_like(raw)
    mapped[:, 0] = raw[:, 0] * scale
    mapped[:, 1] = raw[:, 2] * scale
    mapped[:, 2] = -raw[:, 1] * scale
    offset = height - float(mapped[:, 2].max())
    mapped[:, 2] += offset
    return mapped, scale, offset


def map_from_blender(mapped: np.ndarray, scale: float, offset: float) -> np.ndarray:
    raw = np.empty_like(mapped)
    raw[:, 0] = mapped[:, 0] / scale
    raw[:, 2] = mapped[:, 1] / scale
    raw[:, 1] = -(mapped[:, 2] - offset) / scale
    return raw


def weights(points: np.ndarray, center, radii, inner=0.0, outer=1.0) -> np.ndarray:
    center = np.asarray(center, np.float64)
    radii = np.asarray(radii, np.float64)
    q = np.sqrt(np.sum(((points - center) / radii) ** 2, axis=1))
    result = np.zeros(len(points), np.float64)
    result[q <= inner] = 1.0
    mask = (q > inner) & (q < outer)
    if np.any(mask):
        t = (q[mask] - inner) / (outer - inner + 1e-12)
        result[mask] = 0.5 * (1.0 + np.cos(np.pi * t))
    return result


def shift(array: np.ndarray, ids: np.ndarray, center, radii, delta, inner=0.0, outer=1.0) -> None:
    points = array[ids].copy()
    points += weights(points, center, radii, inner, outer)[:, None] * np.asarray(delta, np.float64)
    array[ids] = points


def scale_region(array: np.ndarray, ids: np.ndarray, center, radii, factors, inner=0.0, outer=1.0) -> None:
    points = array[ids].copy()
    center = np.asarray(center, np.float64)
    blend = weights(points, center, radii, inner, outer)[:, None]
    target = center + (points - center) * np.asarray(factors, np.float64)
    array[ids] = points + blend * (target - points)


def adjacency(vertex_count: int, faces: np.ndarray, valid: np.ndarray):
    neighbours = [set() for _ in range(vertex_count)]
    valid_mask = np.zeros(vertex_count, dtype=bool)
    valid_mask[valid] = True
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        if not (valid_mask[a] and valid_mask[b] and valid_mask[c]):
            continue
        neighbours[a].update((b, c)); neighbours[b].update((a, c)); neighbours[c].update((a, b))
    return neighbours


def smooth_displacement(displacement: np.ndarray, faces: np.ndarray, head: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    neighbours = adjacency(len(displacement), faces, head)
    original = displacement.copy()
    smoothed = displacement.copy()
    anchor_mask = np.zeros(len(displacement), dtype=bool)
    anchor_mask[anchors] = True
    for _ in range(2):
        prior = smoothed.copy()
        for index in head:
            if anchor_mask[index] or not neighbours[index]:
                continue
            mean = prior[list(neighbours[index])].mean(axis=0)
            smoothed[index] = 0.78 * prior[index] + 0.22 * mean
    smoothed[anchors] = original[anchors]
    return 0.88 * original + 0.12 * smoothed


def main() -> None:
    args = parse_args()
    lines, raw, faces = read_obj(args.mesh)
    if len(raw) <= int(K.max()):
        raise RuntimeError("Locked OBJ does not contain required semantic vertex indices")

    roots = component_roots(len(raw), faces)
    head_root = int(roots[int(K[0])])
    head = np.flatnonzero(roots == head_root)

    base, scale_factor, offset = map_to_blender(raw, args.height)
    out = base.copy()

    # 1. Eye fissure, eyelid and orbit continuity.
    eye_specs = (
        (range(36, 42), 36, 39, (37, 38), (40, 41)),
        (range(42, 48), 45, 42, (43, 44), (46, 47)),
    )
    for eye_range, outer_index, inner_index, upper_ids, lower_ids in eye_specs:
        lm = out[K].copy()
        center = lm[list(eye_range)].mean(axis=0)
        scale_region(out, head, center, (0.044, 0.036, 0.027), (1.10, 1.0, 1.18), 0.10, 1.15)
        shift(out, head, center, (0.045, 0.037, 0.029), (0.0, -0.0012, 0.0003), 0.05, 1.14)

        lm = out[K].copy()
        upper = lm[list(upper_ids)].mean(axis=0)
        lower = lm[list(lower_ids)].mean(axis=0)
        shift(out, head, upper, (0.034, 0.028, 0.015), (0.0, -0.0008, 0.0018), 0.02, 1.08)
        shift(out, head, lower, (0.035, 0.029, 0.015), (0.0, -0.0005, -0.0010), 0.02, 1.08)

        lm = out[K].copy()
        outer = lm[outer_index]
        side = -1.0 if outer[0] < 0 else 1.0
        shift(out, head, outer, (0.018, 0.018, 0.014), (side * 0.0012, -0.0004, 0.0014), 0.02, 1.02)
        shift(out, head, lm[inner_index], (0.015, 0.017, 0.013), (-side * 0.0002, -0.0003, 0.0001), 0.02, 1.02)

        lm = out[K].copy()
        center = lm[list(eye_range)].mean(axis=0)
        shift(out, head, (center[0], center[1] + 0.006, center[2] - 0.013),
              (0.041, 0.038, 0.025), (0.0, -0.0012, 0.0007), 0.0, 1.12)
        shift(out, head, (center[0], center[1] + 0.010, center[2] + 0.020),
              (0.044, 0.039, 0.027), (0.0, 0.0025, -0.0005), 0.0, 1.14)

    # 2. Brow ridge and glabella retreat for a softer youthful forehead.
    lm = out[K].copy()
    for brow_range in (range(17, 22), range(22, 27)):
        center = lm[list(brow_range)].mean(axis=0)
        shift(out, head, center, (0.042, 0.032, 0.025), (0.0, 0.0022, -0.0015), 0.0, 1.12)
    shift(out, head, lm[27], (0.032, 0.032, 0.041), (0.0, 0.0015, 0.0), 0.0, 1.08)

    # 3. Delicate nose with real profile depth; Blender front is negative Y.
    lm = out[K].copy()
    root = lm[27]
    bridge = lm[27:31].mean(axis=0)
    tip = lm[30]
    nose_base = lm[31:36].mean(axis=0)
    scale_region(out, head, root, (0.030, 0.030, 0.048), (0.78, 1.0, 0.84), 0.02, 1.15)
    scale_region(out, head, nose_base, (0.029, 0.027, 0.028), (0.78, 1.0, 0.92), 0.02, 1.12)
    shift(out, head, bridge, (0.024, 0.032, 0.046), (0.0, -0.0015, 0.0006), 0.0, 1.10)
    shift(out, head, tip, (0.020, 0.023, 0.022), (0.0, -0.0035, 0.0024), 0.0, 1.06)
    shift(out, head, nose_base, (0.029, 0.030, 0.030), (0.0, -0.0024, 0.0018), 0.0, 1.10)

    # 4. Integrated lips: flatter perimeter, retained centre volume and shorter philtrum feel.
    lm = out[K].copy()
    mouth = lm[48:60].mean(axis=0)
    scale_region(out, head, mouth, (0.048, 0.035, 0.027), (0.96, 1.0, 0.76), 0.08, 1.16)
    shift(out, head, mouth, (0.051, 0.040, 0.035), (0.0, 0.0038, 0.0022), 0.04, 1.16)
    lm = out[K].copy()
    upper_lip = lm[[49, 50, 51, 52, 53]].mean(axis=0)
    lower_lip = lm[[55, 56, 57, 58, 59]].mean(axis=0)
    shift(out, head, upper_lip, (0.034, 0.024, 0.013), (0.0, -0.0012, 0.0006), 0.02, 1.06)
    shift(out, head, lower_lip, (0.036, 0.025, 0.014), (0.0, -0.0015, -0.0002), 0.02, 1.06)
    for corner_index in (48, 54):
        lm = out[K].copy()
        shift(out, head, lm[corner_index], (0.018, 0.018, 0.013), (0.0, 0.0007, 0.0007), 0.0, 1.02)

    # 5. Apple-cheek support and eye-to-cheek surface continuity.
    lm = out[K].copy()
    cheek_centers = ((lm[40] + lm[31] + lm[48]) / 3.0, (lm[46] + lm[35] + lm[54]) / 3.0)
    for center in cheek_centers:
        shift(out, head, center, (0.045, 0.041, 0.040), (0.0, -0.0020, 0.0010), 0.02, 1.12)
        scale_region(out, head, center, (0.047, 0.043, 0.042), (0.98, 1.0, 1.03), 0.0, 1.10)

    # 6. Soft V lower third, rounded chin, tucked ears and clean neck transition.
    lm = out[K].copy()
    mouth_z = float(lm[48:60, 2].mean())
    chin_z = float(lm[8, 2])
    points = out[head].copy()
    upper = mouth_z + 0.020
    lower = chin_z - 0.010
    t = np.clip((upper - points[:, 2]) / max(upper - lower, 1e-6), 0.0, 1.0)
    front = np.exp(-0.5 * ((points[:, 1] + 0.002) / 0.060) ** 4)
    side_gate = np.clip((np.abs(points[:, 0]) - 0.018) / 0.055, 0.0, 1.0)
    points[:, 0] *= 1.0 - 0.10 * (t ** 1.25) * front * (0.25 + 0.75 * side_gate)
    out[head] = points

    lm = out[K].copy()
    chin = lm[8]
    scale_region(out, head, chin, (0.045, 0.047, 0.043), (0.82, 0.93, 0.90), 0.02, 1.10)
    shift(out, head, chin, (0.047, 0.047, 0.043), (0.0, 0.0016, 0.0030), 0.0, 1.05)

    lm = out[K].copy()
    for jaw_index in (4, 12):
        center = lm[jaw_index]
        side = -1.0 if center[0] < 0 else 1.0
        shift(out, head, center, (0.037, 0.046, 0.046), (-side * 0.0022, 0.0012, 0.0008), 0.0, 1.08)

    lm = out[K].copy()
    for ear_index in (0, 16):
        center = lm[ear_index]
        side = -1.0 if center[0] < 0 else 1.0
        scale_region(out, head, center, (0.036, 0.045, 0.058), (0.76, 0.82, 0.76), 0.0, 1.11)
        shift(out, head, center, (0.036, 0.045, 0.058), (-side * 0.0045, 0.0025, 0.0), 0.0, 1.05)

    lm = out[K].copy()
    chin = lm[8]
    under_chin = (chin[0], chin[1] + 0.020, chin[2] - 0.013)
    scale_region(out, head, under_chin, (0.060, 0.063, 0.036), (0.88, 0.95, 1.0), 0.0, 1.08)
    shift(out, head, under_chin, (0.060, 0.063, 0.036), (0.0, 0.0018, -0.0005), 0.0, 1.05)

    displacement = out - base
    displacement = smooth_displacement(displacement, faces, head, K)
    out = base + displacement

    refined_raw = map_from_blender(out, scale_factor, offset)
    write_obj(lines, refined_raw, args.out)

    head_delta = displacement[head]
    magnitudes = np.linalg.norm(head_delta, axis=1)
    report = {
        "product": "AINA direct production mesh refinement",
        "source": str(args.mesh),
        "output": str(args.out),
        "topology_changed": False,
        "vertex_count": int(len(raw)),
        "face_count": int(len(faces)),
        "head_vertex_count": int(len(head)),
        "max_head_displacement_m": float(magnitudes.max()),
        "rms_head_displacement_m": float(np.sqrt(np.mean(magnitudes * magnitudes))),
        "p95_head_displacement_m": float(np.percentile(magnitudes, 95)),
        "areas": ["eyelids_orbits", "brow_glabella", "nose", "lips", "apple_cheeks", "jaw_chin_ears_neck"],
        "identity_lock": False,
        "visual_gate": "actual Blender neutral front + 3Q + expression previews",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
