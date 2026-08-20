#!/usr/bin/env python3
"""AINA direct real-mesh continuity pass.

Edits the existing locked FaceVerse OBJ in vertex space only.  No reference art,
new identity or replacement render is generated.  Vertex order, triangle order
and topology are preserved so the production 52-shape/VRM pipeline can continue
using the same semantic indices.

Focus:
- connect the enlarged real eyes to the surrounding eyelids/orbits;
- soften under-eye and upper-cheek transitions;
- refine cheek-to-jaw continuity without creating a pinched V;
- tuck jaw-angle/ear-side mass and clean the lower head/neck silhouette;
- round the chin while preserving the approved facial proportions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

K = np.array([
    1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,
    10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,
    10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,
    4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,
    6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,
    6817,6550,13695,12973,13422,6543,6537,
], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def map_to_blender(v: np.ndarray, height: float = 1.72) -> tuple[np.ndarray, float]:
    scale = 1.08
    out = np.empty_like(v, dtype=np.float64)
    out[:, 0] = v[:, 0] * scale
    out[:, 1] = v[:, 2] * scale
    out[:, 2] = -v[:, 1] * scale
    offset = height - float(out[:, 2].max())
    out[:, 2] += offset
    return out, offset


def map_from_blender(v: np.ndarray, offset: float) -> np.ndarray:
    scale = 1.08
    out = np.empty_like(v, dtype=np.float64)
    out[:, 0] = v[:, 0] / scale
    out[:, 2] = v[:, 1] / scale
    out[:, 1] = -(v[:, 2] - offset) / scale
    return out


def ellipsoid_weight(points: np.ndarray, center, radii, inner=0.0, outer=1.0) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    q = np.sqrt(np.sum(((points - c) / np.maximum(r, 1e-8)) ** 2, axis=1))
    w = np.zeros(len(points), dtype=np.float64)
    w[q <= inner] = 1.0
    mask = (q > inner) & (q < outer)
    if np.any(mask):
        t = (q[mask] - inner) / max(outer - inner, 1e-8)
        w[mask] = 0.5 * (1.0 + np.cos(np.pi * t))
    return w


def add_shift(delta: np.ndarray, points: np.ndarray, ids: np.ndarray, center, radii, shift,
              inner=0.0, outer=1.0, gate: np.ndarray | None = None) -> None:
    ids = np.asarray(ids, dtype=np.int64)
    w = ellipsoid_weight(points[ids], center, radii, inner, outer)
    if gate is not None:
        w *= gate[ids]
    delta[ids] += w[:, None] * np.asarray(shift, dtype=np.float64)


def add_scale(delta: np.ndarray, points: np.ndarray, ids: np.ndarray, center, radii, factors,
              inner=0.0, outer=1.0, gate: np.ndarray | None = None) -> None:
    ids = np.asarray(ids, dtype=np.int64)
    c = np.asarray(center, dtype=np.float64)
    p = points[ids]
    w = ellipsoid_weight(p, c, radii, inner, outer)
    if gate is not None:
        w *= gate[ids]
    target = c + (p - c) * np.asarray(factors, dtype=np.float64)
    delta[ids] += w[:, None] * (target - p)


def components(n: int, faces: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, c in faces:
        union(int(a), int(b)); union(int(b), int(c)); union(int(c), int(a))
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    groups: dict[int, list[int]] = {}
    for i, root in enumerate(roots):
        groups.setdefault(int(root), []).append(i)
    return roots, {root: np.asarray(ids, dtype=np.int64) for root, ids in groups.items()}


def adjacency(n: int, faces: np.ndarray, allowed: np.ndarray) -> list[list[int]]:
    allow = np.zeros(n, dtype=bool)
    allow[allowed] = True
    nbr: list[set[int]] = [set() for _ in range(n)]
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        if allow[a] and allow[b]: nbr[a].add(b); nbr[b].add(a)
        if allow[b] and allow[c]: nbr[b].add(c); nbr[c].add(b)
        if allow[c] and allow[a]: nbr[c].add(a); nbr[a].add(c)
    return [list(x) for x in nbr]


def smooth_delta(delta: np.ndarray, nbr: list[list[int]], ids: np.ndarray,
                 iterations: int = 3, alpha: float = 0.24) -> np.ndarray:
    out = delta.copy()
    ids = np.asarray(ids, dtype=np.int64)
    for _ in range(iterations):
        old = out.copy()
        for i in ids:
            ns = nbr[int(i)]
            if ns:
                out[i] = (1.0 - alpha) * old[i] + alpha * old[ns].mean(axis=0)
    return 0.82 * delta + 0.18 * out


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# AINA real-mesh orbit/jaw/neck continuity refinement\n")
        for x, y, z in vertices:
            f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        for a, b, c in faces:
            f.write(f"f {int(a)+1} {int(b)+1} {int(c)+1}\n")


def main() -> None:
    args = parse_args()
    mesh = trimesh.load(args.mesh, process=False, maintain_order=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Expected a single triangulated AINA OBJ")

    raw0 = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if int(K.max()) >= len(raw0):
        raise RuntimeError("AINA semantic indices are outside this mesh")

    points, offset = map_to_blender(raw0)
    original = points.copy()
    delta = np.zeros_like(points)
    roots, groups = components(len(points), faces)
    head = max(groups.values(), key=len)
    head_mask = np.zeros(len(points), dtype=bool)
    head_mask[head] = True
    lm = points[K].copy()

    # Avoid moving rear skull/interior components.  Blender front is -Y.
    front_gate = np.clip((0.075 - points[:, 1]) / 0.095, 0.0, 1.0)
    front_gate *= np.clip((points[:, 2] - (lm[8, 2] - 0.040)) / 0.075, 0.0, 1.0)
    front_gate *= head_mask.astype(np.float64)

    # 1. Real eyelid/orbit continuity around the enlarged visible eye geometry.
    for eye_ids in (np.arange(36, 42), np.arange(42, 48)):
        c = lm[eye_ids].mean(axis=0)
        # Open the actual skin aperture modestly and keep the orbit smooth.
        add_scale(delta, points, head, c, (0.039, 0.031, 0.024),
                  (1.045, 1.0, 1.105), 0.08, 1.16, front_gate)
        # Upper lid rises more than lower lid falls: youthful almond, not stare.
        upper_gate = (points[:, 2] >= c[2]).astype(np.float64) * front_gate
        lower_gate = (points[:, 2] < c[2]).astype(np.float64) * front_gate
        add_shift(delta, points, head, c + np.array([0.0, 0.0, 0.004]),
                  (0.036, 0.027, 0.017), (0.0, -0.00045, 0.00130),
                  0.03, 1.12, upper_gate)
        add_shift(delta, points, head, c - np.array([0.0, 0.0, 0.003]),
                  (0.035, 0.027, 0.016), (0.0, -0.00020, -0.00035),
                  0.03, 1.10, lower_gate)
        # Under-eye support removes the hard shelf left by the old orbit.
        add_shift(delta, points, head, c + np.array([0.0, 0.004, -0.015]),
                  (0.041, 0.034, 0.025), (0.0, -0.00075, 0.00035),
                  0.0, 1.12, front_gate)

        side = -1.0 if c[0] < 0 else 1.0
        outer = lm[eye_ids[np.argmin(lm[eye_ids, 0])]] if side < 0 else lm[eye_ids[np.argmax(lm[eye_ids, 0])]]
        add_shift(delta, points, head, outer, (0.017, 0.017, 0.014),
                  (side * 0.00035, -0.00030, 0.00105), 0.0, 1.06, front_gate)

    # 2. Soften glabella/temple transition without flattening the nose bridge.
    brow_center = lm[17:27].mean(axis=0)
    add_shift(delta, points, head, brow_center, (0.072, 0.039, 0.034),
              (0.0, 0.00090, -0.00025), 0.0, 1.13, front_gate)
    for side in (-1.0, 1.0):
        temple = np.array([side * 0.071, brow_center[1] + 0.010, brow_center[2] + 0.004])
        add_scale(delta, points, head, temple, (0.041, 0.043, 0.052),
                  (0.975, 1.0, 1.0), 0.0, 1.10, front_gate)

    # 3. High apple-cheek support blended into a narrower lower cheek.
    lm = (points + delta)[K]
    cheek_r = (lm[40] + lm[31] + lm[48]) / 3.0
    cheek_l = (lm[46] + lm[35] + lm[54]) / 3.0
    for c in (cheek_r, cheek_l):
        add_shift(delta, points, head, c, (0.043, 0.040, 0.038),
                  (0.0, -0.00085, 0.00055), 0.02, 1.10, front_gate)
        lower_c = c + np.array([0.0, 0.005, -0.027])
        add_scale(delta, points, head, lower_c, (0.047, 0.045, 0.040),
                  (0.975, 1.0, 0.99), 0.0, 1.10, front_gate)

    # 4. Continuous cheek -> jaw -> chin taper.  Preserve jaw volume near mouth.
    lm = (points + delta)[K]
    mouth_z = float(lm[48:60, 2].mean())
    chin_z = float(lm[8, 2])
    p = points[head]
    lower_t = np.clip((mouth_z + 0.026 - p[:, 2]) / max(mouth_z + 0.026 - chin_z, 1e-6), 0.0, 1.0)
    lateral = np.clip((np.abs(p[:, 0]) - 0.025) / 0.070, 0.0, 1.0)
    front = np.clip((0.060 - p[:, 1]) / 0.085, 0.0, 1.0)
    taper = 0.055 * (lower_t ** 1.22) * (0.42 + 0.58 * lateral) * front
    target_x = p[:, 0] * (1.0 - taper)
    delta[head, 0] += target_x - p[:, 0]

    # Jaw angles tuck in/back; chin remains small and rounded rather than pointed.
    for idx in (4, 12):
        c = lm[idx]
        side = -1.0 if c[0] < 0 else 1.0
        add_shift(delta, points, head, c, (0.040, 0.045, 0.052),
                  (-side * 0.0024, 0.00120, 0.00025), 0.0, 1.08, front_gate)
    chin = lm[8]
    add_scale(delta, points, head, chin, (0.045, 0.044, 0.040),
              (0.86, 0.97, 0.975), 0.03, 1.10, front_gate)
    add_shift(delta, points, head, chin, (0.046, 0.044, 0.040),
              (0.0, 0.00035, 0.00110), 0.0, 1.06, front_gate)

    # 5. Ear-side / lower-head / neck continuity.  Keep central neck untouched.
    lm = (points + delta)[K]
    for idx in (0, 16):
        c = lm[idx]
        side = -1.0 if c[0] < 0 else 1.0
        add_scale(delta, points, head, c, (0.038, 0.048, 0.060),
                  (0.80, 0.90, 0.86), 0.0, 1.10, head_mask.astype(float))
        add_shift(delta, points, head, c, (0.040, 0.050, 0.062),
                  (-side * 0.0032, 0.0018, -0.0002), 0.0, 1.06,
                  head_mask.astype(float))

    neck_zone = (points[:, 2] < chin_z - 0.018) & head_mask
    neck_side = np.clip((np.abs(points[:, 0]) - 0.038) / 0.055, 0.0, 1.0)
    neck_vertical = np.clip((chin_z - 0.012 - points[:, 2]) / 0.085, 0.0, 1.0)
    neck_taper = 0.032 * neck_side * neck_vertical * neck_zone.astype(float)
    delta[:, 0] += -points[:, 0] * neck_taper

    # Smooth the displacement over the real head topology, not the final shape.
    nbr = adjacency(len(points), faces, head)
    delta = smooth_delta(delta, nbr, head, iterations=3, alpha=0.22)
    delta[~head_mask] = 0.0

    # Production safety clamp: no single vertex may jump more than 4.8 mm.
    length = np.linalg.norm(delta, axis=1)
    scale = np.minimum(1.0, 0.0048 / np.maximum(length, 1e-12))
    delta *= scale[:, None]
    refined = points + delta

    # Validate triangle quality against the locked source.
    tri0 = points[faces]
    tri1 = refined[faces]
    area0 = 0.5 * np.linalg.norm(np.cross(tri0[:, 1] - tri0[:, 0], tri0[:, 2] - tri0[:, 0]), axis=1)
    area1 = 0.5 * np.linalg.norm(np.cross(tri1[:, 1] - tri1[:, 0], tri1[:, 2] - tri1[:, 0]), axis=1)
    ratio = area1 / np.maximum(area0, 1e-12)

    raw1 = map_from_blender(refined, offset)
    write_obj(args.out, raw1, faces)
    reloaded = trimesh.load(args.out, process=False, maintain_order=True)
    if not isinstance(reloaded, trimesh.Trimesh):
        raise RuntimeError("Refined OBJ could not be reloaded")

    lm0 = original[K]
    lm1 = refined[K]
    report = {
        "product": "AINA direct orbit/jaw/ear/neck real-mesh continuity pass",
        "source": str(args.mesh),
        "output": str(args.out),
        "topology_changed": False,
        "vertices_before": int(len(raw0)),
        "vertices_after": int(len(reloaded.vertices)),
        "triangles_before": int(len(faces)),
        "triangles_after": int(len(reloaded.faces)),
        "semantic_vertex_order_preserved": bool(len(reloaded.vertices) == len(raw0)),
        "max_displacement_m": float(np.linalg.norm(delta, axis=1).max()),
        "rms_displacement_m": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
        "moved_vertices_over_0_1mm": int(np.sum(np.linalg.norm(delta, axis=1) > 0.0001)),
        "triangle_area_ratio_p01": float(np.percentile(ratio, 1)),
        "triangle_area_ratio_p99": float(np.percentile(ratio, 99)),
        "right_eye_width_before_m": float(np.linalg.norm(lm0[36] - lm0[39])),
        "right_eye_width_after_m": float(np.linalg.norm(lm1[36] - lm1[39])),
        "left_eye_width_before_m": float(np.linalg.norm(lm0[42] - lm0[45])),
        "left_eye_width_after_m": float(np.linalg.norm(lm1[42] - lm1[45])),
        "jaw_width_before_m": float(abs(lm0[12, 0] - lm0[4, 0])),
        "jaw_width_after_m": float(abs(lm1[12, 0] - lm1[4, 0])),
        "checks": {
            "vertex_count_preserved": len(reloaded.vertices) == len(raw0),
            "triangle_count_preserved": len(reloaded.faces) == len(faces),
            "finite_vertices": bool(np.isfinite(raw1).all()),
            "max_displacement_safe": float(np.linalg.norm(delta, axis=1).max()) <= 0.00481,
            "triangle_quality_safe": float(np.percentile(ratio, 1)) > 0.30 and float(np.percentile(ratio, 99)) < 2.50,
        },
        "visual_lock": False,
        "visual_gate": "actual Blender neutral front and calibrated 20-degree 3Q must be inspected before lock",
    }
    report["pass"] = bool(all(report["checks"].values()))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("AINA direct real-mesh continuity QA failed")


if __name__ == "__main__":
    main()
