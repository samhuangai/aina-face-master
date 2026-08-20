#!/usr/bin/env python3
"""Bounded front-semantic correction for the directly refined AINA real mesh.

This does not create a new identity or image.  It preserves the depth work from
orbit/jaw/profile sculpting, then applies a small camera-plane semantic cage so
the actual OBJ's 68 front landmarks remain aligned with the approved AINA target.
Topology and semantic vertex order are unchanged.
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
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def map_to_blender(v: np.ndarray, height=1.72) -> tuple[np.ndarray, float]:
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


def fit_camera(points: np.ndarray, target: np.ndarray, weights: np.ndarray):
    x = np.c_[points, np.ones(len(points))]
    sw = np.sqrt(weights)[:, None]
    beta = np.linalg.lstsq(x * sw, target * sw, rcond=None)[0]
    a = beta[:3].T
    b = beta[3]
    n1, n2 = np.linalg.norm(a[0]), np.linalg.norm(a[1])
    scale = max(1e-9, 0.5 * (n1 + n2))
    r1 = a[0] / max(n1, 1e-9)
    v2 = a[1] - np.dot(a[1], r1) * r1
    r2 = v2 / max(np.linalg.norm(v2), 1e-9)
    r3 = np.cross(r1, r2)
    r3 /= max(np.linalg.norm(r3), 1e-9)
    r2 = np.cross(r3, r1)
    r = np.stack([r1, r2, r3])
    if np.linalg.det(r) < 0:
        r[2] *= -1
    return r, scale, b


def project(points: np.ndarray, r: np.ndarray, scale: float, b: np.ndarray) -> np.ndarray:
    return scale * (points @ r.T)[:, :2] + b


def feature_weights() -> np.ndarray:
    w = np.ones(68, dtype=np.float64)
    w[:17] = 1.20
    w[17:27] = 0.55
    w[27:36] = 1.85
    w[36:48] = 2.45
    w[48:68] = 2.10
    return w


def handle_strengths() -> np.ndarray:
    s = np.ones(68, dtype=np.float64)
    s[:17] = 0.46       # keep the newly sculpted 3D jaw/profile depth
    s[17:27] = 0.30
    s[27:36] = 0.58
    s[36:48] = 0.72
    s[48:68] = 0.66
    return s


def handle_radii() -> np.ndarray:
    r = np.full(68, 0.017, dtype=np.float64)
    r[:17] = 0.028
    r[17:27] = 0.022
    r[27:36] = 0.018
    r[36:48] = 0.015
    r[48:68] = 0.017
    return r


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
    mask = np.zeros(n, dtype=bool); mask[allowed] = True
    nbr: list[set[int]] = [set() for _ in range(n)]
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        if mask[a] and mask[b]: nbr[a].add(b); nbr[b].add(a)
        if mask[b] and mask[c]: nbr[b].add(c); nbr[c].add(b)
        if mask[c] and mask[a]: nbr[c].add(a); nbr[a].add(c)
    return [list(x) for x in nbr]


def smooth(delta: np.ndarray, nbr: list[list[int]], ids: np.ndarray, iterations=2, alpha=0.16) -> np.ndarray:
    out = delta.copy()
    for _ in range(iterations):
        old = out.copy()
        for i in ids:
            ns = nbr[int(i)]
            if ns:
                out[i] = (1.0 - alpha) * old[i] + alpha * old[ns].mean(axis=0)
    return 0.88 * delta + 0.12 * out


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# AINA bounded front semantic cage correction\n")
        for x, y, z in vertices:
            f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        for a, b, c in faces:
            f.write(f"f {int(a)+1} {int(b)+1} {int(c)+1}\n")


def main() -> None:
    args = parse_args()
    mesh = trimesh.load(args.mesh, process=False, maintain_order=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Expected one triangulated AINA OBJ")
    raw = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if int(K.max()) >= len(raw):
        raise RuntimeError("Semantic indices are outside the refined mesh")

    target_data = json.loads(args.target.read_text(encoding="utf-8"))
    size = np.asarray(target_data["image_size"], dtype=np.float64)
    target_px = np.asarray(target_data["landmarks_xy"], dtype=np.float64)
    target = (target_px - size * 0.5) / (0.5 * max(size))

    points, offset = map_to_blender(raw)
    roots, groups = components(len(points), faces)
    head = max(groups.values(), key=len)
    head_mask = np.zeros(len(points), dtype=bool); head_mask[head] = True
    lm = points[K]
    w = feature_weights()
    r, camera_scale, b = fit_camera(lm, target, w)
    pred0 = project(lm, r, camera_scale, b)
    err0 = np.linalg.norm(pred0 - target, axis=1)
    rmse0 = float(np.sqrt(np.sum(w * err0 * err0) / np.sum(w)))

    residual = target - pred0
    # Convert normalized image-plane residuals into 3D camera-plane handle motion.
    handle = ((residual[:, 0] / camera_scale)[:, None] * r[0][None, :] +
              (residual[:, 1] / camera_scale)[:, None] * r[1][None, :])
    handle *= handle_strengths()[:, None]
    handle_len = np.linalg.norm(handle, axis=1)
    handle *= np.minimum(1.0, 0.00135 / np.maximum(handle_len, 1e-12))[:, None]

    # Diffuse every semantic handle locally over the real head surface.
    delta = np.zeros_like(points)
    weight_sum = np.zeros(len(points), dtype=np.float64)
    radii = handle_radii()
    hp = points[head]
    for j in range(68):
        distance = np.linalg.norm(hp - lm[j], axis=1)
        local = np.exp(-0.5 * (distance / radii[j]) ** 4)
        delta[head] += local[:, None] * handle[j]
        weight_sum[head] += local
    nz = weight_sum > 1e-8
    delta[nz] /= weight_sum[nz, None]
    delta[~head_mask] = 0.0

    # Do not move rear skull or the low neck from a front-landmark correction.
    front_gate = np.clip((0.075 - points[:, 1]) / 0.100, 0.0, 1.0)
    chin_z = float(lm[8, 2])
    vertical_gate = np.clip((points[:, 2] - (chin_z - 0.018)) / 0.065, 0.0, 1.0)
    delta *= (front_gate * vertical_gate)[:, None]

    nbr = adjacency(len(points), faces, head)
    delta = smooth(delta, nbr, head, iterations=2, alpha=0.15)
    length = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1.0, 0.00155 / np.maximum(length, 1e-12))[:, None]
    corrected = points + delta

    lm1 = corrected[K]
    r1, scale1, b1 = fit_camera(lm1, target, w)
    pred1 = project(lm1, r1, scale1, b1)
    err1 = np.linalg.norm(pred1 - target, axis=1)
    rmse1 = float(np.sqrt(np.sum(w * err1 * err1) / np.sum(w)))

    tri0 = points[faces]
    tri1 = corrected[faces]
    area0 = 0.5 * np.linalg.norm(np.cross(tri0[:, 1] - tri0[:, 0], tri0[:, 2] - tri0[:, 0]), axis=1)
    area1 = 0.5 * np.linalg.norm(np.cross(tri1[:, 1] - tri1[:, 0], tri1[:, 2] - tri1[:, 0]), axis=1)
    area_ratio = area1 / np.maximum(area0, 1e-12)

    raw1 = map_from_blender(corrected, offset)
    write_obj(args.out, raw1, faces)
    reloaded = trimesh.load(args.out, process=False, maintain_order=True)
    if not isinstance(reloaded, trimesh.Trimesh):
        raise RuntimeError("Corrected OBJ failed to reload")

    px_scale = 0.5 * max(size)
    report = {
        "product": "AINA bounded front semantic cage on directly refined real mesh",
        "source": str(args.mesh),
        "output": str(args.out),
        "topology_changed": False,
        "semantic_vertex_order_preserved": len(reloaded.vertices) == len(raw),
        "vertices": int(len(raw)),
        "triangles": int(len(faces)),
        "front_weighted_rmse_before_px": rmse0 * px_scale,
        "front_weighted_rmse_after_px": rmse1 * px_scale,
        "front_max_error_before_px": float(err0.max() * px_scale),
        "front_max_error_after_px": float(err1.max() * px_scale),
        "max_handle_m": float(np.linalg.norm(handle, axis=1).max()),
        "max_vertex_displacement_m": float(np.linalg.norm(delta, axis=1).max()),
        "rms_vertex_displacement_m": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
        "triangle_area_ratio_p01": float(np.percentile(area_ratio, 1)),
        "triangle_area_ratio_p99": float(np.percentile(area_ratio, 99)),
        "checks": {
            "vertex_count_preserved": len(reloaded.vertices) == len(raw),
            "triangle_count_preserved": len(reloaded.faces) == len(faces),
            "finite_vertices": bool(np.isfinite(raw1).all()),
            "front_fit_not_worse": rmse1 <= rmse0 + 1e-7,
            "bounded_displacement": float(np.linalg.norm(delta, axis=1).max()) <= 0.00156,
            "triangle_quality_safe": float(np.percentile(area_ratio, 1)) > 0.45 and float(np.percentile(area_ratio, 99)) < 1.80,
        },
        "visual_lock": False,
        "visual_gate": "inspect actual Blender beauty and clay front/20-degree-3Q renders",
    }
    report["pass"] = bool(all(report["checks"].values()))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("AINA front semantic cage QA failed")


if __name__ == "__main__":
    main()
