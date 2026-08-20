#!/usr/bin/env python3
"""Bounded approved-three-quarter semantic cage for the actual AINA mesh.

Uses MediaPipe only to measure the existing approved AINA 3Q reference and an
actual Blender render of the current OBJ.  It does not generate or replace any
reference art.  Small image-plane residuals are converted into real 3D camera-
plane vertex motion around the matching 68 FaceVerse semantic handles.  Depth
work is preserved through conservative regional strengths and a 1.25 mm clamp.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
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

# MediaPipe FaceMesh indices arranged in the standard 68-point order.
MP68 = np.array([
    127,234,93,132,58,172,136,150,152,379,365,397,288,361,323,454,356,
    70,63,105,66,107,336,296,334,293,300,
    168,6,197,195,98,97,2,326,327,
    33,160,158,133,153,144,362,385,387,263,373,380,
    61,40,37,0,267,270,291,321,314,17,84,91,
    78,81,13,311,308,402,14,178,
], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--actual", type=Path, required=True)
    p.add_argument("--approved", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def detect68(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.30,
    ) as detector:
        result = detector.process(rgb)
    if not result.multi_face_landmarks:
        raise RuntimeError(f"MediaPipe did not find a face in {path}")
    points = np.array([[p.x, p.y] for p in result.multi_face_landmarks[0].landmark], dtype=np.float64)
    if int(MP68.max()) >= len(points):
        raise RuntimeError("MediaPipe landmark set is smaller than expected")
    return points[MP68]


def similarity(source: np.ndarray, target: np.ndarray, weights: np.ndarray):
    w = weights / max(weights.sum(), 1e-12)
    cs = np.sum(source * w[:, None], axis=0)
    ct = np.sum(target * w[:, None], axis=0)
    xs = source - cs
    xt = target - ct
    cov = (xs * w[:, None]).T @ xt
    u, singular, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    variance = np.sum(w * np.sum(xs * xs, axis=1))
    scale = float(singular.sum() / max(variance, 1e-12))
    translation = ct - scale * (cs @ r)
    return scale, r, translation


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
    r3 = np.cross(r1, r2); r3 /= max(np.linalg.norm(r3), 1e-9)
    r2 = np.cross(r3, r1)
    r = np.stack([r1, r2, r3])
    if np.linalg.det(r) < 0: r[2] *= -1
    return r, scale, b


def project(points: np.ndarray, r: np.ndarray, scale: float, b: np.ndarray) -> np.ndarray:
    return scale * (points @ r.T)[:, :2] + b


def map_to_blender(v: np.ndarray, height=1.72) -> tuple[np.ndarray, float]:
    out = np.empty_like(v, dtype=np.float64); s = 1.08
    out[:, 0] = v[:, 0] * s
    out[:, 1] = v[:, 2] * s
    out[:, 2] = -v[:, 1] * s
    offset = height - float(out[:, 2].max())
    out[:, 2] += offset
    return out, offset


def map_from_blender(v: np.ndarray, offset: float) -> np.ndarray:
    out = np.empty_like(v, dtype=np.float64); s = 1.08
    out[:, 0] = v[:, 0] / s
    out[:, 2] = v[:, 1] / s
    out[:, 1] = -(v[:, 2] - offset) / s
    return out


def components(n: int, faces: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    parent = np.arange(n, dtype=np.int64)
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = int(parent[x])
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for a, b, c in faces:
        union(int(a), int(b)); union(int(b), int(c)); union(int(c), int(a))
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    groups: dict[int, list[int]] = {}
    for i, root in enumerate(roots): groups.setdefault(int(root), []).append(i)
    return roots, {root: np.asarray(ids, dtype=np.int64) for root, ids in groups.items()}


def adjacency(n: int, faces: np.ndarray, ids: np.ndarray) -> list[list[int]]:
    mask = np.zeros(n, dtype=bool); mask[ids] = True
    nbr: list[set[int]] = [set() for _ in range(n)]
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        if mask[a] and mask[b]: nbr[a].add(b); nbr[b].add(a)
        if mask[b] and mask[c]: nbr[b].add(c); nbr[c].add(b)
        if mask[c] and mask[a]: nbr[c].add(a); nbr[a].add(c)
    return [list(x) for x in nbr]


def smooth(delta: np.ndarray, nbr: list[list[int]], ids: np.ndarray) -> np.ndarray:
    out = delta.copy()
    for _ in range(2):
        old = out.copy()
        for i in ids:
            ns = nbr[int(i)]
            if ns: out[i] = 0.86 * old[i] + 0.14 * old[ns].mean(axis=0)
    return 0.90 * delta + 0.10 * out


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# AINA approved-3Q bounded semantic cage\n")
        for x, y, z in vertices: f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        for a, b, c in faces: f.write(f"f {int(a)+1} {int(b)+1} {int(c)+1}\n")


def main() -> None:
    args = parse_args()
    actual = detect68(args.actual)
    approved = detect68(args.approved)
    weights = np.ones(68, dtype=np.float64)
    weights[:17] = 0.85; weights[17:27] = 0.55; weights[27:36] = 1.80
    weights[36:48] = 1.75; weights[48:68] = 1.65

    # Bring the approved artwork landmarks into the actual render's frame before
    # measuring local identity residuals.  This removes crop/scale differences.
    sim_scale, sim_r, sim_t = similarity(approved, actual, weights)
    target = sim_scale * (approved @ sim_r) + sim_t
    residual2 = target - actual
    rmse_before = float(np.sqrt(np.sum(weights * np.sum(residual2 * residual2, axis=1)) / weights.sum()))

    mesh = trimesh.load(args.mesh, process=False, maintain_order=True)
    if not isinstance(mesh, trimesh.Trimesh): raise RuntimeError("Expected one triangulated AINA OBJ")
    raw = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if int(K.max()) >= len(raw): raise RuntimeError("AINA semantic indices are missing")
    points, offset = map_to_blender(raw)
    roots, groups = components(len(points), faces)
    head = max(groups.values(), key=len)
    head_mask = np.zeros(len(points), dtype=bool); head_mask[head] = True
    lm = points[K]

    camera_r, camera_scale, camera_t = fit_camera(lm, actual, weights)
    projected = project(lm, camera_r, camera_scale, camera_t)
    detector_fit_rmse = float(np.sqrt(np.sum(weights * np.sum((projected - actual) ** 2, axis=1)) / weights.sum()))

    # Residual is in normalized image units. Convert to real camera-plane metres.
    handle = ((residual2[:, 0] / camera_scale)[:, None] * camera_r[0][None, :] +
              (residual2[:, 1] / camera_scale)[:, None] * camera_r[1][None, :])
    strengths = np.ones(68, dtype=np.float64)
    strengths[:17] = 0.25
    strengths[17:27] = 0.20
    strengths[27:36] = 0.62
    strengths[36:48] = 0.34
    strengths[48:68] = 0.54
    handle *= strengths[:, None]
    handle_len = np.linalg.norm(handle, axis=1)
    handle *= np.minimum(1.0, 0.00105 / np.maximum(handle_len, 1e-12))[:, None]

    radii = np.full(68, 0.017, dtype=np.float64)
    radii[:17] = 0.027; radii[17:27] = 0.021; radii[27:36] = 0.017
    radii[36:48] = 0.015; radii[48:68] = 0.017
    hp = points[head]
    delta = np.zeros_like(points)
    total = np.zeros(len(points), dtype=np.float64)
    for j in range(68):
        distance = np.linalg.norm(hp - lm[j], axis=1)
        local = np.exp(-0.5 * (distance / radii[j]) ** 4)
        delta[head] += local[:, None] * handle[j]
        total[head] += local
    nz = total > 1e-8
    delta[nz] /= total[nz, None]
    delta[~head_mask] = 0.0

    # Keep rear skull and low neck fixed; this is a face-only 3Q correction.
    front_gate = np.clip((0.075 - points[:, 1]) / 0.100, 0.0, 1.0)
    chin_z = float(lm[8, 2])
    vertical_gate = np.clip((points[:, 2] - (chin_z - 0.010)) / 0.060, 0.0, 1.0)
    delta *= (front_gate * vertical_gate)[:, None]
    delta = smooth(delta, adjacency(len(points), faces, head), head)
    length = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1.0, 0.00125 / np.maximum(length, 1e-12))[:, None]
    corrected = points + delta

    pred_after = project(corrected[K], camera_r, camera_scale, camera_t)
    rmse_after = float(np.sqrt(np.sum(weights * np.sum((pred_after - target) ** 2, axis=1)) / weights.sum()))

    tri0 = points[faces]; tri1 = corrected[faces]
    area0 = 0.5 * np.linalg.norm(np.cross(tri0[:,1]-tri0[:,0], tri0[:,2]-tri0[:,0]), axis=1)
    area1 = 0.5 * np.linalg.norm(np.cross(tri1[:,1]-tri1[:,0], tri1[:,2]-tri1[:,0]), axis=1)
    ratio = area1 / np.maximum(area0, 1e-12)

    raw1 = map_from_blender(corrected, offset)
    write_obj(args.out, raw1, faces)
    reloaded = trimesh.load(args.out, process=False, maintain_order=True)
    if not isinstance(reloaded, trimesh.Trimesh): raise RuntimeError("3Q-corrected OBJ failed to reload")

    report = {
        "product": "AINA approved-3Q bounded semantic correction on actual real mesh",
        "approved_reference": str(args.approved),
        "actual_render": str(args.actual),
        "source": str(args.mesh),
        "output": str(args.out),
        "topology_changed": False,
        "semantic_vertex_order_preserved": len(reloaded.vertices) == len(raw),
        "mediapipe_approved_to_actual_rmse_before_norm": rmse_before,
        "mesh_to_actual_detector_fit_rmse_norm": detector_fit_rmse,
        "projected_target_rmse_after_norm": rmse_after,
        "max_handle_m": float(np.linalg.norm(handle, axis=1).max()),
        "max_vertex_displacement_m": float(np.linalg.norm(delta, axis=1).max()),
        "rms_vertex_displacement_m": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
        "triangle_area_ratio_p01": float(np.percentile(ratio, 1)),
        "triangle_area_ratio_p99": float(np.percentile(ratio, 99)),
        "checks": {
            "vertex_count_preserved": len(reloaded.vertices) == len(raw),
            "triangle_count_preserved": len(reloaded.faces) == len(faces),
            "finite_vertices": bool(np.isfinite(raw1).all()),
            "bounded_displacement": float(np.linalg.norm(delta, axis=1).max()) <= 0.00126,
            "three_quarter_residual_reduced": rmse_after < rmse_before,
            "triangle_quality_safe": float(np.percentile(ratio, 1)) > 0.45 and float(np.percentile(ratio, 99)) < 1.80,
        },
        "visual_lock": False,
        "visual_gate": "rerender exact corrected OBJ and inspect beauty plus clay front/20-degree-3Q",
    }
    report["pass"] = bool(all(report["checks"].values()))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("AINA approved-3Q semantic cage QA failed")


if __name__ == "__main__":
    main()
