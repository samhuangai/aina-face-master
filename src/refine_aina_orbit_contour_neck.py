#!/usr/bin/env python3
"""Direct real-mesh refinement for the locked AINA v15.5 production head.

The pass edits existing OBJ vertex positions only. It preserves vertex order,
faces, connected components, semantic indices, rig compatibility and the 52
shape-control generation path. Focus: open youthful eyelids, softer eye/orbit
continuity, a visible delicate nose, integrated compact lips, apple-cheek volume,
a rounded V lower third, smaller ears and a cleaner chin-to-neck transition.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

K = np.array([
    1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,
    3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,
    5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,
    2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,
    13695,12973,13422,6543,6537,
], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--height", type=float, default=1.72)
    return ap.parse_args(argv)


def read_obj(path: Path):
    lines = path.read_text(errors="ignore").splitlines()
    verts = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            q = line.split()
            verts.append((float(q[1]), float(q[2]), float(q[3])))
        elif line.startswith("f "):
            ids = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
            for i in range(1, len(ids) - 1):
                faces.append((ids[0], ids[i], ids[i + 1]))
    return lines, np.asarray(verts, np.float64), np.asarray(faces, np.int64)


def write_obj(lines, verts: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    vi = 0
    for line in lines:
        if line.startswith("v "):
            p = verts[vi]
            out.append(f"v {p[0]:.9f} {p[1]:.9f} {p[2]:.9f}")
            vi += 1
        else:
            out.append(line)
    if vi != len(verts):
        raise RuntimeError(f"OBJ vertex rewrite mismatch: {vi} != {len(verts)}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def components(n: int, faces: np.ndarray):
    parent = np.arange(n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    for a, b, c in faces:
        union(a, b); union(b, c); union(c, a)
    roots = np.array([find(i) for i in range(n)], dtype=np.int32)
    return roots


def map_face(raw: np.ndarray, height: float):
    scale = 1.08
    mapped = np.empty_like(raw)
    mapped[:, 0] = raw[:, 0] * scale
    mapped[:, 1] = raw[:, 2] * scale
    mapped[:, 2] = -raw[:, 1] * scale
    offset = height - float(mapped[:, 2].max())
    mapped[:, 2] += offset
    return mapped, scale, offset


def inverse_map(mapped: np.ndarray, scale: float, offset: float):
    raw = np.empty_like(mapped)
    raw[:, 0] = mapped[:, 0] / scale
    raw[:, 2] = mapped[:, 1] / scale
    raw[:, 1] = -(mapped[:, 2] - offset) / scale
    return raw


def weights(points: np.ndarray, center, radii, inner=0.0, outer=1.0):
    center = np.asarray(center, np.float64)
    radii = np.asarray(radii, np.float64)
    q = np.sqrt(np.sum(((points - center) / radii) ** 2, axis=1))
    w = np.zeros(len(points), np.float64)
    w[q <= inner] = 1.0
    m = (q > inner) & (q < outer)
    if np.any(m):
        t = (q[m] - inner) / max(outer - inner, 1e-12)
        w[m] = 0.5 * (1.0 + np.cos(np.pi * t))
    return w


def shift(arr, ids, center, radii, delta, inner=0.0, outer=1.0):
    ids = np.asarray(ids, np.int64)
    p = arr[ids].copy()
    p += weights(p, center, radii, inner, outer)[:, None] * np.asarray(delta, np.float64)
    arr[ids] = p


def scale_region(arr, ids, center, radii, factors, inner=0.0, outer=1.0):
    ids = np.asarray(ids, np.int64)
    p = arr[ids].copy()
    center = np.asarray(center, np.float64)
    w = weights(p, center, radii, inner, outer)[:, None]
    target = center + (p - center) * np.asarray(factors, np.float64)
    arr[ids] = p + w * (target - p)


def refine(mapped: np.ndarray, head_ids: np.ndarray, faces: np.ndarray):
    base = mapped.copy()
    out = mapped.copy()
    h = np.asarray(head_ids, np.int64)
    lm = out[K].copy()

    # Youthful lower-third taper and a slightly narrower upper cranium.
    mouth_z = float(lm[48:60, 2].mean())
    chin_z = float(lm[8, 2])
    eye_z = float(lm[36:48, 2].mean())
    p = out[h].copy()
    t = np.clip((mouth_z + 0.018 - p[:, 2]) / max(mouth_z + 0.018 - (chin_z - 0.010), 1e-6), 0.0, 1.0)
    front = np.exp(-0.5 * ((p[:, 1] + 0.005) / 0.070) ** 4)
    p[:, 0] *= 1.0 - 0.105 * (t ** 1.25) * front
    temple = np.clip((p[:, 2] - eye_z) / 0.095, 0.0, 1.0)
    p[:, 0] *= 1.0 - 0.035 * temple * np.exp(-0.5 * ((p[:, 1] + 0.005) / 0.075) ** 4)
    out[h] = p

    # Actual eyelid/orbit mesh: widen and open the fissure, lift only outer tails,
    # fill the tear trough and retreat the heavy adult brow plane.
    eye_specs = [
        (list(range(36, 42)), 36, 39, [37, 38], [40, 41]),
        (list(range(42, 48)), 45, 42, [43, 44], [46, 47]),
    ]
    for rr, outer_i, inner_i, upper_i, lower_i in eye_specs:
        lm = out[K].copy(); c = lm[rr].mean(0)
        scale_region(out, h, c, (0.044, 0.034, 0.026), (1.09, 1.0, 1.20), 0.10, 1.14)
        shift(out, h, c, (0.045, 0.035, 0.028), (0.0, -0.0011, 0.0002), 0.08, 1.12)
        lm = out[K].copy()
        shift(out, h, lm[upper_i].mean(0), (0.035, 0.027, 0.015), (0.0, -0.0008, 0.0020), 0.02, 1.08)
        shift(out, h, lm[lower_i].mean(0), (0.035, 0.027, 0.015), (0.0, -0.0005, -0.00125), 0.02, 1.08)
        lm = out[K].copy(); outer = lm[outer_i]; side = -1.0 if outer[0] < 0 else 1.0
        shift(out, h, outer, (0.019, 0.018, 0.014), (side * 0.0011, -0.00035, 0.0014), 0.02, 1.03)
        shift(out, h, lm[inner_i], (0.016, 0.017, 0.013), (-side * 0.00015, -0.00025, 0.00015), 0.02, 1.03)
        lm = out[K].copy(); c = lm[rr].mean(0)
        shift(out, h, (c[0], c[1] + 0.006, c[2] - 0.013), (0.041, 0.038, 0.024), (0.0, -0.00115, 0.00055), 0.0, 1.12)
        shift(out, h, (c[0], c[1] + 0.010, c[2] + 0.022), (0.044, 0.039, 0.026), (0.0, 0.0028, -0.00045), 0.0, 1.13)

    lm = out[K].copy()
    for rr in (list(range(17, 22)), list(range(22, 27))):
        shift(out, h, lm[rr].mean(0), (0.043, 0.033, 0.026), (0.0, 0.0025, -0.0008), 0.0, 1.14)
    shift(out, h, lm[27], (0.033, 0.033, 0.041), (0.0, 0.0018, -0.0003), 0.0, 1.10)

    # Delicate but visible nose. Blender front is -Y, so negative Y restores profile.
    lm = out[K].copy(); bridge = lm[27:31].mean(0); tip = lm[30]; nbase = lm[31:36].mean(0)
    scale_region(out, h, bridge, (0.025, 0.032, 0.047), (0.82, 1.0, 0.90), 0.04, 1.15)
    shift(out, h, bridge, (0.026, 0.034, 0.046), (0.0, -0.0014, 0.0006), 0.02, 1.12)
    scale_region(out, h, nbase, (0.030, 0.027, 0.028), (0.78, 1.0, 0.91), 0.04, 1.14)
    shift(out, h, nbase, (0.031, 0.029, 0.030), (0.0, -0.0030, 0.0022), 0.02, 1.12)
    shift(out, h, tip, (0.021, 0.024, 0.023), (0.0, -0.0048, 0.0028), 0.02, 1.08)

    # Compact integrated lips: shorten philtrum, retreat the perimeter, preserve
    # real upper/lower volume and avoid a pasted-on oval silhouette.
    lm = out[K].copy(); mouth = lm[48:60].mean(0)
    scale_region(out, h, mouth, (0.050, 0.036, 0.030), (0.98, 0.85, 0.74), 0.10, 1.16)
    shift(out, h, mouth, (0.052, 0.039, 0.032), (0.0, 0.0038, 0.0014), 0.06, 1.15)
    lm = out[K].copy()
    shift(out, h, lm[[49, 50, 51, 52, 53]].mean(0), (0.033, 0.026, 0.014), (0.0, -0.0012, 0.0005), 0.02, 1.05)
    shift(out, h, lm[[55, 56, 57, 58, 59]].mean(0), (0.035, 0.027, 0.015), (0.0, -0.0015, -0.00015), 0.02, 1.05)
    for idx in (48, 54):
        lm = out[K].copy(); c = lm[idx]; side = -1.0 if c[0] < 0 else 1.0
        shift(out, h, c, (0.018, 0.019, 0.014), (-side * 0.00035, 0.0008, 0.00045), 0.02, 1.04)

    # Apple cheeks and smooth eye-to-cheek transitions without ballooning the face.
    lm = out[K].copy()
    cheeks = ((lm[40] + lm[31] + lm[48]) / 3.0, (lm[46] + lm[35] + lm[54]) / 3.0)
    for c in cheeks:
        shift(out, h, c, (0.046, 0.043, 0.041), (0.0, -0.0022, 0.0009), 0.02, 1.13)
        scale_region(out, h, c, (0.047, 0.044, 0.042), (0.97, 1.0, 1.02), 0.0, 1.10)
    lm = out[K].copy(); mouth = lm[48:60].mean(0)
    for side in (-1.0, 1.0):
        shift(out, h, (side * 0.020, mouth[1] + 0.004, mouth[2] + 0.017), (0.028, 0.032, 0.027), (0.0, 0.0007, 0.00015), 0.0, 1.08)

    # Smaller rounded chin, soft V jaw, tucked ears and cleaner neck transition.
    lm = out[K].copy(); chin = lm[8]
    scale_region(out, h, chin, (0.047, 0.047, 0.043), (0.82, 0.94, 0.90), 0.02, 1.12)
    shift(out, h, chin, (0.048, 0.047, 0.043), (0.0, 0.0018, 0.0036), 0.0, 1.07)
    for idx in (4, 12):
        lm = out[K].copy(); c = lm[idx]; side = -1.0 if c[0] < 0 else 1.0
        shift(out, h, c, (0.040, 0.048, 0.048), (-side * 0.0020, 0.0010, 0.0006), 0.0, 1.10)
    for idx in (0, 16):
        lm = out[K].copy(); c = lm[idx]; side = -1.0 if c[0] < 0 else 1.0
        scale_region(out, h, c, (0.037, 0.046, 0.059), (0.78, 0.82, 0.80), 0.0, 1.12)
        shift(out, h, c, (0.038, 0.047, 0.060), (-side * 0.0045, 0.0024, 0.0), 0.0, 1.06)
    lm = out[K].copy(); chin = lm[8]
    under = (chin[0], chin[1] + 0.020, chin[2] - 0.012)
    scale_region(out, h, under, (0.060, 0.064, 0.036), (0.88, 0.96, 1.0), 0.0, 1.10)
    shift(out, h, under, (0.060, 0.064, 0.036), (0.0, 0.0015, -0.0004), 0.0, 1.06)

    # Smooth the displacement field, not the source surface, so detail and topology
    # remain intact while local edit boundaries stop reading as separate patches.
    raw_delta = out - base
    adjacency = [set() for _ in range(len(out))]
    head_mask = np.zeros(len(out), dtype=bool); head_mask[h] = True
    for a, b, c in faces:
        if head_mask[a] and head_mask[b] and head_mask[c]:
            adjacency[a].update((int(b), int(c)))
            adjacency[b].update((int(a), int(c)))
            adjacency[c].update((int(a), int(b)))
    smooth = raw_delta.copy()
    for _ in range(2):
        old = smooth.copy()
        for i in h:
            nbr = adjacency[i]
            if nbr:
                smooth[i] = 0.78 * raw_delta[i] + 0.22 * old[list(nbr)].mean(axis=0)
    smooth[K] = raw_delta[K]
    return base + smooth, raw_delta, smooth, head_mask


def main() -> None:
    args = parse_args()
    lines, raw, faces = read_obj(args.mesh)
    if len(raw) <= int(K.max()):
        raise RuntimeError("Mesh does not preserve required AINA semantic vertex order")
    roots = components(len(raw), faces)
    head_ids = np.flatnonzero(roots == roots[int(K[0])])
    mapped, scale, offset = map_face(raw, args.height)
    refined, raw_delta, smooth_delta, head_mask = refine(mapped, head_ids, faces)

    tri0 = mapped[faces[head_mask[faces].all(axis=1)]]
    tri1 = refined[faces[head_mask[faces].all(axis=1)]]
    area0 = 0.5 * np.linalg.norm(np.cross(tri0[:, 1] - tri0[:, 0], tri0[:, 2] - tri0[:, 0]), axis=1)
    area1 = 0.5 * np.linalg.norm(np.cross(tri1[:, 1] - tri1[:, 0], tri1[:, 2] - tri1[:, 0]), axis=1)
    ratio = area1 / np.maximum(area0, 1e-12)
    d = smooth_delta[head_ids]
    report = {
        "stage": "AINA direct real-mesh orbit/contour/neck refinement",
        "source": str(args.mesh),
        "output": str(args.out),
        "topology_changed": False,
        "vertices": int(len(raw)),
        "faces": int(len(faces)),
        "head_vertices": int(len(head_ids)),
        "max_displacement_m": float(np.linalg.norm(d, axis=1).max()),
        "rms_displacement_m": float(np.sqrt(np.mean(np.sum(d * d, axis=1)))),
        "triangle_area_ratio_p01": float(np.percentile(ratio, 1)),
        "triangle_area_ratio_p99": float(np.percentile(ratio, 99)),
        "degenerate_head_triangles": int(np.sum(area1 < 1e-12)),
        "real_mesh_edited": True,
        "new_reference_generated": False,
        "visual_identity_lock": False,
        "next_gate": "actual Blender neutral front, shallow 3Q and expression renders",
    }
    write_obj(lines, inverse_map(refined, scale, offset), args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
