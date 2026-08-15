#!/usr/bin/env python3
"""AINA v15.1 — real-mesh identity refinement.

This stage starts from the actual v14 FaceVerse full mesh. It does not create
concept art. The pass is topology-preserving and was tuned against real VTK
front / 45 / profile renders of the approved AINA references.

Goals:
- reduce oversized cranial/temple mass
- make ears smaller and more tucked
- restore a natural almond eye instead of an exaggerated cat-eye
- shorten/retract/narrow the nose in profile
- widen and soften the lips
- shorten the lower third and keep a soft feminine oval/V jaw
- suppress cheek/jaw ripple artifacts without smoothing identity features away

Identity lock intentionally remains false until the generated real-mesh QA
views visually pass against all approved references.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components

# FaceVerse 68 semantic vertex IDs used throughout the AINA pipeline.
K = np.array([
    1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,
    10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,
    10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,
    4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,6640,
    4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,
    6550,13695,12973,13422,6543,6537
], dtype=np.int64)


def gauss(points, center, radii):
    c = np.asarray(center, np.float64)
    r = np.asarray(radii, np.float64)
    return np.exp(-0.5 * np.sum(((points - c) / r) ** 2, axis=1))


def components(nv, faces):
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(nv, nv))
    a = (a + a.T).tocsr()
    n, labels = connected_components(a, directed=False)
    return [np.flatnonzero(labels == i) for i in range(n)]


def local_head_adjacency(faces, head, global_to_local):
    mask = np.zeros(int(faces.max()) + 1, dtype=bool)
    mask[head] = True
    hf = faces[np.flatnonzero(mask[faces].all(1))]
    rows, cols = [], []
    for tri in hf:
        li = [global_to_local[int(q)] for q in tri]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            rows.extend((li[i], li[j]))
            cols.extend((li[j], li[i]))
    a = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(head), len(head))).tocsr()
    a.data[:] = 1.0
    degree = np.asarray(a.sum(1)).ravel()
    degree[degree == 0] = 1.0
    return a, degree


def refine_head(p, kl, adj, degree):
    # 1. Smaller cranial volume. Hair will cover the scalp, but the skull must
    # still have the same delicate scale as the approved AINA head.
    w = np.clip((-p[:, 1] - 0.040) / 0.080, 0.0, 1.0)
    p[:, 0] *= 1.0 - 0.075 * w
    p[:, 1] += 0.0080 * w
    p[:, 2] += (0.020 - p[:, 2]) * 0.035 * w

    # 2. Smaller/tucked ears.
    for side in (-1.0, 1.0):
        c = np.array([side * 0.071, -0.004, 0.027])
        w = gauss(p, c, [0.022, 0.040, 0.038])
        p[:, 0] += w * (side * 0.063 - p[:, 0]) * 0.22
        p[:, 1] += w * (c[1] - p[:, 1]) * 0.12
        p[:, 2] += w * (c[2] - p[:, 2]) * 0.12

    # 3. Eyes: modest aperture increase, then lower the previously over-lifted
    # outer corners so the result reads as a soft almond rather than a cat-eye.
    lm = p[kl].copy()
    for ids, outer in ((np.arange(36, 42), 36), (np.arange(42, 48), 45)):
        c = lm[ids].mean(0)
        w = gauss(p, c, [0.034, 0.020, 0.027])
        p[:, 0] += w * (p[:, 0] - c[0]) * 0.05
        p[:, 1] += w * (p[:, 1] - c[1]) * 0.24
        wt = gauss(p, lm[outer], [0.014, 0.012, 0.018])
        p[:, 1] += 0.0035 * wt

    # Slightly open only the upper eyelids. This avoids the rounded/droopy
    # lower lid artifact seen in stronger eye-opening experiments.
    lm = p[kl].copy()
    for idx in (37, 38, 43, 44):
        p[:, 1] -= 0.00125 * gauss(p, lm[idx], [0.010, 0.008, 0.015])

    # 4. Soften heavy glabella/brow projection.
    lm = p[kl].copy()
    bc = lm[17:27].mean(0)
    p[:, 2] += 0.0028 * gauss(p, bc, [0.055, 0.030, 0.038])

    # 5. Nose: small, short and less projected in side/45 views.
    lm = p[kl].copy()
    bridge = lm[27:31].mean(0)
    tip = lm[30]
    base = lm[31:36].mean(0)
    w = gauss(p, bridge, [0.021, 0.035, 0.027])
    p[:, 2] += 0.0040 * w
    p[:, 0] += w * (bridge[0] - p[:, 0]) * 0.12
    w = gauss(p, tip, [0.020, 0.020, 0.022])
    p[:, 2] += 0.0085 * w
    p[:, 1] -= 0.0040 * w
    p[:, 0] += w * (tip[0] - p[:, 0]) * 0.18
    w = gauss(p, base, [0.025, 0.020, 0.025])
    p[:, 2] += 0.0055 * w
    p[:, 1] -= 0.0025 * w
    p[:, 0] += w * (base[0] - p[:, 0]) * 0.28

    # 6. Centered, softly full apple cheeks.
    for side in (-1.0, 1.0):
        c = np.array([side * 0.033, 0.002, -0.001])
        w = gauss(p, c, [0.040, 0.034, 0.038])
        p[:, 2] -= 0.0010 * w
        p[:, 0] += side * 0.0016 * w

    # 7. Wider, softer mouth and shorter philtrum. A second direct mouth-region
    # scale brings the real lip seam to ~45 mm in FaceVerse model coordinates,
    # close to the approved visual proportion relative to the eye width.
    lm = p[kl].copy()
    mc = lm[48:68].mean(0)
    w = gauss(p, mc, [0.044, 0.028, 0.030])
    p[:, 0] += w * (p[:, 0] - mc[0]) * 0.50
    p[:, 1] -= 0.0024 * w
    p[:, 2] -= 0.0008 * w
    p[:, 1] += w * (p[:, 1] - mc[1]) * 0.16
    for idx in (48, 54):
        c = lm[idx]
        wc = gauss(p, c, [0.016, 0.013, 0.020])
        p[:, 0] += np.sign(c[0]) * 0.0025 * wc

    lm = p[kl].copy()
    mc = lm[48:68].mean(0)
    wm = (
        np.exp(-0.5 * ((p[:, 1] - mc[1]) / 0.016) ** 2)
        * np.exp(-0.5 * ((p[:, 2] - mc[2]) / 0.022) ** 2)
        * np.exp(-0.5 * ((p[:, 0] - mc[0]) / 0.042) ** 2)
    )
    p[:, 0] += wm * (p[:, 0] - mc[0]) * 0.32
    lower_lip = lm[[56, 57, 58, 65, 66, 67]].mean(0)
    p[:, 2] -= 0.0005 * gauss(p, lower_lip, [0.030, 0.012, 0.020])

    # 8. Lower third: shorter and feminine, but avoid the pinched V of v15
    # experiments that were too aggressive.
    mouth_y = mc[1]
    lower = np.clip((p[:, 1] - mouth_y) / 0.060, 0.0, 1.0)
    lower *= np.exp(-0.5 * ((p[:, 2] - 0.018) / 0.075) ** 2)
    p[:, 0] *= 1.0 - 0.045 * lower
    p[:, 1] -= 0.0052 * lower
    for side in (-1.0, 1.0):
        c = np.array([side * 0.052, 0.045, 0.028])
        wj = gauss(p, c, [0.034, 0.035, 0.050])
        p[:, 0] -= side * 0.0018 * wj

    lm = p[kl].copy()
    chin = lm[8]
    wc = gauss(p, chin, [0.036, 0.027, 0.040])
    p[:, 2] += 0.0010 * wc
    p[:, 1] -= 0.0045 * wc
    p[:, 0] += wc * (chin[0] - p[:, 0]) * 0.10

    # 9. Remove the horizontal cheek/jaw ripples visible in v14 45/profile QA,
    # but never smooth the central eye/nose/mouth identity area.
    side_mask = (
        (p[:, 1] > -0.008)
        & (p[:, 1] < 0.065)
        & (np.abs(p[:, 0]) > 0.022)
        & (np.abs(p[:, 0]) < 0.075)
        & (p[:, 2] < 0.075)
    )
    for _ in range(6):
        avg = adj.dot(p) / degree[:, None]
        p[side_mask] = p[side_mask] * 0.76 + avg[side_mask] * 0.24

    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-full', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('output_v151'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(args.base_full, process=False, maintain_order=True)
    v0 = np.asarray(mesh.vertices, np.float64)
    f = np.asarray(mesh.faces, np.int64)
    v = v0.copy()

    comps = components(len(v), f)
    head = max(comps, key=len)
    head_mask = np.zeros(len(v), dtype=bool)
    head_mask[head] = True
    g2l = {int(q): i for i, q in enumerate(head)}
    if not np.all(head_mask[K]):
        raise RuntimeError('AINA FaceVerse semantic vertices are not on the main head shell')
    kl = np.array([g2l[int(q)] for q in K], dtype=np.int64)
    adj, degree = local_head_adjacency(f, head, g2l)

    p = refine_head(v[head].copy(), kl, adj, degree)
    v[head] = p

    # Keep eyeballs and oral pieces coherent with the edited skin shell.
    old = v0[K]
    new = v[K]
    eyes = sorted([q for q in comps if 650 < len(q) < 900], key=lambda q: v0[q].mean(0)[0])
    shifts = [
        new[36:42].mean(0) - old[36:42].mean(0),
        new[42:48].mean(0) - old[42:48].mean(0),
    ]
    for ids, shift in zip(eyes, shifts):
        v[ids] += shift
    mouth_shift = new[48:60].mean(0) - old[48:60].mean(0)
    for ids in comps:
        if np.array_equal(ids, head) or any(np.array_equal(ids, eye) for eye in eyes):
            continue
        v[ids] += mouth_shift

    out = trimesh.Trimesh(vertices=v, faces=f, process=False)
    for ext in ('obj', 'glb', 'ply'):
        out.export(args.out / f'AINA_FACEVERSE_FULL_v15.1_IDENTITY.{ext}')

    # Identity clay = actual edited head shell + actual eyeballs.
    keep = head_mask.copy()
    for ids in eyes:
        keep[ids] = True
    face_ids = np.flatnonzero(keep[f].all(1))
    clay = out.submesh([face_ids], append=True, repair=False)
    for ext in ('obj', 'glb', 'ply'):
        clay.export(args.out / f'AINA_FACEVERSE_IDENTITY_CLAY_v15.1.{ext}')

    d = v[head] - v0[head]
    lm = v[K]
    report = {
        'version': 'AINA Face Master v15.1 Real-Mesh Identity Refinement',
        'base': str(args.base_full),
        'topology_changed': False,
        'head_vertices': int(len(head)),
        'full_vertices': int(len(v)),
        'full_triangles': int(len(f)),
        'mouth_span_m': float(np.ptp(lm[48:60, 0])),
        'right_eye_span_m': float(np.ptp(lm[36:42, 0])),
        'left_eye_span_m': float(np.ptp(lm[42:48, 0])),
        'max_head_delta_m': float(np.linalg.norm(d, axis=1).max()),
        'rms_head_delta_m': float(np.sqrt(np.mean(np.sum(d * d, axis=1)))),
        'identity_lock': False,
        'candidate': True,
        'qa_gate': 'real naked-clay front + left/right 45 + left/right profile vs approved AINA effect-art references',
        'next': 'visual QA; only after pass may identity_lock become true and the head be grafted into the production VRM body',
    }
    (args.out / 'AINA_FACEVERSE_v15.1_REPORT.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
