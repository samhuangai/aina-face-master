#!/usr/bin/env python3
"""AINA v15.3 — real-mesh eyelid/eyeball + multi-view identity convergence.

Topology-preserving 3D edit only. Starts from the actual v15.2 FaceVerse mesh and
uses the approved sparse front landmarks only as a soft geometric guide. Visual
QA remains authoritative where sparse landmark extraction disagrees with the
approved artwork (especially eyelid vertical aperture).

v15.3 fixes a real component-depth error from v15.1/v15.2: the eyeball spheres
were progressively moved too far behind the lid surface, producing tiny dot-like
eyes in clay QA. The spheres are now re-centered behind the reshaped real lids.

Identity lock intentionally remains false until naked-clay front + both 45s +
both profiles visually match the approved AINA references.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components

K = np.array([
    1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,
    10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,
    10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,
    4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,6640,
    4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,
    6550,13695,12973,13422,6543,6537
], dtype=np.int64)


def components(nv: int, faces: np.ndarray):
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv))
    a = (a + a.T).tocsr()
    n, labels = connected_components(a, directed=False)
    return [np.flatnonzero(labels == i) for i in range(n)]


def falloff_ell(p, center, radii, inner=.45, outer=1.45):
    c = np.asarray(center, np.float64)
    r = np.asarray(radii, np.float64)
    q = np.sqrt(np.sum(((p - c) / r) ** 2, axis=1))
    w = np.zeros(len(p), np.float64)
    w[q <= inner] = 1.0
    m = (q > inner) & (q < outer)
    t = (q[m] - inner) / max(outer - inner, 1e-9)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * t))
    return w


def rbf_displace_xy(p, controls_xyz, disp_xy, sigma, strength=1.0, z_sigma=None):
    controls = np.asarray(controls_xyz, np.float64)
    disp = np.asarray(disp_xy, np.float64)
    sig2 = float(sigma) ** 2
    zsig2 = float(z_sigma if z_sigma is not None else sigma) ** 2
    out = np.zeros((len(p), 2), np.float64)
    for start in range(0, len(p), 4096):
        pp = p[start:start + 4096]
        dx = pp[:, None, 0] - controls[None, :, 0]
        dy = pp[:, None, 1] - controls[None, :, 1]
        dz = pp[:, None, 2] - controls[None, :, 2]
        q = (dx * dx + dy * dy) / (2.0 * sig2) + (dz * dz) / (2.0 * zsig2)
        w = np.exp(-q)
        sw = w.sum(axis=1)
        val = (w @ disp) / (sw[:, None] + 1e-12)
        env = 1.0 - np.exp(-sw * 1.2)
        out[start:start + len(pp)] = val * env[:, None] * float(strength)
    p[:, :2] += out
    return out


def axis_fit(lm_xy: np.ndarray, target_xy: np.ndarray):
    sx = np.cov(lm_xy[:, 0], target_xy[:, 0], bias=True)[0, 1] / np.var(lm_xy[:, 0])
    tx = target_xy[:, 0].mean() - sx * lm_xy[:, 0].mean()
    sy = np.cov(lm_xy[:, 1], target_xy[:, 1], bias=True)[0, 1] / np.var(lm_xy[:, 1])
    ty = target_xy[:, 1].mean() - sy * lm_xy[:, 1].mean()
    desired = np.column_stack([
        (target_xy[:, 0] - tx) / sx,
        (target_xy[:, 1] - ty) / sy,
    ])
    return desired, (float(sx), float(sy), float(tx), float(ty))


def target_rms_px(lm_xy: np.ndarray, target_xy: np.ndarray):
    _, (sx, sy, tx, ty) = axis_fit(lm_xy, target_xy)
    pred = np.column_stack([sx * lm_xy[:, 0] + tx, sy * lm_xy[:, 1] + ty])
    return float(np.sqrt(np.mean(np.sum((target_xy - pred) ** 2, axis=1))))


def sculpt_head(p: np.ndarray, kl: np.ndarray, target_xy: np.ndarray):
    p0 = p.copy()
    lm = p[kl].copy()
    desired, _ = axis_fit(lm[:, :2], target_xy)

    for ids, sigma, strength, zsig in (
        (np.arange(0, 17), .0115, .50, .020),
        (np.arange(17, 27), .0090, .20, .014),
        (np.arange(27, 36), .0075, .35, .012),
        (np.arange(48, 68), .0068, .55, .010),
    ):
        lm = p[kl].copy()
        rbf_displace_xy(p, lm[ids], desired[ids] - lm[ids, :2], sigma=sigma, strength=strength, z_sigma=zsig)

    lm = p[kl].copy()
    for ids in (np.arange(36, 42), np.arange(42, 48)):
        des = desired[ids].copy()
        des[:, 1] = lm[ids, 1] + .06 * (des[:, 1] - lm[ids, 1])
        rbf_displace_xy(p, lm[ids], des - lm[ids, :2], sigma=.0052, strength=.66, z_sigma=.009)

    lm = p[kl].copy()
    ids = np.arange(48, 60)
    mc = lm[ids].mean(0)
    tar = lm[ids, :2].copy()
    tar[:, 0] = mc[0] + 1.055 * (lm[ids, 0] - mc[0])
    tar[:, 1] = mc[1] + .92 * (lm[ids, 1] - mc[1])
    rbf_displace_xy(p, lm[ids], tar - lm[ids, :2], sigma=.0055, strength=.55, z_sigma=.009)

    lm = p[kl].copy()
    mc = lm[48:60].mean(0)
    p[:, 2] += .0021 * falloff_ell(p, mc, [.036, .026, .030], .48, 1.45)
    for sign in (-1.0, 1.0):
        c = np.array([sign * .032, .011, .006])
        p[:, 2] += .0013 * falloff_ell(p, c, [.032, .033, .038], .48, 1.42)
    chin = lm[8]
    p[:, 2] += .0018 * falloff_ell(p, chin, [.032, .028, .038], .48, 1.38)
    tip = lm[30]
    p[:, 2] -= .00055 * falloff_ell(p, tip, [.015, .016, .017], .50, 1.25)

    for sign in (-1.0, 1.0):
        c = np.array([sign * .069, -.004, .030])
        w = falloff_ell(p, c, [.020, .032, .030], .40, 1.25)
        p[:, 0] += w * (sign * .064 - p[:, 0]) * .17
        p[:, 1] += w * (-.001 - p[:, 1]) * .04

    lm = p[kl].copy()
    for ids in (np.arange(36, 42), np.arange(42, 48)):
        c = lm[ids].mean(0)
        tar = lm[ids, :2].copy()
        tar[:, 1] = c[1] + 1.10 * (lm[ids, 1] - c[1])
        rbf_displace_xy(p, lm[ids], tar - lm[ids, :2], sigma=.0044, strength=.78, z_sigma=.0075)

    lm = p[kl].copy()
    ids = np.arange(48, 60)
    mc = lm[ids].mean(0)
    tar = lm[ids, :2].copy()
    tar[:, 0] = mc[0] + 1.04 * (lm[ids, 0] - mc[0])
    rbf_displace_xy(p, lm[ids], tar - lm[ids, :2], sigma=.0055, strength=.65, z_sigma=.009)

    lm = p[kl].copy()
    mc = lm[48:60].mean(0)
    p[:, 2] += .0010 * falloff_ell(p, mc, [.036, .026, .030], .48, 1.45)
    for sign in (-1.0, 1.0):
        c = np.array([sign * .032, .011, .006])
        p[:, 2] += .0006 * falloff_ell(p, c, [.032, .033, .038], .48, 1.42)
    chin = lm[8]
    p[:, 2] += .0005 * falloff_ell(p, chin, [.032, .028, .038], .48, 1.38)
    w = falloff_ell(p, chin, [.030, .026, .036], .45, 1.30)
    p[:, 0] += w * (-p[:, 0]) * .025

    return p, p - p0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-full', type=Path, required=True)
    ap.add_argument('--target-landmarks', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('output_v153'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    target_xy = np.asarray(json.loads(args.target_landmarks.read_text())['landmarks_xy'], np.float64)
    if target_xy.shape != (68, 2):
        raise RuntimeError(f'expected 68x2 target landmarks, got {target_xy.shape}')

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

    old_lm = v[K].copy()
    p, head_delta = sculpt_head(v[head].copy(), kl, target_xy)
    v[head] = p
    new_lm = v[K].copy()

    eyes = sorted([q for q in comps if 650 < len(q) < 900], key=lambda q: v0[q].mean(0)[0])
    if len(eyes) != 2:
        raise RuntimeError(f'expected exactly 2 eyeball components, got {len(eyes)}')
    eye_lms = (new_lm[36:42], new_lm[42:48])
    eye_centers_before, eye_centers_after = [], []
    for ids, eye_lm in zip(eyes, eye_lms):
        c = v[ids].mean(0)
        eye_centers_before.append(c.tolist())
        rim = eye_lm.mean(0)
        target_c = np.array([rim[0], rim[1], rim[2] + .0080])
        v[ids] += target_c - c
        eye_centers_after.append(v[ids].mean(0).tolist())

    mouth_shift = new_lm[48:60].mean(0) - old_lm[48:60].mean(0)
    for ids in comps:
        if np.array_equal(ids, head) or any(np.array_equal(ids, eye) for eye in eyes):
            continue
        v[ids] += mouth_shift

    out = trimesh.Trimesh(vertices=v, faces=f, process=False)
    for ext in ('obj', 'glb', 'ply'):
        out.export(args.out / f'AINA_FACEVERSE_FULL_v15.3_IDENTITY.{ext}')

    keep = head_mask.copy()
    for ids in eyes:
        keep[ids] = True
    face_ids = np.flatnonzero(keep[f].all(1))
    clay = out.submesh([face_ids], append=True, repair=False)
    for ext in ('obj', 'glb', 'ply'):
        clay.export(args.out / f'AINA_FACEVERSE_IDENTITY_CLAY_v15.3.{ext}')

    lm = v[K]
    report = {
        'version': 'AINA Face Master v15.3 Real-Mesh Eye/Profile Convergence',
        'base': str(args.base_full),
        'target_landmarks': str(args.target_landmarks),
        'topology_changed': False,
        'head_vertices': int(len(head)),
        'full_vertices': int(len(v)),
        'full_triangles': int(len(f)),
        'target_landmark_rms_px': target_rms_px(lm[:, :2], target_xy),
        'mouth_span_m': float(np.ptp(lm[48:60, 0])),
        'right_eye_span_xy_m': np.ptp(lm[36:42, :2], axis=0).tolist(),
        'left_eye_span_xy_m': np.ptp(lm[42:48, :2], axis=0).tolist(),
        'eye_center_distance_m': float(abs(lm[42:48, 0].mean() - lm[36:42, 0].mean())),
        'chin_to_mouth_center_m': float(lm[8, 1] - lm[48:60, 1].mean()),
        'max_head_delta_from_v152_m': float(np.linalg.norm(head_delta, axis=1).max()),
        'rms_head_delta_from_v152_m': float(np.sqrt(np.mean(np.sum(head_delta * head_delta, axis=1)))),
        'eye_centers_before': eye_centers_before,
        'eye_centers_after': eye_centers_after,
        'eyeball_depth_policy': 'sphere center = current lid centroid depth + 0.008 m',
        'sparse_eye_vertical_override': True,
        'identity_lock': False,
        'candidate': True,
        'qa_gate': 'real naked-clay front + left/right 45 + left/right profile vs approved AINA references',
        'note': 'No new AINA effect/reference image is generated. v15.3 edits only the actual 3D mesh and real eyeball components.',
        'next': 'visual QA; continue v15.x until likeness gate passes, then freeze identity and graft into production VRM body',
    }
    (args.out / 'AINA_FACEVERSE_v15.3_REPORT.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
