#!/usr/bin/env python3
"""AINA v15.2 — real-mesh adult proportion + profile convergence pass.

Starts from the actual v15.1 full FaceVerse mesh. This is a topology-preserving
3D edit only; it does not generate or replace any AINA reference artwork.

Changes are intentionally continuous-region edits instead of sparse hard pulls:
- lengthen the facial mask to remove the toddler/short-lower-third read
- reduce mid/lower-face width and cheek fullness
- bring eye centers inward and reduce eye aperture without collapsing the orbit
- refine nose width/projection while preserving a short delicate tip
- soften/lower/retract the mouth and perioral muzzle
- retract cheeks/chin in 45/profile views
- tuck ears and slightly reduce upper-cranium width

Identity lock remains false until naked-clay front + both 45 + both profiles
visually converge to the approved AINA references.
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


def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def falloff_ell(p, center, radii, inner=.5, outer=1.45):
    c = np.asarray(center, np.float64)
    r = np.asarray(radii, np.float64)
    q = np.sqrt(np.sum(((p - c) / r) ** 2, axis=1))
    w = np.zeros(len(p), np.float64)
    w[q <= inner] = 1.0
    m = (q > inner) & (q < outer)
    t = (q[m] - inner) / max(outer - inner, 1e-9)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * t))
    return w


def rbf_displace_xy(p, controls_xyz, disp_xy, sigma, strength=1.0, envelope_scale=1.0, z_sigma=None):
    controls = np.asarray(controls_xyz, np.float64)
    disp = np.asarray(disp_xy, np.float64)
    sig2 = float(sigma) ** 2
    out = np.zeros((len(p), 2), np.float64)
    for start in range(0, len(p), 4096):
        pp = p[start:start + 4096]
        dx = pp[:, None, 0] - controls[None, :, 0]
        dy = pp[:, None, 1] - controls[None, :, 1]
        dz = pp[:, None, 2] - controls[None, :, 2]
        if z_sigma is None:
            q = (dx * dx + dy * dy + dz * dz) / (2.0 * sig2)
        else:
            q = (dx * dx + dy * dy) / (2.0 * sig2) + (dz * dz) / (2.0 * float(z_sigma) ** 2)
        w = np.exp(-q)
        sw = w.sum(axis=1)
        val = (w @ disp) / (sw[:, None] + 1e-12)
        env = 1.0 - np.exp(-sw * float(envelope_scale))
        out[start:start + len(pp)] = val * env[:, None] * float(strength)
    p[:, :2] += out
    return out


def refine_head(p, kl):
    p0 = p.copy()
    lm0 = p[kl].copy()
    zface = float(np.median(lm0[:, 2]))

    wf = sigmoid((zface + .032 - p[:, 2]) / .008)
    wx = sigmoid((.078 - np.abs(p[:, 0])) / .006)
    wy = sigmoid((.080 - np.abs(p[:, 1] - .006)) / .008)
    wface = wf * wx * wy
    y0 = .010
    p[:, 1] += wface * ((y0 + 1.16 * (p[:, 1] - y0)) - p[:, 1])

    my = np.clip((p[:, 1] + .010) / .060, 0.0, 1.0)
    side = np.clip((np.abs(p[:, 0]) - .018) / .045, 0.0, 1.0)
    zmask = sigmoid((zface + .032 - p[:, 2]) / .010)
    p[:, 0] *= 1.0 - .055 * my * side * zmask

    lm = p[kl].copy()
    for ids, sign in ((np.arange(36, 42), -1.0), (np.arange(42, 48), 1.0)):
        c = lm[ids].mean(0)
        tc = c.copy()
        tc[0] = sign * .0312
        tc[1] = -.0310
        target = lm[ids, :2].copy()
        target[:, 0] = tc[0] + .84 * (lm[ids, 0] - c[0])
        target[:, 1] = tc[1] + .72 * (lm[ids, 1] - c[1])
        rbf_displace_xy(p, lm[ids], target - lm[ids, :2], sigma=.0066,
                        strength=.88, envelope_scale=1.15, z_sigma=.010)

    lm = p[kl].copy()
    bridge = lm[27:31].mean(0)
    tip = lm[30]
    base = lm[31:36].mean(0)
    for c, r, fac in (
        (bridge, [.019, .033, .026], .06),
        (tip,    [.017, .018, .021], .10),
        (base,   [.025, .020, .027], .14),
    ):
        w = falloff_ell(p, c, r, .5, 1.4)
        p[:, 0] += w * (c[0] - p[:, 0]) * fac
    for c, r, amp in (
        (bridge, [.019, .032, .025], .0012),
        (tip,    [.017, .018, .020], .0022),
        (base,   [.024, .020, .025], .0018),
    ):
        p[:, 2] += amp * falloff_ell(p, c, r, .5, 1.4)
    p[:, 1] -= .0010 * falloff_ell(p, tip,  [.018, .019, .022], .5, 1.35)
    p[:, 1] -= .0007 * falloff_ell(p, base, [.026, .021, .027], .5, 1.35)

    lm = p[kl].copy()
    ids = np.arange(48, 60)
    mc = lm[ids].mean(0)
    target = lm[ids, :2].copy()
    target[:, 0] = mc[0] + .95 * (lm[ids, 0] - mc[0])
    target[:, 1] = (mc[1] + .0018) + .88 * (lm[ids, 1] - mc[1])
    rbf_displace_xy(p, lm[ids], target - lm[ids, :2], sigma=.0065,
                    strength=.75, envelope_scale=1.0, z_sigma=.010)
    lm = p[kl].copy()
    mc = lm[48:60].mean(0)
    p[:, 2] += .0028 * falloff_ell(p, mc, [.038, .026, .035], .52, 1.45)

    for sign in (-1.0, 1.0):
        c = np.array([sign * .036, .010, .003])
        p[:, 2] += .0015 * falloff_ell(p, c, [.035, .032, .043], .5, 1.45)
    lm = p[kl].copy()
    chin = lm[8]
    p[:, 2] += .0018 * falloff_ell(p, chin, [.034, .030, .043], .5, 1.4)

    for sign in (-1.0, 1.0):
        c = np.array([sign * .074, -.004, .028])
        w = falloff_ell(p, c, [.019, .034, .030], .45, 1.25)
        p[:, 0] += w * (sign * .064 - p[:, 0]) * .18
        p[:, 2] += w * (.033 - p[:, 2]) * .07
    ws = np.clip((-p[:, 1] - .055) / .065, 0.0, 1.0)
    p[:, 0] *= 1.0 - .012 * ws

    wy2 = sigmoid((p[:, 1] + .042) / .008) * sigmoid((.073 - p[:, 1]) / .008)
    wx2 = sigmoid((.074 - np.abs(p[:, 0])) / .006)
    wz2 = sigmoid((zface + .030 - p[:, 2]) / .008)
    p[:, 0] *= 1.0 - .095 * wy2 * wx2 * wz2

    lm = p[kl].copy()
    ids = np.arange(48, 60)
    mc = lm[ids].mean(0)
    target = lm[ids, :2].copy()
    target[:, 0] = mc[0] + 1.04 * (lm[ids, 0] - mc[0])
    rbf_displace_xy(p, lm[ids], target - lm[ids, :2], sigma=.0058,
                    strength=.55, envelope_scale=1.0, z_sigma=.009)

    lm = p[kl].copy()
    mc = lm[48:60].mean(0)
    chin = lm[8]
    p[:, 2] += .0024 * falloff_ell(p, mc, [.038, .028, .036], .5, 1.5)
    p[:, 2] += .0026 * falloff_ell(p, chin, [.034, .031, .043], .5, 1.5)
    for sign in (-1.0, 1.0):
        c = np.array([sign * .031, .012, .004])
        p[:, 2] += .0018 * falloff_ell(p, c, [.032, .032, .042], .5, 1.45)

    lm = p[kl].copy()
    base = lm[31:36].mean(0)
    tip = lm[30]
    for c, r, fac in (
        (base, [.024, .019, .026], .08),
        (tip,  [.017, .018, .021], .05),
    ):
        w = falloff_ell(p, c, r, .5, 1.35)
        p[:, 0] += w * (c[0] - p[:, 0]) * fac

    return p, p - p0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-full', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('output_v152'))
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

    p, delta = refine_head(v[head].copy(), kl)
    v[head] = p

    old = v0[K]
    new = v[K]
    eyes = sorted([q for q in comps if 650 < len(q) < 900], key=lambda q: v0[q].mean(0)[0])
    shifts = [
        new[36:42].mean(0) - old[36:42].mean(0),
        new[42:48].mean(0) - old[42:48].mean(0),
    ]
    for ids, shift in zip(eyes, shifts):
        v[ids] += shift
        v[ids, 2] += .0022

    mouth_shift = new[48:60].mean(0) - old[48:60].mean(0)
    for ids in comps:
        if np.array_equal(ids, head) or any(np.array_equal(ids, eye) for eye in eyes):
            continue
        v[ids] += mouth_shift

    out = trimesh.Trimesh(vertices=v, faces=f, process=False)
    for ext in ('obj', 'glb', 'ply'):
        out.export(args.out / f'AINA_FACEVERSE_FULL_v15.2_IDENTITY.{ext}')

    keep = head_mask.copy()
    for ids in eyes:
        keep[ids] = True
    face_ids = np.flatnonzero(keep[f].all(1))
    clay = out.submesh([face_ids], append=True, repair=False)
    for ext in ('obj', 'glb', 'ply'):
        clay.export(args.out / f'AINA_FACEVERSE_IDENTITY_CLAY_v15.2.{ext}')

    lm = v[K]
    report = {
        'version': 'AINA Face Master v15.2 Real-Mesh Adult/Profile Convergence',
        'base': str(args.base_full),
        'topology_changed': False,
        'head_vertices': int(len(head)),
        'full_vertices': int(len(v)),
        'full_triangles': int(len(f)),
        'mouth_span_m': float(np.ptp(lm[48:60, 0])),
        'right_eye_span_m': float(np.ptp(lm[36:42, 0])),
        'left_eye_span_m': float(np.ptp(lm[42:48, 0])),
        'eye_center_distance_m': float(abs(lm[42:48, 0].mean() - lm[36:42, 0].mean())),
        'chin_to_mouth_center_m': float(lm[8, 1] - lm[48:60, 1].mean()),
        'max_head_delta_from_v151_m': float(np.linalg.norm(delta, axis=1).max()),
        'rms_head_delta_from_v151_m': float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
        'identity_lock': False,
        'candidate': True,
        'qa_gate': 'real naked-clay front + left/right 45 + left/right profile vs approved AINA references',
        'note': 'v15.2 replaces hard sparse pulls with continuous facial-region deformations; no new reference/effect image is generated.',
        'next': 'visual QA; continue v15.x until real mesh passes likeness gate, then lock identity and graft into production VRM body',
    }
    (args.out / 'AINA_FACEVERSE_v15.2_REPORT.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
