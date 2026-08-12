#!/usr/bin/env python3
"""AINA v10.8.0 — female semantic base + Laplacian-preserving likeness sculpt.

v10.7 proved that simply driving sparse points with local Gaussian warps can
reduce landmark error while damaging facial anatomy. v10.8 starts from a
reproducible FEMALE/ASIAN semantic GNM identity blend, then solves a weighted
Laplacian mesh-deformation problem against the approved front / 3Q / profile
references. Original GNM topology is preserved.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import face_alignment
import numpy as np
import trimesh
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import lsqr

import fit_aina_v101 as core
from gnm.shape import gnm_numpy, gnm_landmarks
from gnm.shape.semantic_sampler import IdentitySampler, Gender, Ethnicity

GNM_TO_STANDARD = np.array([
    0,1,6,5,4,3,2,7,8,9,10,11,12,13,14,15,16,
    *range(17,68)
], dtype=np.int64)

# Reproducible picks from the successful 384-sample FEMALE/ASIAN search.
# Rank 1 (sample 264) is the best raw front-landmark fit; rank 3 (sample 290)
# contributes the narrower lower-face silhouette seen in the approved AINA art.
BASE_SAMPLE_A = 264
BASE_SAMPLE_B = 290
BASE_BLEND_A = 0.65
BASE_BLEND_B = 0.35
RNG_SEED = 20260812


def view_weights(name: str) -> np.ndarray:
    w = np.ones(68, np.float64)
    if name == 'front':
        w[0:17] = 4.2
        w[17:27] = 1.25
        w[27:36] = 4.0
        w[36:48] = 5.6
        w[48:60] = 4.8
        w[60:68] = 3.4
        return 4.0 * w
    if name == 'three_quarter':
        w[0:17] = 2.0
        w[17:27] = 0.70
        w[27:36] = 2.6
        w[36:48] = 2.5
        w[48:60] = 2.3
        w[60:68] = 1.5
        return 1.35 * w
    # Strict profile is used primarily for depth, nose projection, lips, chin,
    # and the visible lower-face contour. Hidden-side detector points are weak.
    w[:] = 0.045
    w[0:17] = 0.36
    w[5:12] = 1.15
    w[27:36] = 2.25
    w[48:60] = 1.05
    w[36:42] = 0.20
    return 0.62 * w


def control_strengths() -> np.ndarray:
    s = np.full(68, 55.0, np.float64)
    s[0:17] = 110.0       # jaw / cheek silhouette
    s[17:27] = 42.0       # brows should follow more softly
    s[27:36] = 125.0      # nose
    s[36:48] = 175.0      # eye opening / eye corners are identity-critical
    s[48:60] = 140.0      # outer lips
    s[60:68] = 105.0      # inner mouth
    return s


def load_targets(args):
    refs = {
        'front': core.load_image_rgb(args.front),
        'three_quarter': core.load_image_rgb(args.three_quarter),
        'side': core.load_image_rgb(args.side),
    }
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device='cpu',
        face_detector='sfd',
    )
    front_json = json.loads(args.front_landmarks.read_text())
    target_px = {
        'front': np.asarray(front_json['landmarks_xy'], dtype=np.float32),
        'three_quarter': core.detect_68(fa, refs['three_quarter']),
        'side': core.detect_68(fa, refs['side']),
    }
    target = {
        name: core.normalize_target(target_px[name], refs[name].shape)
        for name in core.VIEW_ORDER
    }
    return refs, target_px, target


def semantic_base_identity() -> np.ndarray:
    sampler = IdentitySampler()
    rng = np.random.default_rng(RNG_SEED)
    identities = np.asarray(
        sampler.sample_identity(
            Gender.FEMALE,
            Ethnicity.ASIAN,
            num_samples=384,
            rng=rng,
        ),
        dtype=np.float64,
    )
    return BASE_BLEND_A * identities[BASE_SAMPLE_A] + BASE_BLEND_B * identities[BASE_SAMPLE_B]


def control_points(vertices: np.ndarray, idx_std: np.ndarray, bw_std: np.ndarray) -> np.ndarray:
    return (vertices[idx_std] * bw_std[..., None]).sum(axis=-2)


def anchor_vertices(idx_std: np.ndarray, bw_std: np.ndarray) -> np.ndarray:
    best = np.argmax(bw_std, axis=-1)
    return np.take_along_axis(idx_std, best[..., None], axis=-1).squeeze(-1).astype(np.int64)


def fit_cameras(points3: np.ndarray, target: dict[str, np.ndarray]):
    cams = {}
    for name in core.VIEW_ORDER:
        cams[name] = core.scaled_ortho_init(points3, target[name], view_weights(name))
    return cams


def triangulate_desired(current: np.ndarray, target: dict[str,np.ndarray], cameras) -> np.ndarray:
    desired = np.zeros_like(current)
    # Front dominates likeness; the small 3D prior prevents noisy profile points
    # from inventing implausible depth in hidden regions.
    prior = 0.030
    for li in range(68):
        rows = []
        rhs = []
        for name in core.VIEW_ORDER:
            r, scale, trans = cameras[name]
            w = float(view_weights(name)[li])
            if w <= 1e-9:
                continue
            sw = math.sqrt(w)
            for d in range(2):
                rows.append(sw * scale * r[d])
                rhs.append(sw * (target[name][li, d] - trans[d]))
        for d in range(3):
            row = np.zeros(3, np.float64)
            row[d] = math.sqrt(prior)
            rows.append(row)
            rhs.append(math.sqrt(prior) * current[li, d])
        desired[li], *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    return desired


def uniform_laplacian(nv: int, faces: np.ndarray) -> sparse.csr_matrix:
    e = np.vstack([
        faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]],
        faces[:, [1,0]], faces[:, [2,1]], faces[:, [0,2]],
    ])
    data = np.ones(len(e), np.float64)
    adj = sparse.coo_matrix((data, (e[:,0], e[:,1])), shape=(nv,nv)).tocsr()
    adj.data[:] = 1.0
    adj.eliminate_zeros()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.zeros_like(deg)
    good = deg > 0
    inv[good] = 1.0 / deg[good]
    return sparse.eye(nv, format='csr') - sparse.diags(inv) @ adj


def choose_pins(vertices: np.ndarray, controls: np.ndarray, face_scale: float) -> np.ndarray:
    # Anything sufficiently far from sparse facial controls belongs to scalp,
    # back-head, ear outskirts, or neck and should not drift with facial edits.
    dmin = np.full(len(vertices), np.inf, np.float64)
    for c in controls:
        dmin = np.minimum(dmin, np.linalg.norm(vertices - c, axis=1))
    far = np.flatnonzero(dmin > 0.24 * face_scale)
    # A deterministic subset is enough because Laplacian coordinates couple the
    # surface; keeping all far vertices would overconstrain and slow the solve.
    return far[::5]


def build_constraint_rows(nv: int, ids: np.ndarray, weights: np.ndarray) -> sparse.csr_matrix:
    rows = np.arange(len(ids), dtype=np.int64)
    return sparse.coo_matrix((np.sqrt(weights), (rows, ids)), shape=(len(ids), nv)).tocsr()


def laplacian_step(
    current: np.ndarray,
    base: np.ndarray,
    faces: np.ndarray,
    anchor_ids: np.ndarray,
    anchor_targets: np.ndarray,
    pins: np.ndarray,
    L: sparse.csr_matrix,
) -> np.ndarray:
    smooth_weight = 1.0
    ctrl = build_constraint_rows(len(current), anchor_ids, control_strengths())
    pin_weight = np.full(len(pins), 10.0, np.float64)
    pin = build_constraint_rows(len(current), pins, pin_weight)
    A = sparse.vstack([smooth_weight * L, ctrl, pin], format='csr')

    lap_rhs = smooth_weight * (L @ current)
    out = np.empty_like(current)
    for d in range(3):
        b = np.concatenate([
            lap_rhs[:, d],
            np.sqrt(control_strengths()) * anchor_targets[:, d],
            np.sqrt(pin_weight) * base[pins, d],
        ])
        sol = lsqr(A, b, atol=2e-7, btol=2e-7, iter_lim=320, show=False)
        out[:, d] = sol[0]
    return out


def sculpt(base_vertices, faces, idx_std, bw_std, target, outer_steps=4):
    base = base_vertices.copy()
    current = base.copy()
    controls0 = control_points(base, idx_std, bw_std)
    face_scale = max(float(np.linalg.norm(np.ptp(controls0, axis=0))), 1e-6)
    anchors = anchor_vertices(idx_std, bw_std)
    pins = choose_pins(base, controls0, face_scale)
    L = uniform_laplacian(len(base), faces)
    history = []

    group_cap = np.full(68, 0.050 * face_scale, np.float64)
    group_cap[0:17] = 0.060 * face_scale
    group_cap[17:27] = 0.035 * face_scale
    group_cap[27:36] = 0.050 * face_scale
    group_cap[36:48] = 0.045 * face_scale
    group_cap[48:68] = 0.045 * face_scale

    gains = [0.78, 0.62, 0.48, 0.36]
    for outer in range(outer_steps):
        controls = control_points(current, idx_std, bw_std)
        cameras = fit_cameras(controls, target)
        desired = triangulate_desired(controls, target, cameras)
        residual = desired - controls
        mag = np.linalg.norm(residual, axis=1)
        clip = np.minimum(1.0, group_cap / np.maximum(mag, 1e-10))
        residual *= clip[:, None]

        gain = gains[min(outer, len(gains)-1)]
        # The landmark is barycentric; moving its strongest support vertex by
        # the residual gives a stable soft control without tearing triangles.
        anchor_targets = current[anchors] + gain * residual
        current = laplacian_step(current, base, faces, anchors, anchor_targets, pins, L)

        after = control_points(current, idx_std, bw_std)
        rec = {'outer': outer, 'gain': gain}
        for name in core.VIEW_ORDER:
            r,s,t = fit_cameras(after, target)[name]
            pred = core.project_np(after, r, s, t)
            e = np.linalg.norm(pred - target[name], axis=1)
            rec[f'{name}_rmse'] = float(np.sqrt(np.mean(e**2)))
        rec['max_vertex_shift_from_base'] = float(np.max(np.linalg.norm(current-base, axis=1)))
        history.append(rec)
        print(json.dumps(rec))
    return current, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--front', type=Path, required=True)
    ap.add_argument('--three-quarter', type=Path, required=True)
    ap.add_argument('--side', type=Path, required=True)
    ap.add_argument('--front-landmarks', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('output_v1080'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / 'QA'
    qa.mkdir(exist_ok=True)

    refs, target_px, target = load_targets(args)
    g = gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    identity = semantic_base_identity()
    base_vertices = np.asarray(g(identity=identity[None,:]))[0].astype(np.float64)
    faces = np.asarray(g.triangles, dtype=np.int64)

    cfg = gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
    idx = np.asarray(cfg.indices, dtype=np.int64)[GNM_TO_STANDARD]
    bw = np.asarray(cfg.weights, dtype=np.float64)[GNM_TO_STANDARD]

    base_controls = control_points(base_vertices, idx, bw)
    base_cams = fit_cameras(base_controls, target)
    baseline_errors = {}
    for name in core.VIEW_ORDER:
        r,s,t = base_cams[name]
        p = core.project_np(base_controls, r, s, t)
        e = np.linalg.norm(p-target[name], axis=1)
        baseline_errors[name] = float(np.sqrt(np.mean(e**2)))

    vertices, history = sculpt(base_vertices, faces, idx, bw, target, outer_steps=4)
    full = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    full.export(args.out/'AINA_FACE_MASTER_GNM_v10.8.0_FEMALE_LAPLACIAN.obj')
    full.export(args.out/'AINA_FACE_MASTER_GNM_v10.8.0_FEMALE_LAPLACIAN.glb')

    skin_ti = np.asarray(g.triangle_indices_for_group('skin'), dtype=np.int64)
    skin = full.submesh([skin_ti], append=True, repair=False)
    skin.remove_unreferenced_vertices()
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.8.0_FEMALE_LAPLACIAN.obj')
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.8.0_FEMALE_LAPLACIAN.glb')
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.8.0_FEMALE_LAPLACIAN.ply')
    np.save(args.out/'AINA_identity_base_v10.8.0.npy', identity.astype(np.float32))

    final_controls = control_points(vertices, idx, bw)
    cameras = fit_cameras(final_controls, target)
    errors = {}
    cameras_json = {}
    for name in core.VIEW_ORDER:
        r,s,t = cameras[name]
        pred = core.project_np(final_controls, r, s, t)
        e = np.linalg.norm(pred-target[name], axis=1)
        errors[name] = {
            'rmse': float(np.sqrt(np.mean(e**2))),
            'median': float(np.median(e)),
            'p90': float(np.percentile(e,90)),
        }
        cameras_json[name] = {'rotation_rows':r.tolist(),'scale':float(s),'translation':t.tolist(),**errors[name]}
        core.save_overlay(refs[name], target_px[name], pred, qa/f'AINA_{name}_overlay_v10.8.0.png', f'AINA v10.8.0 {name}')

    R = cameras['front'][0]
    sv = np.asarray(skin.vertices)
    sf = np.asarray(skin.faces)
    paths = []
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p = qa/f'AINA_CLAY_{label}_v10.8.0.png'
        core.render_mesh_ortho(sv,sf,R,yaw,p,f'AINA v10.8.0 {label}')
        paths.append(p)
    ims=[Image.open(p).convert('RGB') for p in paths]
    H=max(x.height for x in ims); W=max(x.width for x in ims)
    sheet=Image.new('RGB',(W*5,H),'white')
    for i,im in enumerate(ims):
        sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_CLAY_5VIEW_v10.8.0.png')

    report = {
        'version':'AINA Face Master v10.8.0 Female Base Laplacian Sculpt',
        'base': {
            'semantic':'FEMALE / ASIAN',
            'rng_seed':RNG_SEED,
            'sample_indices':[BASE_SAMPLE_A,BASE_SAMPLE_B],
            'blend_weights':[BASE_BLEND_A,BASE_BLEND_B],
        },
        'topology_changed':False,
        'method':'multi-view sparse 3D targets + weighted Laplacian differential-coordinate preservation',
        'baseline_errors':baseline_errors,
        'errors':errors,
        'history':history,
        'identity_lock':False,
        'note':'Visual likeness, not sparse RMSE alone, decides whether this version can be locked.'
    }
    (args.out/'AINA_v10.8.0_REPORT.json').write_text(json.dumps(report,indent=2))
    (args.out/'AINA_CAMERAS_v10.8.0.json').write_text(json.dumps(cameras_json,indent=2))
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
