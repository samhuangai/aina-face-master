#!/usr/bin/env python3
"""AINA Face Master v10.5 — topology-native eye opening + final nose/jaw lock.

Uses the first 200 GNM expression blendshape dimensions (left/right eye regions)
as a topology-native deformation basis. The solver fits only the approved AINA
eye contour, with lower-face expression dimensions fixed at zero. The resulting
eye displacement is baked into the neutral v10.4 mesh, then the remaining nose
and V-jaw residuals are closed with smooth compact semantic fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import face_alignment
import numpy as np
from PIL import Image
from scipy.optimize import lsq_linear
import trimesh

from gnm.shape import gnm_numpy, gnm_landmarks
import identity_lock_v104 as h


def detect68(fa, im):
    hh, ww = im.shape[:2]
    s = max(1.0, 720.0 / max(hh, ww))
    work = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else im
    preds = fa.get_landmarks_from_image(work)
    if not preds:
        raise RuntimeError("No face detected")
    ctr = np.array([work.shape[1] * .5, work.shape[0] * .5], dtype=np.float64)
    q = min(preds, key=lambda p: np.linalg.norm(np.asarray(p)[:, :2].mean(0) - ctr))
    return np.asarray(q, dtype=np.float64)[:, :2] / s


def norm_target(p, shape):
    hh, ww = shape[:2]
    s = .5 * max(ww, hh)
    return (p - np.array([ww * .5, hh * .5])) / s


def build_eye_system(current_cam, target_cam, eye_basis_cam):
    rows, rhs = [], []
    # Eye contour only. Vertical lid motion is intentionally weighted much more
    # strongly than x so the solver opens the fissure without stretching sockets.
    for i in range(36, 48):
        corner = i in (36, 39, 42, 45)
        for axis in (0, 1):
            if axis == 1:
                w = 3.2 if not corner else 1.55
            else:
                w = 1.55 if corner else .55
            rows.append(eye_basis_cam[:, i, axis] * w)
            rhs.append((target_cam[i, axis] - current_cam[i, axis]) * w)
    return np.stack(rows, axis=0), np.asarray(rhs, dtype=np.float64)


def eye_error(vertices, idx, bw, R, target_cam):
    l = h.lm(vertices, idx, bw) @ R.T
    ids = np.arange(36, 48)
    diff = l[ids, :2] - target_cam[ids]
    # Emphasize vertical aperture and corners.
    weights = np.ones((len(ids), 2), dtype=np.float64)
    for k, i in enumerate(ids):
        weights[k, 1] = 2.2 if i not in (36,39,42,45) else 1.25
        weights[k, 0] = 1.25 if i in (36,39,42,45) else .55
    return float(np.sqrt(np.mean((diff * weights) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-full", type=Path, required=True)
    ap.add_argument("--front", type=Path, required=True)
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("output_v105"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)

    ref = np.asarray(Image.open(args.front).convert("RGB"))
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu", face_detector="sfd")
    target_px = detect68(fa, ref)
    target_norm = norm_target(target_px, ref.shape)
    cams = json.loads(args.cameras.read_text())
    cam = cams["front"]
    R = np.asarray(cam["rotation_rows"], dtype=np.float64)
    target_cam = h.symmetrize(h.target_cam(target_norm, cam))

    g = gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    base_mesh = trimesh.load(args.base_full, process=False)
    vertices = np.asarray(base_mesh.vertices, dtype=np.float64)
    triangles = np.asarray(g.triangles, dtype=np.int64)
    if len(vertices) != len(g.template_vertex_positions):
        raise RuntimeError(f"GNM vertex count mismatch: {len(vertices)}")

    cfg = gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
    idx = np.asarray(cfg.indices, dtype=np.int64)
    bw = np.asarray(cfg.weights, dtype=np.float64)
    current_lm_cam = h.lm(vertices, idx, bw) @ R.T

    expr_basis = np.asarray(g.expression_basis[:200], dtype=np.float64)
    lm_expr_basis = (expr_basis[:, idx, :] * bw[None, ..., None]).sum(axis=-2)
    lm_expr_cam = np.einsum("elc,dc->eld", lm_expr_basis, R)
    A, b = build_eye_system(current_lm_cam, target_cam, lm_expr_cam)
    row_scale = float(np.median(np.linalg.norm(A, axis=1)))
    reg_base = max(row_scale * row_scale, 1e-10)

    ident = np.load(args.identity).astype(np.float64).reshape(1, -1)
    neutral_gnm = np.asarray(g(identity=ident))[0]
    candidates = []
    for reg_mult in (0.012, 0.025, 0.05, 0.10, 0.20):
        lam = reg_base * reg_mult
        A2 = np.vstack([A, np.sqrt(lam) * np.eye(200)])
        b2 = np.concatenate([b, np.zeros(200)])
        sol = lsq_linear(A2, b2, bounds=(-2.35, 2.35), method="trf", tol=1e-9, max_iter=350)
        coeff = sol.x
        expr = np.zeros((1, g.expression_dim), dtype=np.float64)
        expr[0, :200] = coeff
        expressed = np.asarray(g(identity=ident, expression=expr))[0]
        native_delta = expressed - neutral_gnm
        for alpha in (.55, .72, .88, 1.0, 1.12):
            cand = vertices + native_delta * alpha
            err = eye_error(cand, idx, bw, R, target_cam)
            # Mild coefficient/displacement penalty only breaks near-ties.
            rms_coeff = float(np.sqrt(np.mean(coeff * coeff)))
            max_disp = float(np.linalg.norm(native_delta * alpha, axis=1).max())
            score = err + 0.00012 * rms_coeff + 0.010 * max(0.0, max_disp - .009)
            candidates.append((score, err, reg_mult, alpha, rms_coeff, max_disp, coeff.copy(), native_delta.copy()))
    candidates.sort(key=lambda x: x[0])
    best = candidates[0]
    _, eye_fit_error, reg_mult, alpha, coeff_rms, eye_max_disp, coeff, native_delta = best
    v_eye = vertices + native_delta * alpha

    # Skin topology for compact final identity residuals.
    skin_tri_idx = np.asarray(g.triangle_indices_for_group("skin"), dtype=np.int64)
    skin_faces_global = triangles[skin_tri_idx]
    skin_ids = np.unique(skin_faces_global.reshape(-1))
    g2l = {int(x): i for i, x in enumerate(skin_ids)}
    skin_faces = np.vectorize(g2l.get)(skin_faces_global)
    sv = v_eye[skin_ids]
    p = sv @ R.T
    original = p.copy()
    lm_eye = h.lm(v_eye, idx, bw)
    lc = lm_eye @ R.T

    # Remaining lower-jaw residual: close most of the target gap while preserving
    # a smooth mandibular angle. Target is symmetrized, so no one-sided pull.
    jaw_ids = list(range(4, 13))
    jaw_gain = {4:.52,5:.70,6:.88,7:.94,8:.96,9:.94,10:.88,11:.70,12:.52}
    seeds, deltas, radii = [], [], []
    for i in jaw_ids:
        d2 = target_cam[i] - lc[i, :2]
        gg = jaw_gain[i]
        seeds.append(lc[i])
        deltas.append(np.array([d2[0] * gg, d2[1] * gg * .48, 0.0]))
        radii.append(.0125 if 6 <= i <= 10 else .0155)
    p += h.field(p, seeds, deltas, radii, 1.0)

    # Remaining nasal alar/tip gap. The bridge is deliberately left alone.
    seeds, deltas, radii = [], [], []
    for i in (30,31,32,33,34,35):
        d2 = target_cam[i] - lc[i, :2]
        seeds.append(lc[i])
        deltas.append(np.array([d2[0] * .82, d2[1] * .55, 0.0]))
        radii.append(.0062)
    p += h.field(p, seeds, deltas, radii, 1.0)
    nose_center = lc[30:36, :2].mean(axis=0)
    nose_w = np.exp(-.5 * (((p[:,0]-nose_center[0])/.016)**2 + ((p[:,1]-nose_center[1])/.022)**2)**1.25)
    p[:,2] += .00085 * nose_w

    raw = p - original
    cap = .0046
    rn = np.linalg.norm(raw, axis=1)
    over = rn > cap
    if np.any(over):
        raw[over] *= (cap / rn[over])[:, None]
    correction = h.smooth(raw, skin_faces, 3, .12)
    p = original + correction
    v_final = v_eye.copy()
    v_final[skin_ids] = p @ R

    skin_final = trimesh.Trimesh(vertices=v_final[skin_ids], faces=skin_faces, process=False)
    skin_final.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.5.obj")
    skin_final.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.5.ply")
    skin_final.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.5.glb")
    full_final = trimesh.Trimesh(vertices=v_final, faces=triangles, process=False)
    full_final.export(args.out / "AINA_FACE_MASTER_GNM_v10.5_FULL_TOPOLOGY.obj")
    full_final.export(args.out / "AINA_FACE_MASTER_GNM_v10.5_FULL_TOPOLOGY.glb")
    np.save(args.out / "AINA_EYE_EXPRESSION_BAKE_COEFFICIENTS_v10.5.npy", coeff.astype(np.float32))

    # QA: skin and full head. Full head includes eyeballs and therefore is the
    # authoritative eye-aperture preview.
    paths_by_kind = {}
    for mesh_v, mesh_f, kind in ((v_final[skin_ids], skin_faces, "SKIN"), (v_final, triangles, "FULL")):
        paths = []
        for yaw, label in ((-90,"left_profile"),(-45,"left_45"),(0,"front"),(45,"right_45"),(90,"right_profile")):
            outp = qa / f"AINA_{kind}_CLAY_{label}_v10.5.png"
            h.render(mesh_v, mesh_f, R, yaw, outp, f"AINA v10.5 {kind} {label.replace('_',' ')}")
            paths.append(outp)
        paths_by_kind[kind] = paths
        ims = [Image.open(x).convert("RGB") for x in paths]
        H = max(x.height for x in ims); W = max(x.width for x in ims)
        sheet = Image.new("RGB", (W * 5, H), "white")
        for k, im in enumerate(ims):
            sheet.paste(im, (k * W + (W - im.width)//2, (H - im.height)//2))
        sheet.save(qa / f"AINA_{kind}_CLAY_5VIEW_v10.5.png")

    final_lm = h.lm(v_final, idx, bw)
    final_cam = final_lm @ R.T
    actual = Image.open(qa / "AINA_FULL_CLAY_front_v10.5.png").convert("RGB")
    refim = Image.open(args.front).convert("RGB")
    H = max(refim.height, actual.height)
    rw = int(refim.width * H / refim.height); aw = int(actual.width * H / actual.height)
    comp = Image.new("RGB", (rw + aw, H), "white")
    comp.paste(refim.resize((rw,H)), (0,0)); comp.paste(actual.resize((aw,H)), (rw,0))
    comp.save(qa / "AINA_REFERENCE_VS_ACTUAL_FULL_FRONT_v10.5.png")

    def width(x, a, b): return float(abs(x[b,0] - x[a,0]))
    def eye_height(x, ids): return float(abs(x[ids[1:3],1].mean() - x[ids[4:6],1].mean()))
    metrics = {
        "chosen_eye_basis_reg_mult": float(reg_mult),
        "chosen_eye_basis_alpha": float(alpha),
        "eye_basis_coeff_rms": float(coeff_rms),
        "eye_basis_max_native_displacement_m": float(eye_max_disp),
        "eye_fit_error_camera_m": float(eye_fit_error),
        "eye_L_width_target_over_final": width(target_cam,36,39)/max(width(final_cam,36,39),1e-9),
        "eye_R_width_target_over_final": width(target_cam,42,45)/max(width(final_cam,42,45),1e-9),
        "eye_L_height_target_over_final": eye_height(target_cam,[36,37,38,39,40,41])/max(eye_height(final_cam,[36,37,38,39,40,41]),1e-9),
        "eye_R_height_target_over_final": eye_height(target_cam,[42,43,44,45,46,47])/max(eye_height(final_cam,[42,43,44,45,46,47]),1e-9),
        "nose_width_target_over_final": width(target_cam,31,35)/max(width(final_cam,31,35),1e-9),
        "mouth_width_target_over_final": width(target_cam,48,54)/max(width(final_cam,48,54),1e-9),
        "jaw_low_target_over_final": width(target_cam,6,10)/max(width(final_cam,6,10),1e-9),
        "max_final_semantic_correction_m": float(np.linalg.norm(correction,axis=1).max()),
    }
    report = {
        "version": "AINA Face Master v10.5",
        "base": "v10.4 stable full GNM topology",
        "method": "GNM first-200 eye expression basis solved to symmetric AINA eyelids, baked into neutral + residual nose/V-jaw semantic lock",
        "skin_vertices": int(len(skin_ids)),
        "skin_triangles": int(len(skin_faces)),
        "metrics": metrics,
        "identity_lock": False,
        "note": "Full-head Clay five-view is authoritative because it includes eyeballs. Identity remains unlocked until visual comparison passes."
    }
    (args.out / "AINA_v10.5_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
