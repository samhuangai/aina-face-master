#!/usr/bin/env python3
"""AINA v10.1: robust front-priority multi-view fit on Google GNM v3 HEAD.

One neutral 170D head identity is solved against the approved front, 3/4 and
profile effect-art references. The fit uses alternating scaled-orthographic
camera estimation and bounded regularized least squares, with robust per-point
weights. Front identity is deliberately dominant; the profile contributes depth
and silhouette without pulling the frontal likeness off target.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import face_alignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from PIL import Image
from scipy.optimize import lsq_linear
import trimesh

from gnm.shape import gnm_numpy
from gnm.shape import gnm_landmarks

VIEW_ORDER = ("front", "three_quarter", "side")
VIEW_GLOBAL = {"front": 2.45, "three_quarter": 1.15, "side": 0.34}
ROBUST_DELTA = {"front": 0.028, "three_quarter": 0.032, "side": 0.040}


def load_image_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def detect_68(fa: face_alignment.FaceAlignment, image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max(1.0, 720.0 / max(h, w))
    work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.0 else image
    preds = fa.get_landmarks_from_image(work)
    if not preds:
        raise RuntimeError("68-point landmark detector did not find a face")
    center = np.array([work.shape[1] * 0.5, work.shape[0] * 0.5], dtype=np.float32)
    best = min(preds, key=lambda p: float(np.linalg.norm(np.asarray(p)[:, :2].mean(axis=0) - center)))
    pts = np.asarray(best, dtype=np.float32)[:, :2] / scale
    if pts.shape != (68, 2):
        raise RuntimeError(f"Expected 68 landmarks, got {pts.shape}")
    return pts


def normalize_target(points_px: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    s = 0.5 * max(w, h)
    center = np.array([0.5 * w, 0.5 * h], np.float32)
    return (points_px - center) / s


def weights_for_view(name: str) -> np.ndarray:
    w = np.ones(68, np.float64)
    if name == "front":
        w[0:17] = 2.35
        w[17:27] = 1.30
        w[27:36] = 2.25
        w[36:48] = 3.15
        w[48:60] = 2.45
        w[60:68] = 1.85
    elif name == "three_quarter":
        w[0:17] = 1.75
        w[17:27] = 1.05
        w[27:36] = 2.15
        w[36:48] = 2.25
        w[48:60] = 2.00
        w[60:68] = 1.45
    else:
        # Profile is only a depth/silhouette stabilizer. Hidden-side detector
        # points are unreliable in a synthetic strict profile reference.
        w[:] = 0.10
        w[0:17] = 0.75
        w[6:11] = 1.75
        w[27:36] = 2.35
        w[48:60] = 1.20
        w[36:42] = 0.45
        w[17:22] = 0.28
    return w


def scaled_ortho_init(points3: np.ndarray, target2: np.ndarray, point_weights: np.ndarray | None = None):
    x = np.concatenate([points3, np.ones((len(points3), 1), np.float64)], axis=1)
    if point_weights is None:
        beta, *_ = np.linalg.lstsq(x, target2, rcond=None)
    else:
        sw = np.sqrt(np.maximum(point_weights, 1e-8))[:, None]
        beta, *_ = np.linalg.lstsq(x * sw, target2 * sw, rcond=None)
    a = beta[:3, :].T
    b = beta[3, :]
    n1, n2 = np.linalg.norm(a[0]), np.linalg.norm(a[1])
    scale = max(1e-7, 0.5 * (n1 + n2))
    r1 = a[0] / max(n1, 1e-10)
    v2 = a[1] - np.dot(a[1], r1) * r1
    r2 = v2 / max(np.linalg.norm(v2), 1e-10)
    r3 = np.cross(r1, r2)
    r3 /= max(np.linalg.norm(r3), 1e-10)
    r2 = np.cross(r3, r1)
    r = np.stack([r1, r2, r3], axis=0)
    if np.linalg.det(r) < 0:
        r[2] *= -1.0
    return r.astype(np.float64), float(scale), b.astype(np.float64)


def project_np(points: np.ndarray, r: np.ndarray, scale: float, trans: np.ndarray) -> np.ndarray:
    return scale * (points @ r.T)[:, :2] + trans


def huber_point_weight(residual_norm: np.ndarray, delta: float) -> np.ndarray:
    out = np.ones_like(residual_norm, dtype=np.float64)
    bad = residual_norm > delta
    out[bad] = delta / np.maximum(residual_norm[bad], 1e-9)
    return np.clip(out, 0.12, 1.0)


def solve_identity(template_lm, basis, target, cameras, robust, reg_lambda=7.5e-4):
    rows, rhs = [], []
    for name in VIEW_ORDER:
        r, scale, trans = cameras[name]
        base = project_np(template_lm, r, scale, trans)
        bp = scale * np.einsum("ilc,dc->ild", basis, r[:2])
        w = weights_for_view(name) * VIEW_GLOBAL[name] * robust[name]
        for l in range(68):
            if w[l] <= 0:
                continue
            sw = math.sqrt(float(w[l]))
            for d in range(2):
                rows.append(bp[:, l, d] * sw)
                rhs.append((target[name][l, d] - base[l, d]) * sw)
    A = np.stack(rows, axis=0)
    y = np.asarray(rhs, dtype=np.float64)
    A = np.concatenate([A, math.sqrt(reg_lambda) * np.eye(basis.shape[0])], axis=0)
    y = np.concatenate([y, np.zeros(basis.shape[0], dtype=np.float64)], axis=0)
    sol = lsq_linear(A, y, bounds=(-3.35, 3.35), method="trf", tol=1e-8, lsmr_tol="auto", max_iter=250)
    if not sol.success:
        print("lsq_linear warning:", sol.message)
    return sol.x.astype(np.float64)


def fit_alternating(template_lm, basis, target, outer_steps=16):
    identity = np.zeros(basis.shape[0], dtype=np.float64)
    robust = {name: np.ones(68, dtype=np.float64) for name in VIEW_ORDER}
    cameras = {name: scaled_ortho_init(template_lm, target[name], weights_for_view(name)) for name in VIEW_ORDER}
    history = []
    for outer in range(outer_steps):
        landmarks = template_lm + np.einsum("i,ilc->lc", identity, basis)
        for name in VIEW_ORDER:
            cw = weights_for_view(name) * robust[name]
            cameras[name] = scaled_ortho_init(landmarks, target[name], cw)
        new_identity = solve_identity(template_lm, basis, target, cameras, robust)
        identity = (0.80 * new_identity + 0.20 * identity) if outer else new_identity
        landmarks = template_lm + np.einsum("i,ilc->lc", identity, basis)
        rec = {"outer": outer, "identity_rms": float(np.sqrt(np.mean(identity**2))), "max_abs": float(np.max(np.abs(identity)))}
        for name in VIEW_ORDER:
            r, scale, trans = cameras[name]
            pred = project_np(landmarks, r, scale, trans)
            e = np.linalg.norm(pred - target[name], axis=1)
            robust[name] = huber_point_weight(e, ROBUST_DELTA[name])
            if name == "side":
                robust[name][22:27] *= 0.25
                robust[name][42:48] *= 0.25
            rec[f"{name}_rmse"] = float(np.sqrt(np.mean(e**2)))
            rec[f"{name}_median"] = float(np.median(e))
        history.append(rec)
        print(json.dumps(rec))
    landmarks = template_lm + np.einsum("i,ilc->lc", identity, basis)
    for name in VIEW_ORDER:
        cameras[name] = scaled_ortho_init(landmarks, target[name], weights_for_view(name) * robust[name])
    return identity, cameras, history


def save_overlay(image, target_px, pred_norm, out_path, title):
    h, w = image.shape[:2]
    s = 0.5 * max(w, h)
    center = np.array([0.5 * w, 0.5 * h], np.float32)
    pred_px = pred_norm * s + center
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.imshow(image)
    ax.scatter(target_px[:, 0], target_px[:, 1], s=9, marker="o", label="reference 68")
    ax.scatter(pred_px[:, 0], pred_px[:, 1], s=7, marker="x", label="GNM projection")
    for chain in [range(0,17), range(17,22), range(22,27), range(27,36), range(36,42), range(42,48), range(48,60), range(60,68)]:
        ids = list(chain)
        ax.plot(target_px[ids,0], target_px[ids,1], linewidth=0.65, alpha=0.55)
    ax.set_title(title)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_mesh_ortho(vertices, faces, base_r, yaw_deg, out_path, title):
    right, up, forward = base_r[0], base_r[1], base_r[2]
    a = math.radians(yaw_deg)
    right2 = math.cos(a) * right + math.sin(a) * forward
    forward2 = -math.sin(a) * right + math.cos(a) * forward
    basis = np.stack([right2, up, forward2], axis=0)
    p = vertices @ basis.T
    xy = p[:, :2]
    tri3 = p[faces]
    n = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-10)
    # Camera sits on negative projected Z and looks toward +Z.
    visible = n[:, 2] < -0.015
    draw_faces = faces[visible]
    tri3 = tri3[visible]
    n = n[visible]
    if len(draw_faces) > 36000:
        stride = int(math.ceil(len(draw_faces) / 36000))
        draw_faces, tri3, n = draw_faces[::stride], tri3[::stride], n[::stride]
    depth = tri3[:, :, 2].mean(axis=1)
    order = np.argsort(depth)[::-1]
    draw_faces, tri3, n = draw_faces[order], tri3[order], n[order]
    tri2 = xy[draw_faces]
    diffuse = np.clip(-n[:, 2], 0.0, 1.0)
    side_light = np.clip(-0.35*n[:,0] - 0.25*n[:,1] - 0.65*n[:,2], 0, 1)
    intensity = np.clip(0.70 + 0.18*diffuse + 0.10*side_light, 0.58, 0.98)
    colors = np.stack([intensity*0.97, intensity*0.98, intensity], axis=1)
    lo = np.percentile(xy, 2.0, axis=0)
    hi = np.percentile(xy, 98.0, axis=0)
    center = 0.5 * (lo + hi)
    extent = max(float((hi - lo).max()), 1e-6) * 0.57
    fig, ax = plt.subplots(figsize=(5, 5), dpi=180)
    ax.set_facecolor((0.985,0.985,0.985))
    ax.add_collection(PolyCollection(tri2, facecolors=colors, edgecolors="none", closed=True))
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] + extent, center[1] - extent)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=Path, required=True)
    ap.add_argument("--three-quarter", type=Path, required=True)
    ap.add_argument("--side", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("output_v101"))
    ap.add_argument("--outer-steps", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa_dir = args.out / "QA"
    qa_dir.mkdir(exist_ok=True)
    np.random.seed(7)

    refs = {"front": load_image_rgb(args.front), "three_quarter": load_image_rgb(args.three_quarter), "side": load_image_rgb(args.side)}
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu", face_detector="sfd")
    target_px = {name: detect_68(fa, refs[name]) for name in VIEW_ORDER}
    target = {name: normalize_target(target_px[name], refs[name].shape) for name in VIEW_ORDER}

    gnm = gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    lm_cfg = gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
    idx = np.asarray(lm_cfg.indices, dtype=np.int64)
    bw = np.asarray(lm_cfg.weights, dtype=np.float64)
    template_v = np.asarray(gnm.template_vertex_positions, dtype=np.float64)
    id_basis_all = np.asarray(gnm.vertex_identity_basis, dtype=np.float64)
    template_lm = (template_v[idx] * bw[..., None]).sum(axis=-2)
    basis = (id_basis_all[:170, idx, :] * bw[None, ..., None]).sum(axis=-2)

    identity_head, cameras, history = fit_alternating(template_lm, basis, target, args.outer_steps)
    identity = np.zeros(gnm.identity_dim, dtype=np.float64)
    identity[:170] = identity_head
    vertices = np.asarray(gnm(identity=identity[None, :]))[0]
    triangles = np.asarray(gnm.triangles, dtype=np.int64)
    full_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    full_mesh.export(args.out / "AINA_FACE_MASTER_GNM_v10.1.glb")
    full_mesh.export(args.out / "AINA_FACE_MASTER_GNM_v10.1.obj")
    skin_tri_idx = np.asarray(gnm.triangle_indices_for_group("skin"), dtype=np.int64)
    skin_mesh = full_mesh.submesh([skin_tri_idx], append=True, repair=False)
    skin_mesh.remove_unreferenced_vertices()
    skin_mesh.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.1.obj")
    skin_mesh.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.1.ply")
    skin_mesh.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.1.glb")
    np.save(args.out / "AINA_identity_coefficients_v10.1.npy", identity.astype(np.float32))

    landmarks = template_lm + np.einsum("i,ilc->lc", identity_head, basis)
    cameras_json, per_view = {}, {}
    front_r = cameras["front"][0]
    for name in VIEW_ORDER:
        r, scale, trans = cameras[name]
        pred = project_np(landmarks, r, scale, trans)
        err = np.linalg.norm(pred-target[name], axis=1)
        rmse = float(np.sqrt(np.mean(err**2)))
        per_view[name] = {"rmse": rmse, "median": float(np.median(err)), "p90": float(np.percentile(err,90))}
        cameras_json[name] = {"rotation_rows": r.tolist(), "scale": scale, "translation": trans.tolist(), **per_view[name]}
        save_overlay(refs[name], target_px[name], pred, qa_dir/f"AINA_{name}_landmark_overlay_v10.1.png", f"AINA v10.1 {name}: reference vs GNM")

    sv, sf = np.asarray(skin_mesh.vertices), np.asarray(skin_mesh.faces)
    view_specs = [(-90,"left_profile"),(-45,"left_45"),(0,"front"),(45,"right_45"),(90,"right_profile")]
    paths=[]
    for yaw,label in view_specs:
        p=qa_dir/f"AINA_CLAY_{label}_v10.1.png"
        render_mesh_ortho(sv,sf,front_r,yaw,p,f"AINA v10.1 Clay {label.replace('_',' ')}")
        paths.append(p)
    ims=[Image.open(p).convert("RGB") for p in paths]
    h=max(im.height for im in ims); w=max(im.width for im in ims)
    sheet=Image.new("RGB",(w*5,h),"white")
    for i,im in enumerate(ims):
        sheet.paste(im,(i*w+(w-im.width)//2,(h-im.height)//2))
    sheet.save(qa_dir/"AINA_CLAY_5VIEW_v10.1.png")

    report={
        "model":"Google GNM v3 HEAD",
        "version":"AINA Face Master v10.1 robust front-priority fit",
        "identity_dimensions_total":int(gnm.identity_dim),
        "optimized_head_identity_dimensions":170,
        "expression":"neutral / all zeros",
        "vertices":int(len(vertices)),"triangles":int(len(triangles)),
        "skin_vertices":int(len(skin_mesh.vertices)),"skin_triangles":int(len(skin_mesh.faces)),
        "identity_coefficient_rms":float(np.sqrt(np.mean(identity_head**2))),
        "identity_coefficient_max_abs":float(np.max(np.abs(identity_head))),
        "per_view_normalized_landmark_error":per_view,
        "view_global_weights":VIEW_GLOBAL,
        "history":history,
        "acceptance_note":"Actual GNM mesh. This pass is not identity-locked until five-view clay visually matches the approved AINA effect-art face."
    }
    (args.out/"AINA_FIT_REPORT_v10.1.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (args.out/"AINA_CAMERAS_v10.1.json").write_text(json.dumps(cameras_json,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    main()
