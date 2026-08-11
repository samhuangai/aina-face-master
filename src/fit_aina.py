#!/usr/bin/env python3
"""Fit a Google GNM v3 head identity to AINA multi-view effect-art references.

The script intentionally keeps expression at neutral and optimizes only the first
170 GNM head identity coefficients. It fits one shared identity to three images
(front, 3/4, side), each with its own scaled-orthographic camera.
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
import torch
import trimesh

from gnm.shape import gnm_pytorch
from gnm.shape import gnm_landmarks

VIEW_ORDER = ("front", "three_quarter", "side")


def load_image_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def detect_68(fa: face_alignment.FaceAlignment, image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max(1.0, 640.0 / max(h, w))
    work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.0 else image
    preds = fa.get_landmarks_from_image(work)
    if not preds:
        raise RuntimeError("68-point landmark detector did not find a face")
    pts = np.asarray(preds[0], dtype=np.float32)[:, :2] / scale
    if pts.shape != (68, 2):
        raise RuntimeError(f"Expected 68 landmarks, got {pts.shape}")
    return pts


def normalize_target(points_px: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    s = 0.5 * max(w, h)
    center = np.array([0.5 * w, 0.5 * h], np.float32)
    return (points_px - center) / s


def weights_for_view(name: str) -> np.ndarray:
    w = np.ones(68, np.float32)
    w[0:17] *= 1.7
    w[17:27] *= 1.2
    w[27:36] *= 1.7
    w[36:48] *= 2.2
    w[48:68] *= 1.8
    if name == "side":
        w[17:27] *= 0.45
        w[36:48] *= 0.55
        w[0:17] *= 1.35
        w[27:36] *= 1.45
        w[48:68] *= 1.25
    return w


def scaled_ortho_init(points3: np.ndarray, target2: np.ndarray):
    x = np.concatenate([points3, np.ones((len(points3), 1), np.float32)], axis=1)
    beta, *_ = np.linalg.lstsq(x, target2, rcond=None)
    a = beta[:3, :].T
    b = beta[3, :]
    n1, n2 = np.linalg.norm(a[0]), np.linalg.norm(a[1])
    scale = max(1e-6, 0.5 * (n1 + n2))
    r1 = a[0] / max(n1, 1e-8)
    v2 = a[1] - np.dot(a[1], r1) * r1
    r2 = v2 / max(np.linalg.norm(v2), 1e-8)
    r3 = np.cross(r1, r2)
    r3 /= max(np.linalg.norm(r3), 1e-8)
    r2 = np.cross(r3, r1)
    r = np.stack([r1, r2, r3], axis=0).astype(np.float32)
    if np.linalg.det(r) < 0:
        r[2] *= -1.0
    return r, float(scale), b.astype(np.float32)


def rot6d_to_matrix_rows(x: torch.Tensor) -> torch.Tensor:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    r1 = torch.nn.functional.normalize(a1, dim=-1)
    a2o = a2 - (r1 * a2).sum(dim=-1, keepdim=True) * r1
    r2 = torch.nn.functional.normalize(a2o, dim=-1)
    r3 = torch.cross(r1, r2, dim=-1)
    return torch.stack([r1, r2, r3], dim=-2)


def project(points: torch.Tensor, cam6: torch.Tensor, log_scale: torch.Tensor, trans: torch.Tensor):
    r = rot6d_to_matrix_rows(cam6)
    cam = points @ r.transpose(-1, -2)
    return torch.exp(log_scale) * cam[..., :2] + trans


def render_mesh_ortho(vertices: np.ndarray, faces: np.ndarray, base_r: np.ndarray, yaw_deg: float, out_path: Path, title: str):
    right, up, forward = base_r[0], base_r[1], base_r[2]
    a = math.radians(yaw_deg)
    right2 = math.cos(a) * right + math.sin(a) * forward
    forward2 = -math.sin(a) * right + math.cos(a) * forward
    basis = np.stack([right2, up, forward2], axis=0)
    p = vertices @ basis.T
    xy = p[:, :2]
    lo = np.percentile(xy, 0.5, axis=0)
    hi = np.percentile(xy, 99.5, axis=0)
    center = 0.5 * (lo + hi)
    extent = max(float((hi - lo).max()), 1e-6) * 0.58
    draw_faces = faces[::int(math.ceil(len(faces) / 32000))] if len(faces) > 32000 else faces
    tri2 = xy[draw_faces]
    depth = p[draw_faces, 2].mean(axis=1)
    order = np.argsort(depth)
    tri2 = tri2[order]
    tri3 = p[draw_faces][order]
    n = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    intensity = 0.62 + 0.30 * np.clip(np.abs(n[:, 2]), 0, 1) + 0.08 * np.clip(n[:, 1], -1, 1)
    intensity = np.clip(intensity, 0.42, 0.98)
    colors = np.stack([intensity * 0.95, intensity * 0.96, intensity], axis=1)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=180)
    ax.add_collection(PolyCollection(tri2, facecolors=colors, edgecolors="none", closed=True))
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] + extent, center[1] - extent)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_overlay(image: np.ndarray, target_px: np.ndarray, pred_norm: np.ndarray, out_path: Path, title: str):
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
        ax.plot(target_px[ids,0], target_px[ids,1], linewidth=0.7, alpha=0.55)
    ax.set_title(title)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=Path, required=True)
    ap.add_argument("--three-quarter", type=Path, required=True)
    ap.add_argument("--side", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--steps", type=int, default=2600)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa_dir = args.out / "QA"
    qa_dir.mkdir(exist_ok=True)
    torch.manual_seed(7)
    np.random.seed(7)
    device = torch.device("cpu")
    refs = {
        "front": load_image_rgb(args.front),
        "three_quarter": load_image_rgb(args.three_quarter),
        "side": load_image_rgb(args.side),
    }
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu", face_detector="sfd")
    target_px = {name: detect_68(fa, img) for name, img in refs.items()}
    target = {name: normalize_target(target_px[name], refs[name].shape) for name in VIEW_ORDER}

    gnm = gnm_pytorch.GNM.from_local(version=gnm_pytorch.GNMMajorVersion.V3, variant=gnm_pytorch.GNMVariant.HEAD).to(device)
    lm_cfg = gnm_landmarks.load_landmarks(gnm_pytorch.GNMLandmarksType.HEAD_SPARSE_68)
    lm_indices = torch.tensor(lm_cfg.indices, dtype=torch.long, device=device)
    lm_weights = torch.tensor(lm_cfg.weights, dtype=torch.float32, device=device)
    template_lm = (gnm.template_vertex_positions[lm_indices] * lm_weights[..., None]).sum(dim=-2)
    head_lm_basis = (gnm.vertex_identity_basis[:170, lm_indices, :] * lm_weights[None, ..., None]).sum(dim=-2)
    template_lm_np = template_lm.detach().cpu().numpy()

    camera_params = {}
    for name in VIEW_ORDER:
        r, scale, trans = scaled_ortho_init(template_lm_np, target[name])
        cam6 = np.concatenate([r[0], r[1]]).astype(np.float32)
        camera_params[name] = {
            "cam6": torch.nn.Parameter(torch.tensor(cam6, device=device)),
            "log_scale": torch.nn.Parameter(torch.tensor(math.log(max(scale, 1e-6)), device=device)),
            "trans": torch.nn.Parameter(torch.tensor(trans, device=device)),
        }

    head_identity = torch.nn.Parameter(torch.zeros(170, dtype=torch.float32, device=device))
    params = [head_identity]
    for name in VIEW_ORDER:
        params += list(camera_params[name].values())
    optimizer = torch.optim.Adam(params, lr=0.035)
    target_t = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in target.items()}
    weight_t = {k: torch.tensor(weights_for_view(k), dtype=torch.float32, device=device) for k in VIEW_ORDER}
    history = []

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        landmarks = template_lm + torch.einsum("i,ilc->lc", head_identity, head_lm_basis)
        data_loss = torch.tensor(0.0, device=device)
        view_losses = {}
        for name in VIEW_ORDER:
            cp = camera_params[name]
            pred = project(landmarks, cp["cam6"], cp["log_scale"], cp["trans"])
            diff2 = ((pred - target_t[name]) ** 2).sum(dim=-1)
            vl = (diff2 * weight_t[name]).sum() / weight_t[name].sum()
            view_losses[name] = vl
            data_loss = data_loss + {"front": 1.25, "three_quarter": 1.0, "side": 1.15}[name] * vl
        id_prior = (head_identity ** 2).mean()
        limit_penalty = torch.relu(torch.abs(head_identity) - 3.0).pow(2).mean()
        loss = data_loss + 0.0045 * id_prior + 0.12 * limit_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
        if step in (900, 1700):
            for group in optimizer.param_groups:
                group["lr"] *= 0.35
        if step % 100 == 0 or step == args.steps - 1:
            rec = {"step": step, "loss": float(loss.detach()), "id_rms": float(torch.sqrt((head_identity**2).mean()).detach())}
            rec.update({f"{k}_loss": float(v.detach()) for k, v in view_losses.items()})
            history.append(rec)
            print(json.dumps(rec))

    with torch.no_grad():
        identity = torch.zeros((1, gnm.identity_dim), dtype=torch.float32, device=device)
        identity[:, :170] = head_identity
        vertices_t = gnm(identity=identity)
        landmarks_t = template_lm + torch.einsum("i,ilc->lc", head_identity, head_lm_basis)
    vertices = vertices_t[0].detach().cpu().numpy()
    landmarks = landmarks_t.detach().cpu().numpy()
    triangles = np.asarray(gnm.triangles.detach().cpu().numpy(), dtype=np.int64)

    full_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    full_mesh.export(args.out / "AINA_FACE_MASTER_GNM_v10.glb")
    full_mesh.export(args.out / "AINA_FACE_MASTER_GNM_v10.obj")
    skin_tri_idx = gnm.triangle_indices_for_group("skin")
    skin_mesh = full_mesh.submesh([skin_tri_idx], append=True, repair=False)
    skin_mesh.remove_unreferenced_vertices()
    skin_mesh.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.obj")
    skin_mesh.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.ply")
    try:
        skin_mesh.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.glb")
    except Exception as exc:
        print(f"Skin GLB export warning: {exc}")

    identity_np = identity[0].detach().cpu().numpy()
    np.save(args.out / "AINA_identity_coefficients.npy", identity_np)
    cameras_json = {}
    per_view_rmse = {}
    front_r = None
    for name in VIEW_ORDER:
        cp = camera_params[name]
        with torch.no_grad():
            pred = project(torch.tensor(landmarks, dtype=torch.float32), cp["cam6"].detach().cpu(), cp["log_scale"].detach().cpu(), cp["trans"].detach().cpu()).numpy()
            r = rot6d_to_matrix_rows(cp["cam6"].detach().cpu()).numpy()
        if name == "front":
            front_r = r
        rmse = float(np.sqrt(np.mean(np.sum((pred - target[name]) ** 2, axis=1))))
        per_view_rmse[name] = rmse
        cameras_json[name] = {
            "rotation_rows": r.tolist(),
            "scale": float(torch.exp(cp["log_scale"].detach().cpu()).item()),
            "translation": cp["trans"].detach().cpu().numpy().tolist(),
            "normalized_landmark_rmse": rmse,
        }
        save_overlay(refs[name], target_px[name], pred, qa_dir / f"AINA_{name}_landmark_overlay.png", f"AINA {name}: reference vs fitted GNM landmarks")

    skin_vertices = np.asarray(skin_mesh.vertices)
    skin_faces = np.asarray(skin_mesh.faces)
    for yaw, label in [(-90, "left_profile"), (-45, "left_45"), (0, "front"), (45, "right_45"), (90, "right_profile")]:
        render_mesh_ortho(skin_vertices, skin_faces, front_r, yaw, qa_dir / f"AINA_CLAY_{label}.png", f"AINA Clay {label.replace('_', ' ')}")

    contact_names = ["left_profile", "left_45", "front", "right_45", "right_profile"]
    ims = [Image.open(qa_dir / f"AINA_CLAY_{n}.png").convert("RGB") for n in contact_names]
    h = max(im.height for im in ims)
    w = max(im.width for im in ims)
    sheet = Image.new("RGB", (w * 5, h), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, (i * w + (w - im.width)//2, (h - im.height)//2))
    sheet.save(qa_dir / "AINA_CLAY_5VIEW.png")

    coeff = identity_np[:170]
    report = {
        "model": "Google GNM v3 HEAD",
        "identity_dimensions_total": int(gnm.identity_dim),
        "optimized_head_identity_dimensions": 170,
        "expression": "neutral / all zeros",
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "skin_vertices": int(len(skin_mesh.vertices)),
        "skin_triangles": int(len(skin_mesh.faces)),
        "identity_coefficient_rms": float(np.sqrt(np.mean(coeff**2))),
        "identity_coefficient_max_abs": float(np.max(np.abs(coeff))),
        "per_view_normalized_landmark_rmse": per_view_rmse,
        "history": history,
        "acceptance_note": "Automated identity-fit pass only. Final production identity remains blocked until the clay five-view is visually approved against AINA effect art.",
    }
    (args.out / "AINA_FIT_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "AINA_CAMERAS.json").write_text(json.dumps(cameras_json, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
