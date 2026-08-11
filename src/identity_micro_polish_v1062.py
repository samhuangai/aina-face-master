#!/usr/bin/env python3
"""AINA Face Master v10.6.2 — topology-native identity micro polish.

Starts from the clean v10.6.1 reset sculpt. Instead of dragging eyelid polygons,
this pass solves GNM's first 200 eye expression blendshape dimensions directly
against the *true eyelid rim vertices* discovered by skin-to-eyeball proximity.
The eye-basis result is baked into the neutral identity mesh. A small bounded
semantic pass then softens brow projection, shortens/narrows the nose, widens
and softens the lips, tapers cheeks/jaw, and rounds/shortens the chin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree
import trimesh

from gnm.shape import gnm_numpy
from reset_identity_v1061 import (
    gaussian_ellipse,
    make_fiveview,
    smooth_displacement,
    smoothstep01,
)


def connected_groups(mesh: trimesh.Trimesh):
    return trimesh.graph.connected_components(mesh.edges_unique, nodes=np.arange(len(mesh.vertices)), min_len=1)


def discover_eyeball_groups(full: trimesh.Trimesh, R: np.ndarray):
    groups = connected_groups(full)
    out = []
    for g in groups:
        if 350 <= len(g) <= 420:
            cc = np.asarray(full.vertices)[g].mean(axis=0) @ R.T
            if -0.32 < cc[1] < -0.20 and abs(cc[0]) < 0.09:
                out.append((np.asarray(g, dtype=np.int64), cc))
    eyes = []
    for side in (-1, 1):
        candidates = [(g, cc) for g, cc in out if np.sign(cc[0]) == side]
        if not candidates:
            raise RuntimeError(f"No eye component on side {side}")
        # Prefer the 385-vertex globe; fallback to the component farther behind cornea.
        globe = [x for x in candidates if len(x[0]) == 385]
        eyes.append((globe[0] if globe else min(candidates, key=lambda x: float(x[1][2]))))
    return eyes


def discover_rim(skin_v: np.ndarray, eye_v: np.ndarray, R: np.ndarray):
    d, _ = cKDTree(eye_v).query(skin_v, k=1)
    ids = np.where(d < 0.00135)[0]
    if len(ids) < 35:
        raise RuntimeError(f"Only {len(ids)} true eyelid-rim vertices found")
    rp = skin_v[ids] @ R.T
    return ids, rp


def desired_rim_targets(rp: np.ndarray):
    xmin, xmax = np.percentile(rp[:, 0], [2, 98])
    x0 = 0.5 * (xmin + xmax); half = 0.5 * (xmax - xmin)
    ylo, yhi = np.percentile(rp[:, 1], [5, 95])
    cy = 0.5 * (ylo + yhi)
    side = -1.0 if x0 < 0 else 1.0
    upper = rp[:, 1] <= cy
    u = np.clip((rp[:, 0] - x0) / max(half, 1e-8), -1.0, 1.0)
    arch = np.sqrt(np.maximum(0.0, 1.0 - u * u))
    outer = np.clip(side * u, 0.0, 1.0)
    outer_lift = -0.00130 * smoothstep01(outer)

    # Art target: slightly wider-set eyes, larger almond aperture, lifted outer corners.
    shifted_x0 = x0 + side * 0.00225
    target = rp.copy()
    target[:, 0] = shifted_x0 + u * half * 1.14
    target[upper, 1] = cy - 0.00820 * arch[upper] - 0.00075 + outer_lift[upper]
    target[~upper, 1] = cy + 0.00520 * arch[~upper] - 0.00075 + 0.20 * outer_lift[~upper]
    return target, {
        "center_before_x": float(x0),
        "center_target_x": float(shifted_x0),
        "width_before_m": float(2 * half),
        "width_target_m": float(2 * half * 1.14),
        "upper_arch_target_m": 0.00820,
        "lower_arch_target_m": 0.00520,
    }


def solve_eye_basis(base_full_v: np.ndarray, skin_v: np.ndarray, eyes, expression_basis: np.ndarray, R: np.ndarray):
    controls = []
    stats = []
    for eye_group, cc in eyes:
        ids, rp = discover_rim(skin_v, base_full_v[eye_group], R)
        target, st = desired_rim_targets(rp)
        st["rim_vertices"] = int(len(ids)); st["eye_component_center"] = cc.tolist()
        controls.append((ids, rp, target))
        stats.append(st)

    B = expression_basis[:200]
    rows = []; rhs = []
    for ids, rp, target in controls:
        camB = np.einsum("evc,dc->evd", B[:, ids, :], R)
        xmin, xmax = np.percentile(rp[:, 0], [2, 98]); x0 = .5 * (xmin + xmax); half = .5 * (xmax - xmin)
        u = np.clip((rp[:, 0] - x0) / max(half, 1e-8), -1, 1)
        corner = np.abs(u) > .72
        for j in range(len(ids)):
            # Strong y fit for upper/lower arcs; strong x only toward corners.
            wx = 1.45 if corner[j] else .55
            wy = 2.70 if not corner[j] else 1.45
            rows.append(camB[:, j, 0] * wx); rhs.append((target[j, 0] - rp[j, 0]) * wx)
            rows.append(camB[:, j, 1] * wy); rhs.append((target[j, 1] - rp[j, 1]) * wy)
    A = np.stack(rows, axis=0); b = np.asarray(rhs)
    row_scale = max(float(np.median(np.linalg.norm(A, axis=1))), 1e-8)
    best = None
    for reg_mult in (0.010, 0.020, 0.040, 0.080):
        lam = row_scale * row_scale * reg_mult
        A2 = np.vstack([A, np.sqrt(lam) * np.eye(200)])
        b2 = np.concatenate([b, np.zeros(200)])
        sol = lsq_linear(A2, b2, bounds=(-2.1, 2.1), method="trf", tol=1e-9, max_iter=350)
        coeff = sol.x
        pred = A @ coeff
        err = float(np.sqrt(np.mean((pred - b) ** 2)))
        score = err + 0.00008 * float(np.sqrt(np.mean(coeff ** 2)))
        if best is None or score < best[0]:
            best = (score, reg_mult, coeff, err)
    _, reg_mult, coeff, fit_error = best
    delta = np.einsum("e,evc->vc", coeff, B)
    # Global safety cap on expression displacement; topology-native direction remains.
    dn = np.linalg.norm(delta, axis=1)
    over = dn > 0.0075
    if np.any(over):
        delta[over] *= (0.0075 / dn[over])[:, None]
    return delta, coeff, stats, {"reg_mult": float(reg_mult), "weighted_fit_rmse": float(fit_error), "coeff_rms": float(np.sqrt(np.mean(coeff ** 2))), "max_basis_displacement_m": float(np.linalg.norm(delta, axis=1).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-full", type=Path, required=True)
    ap.add_argument("--base-skin", type=Path, required=True)
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--front", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("output_v1062"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True); qa = args.out / "QA"; qa.mkdir(exist_ok=True)

    full = trimesh.load(args.base_full, process=False); skin = trimesh.load(args.base_skin, process=False)
    full_v = np.asarray(full.vertices, dtype=np.float64); full_f = np.asarray(full.faces, dtype=np.int64)
    skin_v = np.asarray(skin.vertices, dtype=np.float64); skin_f = np.asarray(skin.faces, dtype=np.int64)
    n_skin = len(skin_v)
    if not np.allclose(full_v[:n_skin], skin_v, atol=2e-6):
        raise RuntimeError("Full/skin vertex ordering mismatch")
    R = np.asarray(json.loads(args.cameras.read_text())["front"]["rotation_rows"], dtype=np.float64)

    gnm = gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    if len(gnm.template_vertex_positions) != len(full_v):
        raise RuntimeError("GNM/full topology mismatch")
    eyes = discover_eyeball_groups(full, R)
    basis_delta, coeff, eye_stats, solver_stats = solve_eye_basis(full_v, skin_v, eyes, np.asarray(gnm.expression_basis, dtype=np.float64), R)
    eye_baked_v = full_v + basis_delta

    # Small semantic polish on SKIN ONLY, after topology-native eye motion.
    bp = eye_baked_v[:n_skin] @ R.T
    raw = np.zeros_like(bp)
    fg = 1.0 - smoothstep01((bp[:, 2] + 0.095) / 0.115)
    cx = 0.005
    central = np.exp(-0.5 * ((bp[:, 0] - cx) / 0.110) ** 6)

    # Softer youthful cheek/jaw silhouette, intentionally leaving orbital centers alone.
    cheek_band = smoothstep01((bp[:, 1] + 0.245) / 0.050) * (1.0 - smoothstep01((bp[:, 1] + 0.135) / 0.040)) * central * fg
    after = bp[:, 0] + raw[:, 0]
    tx = cx + (after - cx) * (1.0 - 0.050 * cheek_band)
    raw[:, 0] += tx - after
    jaw = smoothstep01((bp[:, 1] + 0.198) / 0.085) * central * fg
    after = bp[:, 0] + raw[:, 0]
    tx = cx + (after - cx) * (1.0 - 0.035 * jaw)
    raw[:, 0] += tx - after

    # Shorter, narrower, less projecting nose.
    nose = gaussian_ellipse(bp, 0.0055, -0.211, 0.028, 0.032, power=2.2) * fg
    lower = smoothstep01((bp[:, 1] + 0.240) / 0.050)
    after = bp[:, 0] + raw[:, 0]
    tx = 0.0055 + (after - 0.0055) * 0.90
    raw[:, 0] += (tx - after) * nose * lower
    raw[:, 2] += 0.0020 * nose
    tip = gaussian_ellipse(bp, 0.0055, -0.204, 0.018, 0.017, power=2.1) * fg
    raw[:, 1] -= 0.0017 * tip

    # Wider/softer AINA mouth rather than the pursed GNM neutral mouth.
    mcx, mcy = 0.0055, -0.174
    lips = gaussian_ellipse(bp, mcx, mcy, 0.045, 0.020, power=2.0) * fg
    after = bp[:, 0] + raw[:, 0]
    tx = mcx + (after - mcx) * 1.08
    raw[:, 0] += (tx - after) * lips
    after_y = bp[:, 1] + raw[:, 1]
    ty = mcy + (after_y - mcy) * 1.08
    raw[:, 1] += (ty - after_y) * lips
    raw[:, 1] -= 0.0007 * lips
    lip_core = gaussian_ellipse(bp, mcx, mcy, 0.032, 0.011, power=2.0) * fg
    raw[:, 2] -= 0.00065 * lip_core

    # Reduce hard brow-ridge projection without changing brow position drastically.
    for ex in (-0.032, 0.043):
        brow = gaussian_ellipse(bp, ex, -0.279, 0.034, 0.020, power=2.1) * fg
        raw[:, 2] += 0.00125 * brow

    # Soft apple-cheek volume and a shorter rounded chin.
    for ccx in (-0.038, 0.048):
        cheek = gaussian_ellipse(bp, ccx, -0.218, 0.030, 0.028, power=2.2) * fg
        raw[:, 2] -= 0.00075 * cheek
    chin = gaussian_ellipse(bp, 0.005, -0.124, 0.038, 0.030, power=2.1) * fg
    raw[:, 1] -= 0.0018 * chin

    rn = np.linalg.norm(raw, axis=1); cap = 0.0060
    over = rn > cap
    if np.any(over): raw[over] *= (cap / rn[over])[:, None]
    disp = smooth_displacement(raw, skin_f, iterations=1, alpha=0.05)
    final_v = eye_baked_v.copy(); final_skin_p = bp + disp; final_v[:n_skin] = final_skin_p @ R

    final_skin = trimesh.Trimesh(vertices=final_v[:n_skin], faces=skin_f, process=False)
    final_full = trimesh.Trimesh(vertices=final_v, faces=full_f, process=False)
    final_skin.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.6.2.obj")
    final_skin.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.6.2.ply")
    final_skin.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.6.2.glb")
    final_full.export(args.out / "AINA_FACE_MASTER_GNM_v10.6.2_FULL_TOPOLOGY.obj")
    final_full.export(args.out / "AINA_FACE_MASTER_GNM_v10.6.2_FULL_TOPOLOGY.glb")
    np.save(args.out / "AINA_EYE_BASIS_COEFFICIENTS_v10.6.2.npy", coeff.astype(np.float32))

    full_sheet = make_fiveview(final_v, full_f, R, qa, "FULL")
    skin_sheet = make_fiveview(final_v[:n_skin], skin_f, R, qa, "SKIN")
    ref = Image.open(args.front).convert("RGB"); actual = Image.open(qa / "AINA_FULL_CLAY_front_v10.6.1.png").convert("RGB") if False else None
    # make_fiveview names are v10.6.1 for shared renderer; copy front to a v10.6.2-labelled comparison source.
    generated_front = qa / "AINA_FULL_CLAY_front_v10.6.1.png"
    if generated_front.exists():
        actual = Image.open(generated_front).convert("RGB")
        actual.save(qa / "AINA_FULL_CLAY_front_v10.6.2.png")
        H = max(ref.height, actual.height); rw = int(ref.width * H / ref.height); aw = int(actual.width * H / actual.height)
        comp = Image.new("RGB", (rw + aw, H), "white"); comp.paste(ref.resize((rw, H)), (0, 0)); comp.paste(actual.resize((aw, H)), (rw, 0))
        comp.save(qa / "AINA_REFERENCE_VS_ACTUAL_FULL_FRONT_v10.6.2.png")

    report = {
        "version": "AINA Face Master v10.6.2 Identity Micro Polish",
        "base": "v10.6.1 clean-reset GNM topology",
        "method": "true eyelid-rim target solved through GNM first-200 eye expression basis + bounded facial semantic micro-polish",
        "full_vertices": int(len(final_v)), "full_triangles": int(len(full_f)),
        "skin_vertices": int(n_skin), "skin_triangles": int(len(skin_f)),
        "eye_solver": solver_stats, "eye_targets": eye_stats,
        "max_semantic_skin_displacement_m": float(np.linalg.norm(disp, axis=1).max()),
        "identity_lock": False,
        "acceptance_note": "Visual Z-buffer front/45/profile likeness decides lock; no hair, makeup or lighting tricks may substitute for geometry."
    }
    (args.out / "AINA_v10.6.2_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
