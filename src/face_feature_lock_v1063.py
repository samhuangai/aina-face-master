#!/usr/bin/env python3
"""AINA Face Master v10.6.3 — Face Feature Lock.

Production sculpt pass built ONLY on the clean v10.6.1 reset mesh.
No v10.6.2 eye-orbit enlargement is inherited.

This pass changes skin vertex positions only and preserves the GNM topology:
- eye fissure: inner/outer corners, upper-lid arc, slight outer-tail lift only;
- brow/glabella: retreat and soften;
- nose: shorter bridge/lower nose, smaller higher tip and alae, less projection;
- lips: +~11% width, clearer cupid bow, fuller lower lip;
- apple cheeks: keep forward volume while reducing lateral spread;
- chin: shorten ~2–3 mm and round the pointed tip;
- profile: forehead forward softness, nose back, lips/chin subtly forward.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import trimesh


def smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def gaussian_ellipse(p: np.ndarray, cx: float, cy: float, rx: float, ry: float, power: float = 2.0) -> np.ndarray:
    q = ((p[:, 0] - cx) / max(rx, 1e-9)) ** 2 + ((p[:, 1] - cy) / max(ry, 1e-9)) ** 2
    return np.exp(-0.5 * np.power(q, power / 2.0))


def build_adjacency(n: int, faces: np.ndarray) -> list[set[int]]:
    adj = [set() for _ in range(n)]
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))
    return adj


def smooth_displacement(raw: np.ndarray, faces: np.ndarray, iterations: int = 2, alpha: float = 0.06) -> np.ndarray:
    adj = build_adjacency(len(raw), faces)
    d = raw.copy()
    for _ in range(iterations):
        old = d.copy()
        for i, nbr in enumerate(adj):
            if nbr:
                d[i] = (1.0 - alpha) * old[i] + alpha * old[list(nbr)].mean(axis=0)
    return 0.90 * raw + 0.10 * d


def find_eyeballs(full_mesh: trimesh.Trimesh, R: np.ndarray) -> list[trimesh.Trimesh]:
    candidates = []
    for comp in full_mesh.split(only_watertight=False):
        nv = len(comp.vertices)
        if 350 <= nv <= 420:
            cc = np.asarray(comp.vertices).mean(axis=0) @ R.T
            if -0.32 < cc[1] < -0.20 and abs(cc[0]) < 0.09:
                candidates.append((comp, cc))
    eyes = []
    for side in (-1, 1):
        items = [(c, cc) for c, cc in candidates if np.sign(cc[0]) == side]
        if not items:
            raise RuntimeError(f"No GNM eye component discovered on side {side}")
        eyes.append(min(items, key=lambda item: float(item[1][2]))[0])
    return eyes


def eye_feature_delta(p: np.ndarray, skin_world: np.ndarray, eye_mesh: trimesh.Trimesh, R: np.ndarray) -> tuple[np.ndarray, dict]:
    eye_world = np.asarray(eye_mesh.vertices, dtype=np.float64)
    d3, _ = cKDTree(eye_world).query(skin_world, k=1)
    rim = np.where(d3 < 0.00118)[0]
    if len(rim) < 30:
        raise RuntimeError(f"Too few eyelid-rim vertices: {len(rim)}")
    rp = p[rim]
    xmin, xmax = np.percentile(rp[:, 0], [2, 98])
    x0 = 0.5 * (xmin + xmax)
    half = 0.5 * (xmax - xmin)
    ylo, yhi = np.percentile(rp[:, 1], [5, 95])
    cy = 0.5 * (ylo + yhi)
    side = -1.0 if x0 < 0 else 1.0

    handle = np.zeros((len(rim), 3), dtype=np.float64)
    upper_count = 0
    for k, i in enumerate(rim):
        u = np.clip((p[i, 0] - x0) / max(half, 1e-9), -1.0, 1.0)
        outer = np.clip(side * u, 0.0, 1.0)
        inner = np.clip(-side * u, 0.0, 1.0)
        arch = math.sqrt(max(0.0, 1.0 - u * u))
        is_upper = p[i, 1] <= cy
        if is_upper:
            upper_count += 1
            # Keep orbit size; only reshape the actual fissure edge.
            dy = -0.00100 * arch - 0.00075 * smoothstep01(outer) + 0.00010 * smoothstep01(inner)
        else:
            dy = 0.00008 * arch - 0.00018 * smoothstep01(outer)
        # Tiny corner-only x correction: no global eye enlargement.
        dx = side * 0.00028 * smoothstep01(outer) + (-side) * 0.00010 * smoothstep01(inner)
        handle[k, 0] = dx
        handle[k, 1] = dy

    tree = cKDTree(rp[:, :2])
    dist, nearest = tree.query(p[:, :2], k=1)
    w = np.exp(-0.5 * (dist / 0.0046) ** 4)
    rim_z = float(np.median(rp[:, 2]))
    w *= np.exp(-0.5 * ((p[:, 2] - rim_z) / 0.012) ** 4)
    delta = handle[nearest] * w[:, None]
    return delta, {
        "rim_vertices": int(len(rim)),
        "upper_vertices": int(upper_count),
        "rim_center_x_m": float(x0),
        "rim_center_y_m": float(cy),
        "rim_width_m": float(2.0 * half),
        "max_handle_displacement_m": float(np.linalg.norm(handle, axis=1).max()),
    }


def render_mesh(vertices: np.ndarray, faces: np.ndarray, R0: np.ndarray, yaw_deg: float, path: Path, title: str) -> None:
    right, up, forward = R0[0], R0[1], R0[2]
    a = math.radians(yaw_deg)
    R = np.stack([
        math.cos(a) * right + math.sin(a) * forward,
        up,
        -math.sin(a) * right + math.cos(a) * forward,
    ])
    p = vertices @ R.T
    xy = p[:, :2]
    tri = p[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    order = np.argsort(tri[:, :, 2].mean(axis=1))[::-1]
    tri2 = xy[faces[order]]
    nn = n[order]
    diffuse = np.clip(np.abs(nn[:, 2]), 0.0, 1.0)
    side = np.clip(-0.30 * nn[:, 0] - 0.20 * nn[:, 1] - 0.70 * nn[:, 2], 0.0, 1.0)
    intensity = np.clip(0.66 + 0.21 * diffuse + 0.10 * side, 0.50, 0.98)
    colors = np.stack([intensity * 0.96, intensity * 0.97, intensity], axis=1)
    lo = np.percentile(xy, 1.5, axis=0); hi = np.percentile(xy, 98.5, axis=0)
    center = 0.5 * (lo + hi); extent = max(float((hi - lo).max()), 1e-6) * 0.57
    fig, ax = plt.subplots(figsize=(5, 5), dpi=190)
    ax.add_collection(PolyCollection(tri2, facecolors=colors, edgecolors="none"))
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] + extent, center[1] - extent)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.12)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_fiveview(vertices: np.ndarray, faces: np.ndarray, R: np.ndarray, qa: Path, prefix: str) -> Path:
    paths = []
    for yaw, label in ((-90, "left_profile"), (-45, "left_45"), (0, "front"), (45, "right_45"), (90, "right_profile")):
        out = qa / f"AINA_{prefix}_CLAY_{label}_v10.6.3.png"
        render_mesh(vertices, faces, R, yaw, out, f"AINA v10.6.3 {prefix} {label.replace('_', ' ')}")
        paths.append(out)
    ims = [Image.open(x).convert("RGB") for x in paths]
    H = max(x.height for x in ims); W = max(x.width for x in ims)
    sheet = Image.new("RGB", (5 * W, H), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, (i * W + (W - im.width) // 2, (H - im.height) // 2))
    result = qa / f"AINA_{prefix}_CLAY_5VIEW_v10.6.3.png"
    sheet.save(result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-full", type=Path, required=True)
    ap.add_argument("--base-skin", type=Path, required=True)
    ap.add_argument("--front", type=Path, required=True)
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("output_v1063"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"; qa.mkdir(exist_ok=True)

    full = trimesh.load(args.base_full, process=False)
    skin = trimesh.load(args.base_skin, process=False)
    if not isinstance(full, trimesh.Trimesh) or not isinstance(skin, trimesh.Trimesh):
        raise RuntimeError("Expected single Trimesh OBJ inputs")
    full_v = np.asarray(full.vertices, dtype=np.float64)
    full_f = np.asarray(full.faces, dtype=np.int64)
    skin_v = np.asarray(skin.vertices, dtype=np.float64)
    skin_f = np.asarray(skin.faces, dtype=np.int64)
    n_skin = len(skin_v)
    if len(full_v) < n_skin or not np.allclose(full_v[:n_skin], skin_v, atol=2e-6):
        raise RuntimeError("Full v10.6.1 mesh no longer starts with the skin vertex block")

    cameras = json.loads(args.cameras.read_text(encoding="utf-8"))
    R = np.asarray(cameras["front"]["rotation_rows"], dtype=np.float64)
    p = skin_v @ R.T
    raw = np.zeros_like(p)
    cx = 0.0050
    central = np.exp(-0.5 * ((p[:, 0] - cx) / 0.115) ** 6)
    front_gate = 1.0 - smoothstep01((p[:, 2] + 0.095) / 0.115)

    # 1. EYES — corners, upper-lid arc and tail only. Orbit dimensions are untouched.
    eye_stats = []
    for eye in find_eyeballs(full, R):
        d, s = eye_feature_delta(p, skin_v, eye, R)
        raw += d
        eye_stats.append(s)

    # 2. BROW / GLABELLA — push hard brow mass backward; restore soft forehead convexity.
    for ccx in (-0.030, 0.041):
        brow = gaussian_ellipse(p, ccx, -0.276, 0.040, 0.027, 2.1) * front_gate
        under_eye = gaussian_ellipse(p, ccx, -0.242, 0.037, 0.022, 2.0) * front_gate
        raw[:, 2] += 0.0030 * brow
        raw[:, 2] -= 0.0008 * under_eye
    glabella = gaussian_ellipse(p, 0.005, -0.273, 0.032, 0.032, 2.0) * front_gate
    forehead = gaussian_ellipse(p, 0.005, -0.318, 0.076, 0.060, 2.0) * front_gate
    raw[:, 2] += 0.0020 * glabella
    raw[:, 2] -= 0.0010 * forehead

    # 3. NOSE — shorten lower nose, lift/smaller tip, narrower alae, less profile projection.
    nose = gaussian_ellipse(p, 0.0055, -0.213, 0.032, 0.042, 2.0) * front_gate
    lower = smoothstep01((p[:, 1] + 0.248) / 0.060)
    nw = nose * lower
    raw[:, 0] += ((0.0055 + (p[:, 0] - 0.0055) * 0.84) - p[:, 0]) * nw
    raw[:, 1] -= 0.0031 * nw
    raw[:, 2] += 0.0034 * nose
    tip = gaussian_ellipse(p, 0.0055, -0.204, 0.019, 0.018, 2.0) * front_gate
    raw[:, 0] += ((0.0055 + (p[:, 0] - 0.0055) * 0.92) - p[:, 0]) * tip
    raw[:, 1] -= 0.0010 * tip
    raw[:, 2] += 0.0012 * tip
    bridge = gaussian_ellipse(p, 0.0055, -0.235, 0.021, 0.030, 2.0) * front_gate
    raw[:, 2] += 0.0013 * bridge

    # 4. LIPS — +~11% visual width, cupid-bow lift, fuller lower lip, remove pursed look.
    lips = gaussian_ellipse(p, 0.0055, -0.177, 0.051, 0.023, 2.0) * front_gate
    raw[:, 0] += ((0.0055 + (p[:, 0] - 0.0055) * 1.11) - p[:, 0]) * lips
    raw[:, 1] -= 0.0004 * lips
    upper_lip = gaussian_ellipse(p, 0.0055, -0.183, 0.033, 0.010, 2.1) * front_gate
    lower_lip = gaussian_ellipse(p, 0.0055, -0.171, 0.034, 0.012, 2.0) * front_gate
    cupid = gaussian_ellipse(p, 0.0055, -0.184, 0.010, 0.0065, 2.0) * front_gate
    raw[:, 1] -= 0.00065 * cupid
    raw[:, 2] -= 0.00065 * upper_lip
    raw[:, 2] -= 0.00135 * lower_lip

    # 5. APPLE CHEEKS — preserve volume, reduce lateral spread.
    for ccx in (-0.039, 0.049):
        cheek = gaussian_ellipse(p, ccx, -0.218, 0.034, 0.032, 2.2) * front_gate
        raw[:, 0] += ((cx + (p[:, 0] - cx) * 0.975) - p[:, 0]) * cheek
        raw[:, 2] -= 0.0010 * cheek

    # 6. CHIN — shorten 2–3 mm and round the central point.
    chin = gaussian_ellipse(p, 0.005, -0.128, 0.044, 0.034, 2.0) * front_gate
    chin_core = gaussian_ellipse(p, 0.005, -0.112, 0.024, 0.021, 2.0) * front_gate
    raw[:, 1] -= 0.0022 * chin + 0.0008 * chin_core
    raw[:, 2] -= 0.0009 * chin
    raw[:, 0] += ((0.005 + (p[:, 0] - 0.005) * 1.06) - p[:, 0]) * chin_core

    # 7. PROFILE FLOW is produced coherently by forehead forward / brow back / nose back /
    # lips and chin forward above, rather than a separate silhouette-only distortion.

    # Bound the pass. v10.6.3 is a feature lock, not a new identity rebuild.
    lengths = np.linalg.norm(raw, axis=1)
    cap = 0.0058
    over = lengths > cap
    if np.any(over):
        raw[over] *= (cap / lengths[over])[:, None]
    d = smooth_displacement(raw, skin_f, iterations=2, alpha=0.06)
    p2 = p + d
    skin_v2 = p2 @ R
    full_v2 = full_v.copy()
    full_v2[:n_skin] = skin_v2

    skin_out = trimesh.Trimesh(vertices=skin_v2, faces=skin_f, process=False)
    full_out = trimesh.Trimesh(vertices=full_v2, faces=full_f, process=False)
    skin_out.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.6.3.obj")
    skin_out.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.6.3.ply")
    skin_out.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.6.3.glb")
    full_out.export(args.out / "AINA_FACE_MASTER_GNM_v10.6.3_FULL_TOPOLOGY.obj")
    full_out.export(args.out / "AINA_FACE_MASTER_GNM_v10.6.3_FULL_TOPOLOGY.glb")

    make_fiveview(skin_v2, skin_f, R, qa, "SKIN")
    make_fiveview(full_v2, full_f, R, qa, "FULL")

    # Front reference comparison is QA only; it is not generated art.
    ref = Image.open(args.front).convert("RGB")
    act = Image.open(qa / "AINA_FULL_CLAY_front_v10.6.3.png").convert("RGB")
    H = max(ref.height, act.height)
    rw = int(ref.width * H / ref.height); aw = int(act.width * H / act.height)
    comp = Image.new("RGB", (rw + aw, H), "white")
    comp.paste(ref.resize((rw, H)), (0, 0)); comp.paste(act.resize((aw, H)), (rw, 0))
    comp.save(qa / "AINA_REFERENCE_VS_ACTUAL_FULL_FRONT_v10.6.3.png")

    report = {
        "version": "AINA Face Master v10.6.3 Face Feature Lock",
        "base": "v10.6.1 clean reset mesh",
        "topology_changed": False,
        "full_vertices": int(len(full_v2)),
        "full_triangles": int(len(full_f)),
        "skin_vertices": int(len(skin_v2)),
        "skin_triangles": int(len(skin_f)),
        "max_feature_displacement_m": float(np.linalg.norm(d, axis=1).max()),
        "rms_feature_displacement_m": float(np.sqrt(np.mean(d * d))),
        "eye_orbit_scaled": False,
        "eye_rims": eye_stats,
        "authored_changes": {
            "eye": "inner/outer corners + upper lid curve + outer-tail lift only",
            "brow": "retreat + softened glabella",
            "nose": "shorter, narrower alae, smaller/lifted tip, reduced projection",
            "lips": "~11% wider visual region, cupid bow, fuller lower lip",
            "cheeks": "forward volume retained, lateral spread reduced",
            "chin": "~2-3 mm shorter central region and rounder tip",
            "profile": "forehead/brow/nose/lip/chin depth relation rebalanced"
        },
        "identity_lock": False,
        "acceptance_note": "Actual topology-preserving model pass. Identity remains unlocked until front, 45-degree and profile clay visually match the approved effect-art face."
    }
    (args.out / "AINA_v10.6.3_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
