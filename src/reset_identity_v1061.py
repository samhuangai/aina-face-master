#!/usr/bin/env python3
"""AINA Face Master v10.6.1 — clean single-pass reset sculpt.

This pass intentionally discards all v10.3-v10.6 stacked vertex deformation.
It starts from the clean v10.1 GNM full-head topology, discovers the *actual*
eyelid rim by skin-to-eyeball proximity, reshapes that rim into AINA's almond
eye aperture, then applies one bounded art-directed facial proportion pass.

The GNM vertex/triangle topology is unchanged. Only skin vertex positions move;
eyeballs, corneas, teeth, tongue and other internal components remain untouched.
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
    q = ((p[:, 0] - cx) / rx) ** 2 + ((p[:, 1] - cy) / ry) ** 2
    return np.exp(-0.5 * np.power(q, power / 2.0))


def build_adjacency(n: int, faces: np.ndarray) -> list[set[int]]:
    adj = [set() for _ in range(n)]
    for a, b, c in faces:
        a = int(a); b = int(b); c = int(c)
        adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))
    return adj


def smooth_displacement(raw: np.ndarray, faces: np.ndarray, iterations: int = 2, alpha: float = 0.07) -> np.ndarray:
    adj = build_adjacency(len(raw), faces)
    d = raw.copy()
    for _ in range(iterations):
        old = d.copy()
        for i, nbr in enumerate(adj):
            if nbr:
                d[i] = (1.0 - alpha) * old[i] + alpha * old[list(nbr)].mean(axis=0)
    # Keep most of the authored displacement; smoothing is only anti-ripple.
    return 0.88 * raw + 0.12 * d


def find_eyeballs(full_mesh: trimesh.Trimesh, R: np.ndarray) -> list[trimesh.Trimesh]:
    comps = full_mesh.split(only_watertight=False)
    candidates = []
    for comp in comps:
        nv = len(comp.vertices)
        if 350 <= nv <= 420:
            c_world = np.asarray(comp.vertices).mean(axis=0)
            c_cam = c_world @ R.T
            # Eyes/corneas are in this region; oral components are not.
            if -0.32 < c_cam[1] < -0.20 and abs(c_cam[0]) < 0.09:
                candidates.append((comp, c_cam))
    if len(candidates) < 4:
        raise RuntimeError(f"Could not discover eyeball/cornea candidates; got {[(len(c.vertices), cc.tolist()) for c, cc in candidates]}")

    eyes = []
    for side in (-1, 1):
        side_items = [(c, cc) for c, cc in candidates if np.sign(cc[0]) == side]
        if not side_items:
            raise RuntimeError(f"No eye candidate on side {side}")
        # In the front camera convention the eyeball sits behind the corneal shell;
        # empirical GNM v3 components separate cleanly by this camera-z ordering.
        chosen = min(side_items, key=lambda x: float(x[1][2]))[0]
        eyes.append(chosen)
    return eyes


def reshape_eye_rim(base_p: np.ndarray, skin_world: np.ndarray, eye_mesh: trimesh.Trimesh, R: np.ndarray) -> tuple[np.ndarray, dict]:
    eye_world = np.asarray(eye_mesh.vertices, dtype=np.float64)
    eye_cam_center = eye_world.mean(axis=0) @ R.T
    tree3 = cKDTree(eye_world)
    d3, _ = tree3.query(skin_world, k=1)
    rim_ids = np.where(d3 < 0.00115)[0]
    if len(rim_ids) < 28:
        raise RuntimeError(f"Too few eyelid-rim vertices discovered: {len(rim_ids)}")

    rp = base_p[rim_ids]
    xmin, xmax = np.percentile(rp[:, 0], [2, 98])
    x0 = 0.5 * (xmin + xmax)
    half = 0.5 * (xmax - xmin)
    ylo, yhi = np.percentile(rp[:, 1], [5, 95])
    cy = 0.5 * (ylo + yhi)
    side_sign = -1.0 if x0 < 0 else 1.0

    # Classify from original rim before moving anything.
    upper = rim_ids[base_p[rim_ids, 1] <= cy]
    lower = rim_ids[base_p[rim_ids, 1] > cy]
    desired_half = half * 1.06

    total = np.zeros_like(base_p)
    influence_union = np.zeros(len(base_p), dtype=np.float64)
    group_stats = []
    for group_name, ids, amp, lower_factor in (
        ("upper", upper, 0.0072, 1.0),
        ("lower", lower, 0.0048, 1.0),
    ):
        gp = base_p[ids]
        u = np.clip((gp[:, 0] - x0) / max(half, 1e-8), -1.0, 1.0)
        arch = np.sqrt(np.maximum(0.0, 1.0 - u * u))
        outer = np.clip(side_sign * u, 0.0, 1.0)
        outer_lift = -0.0012 * smoothstep01(outer)
        target_x = x0 + u * desired_half
        if group_name == "upper":
            target_y = cy - amp * arch + outer_lift
        else:
            target_y = cy + amp * arch + 0.25 * outer_lift
        delta = np.zeros((len(ids), 3), dtype=np.float64)
        delta[:, 0] = target_x - gp[:, 0]
        delta[:, 1] = target_y - gp[:, 1]

        # Each nearby skin vertex follows its nearest true rim handle. Compact XY
        # and depth support keeps the deformation in eyelid tissue only.
        tree2 = cKDTree(gp[:, :2])
        dist2, nearest = tree2.query(base_p[:, :2], k=1)
        wxy = np.exp(-0.5 * (dist2 / 0.0065) ** 4)
        rim_z = float(np.median(gp[:, 2]))
        wz = np.exp(-0.5 * ((base_p[:, 2] - rim_z) / 0.016) ** 4)
        if group_name == "upper":
            halfplane = smoothstep01((cy + 0.006 - base_p[:, 1]) / 0.012)
        else:
            halfplane = smoothstep01((base_p[:, 1] - (cy - 0.006)) / 0.012)
        w = wxy * wz * halfplane * 0.92
        total += delta[nearest] * w[:, None]
        influence_union = np.maximum(influence_union, w)
        group_stats.append({"group": group_name, "rim_vertices": int(len(ids)), "max_handle_displacement_m": float(np.linalg.norm(delta, axis=1).max())})

    return total, {
        "camera_center": eye_cam_center.tolist(),
        "rim_vertices": int(len(rim_ids)),
        "rim_width_before_m": float(2.0 * half),
        "rim_width_target_m": float(2.0 * desired_half),
        "upper_arch_m": 0.0072,
        "lower_arch_m": 0.0048,
        "max_influence": float(influence_union.max()),
        "groups": group_stats,
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
    depth = tri[:, :, 2].mean(axis=1)
    order = np.argsort(depth)[::-1]
    ff = faces[order]
    nn = n[order]
    tri2 = xy[ff]
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
        p = qa / f"AINA_{prefix}_CLAY_{label}_v10.6.1.png"
        render_mesh(vertices, faces, R, yaw, p, f"AINA v10.6.1 {prefix} {label.replace('_', ' ')}")
        paths.append(p)
    ims = [Image.open(p).convert("RGB") for p in paths]
    H = max(im.height for im in ims); W = max(im.width for im in ims)
    sheet = Image.new("RGB", (W * 5, H), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, (i * W + (W - im.width) // 2, (H - im.height) // 2))
    out = qa / f"AINA_{prefix}_CLAY_5VIEW_v10.6.1.png"
    sheet.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-full", type=Path, required=True)
    ap.add_argument("--base-skin", type=Path, required=True)
    ap.add_argument("--front", type=Path, required=True)
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("output_v1061"))
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
        raise RuntimeError("v10.1 full OBJ no longer has skin as its first vertex block")

    cameras = json.loads(args.cameras.read_text(encoding="utf-8"))
    R = np.asarray(cameras["front"]["rotation_rows"], dtype=np.float64)
    base_p = skin_v @ R.T
    p = base_p.copy()
    raw = np.zeros_like(p)

    # A) True eyelid rims from topology/eyeball proximity.
    eye_meshes = find_eyeballs(full, R)
    eye_stats = []
    for eye in eye_meshes:
        d, stat = reshape_eye_rim(base_p, skin_v, eye, R)
        raw += d
        eye_stats.append(stat)

    # B) One coherent youthful heart/V-face proportion pass.
    cx = 0.0050
    central = np.exp(-0.5 * ((base_p[:, 0] - cx) / 0.112) ** 6)
    # Restrict global silhouette deformation to the face/front surface, not scalp/back head.
    front_gate = 1.0 - smoothstep01((base_p[:, 2] + 0.095) / 0.115)

    # Cheek/lower-face narrowing begins under the eyes and strengthens toward chin.
    wface = smoothstep01((base_p[:, 1] + 0.280) / 0.165) * central * front_gate
    target_x = cx + (base_p[:, 0] - cx) * (1.0 - 0.12 * wface)
    raw[:, 0] += target_x - base_p[:, 0]

    jaw = smoothstep01((base_p[:, 1] + 0.205) / 0.100) * central * front_gate
    after_face_x = base_p[:, 0] + raw[:, 0]
    target_jaw_x = cx + (after_face_x - cx) * (1.0 - 0.15 * jaw)
    raw[:, 0] += target_jaw_x - after_face_x

    # Shorter lower face / smaller rounded chin.
    mouth_y = -0.183
    low = smoothstep01((base_p[:, 1] + 0.190) / 0.090) * central * front_gate
    target_y = mouth_y + (base_p[:, 1] - mouth_y) * (1.0 - 0.10 * low)
    raw[:, 1] += target_y - base_p[:, 1]

    # C) Delicate short nose: narrow alae/lower nose, reduce projection, lift tip slightly.
    nose = gaussian_ellipse(base_p, 0.0055, -0.211, 0.030, 0.036, power=2.1) * front_gate
    lower_nose = smoothstep01((base_p[:, 1] + 0.245) / 0.055)
    nose_w = nose * lower_nose
    raw[:, 0] += ((0.0055 + (base_p[:, 0] - 0.0055) * 0.80) - base_p[:, 0]) * nose_w
    raw[:, 2] += 0.0037 * nose
    tip = gaussian_ellipse(base_p, 0.0055, -0.205, 0.020, 0.018, power=2.0) * front_gate
    raw[:, 1] -= 0.0022 * tip

    # D) Small soft lips. Keep volume; compact width only slightly and lift as a unit.
    lips = gaussian_ellipse(base_p, 0.0055, -0.177, 0.045, 0.021, power=2.0) * front_gate
    raw[:, 0] += ((0.0055 + (base_p[:, 0] - 0.0055) * 0.94) - base_p[:, 0]) * lips
    raw[:, 1] -= 0.0010 * lips
    lip_core = gaussian_ellipse(base_p, 0.0055, -0.177, 0.031, 0.012, power=2.0) * front_gate
    raw[:, 2] -= 0.00055 * lip_core

    # E) Apple cheeks: modest forward volume, no broadening.
    for ccx in (-0.039, 0.049):
        cheek = gaussian_ellipse(base_p, ccx, -0.218, 0.035, 0.032, power=2.2) * front_gate
        raw[:, 2] -= 0.0018 * cheek

    # F) Rounded small chin support after shortening.
    chin = gaussian_ellipse(base_p, 0.005, -0.126, 0.040, 0.030, power=2.1) * front_gate
    raw[:, 1] -= 0.0015 * chin
    raw[:, 2] -= 0.0007 * chin

    # Hard safety cap before the very light anti-ripple pass.
    cap = 0.010
    rn = np.linalg.norm(raw, axis=1)
    over = rn > cap
    if np.any(over):
        raw[over] *= (cap / rn[over])[:, None]
    disp = smooth_displacement(raw, skin_f, iterations=2, alpha=0.07)
    dn = np.linalg.norm(disp, axis=1)
    over = dn > cap
    if np.any(over):
        disp[over] *= (cap / dn[over])[:, None]

    final_p = base_p + disp
    final_skin_v = final_p @ R
    final_full_v = full_v.copy()
    final_full_v[:n_skin] = final_skin_v

    final_skin = trimesh.Trimesh(vertices=final_skin_v, faces=skin_f, process=False)
    final_full = trimesh.Trimesh(vertices=final_full_v, faces=full_f, process=False)
    final_skin.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.6.1.obj")
    final_skin.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.6.1.ply")
    final_skin.export(args.out / "AINA_FACE_MASTER_SKIN_CLAY_v10.6.1.glb")
    final_full.export(args.out / "AINA_FACE_MASTER_GNM_v10.6.1_FULL_TOPOLOGY.obj")
    final_full.export(args.out / "AINA_FACE_MASTER_GNM_v10.6.1_FULL_TOPOLOGY.glb")

    skin_sheet = make_fiveview(final_skin_v, skin_f, R, qa, "SKIN")
    full_sheet = make_fiveview(final_full_v, full_f, R, qa, "FULL")

    front_render = qa / "AINA_FULL_CLAY_front_v10.6.1.png"
    ref = Image.open(args.front).convert("RGB")
    actual = Image.open(front_render).convert("RGB")
    H = max(ref.height, actual.height)
    rw = int(ref.width * H / ref.height); aw = int(actual.width * H / actual.height)
    compare = Image.new("RGB", (rw + aw, H), "white")
    compare.paste(ref.resize((rw, H)), (0, 0)); compare.paste(actual.resize((aw, H)), (rw, 0))
    compare.save(qa / "AINA_REFERENCE_VS_ACTUAL_FULL_FRONT_v10.6.1.png")

    report = {
        "version": "AINA Face Master v10.6.1 Reset Sculpt",
        "base": "v10.1 clean Google GNM v3 topology",
        "method": "single-pass reset sculpt: skin-to-eyeball eyelid-rim discovery + almond aperture + art-directed youthful heart/V proportions",
        "stacked_deformations_from_v10_3_to_v10_6": False,
        "full_vertices": int(len(final_full_v)),
        "full_triangles": int(len(full_f)),
        "skin_vertices": int(len(final_skin_v)),
        "skin_triangles": int(len(skin_f)),
        "eye_rims": eye_stats,
        "max_skin_displacement_m": float(np.linalg.norm(disp, axis=1).max()),
        "rms_skin_displacement_m": float(np.sqrt(np.mean(disp * disp))),
        "qa_full_fiveview": str(full_sheet),
        "qa_skin_fiveview": str(skin_sheet),
        "identity_lock": False,
        "acceptance_note": "Actual clean-reset mesh. Do not proceed to hair/expressions/VRM until visual front, 45-degree and profile clay likeness passes."
    }
    (args.out / "AINA_v10.6.1_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
