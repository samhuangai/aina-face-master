#!/usr/bin/env python3
"""AINA Identity Master Reconstruction on Google GNM v3.

This stage deliberately replaces the unsuitable FaceVerse identity base instead
of stacking more local patches on it. It fits a female-Asian GNM prior to the
approved AINA front, three-quarter and side references, then performs bounded
multi-view surface correction without changing topology.

Outputs are real OBJ/GLB/PLY meshes and clay QA renders. Identity remains
unlocked until the generated neutral head is visually accepted.
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
from PIL import Image, ImageDraw
import trimesh

from gnm.shape import gnm_numpy, gnm_landmarks
from gnm.shape.semantic_sampler import IdentitySampler, Gender, Ethnicity


GNM_TO_STANDARD = np.array(
    [0, 1, 6, 5, 4, 3, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, *range(17, 68)],
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--front-target", type=Path, required=True)
    ap.add_argument("--front-ref", type=Path, required=True)
    ap.add_argument("--q3-ref", type=Path, required=True)
    ap.add_argument("--side-ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("output_identity_master"))
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260821)
    return ap.parse_args()


def normalize_target(points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    scale = 0.5 * max(width, height)
    return (np.asarray(points, np.float64) - np.array([width * 0.5, height * 0.5])) / scale


def landmark_weights(view: str) -> np.ndarray:
    w = np.ones(68, np.float64)
    w[:17] = 1.65
    w[17:27] = 0.60
    w[27:36] = 2.65
    w[36:48] = 3.10
    w[48:68] = 2.75
    if view == "side":
        w[:] = 0.45
        w[:17] = 2.30
        w[27:36] = 3.15
        w[36:48] = 0.70
        w[48:68] = 2.75
    elif view == "three_quarter":
        w[:17] = 1.85
        w[27:36] = 2.90
        w[36:48] = 2.80
    return w


def fit_camera(points_3d: np.ndarray, target_2d: np.ndarray, weights: np.ndarray):
    """Weighted weak-perspective camera with orthonormalized rows."""
    x = np.c_[points_3d, np.ones(len(points_3d))]
    sw = np.sqrt(weights)[:, None]
    beta = np.linalg.lstsq(x * sw, target_2d * sw, rcond=None)[0]
    affine = beta[:3].T
    translate = beta[3]
    n1 = np.linalg.norm(affine[0])
    n2 = np.linalg.norm(affine[1])
    scale = max(1e-9, 0.5 * (n1 + n2))
    r1 = affine[0] / max(n1, 1e-9)
    v2 = affine[1] - np.dot(affine[1], r1) * r1
    r2 = v2 / max(np.linalg.norm(v2), 1e-9)
    r3 = np.cross(r1, r2)
    r3 /= max(np.linalg.norm(r3), 1e-9)
    r2 = np.cross(r3, r1)
    rotation = np.stack([r1, r2, r3])
    if np.linalg.det(rotation) < 0:
        rotation[2] *= -1
    return rotation, scale, translate


def project(points: np.ndarray, camera) -> np.ndarray:
    rotation, scale, translate = camera
    return scale * (points @ rotation.T)[:, :2] + translate


def weighted_rmse(pred: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    err2 = np.sum((pred - target) ** 2, axis=1)
    return float(np.sqrt(np.sum(weights * err2) / np.sum(weights)))


def load_targets(multiview_path: Path, front_override_path: Path):
    raw = json.loads(multiview_path.read_text())
    front_override = json.loads(front_override_path.read_text())
    raw["front"] = front_override
    result = {}
    for name in ("front", "three_quarter", "side"):
        item = raw[name]
        result[name] = {
            "points": normalize_target(np.asarray(item["landmarks_xy"], np.float64), tuple(item["image_size"])),
            "size": tuple(item["image_size"]),
        }
    return result


def sparse_landmark_model(gnm):
    config = gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
    indices = np.asarray(config.indices, np.int64)
    blend_weights = np.asarray(config.weights, np.float64)
    template = np.asarray(gnm.template_vertex_positions, np.float64)
    identity_basis = np.asarray(gnm.vertex_identity_basis, np.float64)
    lm_template = (template[indices] * blend_weights[..., None]).sum(-2)[GNM_TO_STANDARD]
    lm_basis = (
        identity_basis[:, indices, :] * blend_weights[None, ..., None]
    ).sum(-2)[:, GNM_TO_STANDARD, :]
    return indices[GNM_TO_STANDARD], blend_weights[GNM_TO_STANDARD], lm_template, lm_basis


def sample_score(landmarks: np.ndarray, targets) -> tuple[float, dict]:
    total = 0.0
    cameras = {}
    view_scale = {"front": 1.0, "three_quarter": 1.15, "side": 1.05}
    for name in ("front", "three_quarter", "side"):
        target = targets[name]["points"]
        weights = landmark_weights(name)
        camera = fit_camera(landmarks, target, weights)
        cameras[name] = camera
        total += view_scale[name] * weighted_rmse(project(landmarks, camera), target, weights)
    return total, cameras


def build_female_prior(samples: np.ndarray, components: int = 72):
    mean = samples.mean(axis=0)
    centered = samples - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    count = min(components, len(singular), vt.shape[0])
    std = singular[:count] / math.sqrt(max(len(samples) - 1, 1))
    std = np.maximum(std, 1e-5)
    axes = vt[:count]
    return mean, axes, std


def identity_to_latent(identity: np.ndarray, mean: np.ndarray, axes: np.ndarray, std: np.ndarray):
    return ((identity - mean) @ axes.T) / std


def latent_to_identity(z: np.ndarray, mean: np.ndarray, axes: np.ndarray, std: np.ndarray):
    return mean + (z * std) @ axes


def solve_identity_prior(
    lm_template: np.ndarray,
    lm_identity_basis: np.ndarray,
    targets,
    prior_mean: np.ndarray,
    prior_axes: np.ndarray,
    prior_std: np.ndarray,
    initial_identity: np.ndarray,
):
    """Alternating camera / linear female-prior solve."""
    z = identity_to_latent(initial_identity, prior_mean, prior_axes, prior_std)
    z = np.clip(z, -2.8, 2.8)

    lm_mean = lm_template + np.einsum("i,ilc->lc", prior_mean, lm_identity_basis)
    identity_axes = prior_axes * prior_std[:, None]
    lm_latent_basis = np.einsum("ki,ilc->klc", identity_axes, lm_identity_basis)

    history = []
    for iteration in range(14):
        landmarks = lm_mean + np.einsum("k,klc->lc", z, lm_latent_basis)
        blocks = []
        rhs = []
        cameras = {}
        before = {}
        for name in ("front", "three_quarter", "side"):
            target = targets[name]["points"]
            weights = landmark_weights(name)
            camera = fit_camera(landmarks, target, weights)
            cameras[name] = camera
            before[name] = weighted_rmse(project(landmarks, camera), target, weights)
            rotation, scale, translate = camera
            base2 = scale * (lm_mean @ rotation.T)[:, :2] + translate
            basis2 = scale * np.einsum("klc,dc->kld", lm_latent_basis, rotation)[:, :, :2]
            matrix = basis2.transpose(1, 2, 0).reshape(-1, len(z))
            vector = (target - base2).reshape(-1)
            sw = np.repeat(np.sqrt(weights), 2)
            blocks.append(matrix * sw[:, None])
            rhs.append(vector * sw)

        regularization = 0.30 + 0.035 * iteration
        blocks.append(np.eye(len(z)) * math.sqrt(regularization))
        rhs.append(np.zeros(len(z)))
        matrix = np.vstack(blocks)
        vector = np.concatenate(rhs)
        solved = np.linalg.lstsq(matrix, vector, rcond=1e-6)[0]
        z = np.clip(0.34 * z + 0.66 * solved, -3.0, 3.0)

        landmarks_after = lm_mean + np.einsum("k,klc->lc", z, lm_latent_basis)
        after = {}
        for name, camera in cameras.items():
            after[name] = weighted_rmse(
                project(landmarks_after, camera),
                targets[name]["points"],
                landmark_weights(name),
            )
        history.append({"iteration": iteration, "before": before, "after": after, "latent_norm": float(np.linalg.norm(z))})

    identity = latent_to_identity(z, prior_mean, prior_axes, prior_std)
    landmarks = lm_template + np.einsum("i,ilc->lc", identity, lm_identity_basis)
    cameras = {}
    metrics = {}
    for name in ("front", "three_quarter", "side"):
        camera = fit_camera(landmarks, targets[name]["points"], landmark_weights(name))
        cameras[name] = camera
        metrics[name] = weighted_rmse(project(landmarks, camera), targets[name]["points"], landmark_weights(name))
    return identity, landmarks, cameras, metrics, history, z


def compute_landmarks(vertices: np.ndarray, indices: np.ndarray, blend_weights: np.ndarray):
    return (vertices[indices] * blend_weights[..., None]).sum(-2)


def landmark_radius(index: int) -> float:
    if index < 17:
        return 0.043
    if index < 27:
        return 0.031
    if index < 36:
        return 0.027
    if index < 48:
        return 0.028
    return 0.030


def infer_mirror_axis(vertices: np.ndarray, mirror_indices: np.ndarray) -> int:
    scores = []
    for axis in range(3):
        mirrored = vertices[mirror_indices].copy()
        mirrored[:, axis] *= -1
        scores.append(float(np.median(np.linalg.norm(vertices - mirrored, axis=1))))
    return int(np.argmin(scores))


def multiview_surface_converge(
    vertices: np.ndarray,
    triangles: np.ndarray,
    landmark_indices: np.ndarray,
    landmark_blend_weights: np.ndarray,
    targets,
    mirror_indices: np.ndarray,
):
    """Bounded RBF surface correction from approved front/3Q/side views."""
    base = vertices.copy()
    out = vertices.copy()
    view_steps = {"front": 0.55, "three_quarter": 0.42, "side": 0.46}
    history = []

    for iteration, global_step in enumerate((1.0, 0.72, 0.52, 0.36, 0.24)):
        landmarks = compute_landmarks(out, landmark_indices, landmark_blend_weights)
        accumulated = np.zeros_like(out)
        denominator = np.zeros(len(out), np.float64)
        view_metrics = {}

        for name in ("front", "three_quarter", "side"):
            target = targets[name]["points"]
            weights = landmark_weights(name)
            camera = fit_camera(landmarks, target, weights)
            prediction = project(landmarks, camera)
            residual = target - prediction
            view_metrics[name] = weighted_rmse(prediction, target, weights)
            rotation, scale, _ = camera
            camera_displacement = np.c_[residual / max(scale, 1e-8), np.zeros(len(residual))]
            world_displacement = camera_displacement @ rotation
            max_anchor = 0.0045 if name != "side" else 0.0052
            lengths = np.linalg.norm(world_displacement, axis=1)
            cap = np.minimum(1.0, max_anchor / np.maximum(lengths, 1e-9))
            world_displacement *= cap[:, None]
            world_displacement *= view_steps[name] * global_step

            for landmark_index, center in enumerate(landmarks):
                radius = landmark_radius(landmark_index)
                distance = np.linalg.norm(out - center, axis=1)
                local = np.exp(-0.5 * (distance / radius) ** 4)
                local[distance > radius * 1.45] = 0.0
                local *= weights[landmark_index]
                accumulated += local[:, None] * world_displacement[landmark_index]
                denominator += local

        delta = accumulated / np.maximum(denominator[:, None], 1e-9)
        delta[denominator < 0.08] = 0.0
        length = np.linalg.norm(delta, axis=1)
        cap = np.minimum(1.0, 0.0028 / np.maximum(length, 1e-9))
        delta *= cap[:, None]
        out += delta

        history.append(
            {
                "iteration": iteration,
                "view_rmse_before": view_metrics,
                "max_vertex_step_m": float(np.linalg.norm(delta, axis=1).max()),
                "rms_vertex_step_m": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
            }
        )

    mirror_axis = infer_mirror_axis(out, mirror_indices)
    mirrored = out[mirror_indices].copy()
    mirrored[:, mirror_axis] *= -1
    support_landmarks = compute_landmarks(out, landmark_indices, landmark_blend_weights)
    face_distance = np.min(
        np.linalg.norm(out[:, None, :] - support_landmarks[None, :, :], axis=2),
        axis=1,
    )
    symmetry_weight = np.clip((0.090 - face_distance) / 0.055, 0.0, 1.0) * 0.42
    out = out * (1.0 - symmetry_weight[:, None]) + 0.5 * (out + mirrored) * symmetry_weight[:, None]

    tri0 = base[triangles]
    tri1 = out[triangles]
    area0 = 0.5 * np.linalg.norm(np.cross(tri0[:, 1] - tri0[:, 0], tri0[:, 2] - tri0[:, 0]), axis=1)
    area1 = 0.5 * np.linalg.norm(np.cross(tri1[:, 1] - tri1[:, 0], tri1[:, 2] - tri1[:, 0]), axis=1)
    ratio = area1 / np.maximum(area0, 1e-12)
    health = {
        "mirror_axis": mirror_axis,
        "max_total_displacement_m": float(np.linalg.norm(out - base, axis=1).max()),
        "rms_total_displacement_m": float(np.sqrt(np.mean(np.sum((out - base) ** 2, axis=1)))),
        "triangle_area_ratio_p01": float(np.percentile(ratio, 1)),
        "triangle_area_ratio_p99": float(np.percentile(ratio, 99)),
        "degenerate_triangles": int(np.sum(area1 < 1e-12)),
    }
    return out, history, health


def render_mesh(vertices: np.ndarray, faces: np.ndarray, rotation: np.ndarray, path: Path, title: str):
    p = vertices @ rotation.T
    xy = p[:, :2]
    tri = p[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    order = np.argsort(tri[:, :, 2].mean(axis=1))[::-1]
    triangles_2d = xy[faces[order]]
    n = normals[order]
    diffuse = np.clip(np.abs(n[:, 2]), 0.0, 1.0)
    side = np.clip(-0.35 * n[:, 0] - 0.20 * n[:, 1] - 0.72 * n[:, 2], 0.0, 1.0)
    intensity = np.clip(0.62 + 0.24 * diffuse + 0.12 * side, 0.42, 0.98)
    colors = np.stack([intensity * 0.95, intensity * 0.97, intensity], axis=1)
    lo = np.percentile(xy, 1.0, axis=0)
    hi = np.percentile(xy, 99.0, axis=0)
    center = 0.5 * (lo + hi)
    extent = max(float((hi - lo).max()), 1e-6) * 0.56
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=190)
    ax.add_collection(PolyCollection(triangles_2d, facecolors=colors, edgecolors="none"))
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] + extent, center[1] - extent)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.08)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def compare(reference: Path, render: Path, output: Path, label: str):
    ref = Image.open(reference).convert("RGB")
    img = Image.open(render).convert("RGB")
    height = max(ref.height, img.height)
    rw = int(ref.width * height / ref.height)
    iw = int(img.width * height / img.height)
    canvas = Image.new("RGB", (rw + iw, height + 36), "white")
    canvas.paste(ref.resize((rw, height)), (0, 0))
    canvas.paste(img.resize((iw, height)), (rw, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, height + 9), f"Approved {label}", fill="black")
    draw.text((rw + 8, height + 9), "AINA Identity Master real clay", fill="black")
    canvas.save(output)


def yaw_rotation(front_rotation: np.ndarray, yaw_degrees: float):
    right, up, forward = front_rotation[0], front_rotation[1], front_rotation[2]
    angle = math.radians(yaw_degrees)
    return np.stack(
        [
            math.cos(angle) * right + math.sin(angle) * forward,
            up,
            -math.sin(angle) * right + math.cos(angle) * forward,
        ]
    )


def export_meshes(gnm, vertices: np.ndarray, out: Path):
    triangles = np.asarray(gnm.triangles, np.int64)
    full = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    for ext in ("obj", "glb", "ply"):
        full.export(out / f"AINA_IDENTITY_MASTER_GNM_FULL.{ext}")

    skin_triangles = np.asarray(gnm.triangle_indices_for_group("skin"), np.int64)
    skin = full.submesh([skin_triangles], append=True, repair=False)
    skin.remove_unreferenced_vertices()
    for ext in ("obj", "glb", "ply"):
        skin.export(out / f"AINA_IDENTITY_MASTER_GNM_SKIN.{ext}")
    return full, skin


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)

    targets = load_targets(args.targets, args.front_target)
    gnm = gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3,
        variant=gnm_numpy.GNMVariant.HEAD,
    )
    lm_indices, lm_blend_weights, lm_template, lm_identity_basis = sparse_landmark_model(gnm)

    sampler = IdentitySampler()
    rng = np.random.default_rng(args.seed)
    samples = sampler.sample_identity(
        Gender.FEMALE,
        Ethnicity.ASIAN,
        num_samples=args.samples,
        rng=rng,
    ).astype(np.float64)

    scored = []
    for index, identity in enumerate(samples):
        landmarks = lm_template + np.einsum("i,ilc->lc", identity, lm_identity_basis)
        score, cameras = sample_score(landmarks, targets)
        scored.append((score, index, cameras))
    scored.sort(key=lambda item: item[0])
    best_score, best_index, _ = scored[0]
    best_identity = samples[best_index]

    prior_mean, prior_axes, prior_std = build_female_prior(samples)
    identity, fitted_landmarks, fitted_cameras, fit_metrics, fit_history, latent = solve_identity_prior(
        lm_template,
        lm_identity_basis,
        targets,
        prior_mean,
        prior_axes,
        prior_std,
        best_identity,
    )

    vertices = np.asarray(gnm(identity=identity[None, :]))[0].astype(np.float64)
    triangles = np.asarray(gnm.triangles, np.int64)
    refined_vertices, surface_history, health = multiview_surface_converge(
        vertices,
        triangles,
        lm_indices,
        lm_blend_weights,
        targets,
        np.asarray(gnm.mirror_indices, np.int64),
    )

    final_landmarks = compute_landmarks(refined_vertices, lm_indices, lm_blend_weights)
    final_cameras = {}
    final_metrics = {}
    for name in ("front", "three_quarter", "side"):
        camera = fit_camera(final_landmarks, targets[name]["points"], landmark_weights(name))
        final_cameras[name] = camera
        final_metrics[name] = weighted_rmse(
            project(final_landmarks, camera),
            targets[name]["points"],
            landmark_weights(name),
        )

    full, skin = export_meshes(gnm, refined_vertices, args.out)
    np.save(args.out / "AINA_IDENTITY_MASTER_GNM_IDENTITY.npy", identity.astype(np.float32))
    np.save(args.out / "AINA_IDENTITY_MASTER_GNM_LATENT.npy", latent.astype(np.float32))

    render_paths = {}
    for name, title in (
        ("front", "AINA Identity Master — front"),
        ("three_quarter", "AINA Identity Master — approved 3Q"),
        ("side", "AINA Identity Master — approved side"),
    ):
        path = qa / f"AINA_IDENTITY_MASTER_{name.upper()}_CLAY.png"
        render_mesh(np.asarray(skin.vertices), np.asarray(skin.faces), final_cameras[name][0], path, title)
        render_paths[name] = path

    compare(args.front_ref, render_paths["front"], qa / "AINA_APPROVED_VS_IDENTITY_MASTER_FRONT.png", "front")
    compare(args.q3_ref, render_paths["three_quarter"], qa / "AINA_APPROVED_VS_IDENTITY_MASTER_3Q.png", "3Q")
    compare(args.side_ref, render_paths["side"], qa / "AINA_APPROVED_VS_IDENTITY_MASTER_SIDE.png", "side")

    front_rotation = final_cameras["front"][0]
    five = []
    for yaw, label in ((-90, "LEFT_PROFILE"), (-45, "LEFT_45"), (0, "FRONT"), (45, "RIGHT_45"), (90, "RIGHT_PROFILE")):
        path = qa / f"AINA_IDENTITY_MASTER_{label}.png"
        render_mesh(np.asarray(skin.vertices), np.asarray(skin.faces), yaw_rotation(front_rotation, yaw), path, f"AINA Identity Master {label}")
        five.append(path)
    images = [Image.open(path).convert("RGB") for path in five]
    width = max(img.width for img in images)
    height = max(img.height for img in images)
    sheet = Image.new("RGB", (5 * width, height), "white")
    for index, img in enumerate(images):
        sheet.paste(img, (index * width + (width - img.width) // 2, (height - img.height) // 2))
    sheet.save(qa / "AINA_IDENTITY_MASTER_5VIEW.png")

    report = {
        "product": "AINA Identity Master Reconstruction",
        "source_model": "Google GNM v3 HEAD",
        "prior": "GNM semantic sampler Female + Asian",
        "sample_count": int(args.samples),
        "best_prior_sample_index": int(best_index),
        "best_prior_multiview_score": float(best_score),
        "identity_dimension": int(gnm.identity_dim),
        "expression_dimension_available": int(gnm.expression_dim),
        "vertices": int(len(refined_vertices)),
        "triangles": int(len(triangles)),
        "skin_vertices": int(len(skin.vertices)),
        "skin_triangles": int(len(skin.faces)),
        "topology_changed": False,
        "new_reference_generated": False,
        "real_mesh_reconstructed": True,
        "female_asian_prior": True,
        "fit_rmse_before_surface": fit_metrics,
        "fit_rmse_after_surface": final_metrics,
        "fit_history": fit_history,
        "surface_history": surface_history,
        "mesh_health": health,
        "identity_lock": False,
        "visual_identity_lock": False,
        "candidate": True,
        "next_gate": "Visually inspect approved front, 3Q and side comparisons. Lock only after the real neutral clay resembles AINA in all views.",
    }
    (args.out / "AINA_IDENTITY_MASTER_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "AINA_IDENTITY_MASTER_CAMERAS.json").write_text(
        json.dumps(
            {
                name: {
                    "rotation_rows": camera[0].tolist(),
                    "scale": float(camera[1]),
                    "translation": np.asarray(camera[2]).tolist(),
                }
                for name, camera in final_cameras.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
