#!/usr/bin/env python3
"""Build AINA Custom Identity Head v1 without using the Rain face.

Inputs
------
* Approved-view FaceVerse identity coefficients and FaceVerse V4 model.
* Dense neutral BFM target reconstructed from approved front/3Q/side views.

Method
------
1. Reconstruct the exact FaceVerse neutral topology from its identity vector.
2. Align the FaceVerse skin to the dense BFM target with trimmed similarity ICP.
3. Compute a confidence-weighted non-rigid target displacement only where the
   dense target actually covers the face.
4. Regularize that displacement on the unchanged FaceVerse topology with a
   bounded Laplacian field.
5. Propagate nearby skin motion to separate facial components.
6. Export the unchanged production topology plus the neutral displacement field
   needed to rebuild all facial expression targets later.

This is a neutral-head identity gate. It does not use the Rain facial topology,
does not generate replacement effect art, and does not export VRM.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from matplotlib.collections import PolyCollection
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT = Path.cwd().resolve()
FVROOT = (ROOT / "vendor/faceverse-onnx").resolve()
sys.path.insert(0, str(FVROOT))
from faceversev4 import FaceVerseModel_torch  # noqa: E402


def normalize_face_height(vertices: np.ndarray, target_height: float = 0.180):
    v = np.asarray(vertices, np.float64).copy()
    centre = np.median(v, axis=0)
    v -= centre
    height = float(np.percentile(v[:, 1], 99.0) - np.percentile(v[:, 1], 1.0))
    scale = target_height / max(height, 1.0e-9)
    return v * scale, scale, centre


def umeyama(source: np.ndarray, target: np.ndarray, with_scale: bool = True):
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    if len(source) != len(target) or len(source) < 3:
        raise ValueError("Umeyama requires paired point sets with at least 3 points")
    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src = source - src_mean
    dst = target - dst_mean
    covariance = (dst.T @ src) / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    if with_scale:
        variance = float(np.sum(src * src) / len(source))
        scale = float(np.trace(np.diag(singular) @ correction) / max(variance, 1.0e-12))
    else:
        scale = 1.0
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray):
    return scale * (np.asarray(points) @ rotation.T) + translation


def robust_subset(points: np.ndarray):
    points = np.asarray(points)
    x_lo, x_hi = np.percentile(points[:, 0], [7.0, 93.0])
    y_lo, y_hi = np.percentile(points[:, 1], [5.0, 95.0])
    return (
        (points[:, 0] >= x_lo)
        & (points[:, 0] <= x_hi)
        & (points[:, 1] >= y_lo)
        & (points[:, 1] <= y_hi)
    )


def trimmed_similarity_icp(source: np.ndarray, target: np.ndarray, iterations: int = 12):
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    tree = cKDTree(target)

    best = None
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                sign = np.array([sx, sy, sz], np.float64)
                current = source * sign
                src_c = np.median(current, axis=0)
                dst_c = np.median(target, axis=0)
                src_h = np.percentile(current[:, 1], 95) - np.percentile(current[:, 1], 5)
                dst_h = np.percentile(target[:, 1], 95) - np.percentile(target[:, 1], 5)
                initial_scale = float(dst_h / max(src_h, 1.0e-9))
                current = (current - src_c) * initial_scale + dst_c

                total_scale = initial_scale
                total_rotation = np.eye(3)
                total_translation = dst_c - initial_scale * src_c
                for _ in range(iterations):
                    distances, nearest = tree.query(current, k=1)
                    cutoff = float(np.quantile(distances, 0.62))
                    keep = distances <= max(cutoff, 1.0e-6)
                    if int(np.sum(keep)) < 100:
                        break
                    scale, rotation, translation = umeyama(current[keep], target[nearest[keep]], with_scale=True)
                    current = apply_similarity(current, scale, rotation, translation)
                    total_translation = scale * (rotation @ total_translation) + translation
                    total_rotation = rotation @ total_rotation
                    total_scale *= scale

                distances, _ = tree.query(current, k=1)
                score = float(np.sqrt(np.mean(np.square(np.sort(distances)[: max(100, int(0.62 * len(distances)))]))))
                candidate = {
                    "score": score,
                    "sign": sign,
                    "points": current,
                    "scale": total_scale,
                    "rotation": total_rotation,
                    "translation": total_translation,
                }
                if best is None or candidate["score"] < best["score"]:
                    best = candidate
    if best is None:
        raise RuntimeError("Similarity ICP failed")
    return best


def adjacency(vertex_count: int, faces: np.ndarray, allowed: np.ndarray):
    allowed_set = set(np.flatnonzero(allowed).tolist())
    result = [set() for _ in range(vertex_count)]
    for tri in np.asarray(faces, np.int64):
        a, b, c = map(int, tri)
        for x, y in ((a, b), (b, c), (c, a)):
            if x in allowed_set and y in allowed_set:
                result[x].add(y)
                result[y].add(x)
    return result


def smooth_displacement(field: np.ndarray, neighbours, confidence: np.ndarray, passes: int = 12):
    result = np.asarray(field, np.float64).copy()
    confidence = np.asarray(confidence, np.float64)
    for _ in range(passes):
        updated = result.copy()
        for index, linked in enumerate(neighbours):
            if not linked:
                continue
            average = np.mean(result[list(linked)], axis=0)
            strength = 0.56 * (0.35 + 0.65 * confidence[index])
            updated[index] = result[index] * (1.0 - strength) + average * strength
        result = updated
    return result


def render_clay(vertices, faces, yaw_deg: float, path: Path, title: str):
    p = np.asarray(vertices, np.float64).copy()
    angle = math.radians(yaw_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    x = cosine * p[:, 0] + sine * p[:, 2]
    z = -sine * p[:, 0] + cosine * p[:, 2]
    p[:, 0], p[:, 2] = x, z
    triangles = p[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-9)
    order = np.argsort(triangles[:, :, 2].mean(axis=1))
    projected = p[faces[order], :2]
    n = normals[order]
    diffuse = np.clip(np.abs(n[:, 2]), 0.0, 1.0)
    side = np.clip(-0.28 * n[:, 0] - 0.12 * n[:, 1] + 0.74 * n[:, 2], 0.0, 1.0)
    intensity = np.clip(0.63 + 0.24 * diffuse + 0.10 * side, 0.48, 0.97)
    colours = np.stack([intensity * 0.96, intensity * 0.98, intensity], axis=1)
    xy = p[:, :2]
    lo = np.percentile(xy, 1.5, axis=0)
    hi = np.percentile(xy, 98.5, axis=0)
    centre = 0.5 * (lo + hi)
    extent = max(float((hi - lo).max()), 1.0e-6) * 0.57
    figure, axis = plt.subplots(figsize=(5, 5), dpi=190)
    axis.add_collection(PolyCollection(projected, facecolors=colours, edgecolors="none", closed=True))
    axis.set_xlim(centre[0] - extent, centre[0] + extent)
    axis.set_ylim(centre[1] - extent, centre[1] + extent)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, fontsize=10)
    figure.tight_layout(pad=0.12)
    figure.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def comparison_sheet(reference_paths: dict[str, Path], render_paths: dict[str, Path], output: Path):
    rows = []
    for label in ("front", "three_quarter", "side"):
        reference = Image.open(reference_paths[label]).convert("RGB")
        actual = Image.open(render_paths[label]).convert("RGB")
        reference.thumbnail((430, 430))
        actual.thumbnail((430, 430))
        panel = Image.new("RGB", (900, 485), "white")
        panel.paste(reference, ((440 - reference.width) // 2, 5))
        panel.paste(actual, (455 + (440 - actual.width) // 2, 5))
        draw = ImageDraw.Draw(panel)
        draw.text((10, 455), f"APPROVED AINA {label.upper()}", fill="black")
        draw.text((465, 455), f"CUSTOM HEAD v1 {label.upper()}", fill="black")
        rows.append(panel)
    sheet = Image.new("RGB", (900, 485 * len(rows)), "white")
    for index, row in enumerate(rows):
        sheet.paste(row, (0, index * 485))
    sheet.save(output)


def main():
    out = ROOT / "output_custom_head_v1"
    qa = out / "QA"
    out.mkdir(exist_ok=True)
    qa.mkdir(exist_ok=True)

    target_mesh = trimesh.load(ROOT / "output_dense_3ddfa/AINA_DENSE_BFM_NEUTRAL_v11.0.obj", process=False, maintain_order=True)
    target_vertices = np.asarray(target_mesh.vertices, np.float64)
    target_faces = np.asarray(target_mesh.faces, np.int64)
    target_vertices, _, _ = normalize_face_height(target_vertices)

    identity = np.load(ROOT / "output_faceverse_v120/AINA_FACEVERSE_IDENTITY_156_v12.0.npy").astype(np.float32)
    model = FaceVerseModel_torch(
        device=torch.device("cpu"),
        facevrsepath=str(FVROOT / "data/faceverse_v4_2.npy"),
        camera_distance=10,
        focal=1000,
        center=128,
    )
    neutral = np.zeros(int(model.all_dims), np.float32)
    neutral[: int(model.id_dims)] = identity
    with torch.no_grad():
        result = model.run(torch.from_numpy(neutral[None]).float(), only_lms=False, use_color=False)
    source_vertices = np.asarray(result["vertices"][0].cpu(), np.float64)
    source_faces = np.asarray(model.tri.cpu(), np.int64)
    if source_faces.min() == 1:
        source_faces -= 1
    source_vertices, source_metric_scale, source_centre = normalize_face_height(source_vertices)
    skin_mask = np.asarray(model.fvd["parsing"]["skin"]).reshape(-1) > 0
    if len(skin_mask) != len(source_vertices):
        raise RuntimeError("FaceVerse skin mask does not match vertex count")

    source_skin_ids = np.flatnonzero(skin_mask)
    source_skin = source_vertices[source_skin_ids]
    source_keep = robust_subset(source_skin)
    target_keep = robust_subset(target_vertices)
    alignment = trimmed_similarity_icp(source_skin[source_keep], target_vertices[target_keep], iterations=12)

    sign = alignment["sign"]
    signed_full = source_vertices * sign
    aligned_full = apply_similarity(signed_full, alignment["scale"], alignment["rotation"], alignment["translation"])
    aligned_skin = aligned_full[source_skin_ids]

    target_tree = cKDTree(target_vertices)
    distances, nearest = target_tree.query(aligned_skin, k=1)
    nearest_points = target_vertices[nearest]
    target_lo = np.percentile(target_vertices, 1.0, axis=0)
    target_hi = np.percentile(target_vertices, 99.0, axis=0)
    inside = (
        (aligned_skin[:, 0] >= target_lo[0] - 0.012)
        & (aligned_skin[:, 0] <= target_hi[0] + 0.012)
        & (aligned_skin[:, 1] >= target_lo[1] - 0.012)
        & (aligned_skin[:, 1] <= target_hi[1] + 0.012)
    )
    confidence_skin = np.exp(-np.square(distances / 0.0175)) * inside.astype(np.float64)
    confidence_skin[distances > 0.036] = 0.0
    raw_skin_delta = (nearest_points - aligned_skin) * confidence_skin[:, None]
    raw_length = np.linalg.norm(raw_skin_delta, axis=1)
    raw_skin_delta *= np.minimum(1.0, 0.012 / np.maximum(raw_length, 1.0e-9))[:, None]

    full_confidence = np.zeros(len(aligned_full), np.float64)
    full_confidence[source_skin_ids] = confidence_skin
    full_delta = np.zeros_like(aligned_full)
    full_delta[source_skin_ids] = raw_skin_delta
    neighbours = adjacency(len(aligned_full), source_faces, skin_mask)
    smoothed = smooth_displacement(full_delta, neighbours, full_confidence, passes=14)
    smoothed[~skin_mask] = 0.0
    smoothed_length = np.linalg.norm(smoothed, axis=1)
    smoothed *= np.minimum(1.0, 0.012 / np.maximum(smoothed_length, 1.0e-9))[:, None]

    final_vertices = aligned_full + smoothed

    skin_tree = cKDTree(aligned_full[source_skin_ids])
    non_skin_ids = np.flatnonzero(~skin_mask)
    if len(non_skin_ids):
        component_distances, component_nearest = skin_tree.query(aligned_full[non_skin_ids], k=1)
        component_weight = np.clip(1.0 - component_distances / 0.030, 0.0, 1.0)
        component_delta = smoothed[source_skin_ids[component_nearest]] * (0.88 * component_weight[:, None])
        final_vertices[non_skin_ids] += component_delta

    full_mesh = trimesh.Trimesh(vertices=final_vertices, faces=source_faces, process=False)
    full_mesh.export(out / "AINA_CUSTOM_HEAD_FULL_v1.obj")
    full_mesh.export(out / "AINA_CUSTOM_HEAD_FULL_v1.glb")

    skin_faces_mask = skin_mask[source_faces].all(axis=1)
    skin_mesh = full_mesh.submesh([np.flatnonzero(skin_faces_mask)], append=True, repair=False)
    skin_mesh.remove_unreferenced_vertices()
    skin_mesh.export(out / "AINA_CUSTOM_HEAD_SKIN_v1.obj")
    skin_mesh.export(out / "AINA_CUSTOM_HEAD_SKIN_v1.glb")
    skin_mesh.export(out / "AINA_CUSTOM_HEAD_SKIN_v1.ply")

    np.savez_compressed(
        out / "AINA_CUSTOM_HEAD_TRANSFER_v1.npz",
        identity=identity,
        source_vertices=source_vertices.astype(np.float32),
        aligned_vertices=aligned_full.astype(np.float32),
        neutral_displacement=(final_vertices - aligned_full).astype(np.float32),
        final_vertices=final_vertices.astype(np.float32),
        faces=source_faces.astype(np.int32),
        skin_mask=skin_mask.astype(np.uint8),
        sign=sign.astype(np.float32),
        scale=np.float32(alignment["scale"]),
        rotation=alignment["rotation"].astype(np.float32),
        translation=alignment["translation"].astype(np.float32),
    )

    skin_vertices = np.asarray(skin_mesh.vertices)
    skin_faces = np.asarray(skin_mesh.faces)
    render_paths = {
        "front": qa / "AINA_CUSTOM_HEAD_FRONT_v1.png",
        "three_quarter": qa / "AINA_CUSTOM_HEAD_3Q_v1.png",
        "side": qa / "AINA_CUSTOM_HEAD_SIDE_v1.png",
    }
    render_clay(skin_vertices, skin_faces, 0.0, render_paths["front"], "AINA Custom Head v1 front")
    render_clay(skin_vertices, skin_faces, 42.0, render_paths["three_quarter"], "AINA Custom Head v1 3Q")
    render_clay(skin_vertices, skin_faces, 88.0, render_paths["side"], "AINA Custom Head v1 side")

    five = []
    for yaw, label in ((-88, "left_profile"), (-42, "left_3q"), (0, "front"), (42, "right_3q"), (88, "right_profile")):
        path = qa / f"AINA_CUSTOM_HEAD_{label}_v1.png"
        render_clay(skin_vertices, skin_faces, yaw, path, f"AINA Custom Head v1 {label}")
        five.append(path)
    images = [Image.open(path).convert("RGB") for path in five]
    height = max(image.height for image in images)
    width = max(image.width for image in images)
    sheet = Image.new("RGB", (5 * width, height), "white")
    for index, image in enumerate(images):
        sheet.paste(image, (index * width + (width - image.width) // 2, (height - image.height) // 2))
    sheet.save(qa / "AINA_CUSTOM_HEAD_5VIEW_v1.png")

    references = {
        "front": ROOT / "references/AINA_APPROVED_FRONT.jpg",
        "three_quarter": ROOT / "references/AINA_APPROVED_3Q.jpg",
        "side": ROOT / "references/AINA_APPROVED_SIDE.jpg",
    }
    comparison_sheet(references, render_paths, qa / "AINA_APPROVED_VS_CUSTOM_HEAD_v1.png")

    final_skin = final_vertices[source_skin_ids]
    final_distances, _ = target_tree.query(final_skin, k=1)
    confident = confidence_skin > 0.20
    report = {
        "product": "AINA Custom Identity Head v1",
        "direction_pivot": True,
        "primary_face_direction": "CUSTOM_FACEVERSE_DENSE_GRAFT",
        "rain_face_used": False,
        "rain_body_or_rig_used": False,
        "source_topology": "FaceVerse V4",
        "dense_identity_target": "3DDFA V2 BFM neutral blended from approved front/3Q/side",
        "replacement_effect_art_generated": False,
        "vertices": int(len(final_vertices)),
        "triangles": int(len(source_faces)),
        "skin_vertices": int(np.sum(skin_mask)),
        "topology_changed": False,
        "vertex_order_preserved": True,
        "neutral_displacement_saved": True,
        "expression_transfer_ready": True,
        "source_metric_scale": float(source_metric_scale),
        "source_metric_centre": source_centre.tolist(),
        "alignment": {
            "trimmed_icp_rmse_m": float(alignment["score"]),
            "sign": sign.tolist(),
            "scale": float(alignment["scale"]),
            "rotation": alignment["rotation"].tolist(),
            "translation": alignment["translation"].tolist(),
        },
        "graft": {
            "confident_skin_vertices": int(np.sum(confident)),
            "confidence_mean": float(confidence_skin.mean()),
            "raw_displacement_max_m": float(np.linalg.norm(raw_skin_delta, axis=1).max()),
            "final_neutral_displacement_max_m": float(np.linalg.norm(final_vertices - aligned_full, axis=1).max()),
            "final_target_rmse_m": float(np.sqrt(np.mean(np.square(final_distances[confident])))) if np.any(confident) else None,
        },
        "identity_lock": False,
        "visual_identity_lock": False,
        "production_release": False,
        "candidate": True,
        "vrm_exported": False,
        "next_gate": "Inspect actual custom-head front, 3Q, profile and five-view clay. If accepted, apply the saved neutral displacement identically to all FaceVerse expression targets and graft this head to the adult body/rig.",
        "files": {
            "full_obj": str(out / "AINA_CUSTOM_HEAD_FULL_v1.obj"),
            "full_glb": str(out / "AINA_CUSTOM_HEAD_FULL_v1.glb"),
            "skin_obj": str(out / "AINA_CUSTOM_HEAD_SKIN_v1.obj"),
            "transfer_npz": str(out / "AINA_CUSTOM_HEAD_TRANSFER_v1.npz"),
            "comparison": str(qa / "AINA_APPROVED_VS_CUSTOM_HEAD_v1.png"),
            "five_view": str(qa / "AINA_CUSTOM_HEAD_5VIEW_v1.png"),
        },
    }
    (out / "AINA_CUSTOM_HEAD_v1_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
