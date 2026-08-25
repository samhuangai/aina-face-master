#!/usr/bin/env python3
"""AINA Direct Reference Sculpt v2.

This stage deliberately stops automatic identity regression. It keeps only the
stable FaceVerse vertex order and expression-capable topology from Custom Head
v1, then sculpts that real Mesh directly against the approved AINA front
landmarks and art-directed front/profile proportions.

The same neutral displacement remains transferable to every later expression
target. No Rain facial topology, replacement effect art, body rig or VRM is
used in this stage.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch
import trimesh

ROOT = Path.cwd().resolve()
FVROOT = (ROOT / "vendor/faceverse-onnx").resolve()
sys.path.insert(0, str(FVROOT))
from faceversev4 import FaceVerseModel_torch


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def ellipsoid(points: np.ndarray, centre, radii, outer=1.0) -> np.ndarray:
    centre = np.asarray(centre, np.float64)
    radii = np.maximum(np.asarray(radii, np.float64), 1.0e-8)
    q = np.sqrt(np.sum(((points - centre) / radii) ** 2, axis=1))
    weight = np.zeros(len(points), np.float64)
    mask = q < outer
    if np.any(mask):
        t = q[mask] / outer
        weight[mask] = 0.5 * (1.0 + np.cos(np.pi * t))
    return weight


def adjacency(vertex_count: int, faces: np.ndarray, active: np.ndarray) -> list[list[int]]:
    active_set = set(np.flatnonzero(active).tolist())
    result = [set() for _ in range(vertex_count)]
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        for x, y in ((a, b), (b, c), (c, a)):
            if x in active_set and y in active_set:
                result[x].add(y)
                result[y].add(x)
    return [list(item) for item in result]


def smooth_field(field: np.ndarray, graph: list[list[int]], preserve: np.ndarray, passes=4) -> np.ndarray:
    result = field.copy()
    active_ids = np.flatnonzero(np.linalg.norm(field, axis=1) > 0)
    for _ in range(passes):
        updated = result.copy()
        for index in active_ids:
            neighbours = graph[int(index)]
            if not neighbours:
                continue
            average = result[neighbours].mean(axis=0)
            alpha = 0.42 * (1.0 - 0.86 * preserve[index])
            updated[index] = result[index] * (1.0 - alpha) + average * alpha
        result = updated
    return result


def target_landmarks(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    width, height = data["image_size"]
    points = np.asarray(data["landmarks_xy"], np.float64)
    points[:, 0] /= float(width)
    points[:, 1] = 1.0 - points[:, 1] / float(height)
    return points


def desired_front_xy(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Use the jaw endpoints as a stable physical scale and the nose bridge as
    # the identity centre. This keeps the head in the existing DCC scale while
    # imposing the approved AINA facial proportions.
    source_width = max(float(np.linalg.norm(source[0, :2] - source[16, :2])), 1.0e-6)
    target_width = max(float(np.linalg.norm(target[0] - target[16])), 1.0e-6)
    source_centre = 0.5 * (source[0, :2] + source[16, :2])
    target_centre = 0.5 * (target[0] + target[16])
    desired = source_centre + (target - target_centre) * (source_width / target_width)

    # Do not inherit target-image framing. Align the nose root vertically to
    # the actual Mesh and let only relative feature proportions change.
    desired += source[27, :2] - desired[27]
    return desired


def landmark_radius(index: int) -> float:
    if index < 17:
        return 0.032
    if index < 27:
        return 0.024
    if index < 36:
        return 0.021
    if index < 48:
        return 0.022
    return 0.025


def apply_landmark_residual(
    vertices: np.ndarray,
    surface_mask: np.ndarray,
    landmark_ids: np.ndarray,
    desired_xy: np.ndarray,
    passes=(0.54, 0.34, 0.20),
) -> tuple[np.ndarray, dict, np.ndarray]:
    current = vertices.copy()
    preserve = np.zeros(len(vertices), np.float64)
    history = []
    surface_ids = np.flatnonzero(surface_mask)

    for gain in passes:
        source_lms = current[landmark_ids]
        residual = desired_xy - source_lms[:, :2]
        accumulation = np.zeros_like(current)
        denominator = np.zeros(len(current), np.float64)

        for index, vertex_id in enumerate(landmark_ids):
            centre = current[int(vertex_id)]
            radius = landmark_radius(index)
            distance = np.linalg.norm(current[surface_ids] - centre, axis=1)
            weight = np.exp(-0.5 * (distance / max(radius, 1.0e-6)) ** 4)
            weight[distance > radius * 1.55] = 0.0
            vector = np.array([residual[index, 0], residual[index, 1], 0.0]) * gain
            length = float(np.linalg.norm(vector))
            if length > 0.0040:
                vector *= 0.0040 / length
            accumulation[surface_ids] += weight[:, None] * vector
            denominator[surface_ids] += weight
            preserve[surface_ids] = np.maximum(preserve[surface_ids], np.clip(weight, 0.0, 1.0))

        step = accumulation / np.maximum(denominator[:, None], 1.0e-9)
        step[denominator < 0.035] = 0.0
        lengths = np.linalg.norm(step, axis=1)
        step *= np.minimum(1.0, 0.0038 / np.maximum(lengths, 1.0e-9))[:, None]
        current += step
        history.append({
            "gain": gain,
            "landmark_rmse_before_m": float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))),
            "max_step_m": float(np.linalg.norm(step, axis=1).max()),
        })

    final_residual = desired_xy - current[landmark_ids, :2]
    return current, {
        "history": history,
        "final_landmark_rmse_m": float(np.sqrt(np.mean(np.sum(final_residual * final_residual, axis=1)))),
        "final_landmark_max_m": float(np.linalg.norm(final_residual, axis=1).max()),
    }, preserve


def art_directed_depth_and_cranium(
    vertices: np.ndarray,
    surface_mask: np.ndarray,
    landmark_ids: np.ndarray,
) -> tuple[np.ndarray, dict, np.ndarray]:
    result = vertices.copy()
    points = result
    lms = points[landmark_ids]
    surface_ids = np.flatnonzero(surface_mask)
    surface = points[surface_ids]
    face_x = float(0.5 * (lms[0, 0] + lms[16, 0]))
    eye_z = float(0.5 * (lms[39, 1] + lms[42, 1]))
    mouth_centre = lms[48:68].mean(axis=0)
    nose_centre = lms[27:36].mean(axis=0)
    chin = lms[8]
    head_top = float(np.percentile(surface[:, 1], 99.2))
    head_bottom = float(np.percentile(surface[:, 1], 0.8))
    head_height = max(head_top - head_bottom, 1.0e-6)
    median_depth = float(np.median(surface[:, 2]))
    front_sign = 1.0 if lms[30, 2] > median_depth else -1.0

    delta = np.zeros_like(points)
    preserve = np.zeros(len(points), np.float64)

    # Young AINA cranium: slightly shorter and narrower, without crushing the
    # brow/temple transition.
    top_origin = eye_z + 0.055 * head_height
    top = np.clip((points[:, 1] - top_origin) / max(head_top - top_origin, 1.0e-6), 0.0, 1.0)
    top_weight = smoothstep(top) * surface_mask.astype(np.float64)
    delta[:, 0] += -(points[:, 0] - face_x) * (0.075 * top) * top_weight
    delta[:, 1] += -(points[:, 1] - top_origin) * (0.080 * top) * top_weight

    # Small, delicate nose: narrow bridge/base and reduce excessive projection.
    bridge = lms[27:31].mean(axis=0)
    bridge_w = ellipsoid(points, bridge, (0.028, 0.043, 0.040), 1.22) * surface_mask
    delta[:, 0] += -(points[:, 0] - face_x) * 0.18 * bridge_w
    tip = lms[30]
    tip_w = ellipsoid(points, tip, (0.025, 0.028, 0.026), 1.15) * surface_mask
    delta[:, 0] += -(points[:, 0] - face_x) * 0.22 * tip_w
    delta[:, 2] += -front_sign * 0.0021 * tip_w
    base = lms[31:36].mean(axis=0)
    base_w = ellipsoid(points, base, (0.037, 0.026, 0.032), 1.15) * surface_mask
    delta[:, 0] += -(points[:, 0] - face_x) * 0.16 * base_w
    preserve = np.maximum(preserve, np.clip(bridge_w + tip_w + base_w, 0.0, 1.0))

    # Compact lips integrated into the face rather than a broad protruding pad.
    lip_w = ellipsoid(points, mouth_centre, (0.055, 0.030, 0.030), 1.18) * surface_mask
    delta[:, 0] += -(points[:, 0] - mouth_centre[0]) * 0.10 * lip_w
    delta[:, 1] += -(points[:, 1] - mouth_centre[1]) * 0.15 * lip_w
    delta[:, 2] += -front_sign * 0.0013 * lip_w
    preserve = np.maximum(preserve, np.clip(lip_w, 0.0, 1.0))

    # Soft V jaw and shorter lower face while keeping a rounded chin.
    lower_start = float(nose_centre[1] - 0.010)
    lower = np.clip((lower_start - points[:, 1]) / max(lower_start - chin[1], 1.0e-6), 0.0, 1.0)
    lower_w = smoothstep(lower) * surface_mask
    taper = 1.0 - 0.115 * np.power(lower, 1.18)
    delta[:, 0] += ((points[:, 0] - face_x) * taper - (points[:, 0] - face_x)) * lower_w
    target_y = lower_start + (points[:, 1] - lower_start) * 0.95
    delta[:, 1] += (target_y - points[:, 1]) * lower_w * 0.66
    chin_w = ellipsoid(points, chin, (0.043, 0.038, 0.038), 1.18) * surface_mask
    delta[:, 0] += -(points[:, 0] - face_x) * 0.13 * chin_w
    delta[:, 1] += 0.0010 * chin_w
    preserve = np.maximum(preserve, np.clip(chin_w, 0.0, 1.0))

    # High apple-cheek support gives the approved youthful identity without
    # widening the mid-face.
    for eye_indices in (range(36, 42), range(42, 48)):
        eye = lms[list(eye_indices)].mean(axis=0)
        side = -1.0 if eye[0] < face_x else 1.0
        cheek = np.array([
            eye[0] + side * 0.004,
            mouth_centre[1] + 0.56 * (eye[1] - mouth_centre[1]),
            nose_centre[2] - front_sign * 0.004,
        ])
        weight = ellipsoid(points, cheek, (0.050, 0.047, 0.040), 1.18) * surface_mask
        delta[:, 2] += front_sign * 0.0015 * weight
        delta[:, 0] += -side * 0.00035 * weight

    lengths = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1.0, 0.0060 / np.maximum(lengths, 1.0e-9))[:, None]
    result += delta
    return result, {
        "front_sign": front_sign,
        "max_raw_depth_sculpt_m": float(np.linalg.norm(delta, axis=1).max()),
        "moved_vertices_over_0_25mm": int(np.sum(np.linalg.norm(delta, axis=1) > 0.00025)),
    }, preserve


def propagate_to_components(
    before: np.ndarray,
    after: np.ndarray,
    surface_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    result = after.copy()
    displacement = after - before
    surface_ids = np.flatnonzero(surface_mask)
    other_ids = np.flatnonzero(~surface_mask)
    tree = cKDTree(before[surface_ids])
    distance, nearest = tree.query(before[other_ids], k=1)
    weight = np.clip(1.0 - distance / 0.028, 0.0, 1.0)
    result[other_ids] += displacement[surface_ids[nearest]] * (0.86 * weight[:, None])
    return result, {
        "moved_non_surface_vertices": int(np.sum(weight > 0.0)),
        "max_follow_weight": float(weight.max()) if len(weight) else 0.0,
    }


def render(vertices: np.ndarray, faces: np.ndarray, yaw: float, path: Path, title: str) -> None:
    angle = math.radians(yaw)
    c, s = math.cos(angle), math.sin(angle)
    points = vertices.copy()
    x = c * points[:, 0] + s * points[:, 2]
    z = -s * points[:, 0] + c * points[:, 2]
    points[:, 0], points[:, 2] = x, z
    triangles = points[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-9)
    order = np.argsort(triangles[:, :, 2].mean(axis=1))
    projected = points[faces[order], :2]
    normal = normals[order]
    diffuse = np.clip(np.abs(normal[:, 2]), 0.0, 1.0)
    intensity = np.clip(0.58 + 0.35 * diffuse, 0.36, 0.98)
    colours = np.stack([intensity * 0.96, intensity * 0.98, intensity], axis=1)
    xy = points[:, :2]
    lo, hi = np.percentile(xy, 1.2, axis=0), np.percentile(xy, 98.8, axis=0)
    centre = 0.5 * (lo + hi)
    extent = max(float((hi - lo).max()), 1.0e-6) * 0.56
    fig, axis = plt.subplots(figsize=(5, 5), dpi=190)
    axis.add_collection(PolyCollection(projected, facecolors=colours, edgecolors="none"))
    axis.set_xlim(centre[0] - extent, centre[0] + extent)
    axis.set_ylim(centre[1] - extent, centre[1] + extent)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.10)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def comparison_sheet(qa: Path) -> None:
    pairs = [
        (ROOT / "references/AINA_APPROVED_FRONT.jpg", qa / "AINA_DIRECT_SCULPT_FRONT_v2.png", "FRONT"),
        (ROOT / "references/AINA_APPROVED_3Q.jpg", qa / "AINA_DIRECT_SCULPT_3Q_v2.png", "3Q"),
        (ROOT / "references/AINA_APPROVED_SIDE.jpg", qa / "AINA_DIRECT_SCULPT_SIDE_v2.png", "SIDE"),
    ]
    panels = []
    for reference_path, model_path, label in pairs:
        reference = Image.open(reference_path).convert("RGB")
        model = Image.open(model_path).convert("RGB")
        reference.thumbnail((430, 430))
        model.thumbnail((430, 430))
        panel = Image.new("RGB", (900, 480), "white")
        panel.paste(reference, ((440 - reference.width) // 2, 5))
        panel.paste(model, (455 + (440 - model.width) // 2, 5))
        draw = ImageDraw.Draw(panel)
        draw.text((10, 450), f"APPROVED AINA {label}", fill="black")
        draw.text((465, 450), f"DIRECT SCULPT v2 {label}", fill="black")
        panels.append(panel)
    sheet = Image.new("RGB", (900, 480 * len(panels)), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * 480))
    sheet.save(qa / "AINA_APPROVED_VS_DIRECT_SCULPT_v2.png")


def main() -> None:
    out = ROOT / "output_direct_sculpt_v2"
    qa = out / "QA"
    out.mkdir(exist_ok=True)
    qa.mkdir(exist_ok=True)

    source = np.load(ROOT / "source_v1/AINA_CUSTOM_HEAD_TRANSFER_v1.npz")
    initial = np.asarray(source["final_vertices"], np.float64)
    faces = np.asarray(source["faces"], np.int64)
    surface_mask = np.asarray(source["skin_mask"], np.uint8) > 0
    identity = np.asarray(source["identity"], np.float32)

    model = FaceVerseModel_torch(
        device=torch.device("cpu"),
        facevrsepath=str(FVROOT / "data/faceverse_v4_2.npy"),
        camera_distance=10,
        focal=1000,
        center=128,
    )
    landmark_ids = np.asarray(model.fvd["keypoints_68"], np.int64).reshape(-1)
    if len(landmark_ids) != 68:
        raise RuntimeError(f"Expected 68 FaceVerse keypoints, got {len(landmark_ids)}")

    target = target_landmarks(ROOT / "references/AINA_TARGET_3DDFA_SPARSE_68.json")
    desired = desired_front_xy(initial[landmark_ids], target)
    landmark_stage, landmark_report, landmark_preserve = apply_landmark_residual(
        initial, surface_mask, landmark_ids, desired
    )
    depth_stage, depth_report, depth_preserve = art_directed_depth_and_cranium(
        landmark_stage, surface_mask, landmark_ids
    )

    total_raw = depth_stage - initial
    graph = adjacency(len(initial), faces, surface_mask)
    preserve = np.maximum(landmark_preserve, depth_preserve)
    total_smoothed = smooth_field(total_raw, graph, preserve, passes=3)
    lengths = np.linalg.norm(total_smoothed, axis=1)
    total_smoothed *= np.minimum(1.0, 0.010 / np.maximum(lengths, 1.0e-9))[:, None]
    surface_final = initial + total_smoothed
    final, propagation = propagate_to_components(initial, surface_final, surface_mask)

    surface_faces_mask = surface_mask[faces].all(axis=1)
    surface_faces = faces[surface_faces_mask]
    full_mesh = trimesh.Trimesh(vertices=final, faces=faces, process=False)
    full_mesh.export(out / "AINA_DIRECT_REFERENCE_HEAD_FULL_v2.obj")
    full_mesh.export(out / "AINA_DIRECT_REFERENCE_HEAD_FULL_v2.glb")
    surface_mesh = full_mesh.submesh([np.flatnonzero(surface_faces_mask)], append=True, repair=False)
    surface_mesh.remove_unreferenced_vertices()
    surface_mesh.export(out / "AINA_DIRECT_REFERENCE_HEAD_SURFACE_v2.obj")
    surface_mesh.export(out / "AINA_DIRECT_REFERENCE_HEAD_SURFACE_v2.glb")
    surface_mesh.export(out / "AINA_DIRECT_REFERENCE_HEAD_SURFACE_v2.ply")

    np.savez_compressed(
        out / "AINA_DIRECT_REFERENCE_TRANSFER_v2.npz",
        identity=identity,
        source_vertices=initial.astype(np.float32),
        neutral_displacement=(final - initial).astype(np.float32),
        final_vertices=final.astype(np.float32),
        faces=faces.astype(np.int32),
        surface_mask=surface_mask.astype(np.uint8),
        landmark_ids=landmark_ids.astype(np.int32),
        desired_front_xy=desired.astype(np.float32),
    )

    visible_vertices = np.asarray(surface_mesh.vertices)
    visible_faces = np.asarray(surface_mesh.faces)
    views = []
    for yaw, label in ((-90, "left_profile"), (-42, "left_3q"), (0, "front"), (42, "right_3q"), (90, "right_profile")):
        path = qa / f"AINA_DIRECT_SCULPT_{label}_v2.png"
        render(visible_vertices, visible_faces, yaw, path, f"AINA Direct Reference Sculpt v2 {label}")
        views.append(path)
    # Stable aliases for the three direct comparison views.
    (qa / "AINA_DIRECT_SCULPT_FRONT_v2.png").write_bytes((qa / "AINA_DIRECT_SCULPT_front_v2.png").read_bytes())
    (qa / "AINA_DIRECT_SCULPT_3Q_v2.png").write_bytes((qa / "AINA_DIRECT_SCULPT_right_3q_v2.png").read_bytes())
    (qa / "AINA_DIRECT_SCULPT_SIDE_v2.png").write_bytes((qa / "AINA_DIRECT_SCULPT_right_profile_v2.png").read_bytes())

    images = [Image.open(path).convert("RGB") for path in views]
    height = max(image.height for image in images)
    width = max(image.width for image in images)
    sheet = Image.new("RGB", (width * len(images), height), "white")
    for index, image in enumerate(images):
        sheet.paste(image, (index * width + (width - image.width) // 2, (height - image.height) // 2))
    sheet.save(qa / "AINA_DIRECT_SCULPT_5VIEW_v2.png")
    comparison_sheet(qa)

    final_lms = final[landmark_ids]
    final_residual = desired - final_lms[:, :2]
    report = {
        "product": "AINA Direct Reference Sculpt v2",
        "direction": "DIRECT_APPROVED_IMAGE_SPACE_SCULPT",
        "automatic_identity_regression_used": False,
        "rain_face_used": False,
        "source_topology": "FaceVerse V4 preserved vertex order",
        "vertices": int(len(final)),
        "triangles": int(len(faces)),
        "surface_vertices": int(surface_mask.sum()),
        "topology_changed": False,
        "vertex_order_preserved": True,
        "neutral_displacement_saved": True,
        "expression_transfer_ready": True,
        "landmark_fit": landmark_report,
        "depth_and_cranium": depth_report,
        "propagation": propagation,
        "total_neutral_displacement_max_m": float(np.linalg.norm(final - initial, axis=1).max()),
        "total_neutral_displacement_rms_m": float(np.sqrt(np.mean(np.sum((final - initial) ** 2, axis=1)))),
        "final_68_landmark_rmse_m": float(np.sqrt(np.mean(np.sum(final_residual * final_residual, axis=1)))),
        "identity_lock": False,
        "visual_identity_lock": False,
        "production_release": False,
        "candidate": True,
        "vrm_exported": False,
        "next_gate": "Directly inspect approved-vs-v2 front, 3Q and side. Only after visual acceptance transfer the same neutral displacement to all FaceVerse expression targets.",
    }
    (out / "AINA_DIRECT_REFERENCE_SCULPT_v2_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
