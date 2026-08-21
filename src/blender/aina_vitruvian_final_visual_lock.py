#!/usr/bin/env python3
"""Final bounded visual-identity convergence for the real AINA FACS head.

The input is the latest real Vitruvian/CharMorph AINA candidate BLEND.  This
stage does not generate replacement artwork.  It projects the actual neutral
Basis from front, three-quarter and side cameras, measures residuals against the
already-approved AINA references, and applies one small topology-preserving
surface correction.  The identical neutral displacement is added to every
existing shape key, so all FACS/viseme deltas remain bit-for-bit equivalent up
to floating-point storage precision.  Separate eye, tearline, caruncle, brow and
mouth anatomy is moved coherently.  The result is an editable BLEND and a
morph-preserving GLB plus real-model QA renders.  VRM export remains a later
stage and is blocked unless this visual gate passes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def key_array(key) -> np.ndarray:
    values = np.empty(len(key.data) * 3, dtype=np.float64)
    key.data.foreach_get("co", values)
    return values.reshape(-1, 3)


def set_key_array(key, values: np.ndarray) -> None:
    key.data.foreach_set("co", np.asarray(values, dtype=np.float32).ravel())


def local_vertices(obj) -> np.ndarray:
    return np.asarray([vertex.co[:] for vertex in obj.data.vertices], dtype=np.float64)


def world_vertices(obj, points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    homogeneous = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (homogeneous @ matrix.T)[:, :3]


def world_to_local_vectors(obj, vectors: np.ndarray) -> np.ndarray:
    rotation_scale = np.asarray(obj.matrix_world.to_3x3(), dtype=np.float64)
    return vectors @ np.linalg.inv(rotation_scale).T


def object_world_center(obj) -> np.ndarray:
    points = local_vertices(obj)
    if not len(points):
        return np.asarray(obj.matrix_world.translation[:], dtype=np.float64)
    return world_vertices(obj, points).mean(axis=0)


def is_eye_name(name: str) -> bool:
    value = name.lower()
    return any(token in value for token in ("eye", "iris", "pupil", "sclera", "tear", "caruncle", "cornea"))


def is_mouth_name(name: str) -> bool:
    value = name.lower()
    return any(token in value for token in ("mouth", "teeth", "tongue", "gum"))


def is_brow_name(name: str) -> bool:
    value = name.lower()
    return "brow" in value or "eyebrow" in value


def mesh_bounds(meshes) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for obj in meshes:
        points = local_vertices(obj)
        if len(points):
            values.append(world_vertices(obj, points))
    if not values:
        raise RuntimeError("No mesh vertices found")
    points = np.concatenate(values, axis=0)
    return points.min(axis=0), points.max(axis=0)


def identify_skin(meshes):
    candidates = [
        obj
        for obj in meshes
        if obj.data.shape_keys and len(obj.data.shape_keys.key_blocks) >= 12
    ]
    if not candidates:
        raise RuntimeError("Could not identify the real FACS skin mesh")
    return max(candidates, key=lambda obj: len(obj.data.vertices))


def build_setup(scene, meshes):
    lo, hi = mesh_bounds(meshes)
    center = (lo + hi) * 0.5
    size = hi - lo
    eye_centers = [object_world_center(obj) for obj in meshes if is_eye_name(obj.name)]
    if eye_centers:
        eye_center = np.mean(eye_centers, axis=0)
        forward_sign = -1.0 if eye_center[1] < center[1] else 1.0
        target = np.asarray([eye_center[0], center[1], eye_center[2] - 0.018 * max(size[2] / 0.32, 0.5)])
    else:
        forward_sign = -1.0
        target = center.copy()
        target[2] += 0.08 * max(size[2], 0.1)
    distance = max(float(size[2]) * 2.65, float(size[0]) * 2.85, 0.85)
    front = np.asarray([target[0], center[1] + forward_sign * distance, target[2]])
    return {
        "lo": lo,
        "hi": hi,
        "center": center,
        "size": size,
        "target": target,
        "forward_sign": forward_sign,
        "distance": distance,
        "locations": {
            "front": front,
            "three_quarter": front + np.asarray([0.42 * distance, -forward_sign * 0.10 * distance, 0.0]),
            "side": center + np.asarray([distance, 0.0, target[2] - center[2]]),
            "left_45": front + np.asarray([-0.50 * distance, -forward_sign * 0.15 * distance, 0.0]),
            "right_45": front + np.asarray([0.50 * distance, -forward_sign * 0.15 * distance, 0.0]),
        },
    }


def camera_for_view(scene, view: str, setup, lens: float = 82.0):
    name = f"AINA_FinalLock_Camera_{view}"
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    data = bpy.data.cameras.new(name)
    camera = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(camera)
    camera.data.lens = lens
    camera.location = setup["locations"][view]
    camera.rotation_euler = (Vector(setup["target"]) - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    return camera


def project_points(scene, camera, obj, local_points: np.ndarray) -> np.ndarray:
    width = scene.render.resolution_x * scene.render.resolution_percentage / 100.0
    height = scene.render.resolution_y * scene.render.resolution_percentage / 100.0
    projected = np.empty((len(local_points), 3), dtype=np.float64)
    matrix = obj.matrix_world
    for index, point in enumerate(local_points):
        ndc = world_to_camera_view(scene, camera, matrix @ Vector(point))
        projected[index] = (ndc.x * width, (1.0 - ndc.y) * height, ndc.z)
    return projected


def semantic_weights(view: str) -> np.ndarray:
    weights = np.ones(68, dtype=np.float64)
    weights[:17] = 1.45
    weights[17:27] = 0.65
    weights[27:36] = 2.55
    weights[36:48] = 3.00
    weights[48:68] = 2.65
    if view == "three_quarter":
        weights[:17] = 1.70
        weights[27:36] = 2.80
        weights[36:48] = 2.75
    elif view == "side":
        weights[:] = 0.35
        weights[:17] = 2.00
        weights[27:36] = 3.00
        weights[36:48] = 0.45
        weights[48:68] = 2.30
    return weights


def weighted_similarity(source: np.ndarray, destination: np.ndarray, weights: np.ndarray):
    """Align source 2D landmarks into destination image coordinates."""
    w = np.asarray(weights, dtype=np.float64)
    w /= max(float(w.sum()), 1e-9)
    source_center = np.sum(source * w[:, None], axis=0)
    destination_center = np.sum(destination * w[:, None], axis=0)
    x = source - source_center
    y = destination - destination_center
    covariance = (x * w[:, None]).T @ y
    u, singular, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    denominator = np.sum(w * np.sum(x * x, axis=1))
    scale = float(np.sum(singular) / max(float(denominator), 1e-9))
    mapped = scale * (x @ rotation.T) + destination_center
    return mapped, {
        "scale": scale,
        "rotation": rotation.tolist(),
        "source_center": source_center.tolist(),
        "destination_center": destination_center.tolist(),
    }


def choose_anchor_vertices(projected: np.ndarray, landmarks: np.ndarray, k: int = 48) -> np.ndarray:
    result = []
    valid = projected[:, 2] > 0.0
    valid_ids = np.flatnonzero(valid)
    if not len(valid_ids):
        valid_ids = np.arange(len(projected))
    visible = projected[valid_ids]
    for point in landmarks:
        distance2 = np.sum((visible[:, :2] - point) ** 2, axis=1)
        count = min(k, len(distance2))
        candidate_local = np.argpartition(distance2, count - 1)[:count]
        candidates = valid_ids[candidate_local]
        # Prefer the frontmost candidate among similarly projected vertices.
        depth = projected[candidates, 2]
        depth_term = 0.04 * np.square(depth - depth.min())
        score = np.sum((projected[candidates, :2] - point) ** 2, axis=1) + depth_term
        result.append(int(candidates[np.argmin(score)]))
    return np.asarray(result, dtype=np.int64)


def world_delta_from_pixels(scene, camera, anchor_world: np.ndarray, residual: np.ndarray) -> np.ndarray:
    depth = max(float(np.linalg.norm(anchor_world - np.asarray(camera.location[:], dtype=np.float64))), 1e-6)
    width = scene.render.resolution_x * scene.render.resolution_percentage / 100.0
    height = scene.render.resolution_y * scene.render.resolution_percentage / 100.0
    scale_x = 2.0 * depth * math.tan(camera.data.angle_x * 0.5) / max(width, 1.0)
    scale_y = 2.0 * depth * math.tan(camera.data.angle_y * 0.5) / max(height, 1.0)
    matrix = camera.matrix_world.to_3x3()
    right = np.asarray(matrix.col[0][:], dtype=np.float64)
    up = np.asarray(matrix.col[1][:], dtype=np.float64)
    return right * residual[0] * scale_x - up * residual[1] * scale_y


def landmark_radius(index: int, unit: float) -> float:
    if index < 17:
        return 0.036 * unit
    if index < 27:
        return 0.028 * unit
    if index < 36:
        return 0.024 * unit
    if index < 48:
        return 0.026 * unit
    return 0.027 * unit


def adjacency_from_mesh(obj):
    adjacency = [set() for _ in obj.data.vertices]
    for polygon in obj.data.polygons:
        vertices = list(polygon.vertices)
        for index, a in enumerate(vertices):
            b = vertices[(index + 1) % len(vertices)]
            adjacency[a].add(b)
            adjacency[b].add(a)
    return [np.asarray(sorted(items), dtype=np.int64) for items in adjacency]


def smooth_delta(delta: np.ndarray, influence: np.ndarray, adjacency, rounds: int = 2) -> np.ndarray:
    result = delta.copy()
    for _ in range(rounds):
        updated = result.copy()
        for index, neighbors in enumerate(adjacency):
            if influence[index] < 0.03 or not len(neighbors):
                continue
            average = result[neighbors].mean(axis=0)
            amount = min(0.40, 0.18 + 0.25 * influence[index])
            updated[index] = result[index] * (1.0 - amount) + average * amount
        result = updated
    return result


def bounded_multiview_pass(scene, skin, data, setup):
    keys = skin.data.shape_keys.key_blocks
    basis = keys.get("Basis") or keys[0]
    adjacency = adjacency_from_mesh(skin)
    unit = max(float(setup["size"][2]) / 0.32, 0.55)
    original_deltas = {
        key.name: key_array(key) - key_array(basis)
        for key in keys
        if key.name != basis.name
    }
    history = []
    front_anchor_world = None
    front_anchor_delta = None
    front_model = None
    front_target = None

    for iteration, step_scale in enumerate((0.42, 0.24)):
        current_local = key_array(basis)
        current_world = world_vertices(skin, current_local)
        accumulated = np.zeros_like(current_world)
        denominator = np.zeros(len(current_world), dtype=np.float64)
        per_view = {}

        for view, view_scale in (("front", 1.0), ("three_quarter", 0.68), ("side", 0.32)):
            camera = camera_for_view(scene, view, setup)
            projected = project_points(scene, camera, skin, current_local)
            model_item = data["model"][view]
            approved_item = data["approved"][view]
            model_points = np.asarray(model_item["landmarks_xy"], dtype=np.float64)
            approved_points = np.asarray(approved_item["landmarks_xy"], dtype=np.float64)
            weights = semantic_weights(view)
            target_points, similarity = weighted_similarity(approved_points, model_points, weights)
            anchors = choose_anchor_vertices(projected, model_points)
            anchor_pixels = projected[anchors, :2]
            residual_pixels = target_points - anchor_pixels
            face_width = max(float(np.linalg.norm(target_points[0] - target_points[16])), 1.0)
            before_rmse = float(np.sqrt(np.sum(weights * np.sum(residual_pixels ** 2, axis=1)) / np.sum(weights)) / face_width)
            anchor_world = current_world[anchors]
            anchor_delta = np.zeros((68, 3), dtype=np.float64)
            max_anchor = (0.0034 if view != "side" else 0.0030) * unit
            for index in range(68):
                value = world_delta_from_pixels(scene, camera, anchor_world[index], residual_pixels[index])
                length = float(np.linalg.norm(value))
                if length > max_anchor:
                    value *= max_anchor / max(length, 1e-9)
                anchor_delta[index] = value * step_scale * view_scale

            if view == "front" and iteration == 0:
                front_anchor_world = anchor_world.copy()
                front_anchor_delta = anchor_delta.copy()
                front_model = model_points.copy()
                front_target = target_points.copy()

            for index, center in enumerate(anchor_world):
                radius = landmark_radius(index, unit)
                distance = np.linalg.norm(current_world - center, axis=1)
                local_weight = np.exp(-0.5 * (distance / max(radius, 1e-8)) ** 4)
                local_weight[distance > radius * 1.45] = 0.0
                local_weight *= weights[index]
                accumulated += local_weight[:, None] * anchor_delta[index]
                denominator += local_weight

            per_view[view] = {
                "normalized_rmse_before": before_rmse,
                "similarity": similarity,
                "maximum_anchor_delta_m": float(np.linalg.norm(anchor_delta, axis=1).max()),
            }

        world_delta = accumulated / np.maximum(denominator[:, None], 1e-9)
        world_delta[denominator < 0.08] = 0.0
        influence = np.clip(denominator / max(float(np.percentile(denominator[denominator > 0], 92)) if np.any(denominator > 0) else 1.0, 1e-8), 0.0, 1.0)
        max_vertex = 0.00215 * unit
        lengths = np.linalg.norm(world_delta, axis=1)
        world_delta *= np.minimum(1.0, max_vertex / np.maximum(lengths, 1e-9))[:, None]
        world_delta = smooth_delta(world_delta, influence, adjacency, rounds=2)
        local_delta = world_to_local_vectors(skin, world_delta)

        # Adding one identical neutral displacement to every key preserves every
        # expression delta by construction.
        for key in keys:
            set_key_array(key, key_array(key) + local_delta)
        bpy.context.view_layer.update()
        history.append(
            {
                "iteration": iteration,
                "step_scale": step_scale,
                "views": per_view,
                "affected_vertices": int(np.sum(influence > 0.03)),
                "max_vertex_step_m": float(np.linalg.norm(world_delta, axis=1).max()),
                "rms_vertex_step_m": float(np.sqrt(np.mean(np.sum(world_delta * world_delta, axis=1)))),
            }
        )

    final_basis = key_array(basis)
    delta_drift = {}
    for key in keys:
        if key.name == basis.name:
            continue
        now = key_array(key) - final_basis
        reference = original_deltas[key.name]
        delta_drift[key.name] = float(np.max(np.abs(now - reference)))

    return {
        "history": history,
        "expression_delta_max_drift": max(delta_drift.values(), default=0.0),
        "expression_delta_drift_by_key": delta_drift,
        "front_anchor_world": front_anchor_world,
        "front_anchor_delta": front_anchor_delta,
        "front_model": front_model,
        "front_target": front_target,
    }


def move_world(obj, delta: np.ndarray, scale_factor: float | None = None):
    matrix = obj.matrix_world.copy()
    matrix.translation = matrix.translation + Vector(delta.tolist())
    obj.matrix_world = matrix
    if scale_factor is not None:
        bounded = float(np.clip(scale_factor, 0.975, 1.040))
        obj.scale = tuple(float(value) * bounded for value in obj.scale)


def move_accessories(meshes, skin, convergence):
    anchor_world = convergence.get("front_anchor_world")
    anchor_delta = convergence.get("front_anchor_delta")
    model = convergence.get("front_model")
    target = convergence.get("front_target")
    if anchor_world is None or anchor_delta is None:
        return {"moved": []}

    eye_groups = [np.arange(36, 42), np.arange(42, 48)]
    eye_centers = [anchor_world[group].mean(axis=0) for group in eye_groups]
    eye_deltas = [anchor_delta[group].mean(axis=0) for group in eye_groups]
    eye_scales = []
    for group in eye_groups:
        source_width = float(np.linalg.norm(model[group[0]] - model[group[3]]))
        target_width = float(np.linalg.norm(target[group[0]] - target[group[3]]))
        eye_scales.append(target_width / max(source_width, 1e-6))

    mouth_delta = anchor_delta[48:68].mean(axis=0)
    brow_delta = anchor_delta[17:27].mean(axis=0)
    moved = []
    for obj in meshes:
        if obj == skin:
            continue
        if is_eye_name(obj.name):
            center = object_world_center(obj)
            index = int(np.argmin([np.linalg.norm(center - value) for value in eye_centers]))
            move_world(obj, eye_deltas[index], eye_scales[index])
            moved.append({"object": obj.name, "region": f"eye_{index}", "delta": eye_deltas[index].tolist(), "scale": float(np.clip(eye_scales[index], 0.975, 1.040))})
        elif is_mouth_name(obj.name):
            move_world(obj, mouth_delta)
            moved.append({"object": obj.name, "region": "mouth", "delta": mouth_delta.tolist()})
        elif is_brow_name(obj.name):
            move_world(obj, brow_delta)
            moved.append({"object": obj.name, "region": "brow", "delta": brow_delta.tolist()})
    bpy.context.view_layer.update()
    return {"moved": moved}


def normalized_key_name(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def activate_case(skin, tokens, strength: float = 0.75):
    keys = skin.data.shape_keys.key_blocks
    for key in keys:
        if key.name != "Basis":
            key.value = 0.0
    activated = []
    for token in tokens:
        needle = normalized_key_name(token)
        matches = [key for key in keys if key.name != "Basis" and needle in normalized_key_name(key.name)]
        if matches:
            key = matches[0]
            key.value = strength
            activated.append(key.name)
    return activated


def clear_shapes(skin):
    if not skin.data.shape_keys:
        return
    for key in skin.data.shape_keys.key_blocks:
        if key.name != "Basis":
            key.value = 0.0


def material(name, color, roughness=0.48, metallic=0.0):
    existing = bpy.data.materials.get(name)
    if existing:
        return existing
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = tuple(color)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
    if shader:
        shader.inputs["Base Color"].default_value = tuple(color)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metallic
    return mat


def create_light(name, location, energy, size, target):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
    return obj


def setup_render(scene, setup, output: Path):
    for obj in list(scene.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.world.color = (0.022, 0.027, 0.040)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = -0.35
    except Exception:
        pass
    size = setup["size"]
    front = setup["locations"]["front"]
    target = setup["target"]
    create_light("AINA_Final_Key", tuple(front + np.asarray([0.60 * size[0], 0.0, 0.52 * size[2]])), 360, 2.2, target)
    create_light("AINA_Final_Fill", tuple(front + np.asarray([-0.70 * size[0], 0.18 * setup["distance"], 0.12 * size[2]])), 170, 2.6, target)
    create_light("AINA_Final_Rim", tuple(setup["center"] + np.asarray([0.0, -setup["forward_sign"] * 0.62 * setup["distance"], 0.50 * size[2]])), 240, 2.1, target)
    (output / "Preview").mkdir(parents=True, exist_ok=True)


def render_view(scene, setup, output: Path, view: str, filename: str):
    camera = camera_for_view(scene, view, setup)
    scene.camera = camera
    path = output / "Preview" / filename
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    return path


def final_metrics(scene, skin, data, setup):
    result = {}
    basis = skin.data.shape_keys.key_blocks.get("Basis") or skin.data.shape_keys.key_blocks[0]
    points = key_array(basis)
    for view in ("front", "three_quarter", "side"):
        camera = camera_for_view(scene, view, setup)
        projected = project_points(scene, camera, skin, points)
        model = np.asarray(data["model"][view]["landmarks_xy"], dtype=np.float64)
        approved = np.asarray(data["approved"][view]["landmarks_xy"], dtype=np.float64)
        weights = semantic_weights(view)
        target, _ = weighted_similarity(approved, model, weights)
        anchors = choose_anchor_vertices(projected, model)
        residual = target - projected[anchors, :2]
        face_width = max(float(np.linalg.norm(target[0] - target[16])), 1.0)
        rmse = float(np.sqrt(np.sum(weights * np.sum(residual ** 2, axis=1)) / np.sum(weights)) / face_width)
        result[view] = rmse
    return result


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)
    data = json.loads(args.landmarks.read_text())

    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    skin = identify_skin(meshes)
    clear_shapes(skin)
    original_vertex_count = len(skin.data.vertices)
    original_shape_names = [key.name for key in skin.data.shape_keys.key_blocks]
    setup = build_setup(scene, meshes)
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100

    convergence = bounded_multiview_pass(scene, skin, data, setup)
    accessories = move_accessories(meshes, skin, convergence)
    setup = build_setup(scene, meshes)
    metrics = final_metrics(scene, skin, data, setup)

    setup_render(scene, setup, args.out)
    renders = {}
    clear_shapes(skin)
    renders["neutral_front"] = str(render_view(scene, setup, args.out, "front", "AINA_FINAL_LOCK_NEUTRAL_FRONT.png"))
    renders["neutral_three_quarter"] = str(render_view(scene, setup, args.out, "three_quarter", "AINA_FINAL_LOCK_NEUTRAL_THREE_QUARTER.png"))
    renders["neutral_side"] = str(render_view(scene, setup, args.out, "side", "AINA_FINAL_LOCK_NEUTRAL_SIDE.png"))
    renders["neutral_left_45"] = str(render_view(scene, setup, args.out, "left_45", "AINA_FINAL_LOCK_NEUTRAL_LEFT_45.png"))
    renders["neutral_right_45"] = str(render_view(scene, setup, args.out, "right_45", "AINA_FINAL_LOCK_NEUTRAL_RIGHT_45.png"))

    cases = {
        "happy": (["happy", "smile"], 0.72),
        "sad": (["sad", "cornersdown"], 0.70),
        "angry": (["angry", "snarl"], 0.68),
        "blink": (["eyesclosedmax", "blink"], 0.90),
        "aa": (["aa", "jawlower", "mouthlargeopened"], 0.62),
        "ou": (["ow", "pucker", "kiss", "funneler"], 0.62),
    }
    activated = {}
    for name, (tokens, strength) in cases.items():
        activated[name] = activate_case(skin, tokens, strength)
        renders[name] = str(render_view(scene, setup, args.out, "front", f"AINA_FINAL_LOCK_{name.upper()}.png"))
    clear_shapes(skin)

    blend_path = args.out / "AINA_IDENTITY_MASTER_VISUAL_LOCK_CANDIDATE.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_IDENTITY_MASTER_VISUAL_LOCK_CANDIDATE.glb"
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    final_shape_names = [key.name for key in skin.data.shape_keys.key_blocks]
    quantitative_gate = (
        metrics.get("front", 1.0) <= 0.050
        and metrics.get("three_quarter", 1.0) <= 0.070
        and metrics.get("side", 1.0) <= 0.095
        and convergence["expression_delta_max_drift"] <= 2.5e-6
        and original_vertex_count == len(skin.data.vertices)
        and original_shape_names == final_shape_names
    )
    report = {
        "product": "AINA Real Identity Master Final Visual Lock Candidate",
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "source_topology": "CC0 Vitruvian/Antonia FACS head",
        "topology_changed": False,
        "vertices_before": original_vertex_count,
        "vertices_after": len(skin.data.vertices),
        "shape_keys_before": original_shape_names,
        "shape_keys_after": final_shape_names,
        "shape_key_count": max(0, len(final_shape_names) - 1),
        "convergence": {
            "history": convergence["history"],
            "expression_delta_max_drift": convergence["expression_delta_max_drift"],
        },
        "accessories": accessories,
        "final_normalized_landmark_rmse": metrics,
        "quantitative_identity_gate": quantitative_gate,
        "manual_visual_gate_required": True,
        "identity_lock": False,
        "visual_identity_lock": False,
        "candidate": True,
        "vrm_exported": False,
        "activated_expression_keys": activated,
        "files": {
            "blend": str(blend_path),
            "glb": str(glb_path),
            "renders": renders,
        },
        "next_gate": "Inspect approved-reference versus actual front, 3Q and side renders. Only after the real head is visually accepted may visual_identity_lock become true and final VRM assembly begin.",
    }
    (qa / "AINA_FINAL_VISUAL_LOCK_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
