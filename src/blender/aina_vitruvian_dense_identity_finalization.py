#!/usr/bin/env python3
"""Dense final identity convergence on the real 17,161-vertex AINA FACS head.

Input is the successful sparse front/3Q/side visual-lock BLEND.  MediaPipe dense
landmarks are measured from the already-approved AINA references and from the
actual sparse-lock Blender renders.  Two small bounded camera-plane corrections
are applied to the true neutral Basis.  The exact same local displacement is
added to every existing FACS/viseme key so the expression deltas are preserved.
Separate eye and mouth anatomy is moved coherently.  No replacement image is
generated and VRM export remains blocked until the resulting real renders pass
manual visual inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

import aina_vitruvian_final_visual_lock as lock


FACE_OVAL = {
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
}
LIPS = {
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
    402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311,
    310, 415, 308, 76, 62, 183, 42, 41, 38, 12, 268, 271, 272, 407, 306,
    77, 96, 89, 179, 86, 15, 316, 403, 320, 325, 307,
}
LEFT_EYE = {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246}
RIGHT_EYE = {362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398}
LEFT_BROW = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46}
RIGHT_BROW = {336, 296, 334, 293, 300, 285, 295, 282, 283, 276}
NOSE = {1, 2, 98, 327, 168, 197, 5, 4, 45, 275, 440, 220, 115, 344, 195, 19, 94}
FEATURE = FACE_OVAL | LIPS | LEFT_EYE | RIGHT_EYE | LEFT_BROW | RIGHT_BROW | NOSE


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-landmarks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def key_array(key) -> np.ndarray:
    values = np.empty(len(key.data) * 3, dtype=np.float64)
    key.data.foreach_get("co", values)
    return values.reshape(-1, 3)


def set_key_array(key, values: np.ndarray) -> None:
    key.data.foreach_set("co", np.asarray(values, dtype=np.float32).ravel())


def selected_indices(count: int) -> np.ndarray:
    # Feature contours are complete; a deterministic coarse sampling of the
    # remaining surface stabilizes forehead/cheek continuity without overfitting.
    values = set(index for index in FEATURE if index < count)
    values.update(range(0, min(count, 468), 5))
    return np.asarray(sorted(values), dtype=np.int64)


def point_weight(index: int, view: str) -> float:
    if index in LEFT_EYE or index in RIGHT_EYE:
        value = 3.20
    elif index in LIPS:
        value = 3.00
    elif index in NOSE:
        value = 2.75
    elif index in FACE_OVAL:
        value = 1.70
    elif index in LEFT_BROW or index in RIGHT_BROW:
        value = 1.25
    else:
        value = 0.38
    if view == "three_quarter":
        if index in FACE_OVAL or index in NOSE:
            value *= 1.16
        if index in LEFT_EYE or index in RIGHT_EYE:
            value *= 0.94
    elif view == "side":
        if index in FACE_OVAL or index in NOSE or index in LIPS:
            value *= 1.30
        else:
            value *= 0.25
    return value


def point_radius(index: int, unit: float) -> float:
    if index in LEFT_EYE or index in RIGHT_EYE:
        return 0.0145 * unit
    if index in LIPS:
        return 0.0140 * unit
    if index in NOSE:
        return 0.0165 * unit
    if index in LEFT_BROW or index in RIGHT_BROW:
        return 0.0180 * unit
    if index in FACE_OVAL:
        return 0.0260 * unit
    return 0.0210 * unit


def adjacency(obj) -> list[np.ndarray]:
    values = [set() for _ in obj.data.vertices]
    for polygon in obj.data.polygons:
        vertices = list(polygon.vertices)
        for index, left in enumerate(vertices):
            right = vertices[(index + 1) % len(vertices)]
            values[left].add(right)
            values[right].add(left)
    return [np.asarray(sorted(item), dtype=np.int64) for item in values]


def smooth_delta(delta: np.ndarray, influence: np.ndarray, graph, rounds: int = 2) -> np.ndarray:
    result = delta.copy()
    for _ in range(rounds):
        updated = result.copy()
        for index, neighbors in enumerate(graph):
            if influence[index] < 0.035 or not len(neighbors):
                continue
            amount = min(0.34, 0.12 + 0.24 * influence[index])
            updated[index] = result[index] * (1.0 - amount) + result[neighbors].mean(axis=0) * amount
        result = updated
    return result


def weighted_rmse(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    face_width = max(float(np.linalg.norm(target[0] - target[min(16, len(target) - 1)])), 1.0)
    value = np.sum(weights * np.sum((prediction - target) ** 2, axis=1)) / max(float(weights.sum()), 1e-9)
    return float(np.sqrt(value) / face_width)


def view_payload(data: dict, view: str):
    approved = data.get("approved", {}).get(view)
    model = data.get("model", {}).get(view)
    if not approved or not model:
        return None
    approved_points = np.asarray(approved["landmarks_xy"], dtype=np.float64)
    model_points = np.asarray(model["landmarks_xy"], dtype=np.float64)
    count = min(len(approved_points), len(model_points), 468)
    if count < 300:
        return None
    return approved_points[:count], model_points[:count]


def world_to_local_vectors(obj, values: np.ndarray) -> np.ndarray:
    return lock.world_to_local_vectors(obj, values)


def apply_dense_pass(scene, skin, meshes, data, setup):
    keys = skin.data.shape_keys.key_blocks
    basis = keys.get("Basis") or keys[0]
    original_basis = key_array(basis)
    original_deltas = {
        key.name: key_array(key) - original_basis
        for key in keys
        if key.name != basis.name
    }
    graph = adjacency(skin)
    unit = max(float(setup["size"][2]) / 0.32, 0.55)
    history = []
    accessory_samples = {}

    available_views = [view for view in ("front", "three_quarter", "side") if view_payload(data, view) is not None]
    if "front" not in available_views or "three_quarter" not in available_views:
        raise RuntimeError(f"Dense front and three-quarter data are mandatory: {available_views}")

    for iteration, step in enumerate((0.48, 0.27)):
        basis_local = key_array(basis)
        basis_world = lock.world_vertices(skin, basis_local)
        accumulated = np.zeros_like(basis_world)
        denominator = np.zeros(len(basis_world), dtype=np.float64)
        per_view = {}

        for view in available_views:
            approved_all, model_all = view_payload(data, view)
            indices = selected_indices(len(model_all))
            approved = approved_all[indices]
            model = model_all[indices]
            weights = np.asarray([point_weight(int(index), view) for index in indices], dtype=np.float64)
            target, similarity = lock.weighted_similarity(approved, model, weights)

            camera = lock.camera_for_view(scene, view, setup)
            projected = lock.project_points(scene, camera, skin, basis_local)
            anchors = lock.choose_anchor_vertices(projected, model, k=36)
            anchor_pixels = projected[anchors, :2]
            residual = target - anchor_pixels
            before = weighted_rmse(anchor_pixels, target, weights)
            anchor_world = basis_world[anchors]
            anchor_delta = np.zeros((len(indices), 3), dtype=np.float64)
            view_scale = {"front": 1.0, "three_quarter": 0.76, "side": 0.34}[view]
            cap_value = {"front": 0.00195, "three_quarter": 0.00175, "side": 0.00145}[view] * unit
            for local_index in range(len(indices)):
                value = lock.world_delta_from_pixels(scene, camera, anchor_world[local_index], residual[local_index])
                length = float(np.linalg.norm(value))
                if length > cap_value:
                    value *= cap_value / max(length, 1e-9)
                anchor_delta[local_index] = value * step * view_scale

            if iteration == 0:
                accessory_samples[view] = {
                    "indices": indices.copy(),
                    "anchor_world": anchor_world.copy(),
                    "anchor_delta": anchor_delta.copy(),
                    "model": model.copy(),
                    "target": target.copy(),
                }

            for local_index, dense_index in enumerate(indices):
                center = anchor_world[local_index]
                radius = point_radius(int(dense_index), unit)
                distance = np.linalg.norm(basis_world - center, axis=1)
                local_weight = np.exp(-0.5 * (distance / max(radius, 1e-8)) ** 4)
                local_weight[distance > radius * 1.48] = 0.0
                local_weight *= weights[local_index]
                accumulated += local_weight[:, None] * anchor_delta[local_index]
                denominator += local_weight

            per_view[view] = {
                "selected_landmarks": int(len(indices)),
                "normalized_rmse_before": before,
                "similarity": similarity,
                "max_anchor_step_m": float(np.linalg.norm(anchor_delta, axis=1).max()),
            }

        world_delta = accumulated / np.maximum(denominator[:, None], 1e-9)
        world_delta[denominator < 0.085] = 0.0
        positive = denominator[denominator > 0]
        scale = float(np.percentile(positive, 92)) if len(positive) else 1.0
        influence = np.clip(denominator / max(scale, 1e-9), 0.0, 1.0)
        max_vertex = 0.00135 * unit
        lengths = np.linalg.norm(world_delta, axis=1)
        world_delta *= np.minimum(1.0, max_vertex / np.maximum(lengths, 1e-9))[:, None]
        world_delta = smooth_delta(world_delta, influence, graph, rounds=2)
        local_delta = world_to_local_vectors(skin, world_delta)
        for key in keys:
            set_key_array(key, key_array(key) + local_delta)
        bpy.context.view_layer.update()
        history.append({
            "iteration": iteration,
            "step_scale": step,
            "views": per_view,
            "affected_vertices": int(np.sum(influence > 0.035)),
            "max_vertex_step_m": float(np.linalg.norm(world_delta, axis=1).max()),
            "rms_vertex_step_m": float(np.sqrt(np.mean(np.sum(world_delta * world_delta, axis=1)))),
        })

    final_basis = key_array(basis)
    drift = {}
    for key in keys:
        if key.name == basis.name:
            continue
        current = key_array(key) - final_basis
        drift[key.name] = float(np.max(np.abs(current - original_deltas[key.name])))
    return {
        "history": history,
        "original_basis": original_basis,
        "final_basis": final_basis,
        "expression_delta_drift_by_key": drift,
        "expression_delta_max_drift": max(drift.values(), default=0.0),
        "accessory_samples": accessory_samples,
        "available_views": available_views,
    }


def move_object_world(obj, delta: np.ndarray, scale: float | None = None) -> None:
    matrix = obj.matrix_world.copy()
    matrix.translation = matrix.translation + Vector(delta.tolist())
    obj.matrix_world = matrix
    if scale is not None:
        factor = float(np.clip(scale, 0.985, 1.030))
        obj.scale = tuple(float(value) * factor for value in obj.scale)


def group_sample(sample: dict, group: set[int]):
    dense_indices = sample["indices"]
    locations = [position for position, value in enumerate(dense_indices) if int(value) in group]
    if not locations:
        return None
    locations = np.asarray(locations, dtype=np.int64)
    return {
        "center": sample["anchor_world"][locations].mean(axis=0),
        "delta": sample["anchor_delta"][locations].mean(axis=0),
        "model": sample["model"][locations],
        "target": sample["target"][locations],
    }


def move_accessories(meshes, skin, convergence):
    sample = convergence["accessory_samples"].get("front")
    if not sample:
        return {"moved": []}
    left = group_sample(sample, LEFT_EYE)
    right = group_sample(sample, RIGHT_EYE)
    mouth = group_sample(sample, LIPS)
    brow_left = group_sample(sample, LEFT_BROW)
    brow_right = group_sample(sample, RIGHT_BROW)
    eye_groups = [value for value in (left, right) if value]
    brow_groups = [value for value in (brow_left, brow_right) if value]
    moved = []
    for obj in meshes:
        if obj == skin:
            continue
        center = lock.object_world_center(obj)
        if lock.is_eye_name(obj.name) and eye_groups:
            group = min(eye_groups, key=lambda item: np.linalg.norm(center - item["center"]))
            model_width = float(np.ptp(group["model"][:, 0]))
            target_width = float(np.ptp(group["target"][:, 0]))
            scale = target_width / max(model_width, 1e-6)
            move_object_world(obj, group["delta"], scale)
            moved.append({"object": obj.name, "region": "eye", "delta": group["delta"].tolist(), "scale": float(np.clip(scale, 0.985, 1.030))})
        elif lock.is_mouth_name(obj.name) and mouth:
            move_object_world(obj, mouth["delta"])
            moved.append({"object": obj.name, "region": "mouth", "delta": mouth["delta"].tolist()})
        elif lock.is_brow_name(obj.name) and brow_groups:
            group = min(brow_groups, key=lambda item: np.linalg.norm(center - item["center"]))
            move_object_world(obj, group["delta"])
            moved.append({"object": obj.name, "region": "brow", "delta": group["delta"].tolist()})
    bpy.context.view_layer.update()
    return {"moved": moved}


def final_metrics(scene, skin, data, setup):
    basis = skin.data.shape_keys.key_blocks.get("Basis") or skin.data.shape_keys.key_blocks[0]
    local = key_array(basis)
    result = {}
    for view in ("front", "three_quarter", "side"):
        payload = view_payload(data, view)
        if payload is None:
            continue
        approved_all, model_all = payload
        indices = selected_indices(len(model_all))
        approved = approved_all[indices]
        model = model_all[indices]
        weights = np.asarray([point_weight(int(index), view) for index in indices], dtype=np.float64)
        target, _ = lock.weighted_similarity(approved, model, weights)
        camera = lock.camera_for_view(scene, view, setup)
        projected = lock.project_points(scene, camera, skin, local)
        anchors = lock.choose_anchor_vertices(projected, model, k=36)
        result[view] = weighted_rmse(projected[anchors, :2], target, weights)
    return result


def triangle_health(before: np.ndarray, after: np.ndarray, triangles: np.ndarray):
    first = before[triangles]
    second = after[triangles]
    area_before = 0.5 * np.linalg.norm(np.cross(first[:, 1] - first[:, 0], first[:, 2] - first[:, 0]), axis=1)
    area_after = 0.5 * np.linalg.norm(np.cross(second[:, 1] - second[:, 0], second[:, 2] - second[:, 0]), axis=1)
    ratio = area_after / np.maximum(area_before, 1e-12)
    return {
        "degenerate_triangles": int(np.sum(area_after < 1e-12)),
        "area_ratio_p01": float(np.percentile(ratio, 1)),
        "area_ratio_p50": float(np.percentile(ratio, 50)),
        "area_ratio_p99": float(np.percentile(ratio, 99)),
        "max_total_displacement_m": float(np.linalg.norm(after - before, axis=1).max()),
        "rms_total_displacement_m": float(np.sqrt(np.mean(np.sum((after - before) ** 2, axis=1)))),
    }


def clay_material():
    mat = bpy.data.materials.get("AINA_Dense_Clay") or bpy.data.materials.new("AINA_Dense_Clay")
    mat.diffuse_color = (0.42, 0.47, 0.58, 1.0)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
    if shader:
        shader.inputs["Base Color"].default_value = (0.42, 0.47, 0.58, 1.0)
        shader.inputs["Roughness"].default_value = 0.52
        shader.inputs["Metallic"].default_value = 0.0
    return mat


def save_materials(meshes):
    return {obj.name: [slot.material for slot in obj.material_slots] for obj in meshes}


def restore_materials(meshes, saved):
    for obj in meshes:
        materials = saved.get(obj.name, [])
        for index, material in enumerate(materials):
            if index < len(obj.data.materials):
                obj.data.materials[index] = material


def apply_clay(meshes, mat):
    for obj in meshes:
        if not len(obj.data.materials):
            obj.data.materials.append(mat)
        else:
            for index in range(len(obj.data.materials)):
                obj.data.materials[index] = mat


def clear_shapes(skin):
    if not skin.data.shape_keys:
        return
    for key in skin.data.shape_keys.key_blocks:
        if key.name != "Basis":
            key.value = 0.0


def render_suite(scene, skin, meshes, setup, output: Path):
    lock.setup_render(scene, setup, output)
    clear_shapes(skin)
    renders = {}
    for view, name in (
        ("front", "AINA_DENSE_FINAL_NEUTRAL_FRONT.png"),
        ("three_quarter", "AINA_DENSE_FINAL_NEUTRAL_THREE_QUARTER.png"),
        ("side", "AINA_DENSE_FINAL_NEUTRAL_SIDE.png"),
        ("left_45", "AINA_DENSE_FINAL_NEUTRAL_LEFT_45.png"),
        ("right_45", "AINA_DENSE_FINAL_NEUTRAL_RIGHT_45.png"),
    ):
        renders[f"beauty_{view}"] = str(lock.render_view(scene, setup, output, view, name))

    cases = {
        "happy": (["happy", "smile"], 0.74),
        "sad": (["sad", "cornersdown"], 0.72),
        "angry": (["angry", "snarl"], 0.70),
        "blink": (["eyesclosedmax", "blink"], 0.92),
        "aa": (["aa", "jawlower", "mouthlargeopened"], 0.64),
        "ou": (["ow", "pucker", "kiss", "funneler"], 0.64),
    }
    activated = {}
    for name, (tokens, strength) in cases.items():
        activated[name] = lock.activate_case(skin, tokens, strength)
        renders[name] = str(lock.render_view(scene, setup, output, "front", f"AINA_DENSE_FINAL_{name.upper()}.png"))
    clear_shapes(skin)

    saved = save_materials(meshes)
    apply_clay(meshes, clay_material())
    for view, name in (
        ("front", "AINA_DENSE_FINAL_CLAY_FRONT.png"),
        ("three_quarter", "AINA_DENSE_FINAL_CLAY_THREE_QUARTER.png"),
        ("side", "AINA_DENSE_FINAL_CLAY_SIDE.png"),
    ):
        renders[f"clay_{view}"] = str(lock.render_view(scene, setup, output, view, name))
    restore_materials(meshes, saved)
    clear_shapes(skin)
    return renders, activated


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)
    data = json.loads(args.dense_landmarks.read_text())

    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    skin = lock.identify_skin(meshes)
    if len(skin.data.vertices) != 17161:
        raise RuntimeError(f"Unexpected FACS skin vertex count: {len(skin.data.vertices)}")
    if not skin.data.shape_keys or len(skin.data.shape_keys.key_blocks) < 27:
        raise RuntimeError("The real source FACS shape keys are missing")
    original_shape_names = [key.name for key in skin.data.shape_keys.key_blocks]
    setup = lock.build_setup(scene, meshes)
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100

    convergence = apply_dense_pass(scene, skin, meshes, data, setup)
    accessories = move_accessories(meshes, skin, convergence)
    setup = lock.build_setup(scene, meshes)
    metrics = final_metrics(scene, skin, data, setup)
    triangles = np.asarray([[vertex for vertex in polygon.vertices] for polygon in skin.data.polygons if len(polygon.vertices) == 3], dtype=np.int64)
    health = triangle_health(convergence["original_basis"], convergence["final_basis"], triangles)
    renders, activated = render_suite(scene, skin, meshes, setup, args.out)

    blend = args.out / "AINA_DENSE_IDENTITY_MASTER.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    glb = args.out / "AINA_DENSE_IDENTITY_MASTER.glb"
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(glb),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    final_shape_names = [key.name for key in skin.data.shape_keys.key_blocks]
    quantitative_gate = (
        metrics.get("front", 1.0) <= 0.043
        and metrics.get("three_quarter", 1.0) <= 0.060
        and ("side" not in metrics or metrics["side"] <= 0.095)
        and convergence["expression_delta_max_drift"] <= 2.5e-6
        and health["degenerate_triangles"] == 0
        and health["area_ratio_p01"] >= 0.18
        and health["area_ratio_p99"] <= 5.5
        and original_shape_names == final_shape_names
    )
    report = {
        "product": "AINA Dense Real Identity Master Candidate",
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "source_topology": "CC0 Vitruvian/Antonia FACS head",
        "vertices": len(skin.data.vertices),
        "triangles": int(len(triangles)),
        "topology_changed": False,
        "shape_keys_before": original_shape_names,
        "shape_keys_after": final_shape_names,
        "source_shape_key_count": max(0, len(final_shape_names) - 1),
        "dense_views": convergence["available_views"],
        "convergence_history": convergence["history"],
        "expression_delta_max_drift": convergence["expression_delta_max_drift"],
        "accessories": accessories,
        "dense_normalized_rmse": metrics,
        "mesh_health": health,
        "quantitative_identity_gate": quantitative_gate,
        "manual_visual_gate_required": True,
        "identity_lock": False,
        "visual_identity_lock": False,
        "candidate": True,
        "vrm_exported": False,
        "activated_expression_keys": activated,
        "files": {"blend": str(blend), "glb": str(glb), "renders": renders},
        "next_gate": "Inspect the actual approved-versus-real dense front, 3Q and side renders. Only the accepted real head may proceed unchanged into full-body VRM production.",
    }
    (qa / "AINA_DENSE_IDENTITY_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
