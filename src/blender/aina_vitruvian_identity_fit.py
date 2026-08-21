#!/usr/bin/env python3
"""Fit the real CC0 Vitruvian FACS topology to the approved AINA identity.

The operation deforms the existing neutral Basis and applies the identical
per-vertex correction to all existing shape keys, preserving expression deltas.
Front, three-quarter and side approved references jointly constrain the mesh.
No replacement effect art is generated and no VRM is exported.
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
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def shape_array(key):
    values = np.empty(len(key.data) * 3, dtype=np.float64)
    key.data.foreach_get("co", values)
    return values.reshape(-1, 3)


def set_shape_array(key, values):
    key.data.foreach_set("co", np.asarray(values, dtype=np.float32).ravel())


def world_vertices(obj, local_points):
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    homogeneous = np.c_[local_points, np.ones(len(local_points))]
    return (homogeneous @ matrix.T)[:, :3]


def local_vectors(obj, world_delta):
    rotation = np.array(obj.matrix_world.to_3x3(), dtype=np.float64)
    return world_delta @ np.linalg.inv(rotation).T


def mesh_world_bounds(objects):
    points = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        local = np.array([vertex.co[:] for vertex in obj.data.vertices], dtype=np.float64)
        points.append(world_vertices(obj, local))
    points = np.concatenate(points, axis=0)
    return points.min(axis=0), points.max(axis=0)


def eye_objects(scene):
    result = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        name = obj.name.lower()
        if any(token in name for token in ("eye", "iris", "pupil", "sclera", "tear", "caruncle")):
            result.append(obj)
    return result


def build_view_setup(scene, meshes):
    lo, hi = mesh_world_bounds(meshes)
    center = (lo + hi) * 0.5
    size = hi - lo
    eyes = eye_objects(scene)
    eye_centers = []
    for obj in eyes:
        local = np.array([vertex.co[:] for vertex in obj.data.vertices], dtype=np.float64)
        if len(local):
            eye_centers.append(world_vertices(obj, local).mean(axis=0))
    if eye_centers:
        eye_center = np.mean(eye_centers, axis=0)
        forward_sign = -1.0 if eye_center[1] < center[1] else 1.0
        target = np.array([eye_center[0], center[1], eye_center[2] - 0.018])
    else:
        forward_sign = -1.0
        target = center.copy()
        target[2] += 0.08 * max(size[2], 0.1)
    distance = max(float(size[2]) * 2.65, float(size[0]) * 2.8, 0.85)
    front = np.array([target[0], center[1] + forward_sign * distance, target[2]])
    locations = {
        "front": front,
        "three_quarter": front
        + np.array([0.42 * distance, -forward_sign * 0.10 * distance, 0.0]),
        "side": center + np.array([distance, 0.0, target[2] - center[2]]),
    }
    return {
        "bounds_min": lo,
        "bounds_max": hi,
        "center": center,
        "target": target,
        "forward_sign": forward_sign,
        "distance": distance,
        "locations": locations,
    }


def camera_for_view(scene, name, setup, lens=82.0):
    camera_name = f"AINA_Fit_Camera_{name}"
    old = bpy.data.objects.get(camera_name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    data = bpy.data.cameras.new(camera_name)
    camera = bpy.data.objects.new(camera_name, data)
    bpy.context.collection.objects.link(camera)
    camera.data.lens = lens
    camera.location = setup["locations"][name]
    camera.rotation_euler = (
        Vector(setup["target"]) - camera.location
    ).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    return camera


def project_points(scene, camera, obj, local_points):
    width = scene.render.resolution_x * scene.render.resolution_percentage / 100
    height = scene.render.resolution_y * scene.render.resolution_percentage / 100
    projected = np.zeros((len(local_points), 3), dtype=np.float64)
    matrix = obj.matrix_world
    for index, point in enumerate(local_points):
        ndc = world_to_camera_view(scene, camera, matrix @ Vector(point))
        projected[index] = (ndc.x * width, (1.0 - ndc.y) * height, ndc.z)
    return projected


def landmark_weights(view):
    weights = np.ones(68, dtype=np.float64)
    weights[:17] = 1.55
    weights[17:27] = 0.62
    weights[27:36] = 2.4
    weights[36:48] = 2.8
    weights[48:68] = 2.5
    if view == "side":
        weights[:] = 0.40
        weights[:17] = 2.1
        weights[27:36] = 3.0
        weights[36:48] = 0.65
        weights[48:68] = 2.5
    return weights


def weighted_similarity(source, destination, weights):
    """Map source 2D points into destination image coordinates."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(weights.sum(), 1e-9)
    source_center = np.sum(source * weights[:, None], axis=0)
    destination_center = np.sum(destination * weights[:, None], axis=0)
    x = source - source_center
    y = destination - destination_center
    covariance = (x * weights[:, None]).T @ y
    u, singular, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    denominator = np.sum(weights * np.sum(x * x, axis=1))
    scale = float(np.sum(singular) / max(denominator, 1e-9))
    mapped = scale * (x @ rotation.T) + destination_center
    return mapped, {
        "scale": scale,
        "rotation": rotation.tolist(),
        "source_center": source_center.tolist(),
        "destination_center": destination_center.tolist(),
    }


def choose_anchor_vertices(projected_vertices, landmark_pixels, k=40):
    indices = []
    for point in landmark_pixels:
        distance2 = np.sum((projected_vertices[:, :2] - point) ** 2, axis=1)
        count = min(k, len(distance2))
        candidates = np.argpartition(distance2, count - 1)[:count]
        score = distance2[candidates] + 0.06 * np.square(
            projected_vertices[candidates, 2] - projected_vertices[candidates, 2].min()
        )
        indices.append(int(candidates[np.argmin(score)]))
    return np.asarray(indices, dtype=np.int64)


def radius_for_landmark(index):
    if index < 17:
        return 0.040
    if index < 27:
        return 0.028
    if index < 36:
        return 0.024
    if index < 48:
        return 0.026
    return 0.028


def world_delta_from_pixels(scene, camera, anchor_world, residual):
    direction = anchor_world - np.array(camera.location[:], dtype=np.float64)
    depth = max(float(np.linalg.norm(direction)), 1e-5)
    width = scene.render.resolution_x * scene.render.resolution_percentage / 100
    height = scene.render.resolution_y * scene.render.resolution_percentage / 100
    world_per_pixel_x = 2.0 * depth * math.tan(camera.data.angle_x * 0.5) / width
    world_per_pixel_y = 2.0 * depth * math.tan(camera.data.angle_y * 0.5) / height
    rotation = camera.matrix_world.to_3x3()
    right = np.array((rotation @ Vector((1, 0, 0)))[:], dtype=np.float64)
    up = np.array((rotation @ Vector((0, 1, 0)))[:], dtype=np.float64)
    return right * residual[0] * world_per_pixel_x - up * residual[1] * world_per_pixel_y


def apply_delta_to_all_keys(skin, delta_local):
    keys = skin.data.shape_keys.key_blocks
    for key in keys:
        values = shape_array(key)
        set_shape_array(key, values + delta_local)
    skin.data.update()


def fit_identity(scene, skin, data, setup):
    keys = skin.data.shape_keys
    if not keys or "Basis" not in keys.key_blocks:
        raise RuntimeError("Skin object has no Basis shape key")
    basis = keys.key_blocks["Basis"]
    base_coordinates = shape_array(basis)
    original_coordinates = base_coordinates.copy()

    cameras = {}
    desired = {}
    anchor_indices = {}
    alignments = {}
    initial_model_landmarks = {}

    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100

    for view in ("front", "three_quarter", "side"):
        camera = camera_for_view(scene, view, setup)
        cameras[view] = camera
        model_points = np.asarray(data["model"][view]["landmarks_xy"], dtype=np.float64)
        target_points = np.asarray(data["approved"][view]["landmarks_xy"], dtype=np.float64)
        model_size = np.asarray(data["model"][view]["image_size"], dtype=np.float64)
        target_size = np.asarray(data["approved"][view]["image_size"], dtype=np.float64)
        model_points = model_points * np.array([768.0 / model_size[0], 768.0 / model_size[1]])
        target_points = target_points * np.array([768.0 / target_size[0], 768.0 / target_size[1]])
        mapped_target, alignment = weighted_similarity(
            target_points, model_points, landmark_weights(view)
        )
        desired[view] = mapped_target
        alignments[view] = alignment
        projected = project_points(scene, camera, skin, base_coordinates)
        anchors = choose_anchor_vertices(projected, model_points)
        anchor_indices[view] = anchors
        initial_model_landmarks[view] = projected[anchors, :2]

    iteration_reports = []
    view_steps = {"front": 0.68, "three_quarter": 0.48, "side": 0.56}
    iteration_steps = (1.0, 0.72, 0.48, 0.30)

    for iteration, iteration_step in enumerate(iteration_steps):
        basis_coordinates = shape_array(basis)
        accumulator = np.zeros_like(basis_coordinates)
        denominator = np.zeros(len(basis_coordinates), dtype=np.float64)
        view_report = {}

        for view in ("front", "three_quarter", "side"):
            camera = cameras[view]
            projected = project_points(scene, camera, skin, basis_coordinates)
            anchors = anchor_indices[view]
            current = projected[anchors, :2]
            residual = desired[view] - current
            weights = landmark_weights(view)
            rmse = float(
                np.sqrt(
                    np.sum(weights * np.sum(residual * residual, axis=1))
                    / np.sum(weights)
                )
            )
            view_report[view] = {"rmse_px": rmse, "max_px": float(np.linalg.norm(residual, axis=1).max())}

            basis_world = world_vertices(skin, basis_coordinates)
            for landmark_index, vertex_index in enumerate(anchors):
                world_delta = world_delta_from_pixels(
                    scene, camera, basis_world[vertex_index], residual[landmark_index]
                )
                length = np.linalg.norm(world_delta)
                cap = 0.0075 if view != "side" else 0.0085
                if length > cap:
                    world_delta *= cap / length
                world_delta *= view_steps[view] * iteration_step
                local_delta = local_vectors(skin, world_delta[None, :])[0]
                center = basis_coordinates[vertex_index]
                radius = radius_for_landmark(landmark_index)
                distance = np.linalg.norm(basis_coordinates - center, axis=1)
                local_weight = np.exp(-0.5 * (distance / radius) ** 4)
                local_weight[distance > radius * 1.55] = 0.0
                local_weight *= weights[landmark_index]
                accumulator += local_weight[:, None] * local_delta
                denominator += local_weight

        delta = accumulator / np.maximum(denominator[:, None], 1e-9)
        delta[denominator < 0.05] = 0.0
        length = np.linalg.norm(delta, axis=1)
        cap = np.minimum(1.0, 0.0038 / np.maximum(length, 1e-9))
        delta *= cap[:, None]
        apply_delta_to_all_keys(skin, delta)
        bpy.context.view_layer.update()
        iteration_reports.append(
            {
                "iteration": iteration,
                "views": view_report,
                "max_vertex_step_m": float(np.linalg.norm(delta, axis=1).max()),
                "rms_vertex_step_m": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
            }
        )

    final_coordinates = shape_array(basis)
    final_report = {}
    for view in ("front", "three_quarter", "side"):
        projected = project_points(scene, cameras[view], skin, final_coordinates)
        residual = desired[view] - projected[anchor_indices[view], :2]
        weights = landmark_weights(view)
        final_report[view] = {
            "rmse_px": float(
                np.sqrt(
                    np.sum(weights * np.sum(residual * residual, axis=1))
                    / np.sum(weights)
                )
            ),
            "max_px": float(np.linalg.norm(residual, axis=1).max()),
        }

    total_delta = final_coordinates - original_coordinates
    eye_shift_world = {}
    for side, indices in (("R", range(36, 42)), ("L", range(42, 48))):
        anchor = anchor_indices["front"][list(indices)]
        local_shift = total_delta[anchor].mean(axis=0)
        world_shift = local_shift @ np.array(skin.matrix_world.to_3x3(), dtype=np.float64).T
        eye_shift_world[side] = world_shift
        for obj in eye_objects(scene):
            local = np.array([vertex.co[:] for vertex in obj.data.vertices], dtype=np.float64)
            center_world = world_vertices(obj, local).mean(axis=0) if len(local) else np.array(obj.location[:])
            matches = center_world[0] < setup["target"][0] if side == "R" else center_world[0] >= setup["target"][0]
            if matches:
                obj.location += Vector(world_shift.tolist())

    return {
        "alignments": alignments,
        "anchor_indices": {view: value.tolist() for view, value in anchor_indices.items()},
        "initial_model_landmarks": {
            view: value.tolist() for view, value in initial_model_landmarks.items()
        },
        "desired_landmarks": {view: value.tolist() for view, value in desired.items()},
        "iterations": iteration_reports,
        "final_metrics": final_report,
        "max_total_displacement_m": float(np.linalg.norm(total_delta, axis=1).max()),
        "rms_total_displacement_m": float(
            np.sqrt(np.mean(np.sum(total_delta * total_delta, axis=1)))
        ),
        "eye_shift_world": {side: value.tolist() for side, value in eye_shift_world.items()},
    }


def make_material(name, color, roughness=0.50):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = tuple(color)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
    if shader:
        shader.inputs["Base Color"].default_value = tuple(color)
        shader.inputs["Roughness"].default_value = roughness
    return mat


def assign_material(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
        polygon.use_smooth = True


def setup_lighting(scene, target, front_location):
    for obj in list(scene.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    def area(name, location, energy, size):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()

    area("AINA_Fit_Key", tuple(front_location + np.array([0.35, 0.0, 0.35])), 260, 2.4)
    area("AINA_Fit_Fill", tuple(front_location + np.array([-0.45, 0.15, 0.10])), 135, 2.8)
    area("AINA_Fit_Rim", tuple(np.array(target) + np.array([0.0, 0.75, 0.35])), 190, 2.2)


def render_views(scene, skin, setup, out):
    preview = out / "Preview"
    preview.mkdir(parents=True, exist_ok=True)
    setup_lighting(scene, setup["target"], setup["locations"]["front"])
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.world.color = (0.025, 0.030, 0.045)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = -0.65
    except Exception:
        pass

    clay = make_material("AINA_Fitted_Clay", (0.22, 0.25, 0.31, 1.0), 0.56)
    eye = make_material("AINA_Fitted_Eye", (0.62, 0.70, 0.80, 1.0), 0.30)
    mouth = make_material("AINA_Fitted_Mouth", (0.055, 0.012, 0.018, 1.0), 0.48)
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        name = obj.name.lower()
        if any(token in name for token in ("eye", "iris", "pupil", "sclera", "tear", "caruncle")):
            assign_material(obj, eye)
        elif any(token in name for token in ("mouth", "teeth", "tongue", "gum")):
            assign_material(obj, mouth)
        else:
            assign_material(obj, clay)

    outputs = {}
    for view in ("front", "three_quarter", "side"):
        camera = camera_for_view(scene, view, setup)
        scene.camera = camera
        path = preview / f"AINA_VITRUVIAN_FITTED_{view.upper()}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        outputs[view] = str(path)
    return outputs


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)
    data = json.loads(args.landmarks.read_text())

    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    skin = bpy.data.objects.get("cm_vitruvian")
    if not skin:
        candidates = [
            obj
            for obj in meshes
            if obj.data.shape_keys and len(obj.data.shape_keys.key_blocks) > 8
        ]
        if not candidates:
            raise RuntimeError("Could not identify Vitruvian skin mesh")
        skin = max(candidates, key=lambda obj: len(obj.data.vertices))

    for obj in meshes:
        if obj.data.shape_keys:
            for key in obj.data.shape_keys.key_blocks:
                key.value = 0.0

    setup = build_view_setup(scene, meshes)
    fitting = fit_identity(scene, skin, data, setup)
    renders = render_views(scene, skin, setup, args.out)

    blend_path = args.out / "AINA_VITRUVIAN_IDENTITY_FIT.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_VITRUVIAN_IDENTITY_FIT.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    shape_keys = [
        key.name for key in skin.data.shape_keys.key_blocks
    ] if skin.data.shape_keys else []
    report = {
        "product": "AINA Vitruvian Identity Fit Candidate",
        "real_3d_model": True,
        "source_topology": "CC0 Vitruvian/Antonia head",
        "replacement_effect_art_generated": False,
        "topology_changed": False,
        "skin_object": skin.name,
        "vertices": len(skin.data.vertices),
        "shape_key_count": max(0, len(shape_keys) - 1),
        "shape_keys": shape_keys,
        "fitting": fitting,
        "identity_lock": False,
        "visual_identity_lock": False,
        "candidate": True,
        "vrm_exported": False,
        "files": {
            "blend": str(blend_path),
            "glb": str(glb_path),
            "renders": renders,
        },
        "next_gate": "Inspect real front, 3Q and side clay against approved AINA. Continue sculpting until all three views match before expression remapping.",
    }
    (qa / "AINA_VITRUVIAN_IDENTITY_FIT_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
