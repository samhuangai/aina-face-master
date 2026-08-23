#!/usr/bin/env python3
"""Reconstruct the real AINA identity on the Blender Studio Rain production rig.

This stage uses Rain's actual rigged meshes rather than another generic face
base. It performs a bounded, topology-preserving multi-view neutral deformation
against the already-approved AINA front, three-quarter and side references.
Existing skin weights, UVs, materials, armature relationships and shape-key
deltas are preserved. Eye and mouth anatomy are moved coherently. The output is
an editable BLEND and morph-preserving GLB candidate with real-model QA renders.

The script deliberately leaves identity_lock and visual_identity_lock false.
Those flags may only be changed after direct inspection of the generated real
front/3Q/profile and expression renders. No replacement effect art is generated
and no VRM is exported at this stage.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


HEAD_BONE_NAMES = (
    "DEF-Head",
    "DEF-head",
    "Head",
    "head",
    "mixamorig:Head",
    "ORG-Head",
    "ORG-head",
)
FACE_GROUP_TOKENS = (
    "head", "face", "jaw", "chin", "cheek", "brow", "eye", "lid",
    "nose", "nostril", "lip", "mouth", "temple", "forehead",
)
HAIR_TOKENS = ("hair", "bang", "fringe", "ponytail", "braid", "strand", "scalp")
EYE_TOKENS = ("eye", "iris", "pupil", "sclera", "cornea", "tear", "caruncle", "lash")
MOUTH_TOKENS = ("teeth", "tooth", "tongue", "gum", "mouth", "oral")
EXCLUDE_SKIN_TOKENS = HAIR_TOKENS + EYE_TOKENS + MOUTH_TOKENS + (
    "cloth", "shirt", "dress", "shoe", "boot", "sock", "jacket", "skirt",
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("source", "fit"), required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--landmarks", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def mesh_local_array(obj) -> np.ndarray:
    values = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", values)
    return values.reshape(-1, 3)


def set_mesh_local_array(obj, values: np.ndarray) -> None:
    obj.data.vertices.foreach_set("co", np.asarray(values, dtype=np.float32).ravel())
    obj.data.update()


def key_array(key) -> np.ndarray:
    values = np.empty(len(key.data) * 3, dtype=np.float64)
    key.data.foreach_get("co", values)
    return values.reshape(-1, 3)


def set_key_array(key, values: np.ndarray) -> None:
    key.data.foreach_set("co", np.asarray(values, dtype=np.float32).ravel())


def to_world(obj, local: np.ndarray) -> np.ndarray:
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    homogeneous = np.c_[local, np.ones(len(local))]
    return (homogeneous @ matrix.T)[:, :3]


def to_local(obj, world: np.ndarray) -> np.ndarray:
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    inverse = np.linalg.inv(matrix)
    homogeneous = np.c_[world, np.ones(len(world))]
    return (homogeneous @ inverse.T)[:, :3]


def world_vector_to_local(obj, delta: np.ndarray) -> np.ndarray:
    rotation = np.asarray(obj.matrix_world, dtype=np.float64)[:3, :3]
    return delta @ np.linalg.inv(rotation).T


def world_vertices(obj) -> np.ndarray:
    if obj.data.shape_keys:
        basis = obj.data.shape_keys.key_blocks.get("Basis") or obj.data.shape_keys.key_blocks[0]
        return to_world(obj, key_array(basis))
    return to_world(obj, mesh_local_array(obj))


def object_tokens(obj) -> str:
    names = [obj.name.lower()]
    names.extend(material.name.lower() for material in obj.data.materials if material)
    return " ".join(names)


def is_hair(obj) -> bool:
    text = object_tokens(obj)
    return any(token in text for token in HAIR_TOKENS)


def is_eye(obj) -> bool:
    text = object_tokens(obj)
    return any(token in text for token in EYE_TOKENS)


def is_mouth(obj) -> bool:
    text = object_tokens(obj)
    return any(token in text for token in MOUTH_TOKENS)


def reset_shape_keys(objects) -> None:
    for obj in objects:
        if not obj.data.shape_keys:
            continue
        for key in obj.data.shape_keys.key_blocks:
            key.value = 0.0


def find_armature(scene):
    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("Rain import contains no armature")
    return max(armatures, key=lambda obj: len(obj.data.bones))


def find_head_bone(armature):
    for name in HEAD_BONE_NAMES:
        bone = armature.pose.bones.get(name)
        if bone:
            return bone
    candidates = []
    for bone in armature.pose.bones:
        lower = bone.name.lower()
        if "head" not in lower:
            continue
        penalty = 0
        if any(token in lower for token in ("forehead", "head_ctrl", "headmaster", "mechanism", "mch")):
            penalty += 20
        if lower.startswith("def-"):
            penalty -= 10
        candidates.append((penalty, len(lower), bone))
    if not candidates:
        raise RuntimeError("Could not locate a Rain head bone")
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def bone_world_point(armature, bone) -> np.ndarray:
    head = armature.matrix_world @ bone.head
    tail = armature.matrix_world @ bone.tail
    return np.asarray((head * 0.42 + tail * 0.58)[:], dtype=np.float64)


def all_mesh_bounds(meshes):
    arrays = [world_vertices(obj) for obj in meshes if len(obj.data.vertices)]
    points = np.concatenate(arrays, axis=0)
    return points.min(axis=0), points.max(axis=0)


def vertex_group_face_score(obj) -> float:
    score = 0.0
    for group in obj.vertex_groups:
        lower = group.name.lower()
        if any(token in lower for token in FACE_GROUP_TOKENS):
            score += 1.0
        if lower in {name.lower() for name in HEAD_BONE_NAMES}:
            score += 8.0
    return score


def identify_skin(scene, head_point: np.ndarray):
    meshes = [obj for obj in scene.objects if obj.type == "MESH" and len(obj.data.vertices)]
    if not meshes:
        raise RuntimeError("Rain import contains no meshes")
    lo, hi = all_mesh_bounds(meshes)
    character_height = max(float(hi[2] - lo[2]), 0.5)
    scored = []
    for obj in meshes:
        text = object_tokens(obj)
        if any(token in text for token in EXCLUDE_SKIN_TOKENS):
            exclusion = 28.0
        else:
            exclusion = 0.0
        points = world_vertices(obj)
        distance = np.linalg.norm(points - head_point[None, :], axis=1)
        near = float(np.mean(distance < character_height * 0.18))
        group_score = vertex_group_face_score(obj)
        material_score = 6.0 if any(token in text for token in ("skin", "face", "head", "body")) else 0.0
        shape_score = 4.0 if obj.data.shape_keys and len(obj.data.shape_keys.key_blocks) > 1 else 0.0
        size_score = min(math.log10(max(len(obj.data.vertices), 10)), 5.0)
        score = 42.0 * near + group_score + material_score + shape_score + size_score - exclusion
        scored.append((score, obj.name, obj))
    scored.sort(key=lambda item: item[0], reverse=True)
    skin = scored[0][2]
    return skin, {
        "character_bounds_min": lo.tolist(),
        "character_bounds_max": hi.tolist(),
        "character_height_m": character_height,
        "skin_ranking": [
            {"name": name, "score": float(score), "vertices": len(obj.data.vertices)}
            for score, name, obj in scored[:10]
        ],
    }


def face_group_vertex_weights(obj) -> np.ndarray:
    result = np.zeros(len(obj.data.vertices), dtype=np.float64)
    accepted = {
        group.index for group in obj.vertex_groups
        if any(token in group.name.lower() for token in FACE_GROUP_TOKENS)
    }
    for vertex in obj.data.vertices:
        total = 0.0
        for membership in vertex.groups:
            if membership.group in accepted:
                total += membership.weight
        result[vertex.index] = min(total, 1.0)
    return result


def eye_objects(scene, head_point: np.ndarray, character_height: float):
    result = []
    radius = character_height * 0.22
    for obj in scene.objects:
        if obj.type != "MESH" or not len(obj.data.vertices) or not is_eye(obj):
            continue
        centre = world_vertices(obj).mean(axis=0)
        if np.linalg.norm(centre - head_point) < radius:
            result.append(obj)
    return result


def eye_centres(objects) -> list[np.ndarray]:
    centres = [world_vertices(obj).mean(axis=0) for obj in objects if len(obj.data.vertices)]
    if not centres:
        return []
    # Collapse several cornea/iris/sclera objects into two eye centres.
    x_values = np.asarray([centre[0] for centre in centres])
    median = float(np.median(x_values))
    left = [centre for centre in centres if centre[0] <= median]
    right = [centre for centre in centres if centre[0] > median]
    output = []
    if left:
        output.append(np.mean(left, axis=0))
    if right:
        output.append(np.mean(right, axis=0))
    output.sort(key=lambda point: point[0])
    return output


def head_region(skin, head_point: np.ndarray, eyes: list[np.ndarray], character_height: float):
    points = world_vertices(skin)
    group = face_group_vertex_weights(skin)
    if eyes:
        eye_average = np.mean(eyes, axis=0)
        centre = np.array([eye_average[0], head_point[1], eye_average[2] - 0.015 * character_height])
    else:
        centre = head_point + np.array([0.0, 0.0, 0.025 * character_height])
    radii = np.array([
        max(0.075, 0.105 * character_height),
        max(0.090, 0.115 * character_height),
        max(0.105, 0.145 * character_height),
    ])
    q = np.sqrt(np.sum(((points - centre) / radii) ** 2, axis=1))
    spatial = np.clip((1.25 - q) / 0.35, 0.0, 1.0)
    weights = np.maximum(spatial, group)
    ids = np.where(weights > 0.08)[0]
    if len(ids) < 1500:
        ids = np.where(q < 1.45)[0]
    if len(ids) < 500:
        raise RuntimeError(f"Rain head region is too small: {len(ids)} vertices")
    return ids.astype(np.int64), weights, centre, radii


def capture_shape_deltas(obj) -> dict[str, np.ndarray]:
    if not obj.data.shape_keys:
        return {}
    keys = obj.data.shape_keys.key_blocks
    basis = key_array(keys.get("Basis") or keys[0])
    return {key.name: key_array(key) - basis for key in keys if key.name != "Basis"}


def validate_shape_deltas(obj, original: dict[str, np.ndarray]) -> dict:
    if not original:
        return {
            "shape_key_count": 0,
            "shape_delta_preservation_max_m": 0.0,
            "shape_delta_preservation_rms_m": 0.0,
        }
    keys = obj.data.shape_keys.key_blocks
    basis = key_array(keys.get("Basis") or keys[0])
    max_error = 0.0
    squared = 0.0
    count = 0
    for name, expected in original.items():
        key = keys.get(name)
        if key is None:
            raise RuntimeError(f"Shape key disappeared during Rain fit: {name}")
        error = (key_array(key) - basis) - expected
        max_error = max(max_error, float(np.linalg.norm(error, axis=1).max()))
        squared += float(np.sum(error * error))
        count += error.size
    return {
        "shape_key_count": len(original),
        "shape_delta_preservation_max_m": max_error,
        "shape_delta_preservation_rms_m": math.sqrt(squared / max(count, 1)),
    }


def apply_world_delta(obj, world_delta: np.ndarray) -> None:
    local_delta = world_vector_to_local(obj, world_delta)
    if obj.data.shape_keys:
        for key in obj.data.shape_keys.key_blocks:
            set_key_array(key, key_array(key) + local_delta)
        obj.data.update()
    else:
        set_mesh_local_array(obj, mesh_local_array(obj) + local_delta)


def clear_render_objects(scene) -> None:
    for obj in list(scene.objects):
        if obj.type in {"LIGHT", "CAMERA"} and obj.name.startswith("AINA_Rain_"):
            bpy.data.objects.remove(obj, do_unlink=True)


def create_light(name: str, location, energy: float, size: float, target) -> None:
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def setup_cameras(scene, skin, head_ids, eyes, head_point, character_height):
    points = world_vertices(skin)[head_ids]
    lo, hi = points.min(axis=0), points.max(axis=0)
    centre = (lo + hi) * 0.5
    size = hi - lo
    eye_average = np.mean(eyes, axis=0) if eyes else centre + np.array([0.0, 0.0, 0.03 * character_height])
    forward_sign = -1.0 if eye_average[1] < centre[1] else 1.0
    target = np.array([eye_average[0], centre[1], eye_average[2] - 0.035 * max(size[2], 0.1)])
    distance = max(float(size[2]) * 2.75, float(size[0]) * 3.15, 0.72)
    front = np.array([target[0], centre[1] + forward_sign * distance, target[2]])
    locations = {
        "front": front,
        "three_quarter": front + np.array([0.43 * distance, -forward_sign * 0.10 * distance, 0.0]),
        "side": centre + np.array([distance, 0.0, target[2] - centre[2]]),
        "left_45": front + np.array([-0.49 * distance, -forward_sign * 0.13 * distance, 0.0]),
        "right_45": front + np.array([0.49 * distance, -forward_sign * 0.13 * distance, 0.0]),
    }
    clear_render_objects(scene)
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.world.color = (0.025, 0.030, 0.045)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = -0.25
    except Exception:
        pass
    create_light(
        "AINA_Rain_Key",
        tuple(front + np.array([0.55 * size[0], 0.0, 0.55 * size[2]])),
        650,
        2.2,
        target,
    )
    create_light(
        "AINA_Rain_Fill",
        tuple(front + np.array([-0.75 * size[0], 0.18 * distance, 0.10 * size[2]])),
        300,
        2.7,
        target,
    )
    create_light(
        "AINA_Rain_Rim",
        tuple(centre + np.array([0.0, -forward_sign * distance * 0.70, 0.48 * size[2]])),
        420,
        2.0,
        target,
    )
    cameras = {}
    for name, location in locations.items():
        data = bpy.data.cameras.new(f"AINA_Rain_Camera_{name}")
        camera = bpy.data.objects.new(f"AINA_Rain_Camera_{name}", data)
        bpy.context.collection.objects.link(camera)
        camera.data.lens = 88
        camera.location = location
        camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
        cameras[name] = camera
    scene.camera = cameras["front"]
    return cameras, {
        "head_bounds_min": lo.tolist(),
        "head_bounds_max": hi.tolist(),
        "head_centre": centre.tolist(),
        "target": target.tolist(),
        "forward_sign_y": forward_sign,
        "distance": distance,
        "locations": {name: value.tolist() for name, value in locations.items()},
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
    }


def set_hair_visible(scene, visible: bool) -> dict[str, tuple[bool, bool]]:
    previous = {}
    for obj in scene.objects:
        if obj.type != "MESH" or not is_hair(obj):
            continue
        previous[obj.name] = (obj.hide_render, obj.hide_viewport)
        obj.hide_render = not visible
        obj.hide_viewport = not visible
    return previous


def restore_visibility(scene, previous: dict[str, tuple[bool, bool]]) -> None:
    for name, state in previous.items():
        obj = scene.objects.get(name)
        if obj:
            obj.hide_render, obj.hide_viewport = state


def render(scene, camera, path: Path) -> None:
    scene.camera = camera
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_source(scene, cameras, out: Path) -> dict:
    source = out / "Source"
    source.mkdir(exist_ok=True)
    previous = set_hair_visible(scene, False)
    outputs = {}
    for view in ("front", "three_quarter", "side"):
        path = source / f"AINA_RAIN_SOURCE_{view.upper()}.png"
        render(scene, cameras[view], path)
        outputs[view] = str(path)
    restore_visibility(scene, previous)
    return outputs


def material(name: str, color, roughness=0.50, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = tuple(color)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
    if shader:
        shader.inputs["Base Color"].default_value = tuple(color)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metallic
    return mat


def assign_single_material(obj, mat) -> list:
    old = [slot.material for slot in obj.material_slots]
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
        polygon.use_smooth = True
    return old


def restore_materials(obj, old) -> None:
    obj.data.materials.clear()
    for mat in old:
        if mat:
            obj.data.materials.append(mat)


def render_final_suite(scene, cameras, skin, out: Path) -> dict:
    preview = out / "Preview"
    preview.mkdir(exist_ok=True)
    previous_visibility = set_hair_visible(scene, False)
    outputs = {"beauty": {}, "clay": {}, "expressions": {}}
    reset_shape_keys([obj for obj in scene.objects if obj.type == "MESH"])
    for view in ("front", "three_quarter", "side", "left_45", "right_45"):
        path = preview / f"AINA_RAIN_IDENTITY_MASTER_{view.upper()}.png"
        render(scene, cameras[view], path)
        outputs["beauty"][view] = str(path)

    clay = material("AINA_Rain_Identity_Clay", (0.30, 0.34, 0.42, 1.0), 0.58)
    eye = material("AINA_Rain_Identity_Eye", (0.68, 0.74, 0.82, 1.0), 0.28)
    mouth = material("AINA_Rain_Identity_Mouth", (0.075, 0.012, 0.020, 1.0), 0.44)
    old_materials = {}
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        if is_hair(obj):
            continue
        old_materials[obj.name] = assign_single_material(obj, eye if is_eye(obj) else mouth if is_mouth(obj) else clay)
    for view in ("front", "three_quarter", "side", "left_45", "right_45"):
        path = preview / f"AINA_RAIN_IDENTITY_MASTER_CLAY_{view.upper()}.png"
        render(scene, cameras[view], path)
        outputs["clay"][view] = str(path)
    for name, old in old_materials.items():
        obj = scene.objects.get(name)
        if obj:
            restore_materials(obj, old)

    # Render available real shape keys without inventing expression geometry.
    if skin.data.shape_keys:
        keys = skin.data.shape_keys.key_blocks
        candidates = {
            "happy": ("smile", "happy", "joy"),
            "blink": ("blink", "closed", "eyeclose"),
            "aa": ("jawopen", "mouthopen", "aa", "jaw_open"),
            "sad": ("sad", "frown", "sorrow"),
        }
        for label, tokens in candidates.items():
            reset_shape_keys([skin])
            chosen = None
            for key in keys:
                lower = key.name.lower().replace("_", "")
                if any(token.replace("_", "") in lower for token in tokens):
                    chosen = key
                    break
            if chosen:
                chosen.value = 1.0
                path = preview / f"AINA_RAIN_IDENTITY_MASTER_{label.upper()}.png"
                render(scene, cameras["front"], path)
                outputs["expressions"][label] = {"file": str(path), "shape_key": chosen.name}
        reset_shape_keys([skin])
    restore_visibility(scene, previous_visibility)
    return outputs


def normalize_points(item) -> np.ndarray:
    width, height = item["image_size"]
    points = np.asarray(item["landmarks_xy"], dtype=np.float64)
    return points / np.array([width, height], dtype=np.float64)


def landmark_weights(view: str) -> np.ndarray:
    weight = np.ones(68, dtype=np.float64)
    weight[:17] = 1.65
    weight[17:27] = 0.62
    weight[27:36] = 2.75
    weight[36:48] = 3.15
    weight[48:68] = 2.80
    if view == "side":
        weight[:] = 0.35
        weight[:17] = 2.45
        weight[27:36] = 3.30
        weight[36:48] = 0.55
        weight[48:68] = 2.40
    return weight


def weighted_similarity(source: np.ndarray, destination: np.ndarray, weight: np.ndarray) -> np.ndarray:
    weight = np.asarray(weight, dtype=np.float64)
    weight /= max(float(weight.sum()), 1e-9)
    source_centre = np.sum(source * weight[:, None], axis=0)
    destination_centre = np.sum(destination * weight[:, None], axis=0)
    x = source - source_centre
    y = destination - destination_centre
    covariance = (x * weight[:, None]).T @ y
    u, singular, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    denominator = np.sum(weight * np.sum(x * x, axis=1))
    scale = float(np.sum(singular) / max(denominator, 1e-9))
    return scale * (x @ rotation.T) + destination_centre


def project_points(scene, camera, obj, local_points: np.ndarray) -> np.ndarray:
    width = scene.render.resolution_x * scene.render.resolution_percentage / 100.0
    height = scene.render.resolution_y * scene.render.resolution_percentage / 100.0
    output = np.zeros((len(local_points), 3), dtype=np.float64)
    matrix = obj.matrix_world
    for index, point in enumerate(local_points):
        ndc = world_to_camera_view(scene, camera, matrix @ Vector(point))
        output[index] = (ndc.x * width, (1.0 - ndc.y) * height, ndc.z)
    return output


def choose_anchor_vertices(projected: np.ndarray, target_pixels: np.ndarray) -> np.ndarray:
    anchors = []
    for point in target_pixels:
        distance2 = np.sum((projected[:, :2] - point) ** 2, axis=1)
        count = min(96, len(distance2))
        candidates = np.argpartition(distance2, count - 1)[:count]
        depth = projected[candidates, 2]
        valid = depth > 0
        if np.any(valid):
            candidates = candidates[valid]
            depth = depth[valid]
        depth_norm = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-8)
        score = distance2[candidates] + 18.0 * depth_norm * depth_norm
        anchors.append(int(candidates[np.argmin(score)]))
    return np.asarray(anchors, dtype=np.int64)


def pixel_residual_to_world(scene, camera, anchor_world: np.ndarray, residual: np.ndarray) -> np.ndarray:
    width = scene.render.resolution_x * scene.render.resolution_percentage / 100.0
    height = scene.render.resolution_y * scene.render.resolution_percentage / 100.0
    camera_position = np.asarray(camera.matrix_world.translation[:], dtype=np.float64)
    depth = max(float(np.linalg.norm(anchor_world - camera_position)), 1e-5)
    world_x = 2.0 * depth * math.tan(camera.data.angle_x * 0.5) / max(width, 1.0)
    world_y = 2.0 * depth * math.tan(camera.data.angle_y * 0.5) / max(height, 1.0)
    rotation = camera.matrix_world.to_3x3()
    right = np.asarray((rotation @ Vector((1.0, 0.0, 0.0)))[:], dtype=np.float64)
    up = np.asarray((rotation @ Vector((0.0, 1.0, 0.0)))[:], dtype=np.float64)
    return residual[0] * world_x * right - residual[1] * world_y * up


def landmark_radius(index: int, face_width: float) -> float:
    if index < 17:
        return 0.22 * face_width
    if index < 27:
        return 0.15 * face_width
    if index < 36:
        return 0.14 * face_width
    if index < 48:
        return 0.15 * face_width
    return 0.16 * face_width


def adjacency_for_region(obj, region_ids: np.ndarray):
    lookup = {int(vertex_id): index for index, vertex_id in enumerate(region_ids)}
    adjacency = [set() for _ in region_ids]
    for polygon in obj.data.polygons:
        vertices = list(polygon.vertices)
        for i, source in enumerate(vertices):
            a = lookup.get(int(source))
            if a is None:
                continue
            for target in vertices[i + 1 :] + vertices[:i]:
                b = lookup.get(int(target))
                if b is not None and a != b:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
    return adjacency


def smooth_region_delta(delta: np.ndarray, adjacency, preserve: np.ndarray, passes=2) -> np.ndarray:
    result = delta.copy()
    for _ in range(passes):
        updated = result.copy()
        for index, neighbours in enumerate(adjacency):
            if not neighbours:
                continue
            average = np.mean(result[list(neighbours)], axis=0)
            strength = 0.30 * (1.0 - 0.80 * preserve[index])
            updated[index] = result[index] * (1.0 - strength) + average * strength
        result = updated
    return result


def rmse(prediction: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    error = np.sum((prediction - target) ** 2, axis=1)
    return float(np.sqrt(np.sum(weight * error) / max(float(weight.sum()), 1e-9)))


def ellipsoid_weight(points: np.ndarray, centre, radii, outer=1.0) -> np.ndarray:
    centre = np.asarray(centre, dtype=np.float64)
    radii = np.maximum(np.asarray(radii, dtype=np.float64), 1e-7)
    q = np.sqrt(np.sum(((points - centre) / radii) ** 2, axis=1))
    result = np.zeros(len(points), dtype=np.float64)
    mask = q < outer
    if np.any(mask):
        t = q[mask] / outer
        result[mask] = 0.5 * (1.0 + np.cos(np.pi * t))
    return result


def art_directed_residual(
    points: np.ndarray,
    anchors_world: np.ndarray,
    model_front: np.ndarray,
    desired_front: np.ndarray,
    forward_sign: float,
) -> tuple[np.ndarray, dict]:
    delta = np.zeros_like(points)
    face_width = max(float(np.linalg.norm(anchors_world[0] - anchors_world[16])), 0.12)

    def distance(array, a, b):
        return max(float(np.linalg.norm(array[a] - array[b])), 1e-6)

    model_eye_width = 0.5 * (distance(model_front, 36, 39) + distance(model_front, 42, 45))
    target_eye_width = 0.5 * (distance(desired_front, 36, 39) + distance(desired_front, 42, 45))
    model_eye_height = 0.25 * (
        distance(model_front, 37, 41) + distance(model_front, 38, 40)
        + distance(model_front, 43, 47) + distance(model_front, 44, 46)
    )
    target_eye_height = 0.25 * (
        distance(desired_front, 37, 41) + distance(desired_front, 38, 40)
        + distance(desired_front, 43, 47) + distance(desired_front, 44, 46)
    )
    eye_x = float(np.clip(target_eye_width / model_eye_width, 0.90, 1.16))
    eye_z = float(np.clip(target_eye_height / model_eye_height, 0.88, 1.20))
    for indices in (range(36, 42), range(42, 48)):
        centre = anchors_world[list(indices)].mean(axis=0)
        weight = ellipsoid_weight(points, centre, (0.25 * face_width, 0.28 * face_width, 0.19 * face_width), 1.15)
        target = points.copy()
        target[:, 0] = centre[0] + (points[:, 0] - centre[0]) * (1.0 + 0.35 * (eye_x - 1.0))
        target[:, 2] = centre[2] + (points[:, 2] - centre[2]) * (1.0 + 0.35 * (eye_z - 1.0))
        delta += weight[:, None] * (target - points)

    nose_centre = anchors_world[27:36].mean(axis=0)
    model_nose = distance(model_front, 31, 35)
    target_nose = distance(desired_front, 31, 35)
    nose_scale = float(np.clip(target_nose / model_nose, 0.82, 1.10))
    weight = ellipsoid_weight(points, nose_centre, (0.15 * face_width, 0.22 * face_width, 0.24 * face_width), 1.12)
    target = points.copy()
    target[:, 0] = nose_centre[0] + (points[:, 0] - nose_centre[0]) * (1.0 + 0.45 * (nose_scale - 1.0))
    delta += weight[:, None] * (target - points)
    delta[:, 1] += forward_sign * 0.0018 * weight
    tip_weight = ellipsoid_weight(points, anchors_world[30], (0.10 * face_width, 0.13 * face_width, 0.12 * face_width), 1.05)
    delta[:, 1] += forward_sign * 0.0018 * tip_weight
    delta[:, 2] += 0.0010 * tip_weight

    mouth_centre = anchors_world[48:68].mean(axis=0)
    model_mouth = distance(model_front, 48, 54)
    target_mouth = distance(desired_front, 48, 54)
    mouth_scale = float(np.clip(target_mouth / model_mouth, 0.88, 1.14))
    weight = ellipsoid_weight(points, mouth_centre, (0.27 * face_width, 0.20 * face_width, 0.13 * face_width), 1.12)
    target = points.copy()
    target[:, 0] = mouth_centre[0] + (points[:, 0] - mouth_centre[0]) * (1.0 + 0.40 * (mouth_scale - 1.0))
    target[:, 2] = mouth_centre[2] + (points[:, 2] - mouth_centre[2]) * 1.06
    delta += weight[:, None] * (target - points)
    delta[:, 1] += forward_sign * 0.0015 * weight

    model_lower = abs(float(model_front[8, 1] - model_front[33, 1]))
    target_lower = abs(float(desired_front[8, 1] - desired_front[33, 1]))
    lower_scale = float(np.clip(target_lower / max(model_lower, 1e-6), 0.82, 1.06))
    nose_z = float(anchors_world[33, 2])
    chin_z = float(anchors_world[8, 2])
    lower = np.clip((nose_z - points[:, 2]) / max(nose_z - chin_z, 1e-6), 0.0, 1.0)
    centre_x = float(anchors_world[27:36, 0].mean())
    face_front = ellipsoid_weight(
        points,
        (centre_x, mouth_centre[1], 0.5 * (nose_z + chin_z)),
        (0.55 * face_width, 0.55 * face_width, 0.60 * abs(nose_z - chin_z)),
        1.25,
    )
    compressed = nose_z + (points[:, 2] - nose_z) * (1.0 + 0.55 * (lower_scale - 1.0))
    delta[:, 2] += (compressed - points[:, 2]) * lower * face_front

    model_jaw = distance(model_front, 4, 12)
    target_jaw = distance(desired_front, 4, 12)
    jaw_scale = float(np.clip(target_jaw / model_jaw, 0.82, 1.05))
    taper = 1.0 + 0.55 * (jaw_scale - 1.0) * np.power(lower, 1.20)
    delta[:, 0] += ((points[:, 0] - centre_x) * taper - (points[:, 0] - centre_x)) * face_front

    chin_weight = ellipsoid_weight(points, anchors_world[8], (0.22 * face_width, 0.26 * face_width, 0.22 * face_width), 1.10)
    delta[:, 0] += -(points[:, 0] - anchors_world[8, 0]) * 0.08 * chin_weight
    delta[:, 2] += 0.0018 * chin_weight

    lengths = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1.0, 0.0065 / np.maximum(lengths, 1e-9))[:, None]
    return delta, {
        "eye_width_ratio": eye_x,
        "eye_height_ratio": eye_z,
        "nose_width_ratio": nose_scale,
        "mouth_width_ratio": mouth_scale,
        "lower_face_ratio": lower_scale,
        "jaw_width_ratio": jaw_scale,
        "max_residual_m": float(np.linalg.norm(delta, axis=1).max()),
        "rms_residual_m": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
    }


def move_related_objects(scene, skin, initial_world: np.ndarray, final_world: np.ndarray, eyes_before: list[np.ndarray]) -> dict:
    skin_delta = final_world - initial_world
    moved = []
    max_shift = 0.0
    face_width = max(float(final_world[:, 0].max() - final_world[:, 0].min()), 0.12)
    for obj in scene.objects:
        if obj.type != "MESH" or obj == skin or not (is_eye(obj) or is_mouth(obj)):
            continue
        vertices = world_vertices(obj)
        centre = vertices.mean(axis=0)
        distances = np.linalg.norm(initial_world - centre[None, :], axis=1)
        radius = 0.24 * face_width if is_eye(obj) else 0.30 * face_width
        weight = np.exp(-0.5 * (distances / max(radius, 1e-6)) ** 4)
        shift = np.sum(skin_delta * weight[:, None], axis=0) / max(float(weight.sum()), 1e-9)
        shift_length = float(np.linalg.norm(shift))
        if shift_length > 0.010:
            shift *= 0.010 / shift_length
            shift_length = 0.010
        result = vertices + shift
        set_mesh_local_array(obj, to_local(obj, result))
        moved.append({"object": obj.name, "shift_m": shift.tolist()})
        max_shift = max(max_shift, shift_length)
    return {"objects": moved, "max_shift_m": max_shift}


def fit_identity(scene, skin, head_ids, cameras, data, forward_sign: float) -> dict:
    available = [view for view in data["available_views"] if view in cameras]
    original_deltas = capture_shape_deltas(skin)
    basis_key = skin.data.shape_keys.key_blocks.get("Basis") if skin.data.shape_keys else None
    initial_local = key_array(basis_key) if basis_key else mesh_local_array(skin)
    initial_world_full = to_world(skin, initial_local)
    region_world = initial_world_full[head_ids].copy()
    adjacency = adjacency_for_region(skin, head_ids)
    width = scene.render.resolution_x
    height = scene.render.resolution_y

    view_data = {}
    for view in available:
        approved = normalize_points(data["approved"][view])
        model = normalize_points(data["model"][view])
        weight = landmark_weights(view)
        desired = weighted_similarity(approved, model, weight)
        model_pixels = model * np.array([width, height])
        desired_pixels = desired * np.array([width, height])
        projected = project_points(scene, cameras[view], skin, initial_local[head_ids])
        anchors_local = choose_anchor_vertices(projected, model_pixels)
        view_data[view] = {
            "approved_norm": approved,
            "model_norm": model,
            "desired_norm": desired,
            "model_pixels": model_pixels,
            "desired_pixels": desired_pixels,
            "weight": weight,
            "anchors_local": anchors_local,
        }

    history = []
    for iteration, step in enumerate((0.72, 0.54, 0.40, 0.28, 0.18)):
        current_local = key_array(basis_key) if basis_key else mesh_local_array(skin)
        current_world_full = to_world(skin, current_local)
        current_region = current_world_full[head_ids]
        accumulated = np.zeros_like(current_region)
        denominator = np.zeros(len(current_region), dtype=np.float64)
        preserve = np.zeros(len(current_region), dtype=np.float64)
        metrics = {}
        for view in available:
            item = view_data[view]
            camera = cameras[view]
            projected = project_points(scene, camera, skin, current_local[head_ids])
            anchor_indices = item["anchors_local"]
            anchor_pixels = projected[anchor_indices, :2]
            residuals = item["desired_pixels"] - anchor_pixels
            metrics[view] = rmse(anchor_pixels / np.array([width, height]), item["desired_norm"], item["weight"])
            anchors_world = current_region[anchor_indices]
            face_width = max(float(np.linalg.norm(anchors_world[0] - anchors_world[16])), 0.12)
            for landmark_index, (anchor, residual) in enumerate(zip(anchors_world, residuals)):
                anchor_delta = pixel_residual_to_world(scene, camera, anchor, residual)
                length = float(np.linalg.norm(anchor_delta))
                max_anchor = 0.0070 if view == "side" else 0.0055
                if length > max_anchor:
                    anchor_delta *= max_anchor / length
                radius = landmark_radius(landmark_index, face_width)
                distance = np.linalg.norm(current_region - anchor[None, :], axis=1)
                local = np.exp(-0.5 * (distance / max(radius, 1e-6)) ** 4)
                local[distance > radius * 1.50] = 0.0
                local *= item["weight"][landmark_index]
                accumulated += local[:, None] * anchor_delta * step
                denominator += local
                preserve = np.maximum(preserve, np.clip(local / max(float(local.max()), 1e-9), 0.0, 1.0))
        region_delta = accumulated / np.maximum(denominator[:, None], 1e-9)
        region_delta[denominator < 0.08] = 0.0
        region_delta = smooth_region_delta(region_delta, adjacency, preserve, passes=2)
        lengths = np.linalg.norm(region_delta, axis=1)
        region_delta *= np.minimum(1.0, 0.0038 / np.maximum(lengths, 1e-9))[:, None]
        full_delta = np.zeros_like(current_world_full)
        full_delta[head_ids] = region_delta
        apply_world_delta(skin, full_delta)
        history.append({
            "iteration": iteration,
            "view_rmse_before_step": metrics,
            "max_vertex_step_m": float(np.linalg.norm(region_delta, axis=1).max()),
            "rms_vertex_step_m": float(np.sqrt(np.mean(np.sum(region_delta * region_delta, axis=1)))),
        })

    current_local = key_array(basis_key) if basis_key else mesh_local_array(skin)
    current_world_full = to_world(skin, current_local)
    current_region = current_world_full[head_ids]
    front_item = view_data["front"]
    front_anchors = current_region[front_item["anchors_local"]]
    residual, art_report = art_directed_residual(
        current_region,
        front_anchors,
        front_item["model_norm"],
        front_item["desired_norm"],
        forward_sign,
    )
    residual = smooth_region_delta(residual, adjacency, np.zeros(len(residual)), passes=2)
    full_residual = np.zeros_like(current_world_full)
    full_residual[head_ids] = residual
    apply_world_delta(skin, full_residual)

    final_local = key_array(basis_key) if basis_key else mesh_local_array(skin)
    final_world_full = to_world(skin, final_local)
    final_metrics = {}
    for view in available:
        item = view_data[view]
        projected = project_points(scene, cameras[view], skin, final_local[head_ids])
        anchor_pixels = projected[item["anchors_local"], :2]
        final_metrics[view] = rmse(
            anchor_pixels / np.array([width, height]),
            item["desired_norm"],
            item["weight"],
        )
    preservation = validate_shape_deltas(skin, original_deltas)
    return {
        "available_views": available,
        "history": history,
        "final_metrics": final_metrics,
        "art_directed_residual": art_report,
        "expression_preservation": preservation,
        "initial_world": initial_world_full,
        "final_world": final_world_full,
        "total_max_displacement_m": float(np.linalg.norm(final_world_full - initial_world_full, axis=1).max()),
        "total_rms_displacement_m": float(np.sqrt(np.mean(np.sum((final_world_full - initial_world_full) ** 2, axis=1)))),
    }


def main_source(args) -> None:
    if not args.source or not args.source.exists():
        raise RuntimeError("--source Rain GLB is required for source stage")
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(args.source))
    bpy.context.view_layer.update()
    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH" and len(obj.data.vertices)]
    reset_shape_keys(meshes)
    armature = find_armature(scene)
    head_bone = find_head_bone(armature)
    head_point = bone_world_point(armature, head_bone)
    skin, skin_report = identify_skin(scene, head_point)
    character_height = skin_report["character_height_m"]
    eyes_objects = eye_objects(scene, head_point, character_height)
    eyes = eye_centres(eyes_objects)
    head_ids, region_weight, region_centre, region_radii = head_region(
        skin, head_point, eyes, character_height
    )
    cameras, camera_report = setup_cameras(
        scene, skin, head_ids, eyes, head_point, character_height
    )
    renders = render_source(scene, cameras, args.out)
    source_blend = args.out / "AINA_RAIN_IDENTITY_SOURCE.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(source_blend))
    report = {
        "product": "AINA Rain Identity Source",
        "source": str(args.source),
        "source_character": "Blender Studio Rain v3",
        "source_license": "CC BY 4.0",
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "armature": armature.name,
        "head_bone": head_bone.name,
        "skin_object": skin.name,
        "skin_vertices": len(skin.data.vertices),
        "skin_triangles": sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons),
        "skin_shape_key_count": max(0, len(skin.data.shape_keys.key_blocks) - 1) if skin.data.shape_keys else 0,
        "head_region_vertices": len(head_ids),
        "head_region_weight_max": float(region_weight.max()),
        "head_region_centre": region_centre.tolist(),
        "head_region_radii": region_radii.tolist(),
        "eye_objects": [obj.name for obj in eyes_objects],
        "eye_centres": [point.tolist() for point in eyes],
        "skin_detection": skin_report,
        "camera": camera_report,
        "renders": renders,
        "identity_lock": False,
        "visual_identity_lock": False,
        "vrm_exported": False,
    }
    (qa / "AINA_RAIN_IDENTITY_SOURCE_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def main_fit(args) -> None:
    if not args.landmarks or not args.landmarks.exists():
        raise RuntimeError("--landmarks is required for fit stage")
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)
    data = json.loads(args.landmarks.read_text(encoding="utf-8"))
    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH" and len(obj.data.vertices)]
    reset_shape_keys(meshes)
    armature = find_armature(scene)
    head_bone = find_head_bone(armature)
    head_point = bone_world_point(armature, head_bone)
    skin, skin_report = identify_skin(scene, head_point)
    character_height = skin_report["character_height_m"]
    eyes_objects = eye_objects(scene, head_point, character_height)
    eyes_before = eye_centres(eyes_objects)
    head_ids, _, _, _ = head_region(skin, head_point, eyes_before, character_height)
    cameras = {
        view: bpy.data.objects.get(f"AINA_Rain_Camera_{view}")
        for view in ("front", "three_quarter", "side", "left_45", "right_45")
    }
    if not all(cameras.values()):
        cameras, camera_report = setup_cameras(
            scene, skin, head_ids, eyes_before, head_point, character_height
        )
    else:
        eye_average = np.mean(eyes_before, axis=0) if eyes_before else head_point
        skin_points = world_vertices(skin)[head_ids]
        centre = skin_points.mean(axis=0)
        camera_report = {
            "forward_sign_y": -1.0 if eye_average[1] < centre[1] else 1.0,
            "reused_source_cameras": True,
        }
    forward_sign = float(camera_report["forward_sign_y"])
    fitting = fit_identity(scene, skin, head_ids, cameras, data, forward_sign)
    related = move_related_objects(
        scene,
        skin,
        fitting.pop("initial_world"),
        fitting.pop("final_world"),
        eyes_before,
    )
    bpy.context.view_layer.update()
    eyes_after = eye_centres(eye_objects(scene, head_point, character_height))
    renders = render_final_suite(scene, cameras, skin, args.out)

    blend_path = args.out / "AINA_RAIN_IDENTITY_MASTER_CANDIDATE.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_RAIN_IDENTITY_MASTER_CANDIDATE.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    report = {
        "product": "AINA Rain Identity Master Candidate",
        "source_character": "Blender Studio Rain v3",
        "source_license": "CC BY 4.0",
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "topology_changed": False,
        "armature_preserved": True,
        "skin_weights_preserved": True,
        "uvs_preserved": True,
        "skin_object": skin.name,
        "vertices": len(skin.data.vertices),
        "triangles": sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons),
        "head_region_vertices": len(head_ids),
        "eye_centres_before": [point.tolist() for point in eyes_before],
        "eye_centres_after": [point.tolist() for point in eyes_after],
        "fitting": fitting,
        "related_anatomy": related,
        "renders": renders,
        "identity_lock": False,
        "visual_identity_lock": False,
        "candidate": True,
        "vrm_exported": False,
        "next_gate": "Directly inspect approved-vs-real front, 3Q and profile. Continue only on this same Rain topology if the real model is materially closer to approved AINA; otherwise stop base swapping and commission a dedicated AINA topology.",
        "files": {"blend": str(blend_path), "glb": str(glb_path)},
    }
    (qa / "AINA_RAIN_IDENTITY_MASTER_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()
    if args.stage == "source":
        main_source(args)
    else:
        main_fit(args)


if __name__ == "__main__":
    main()
