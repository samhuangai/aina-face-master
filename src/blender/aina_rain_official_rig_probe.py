#!/usr/bin/env python3
"""Probe Blender Studio's CC-BY Rain rig as AINA's next identity base.

The script opens the real Rain .blend, reveals the production geometry, resets
shape keys and the rig to a neutral rest state, inventories topology/face rig,
and renders actual beauty plus hair-hidden clay head views. It does not generate
replacement effect art, alter the mesh, or export VRM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def reveal_scene(scene) -> None:
    for collection in bpy.data.collections:
        collection.hide_viewport = False
        collection.hide_render = False
    for layer_collection in scene.view_layers[0].layer_collection.children:
        layer_collection.exclude = False
        layer_collection.hide_viewport = False
    for obj in scene.objects:
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.hide_render = False
        if obj.type == "ARMATURE":
            obj.data.pose_position = "REST"
        if obj.type == "MESH" and obj.data.shape_keys:
            for key in obj.data.shape_keys.key_blocks:
                key.value = 0.0
    scene.frame_set(1)
    bpy.context.view_layer.update()


def local_array(obj) -> np.ndarray:
    values = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", values)
    return values.reshape(-1, 3)


def world_array(obj) -> np.ndarray:
    local = local_array(obj)
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    homogeneous = np.c_[local, np.ones(len(local))]
    return (homogeneous @ matrix.T)[:, :3]


def object_text(obj) -> str:
    return " ".join(
        [obj.name]
        + [material.name for material in getattr(obj.data, "materials", []) if material]
    ).lower()


def is_control_geometry(obj) -> bool:
    text = object_text(obj)
    return any(token in text for token in (
        "widget", "wgts", "wgt-", "control", "picker", "ui_", "metarig",
    ))


def production_meshes(scene):
    meshes = []
    for obj in scene.objects:
        if obj.type != "MESH" or not len(obj.data.vertices):
            continue
        if is_control_geometry(obj):
            obj.hide_render = True
            continue
        meshes.append(obj)
    return meshes


def find_head_bone(armatures):
    preferred = ("head", "def-head", "org-head", "head.x", "head_fk")
    candidates = []
    for armature in armatures:
        for bone in armature.data.bones:
            name = bone.name.lower()
            score = 99
            for index, token in enumerate(preferred):
                if name == token:
                    score = index
                    break
                if token in name:
                    score = 10 + index
                    break
            if score < 99:
                world = armature.matrix_world @ bone.head_local
                candidates.append((score, -float(bone.length), armature, bone, np.array(world[:], dtype=np.float64)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0]


def bounds(meshes):
    arrays = [world_array(obj) for obj in meshes if len(obj.data.vertices)]
    points = np.concatenate(arrays, axis=0)
    return points.min(axis=0), points.max(axis=0), points


def make_material(name, color, roughness=0.52, metallic=0.0):
    material = bpy.data.materials.new(name)
    material.diffuse_color = tuple(color)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF") if material.node_tree else None
    if shader:
        shader.inputs["Base Color"].default_value = tuple(color)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metallic
    return material


def assign_single_material(obj, material):
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
        polygon.use_smooth = True


def create_light(name, location, energy, size, target):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = tuple(location)
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
    return obj


def remove_render_helpers(scene):
    for obj in list(scene.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def render_view(scene, camera, location, target, output):
    camera.location = tuple(location)
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def setup_render(scene, center, target, size, distance):
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 800
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.world.color = (0.035, 0.040, 0.055)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = -0.15
    except Exception:
        pass
    remove_render_helpers(scene)
    key_location = center + np.array([0.65 * size[0], distance * 0.80, 0.55 * size[2]])
    fill_location = center + np.array([-0.80 * size[0], distance * 0.95, 0.15 * size[2]])
    rim_location = center + np.array([0.0, -distance * 0.75, 0.55 * size[2]])
    create_light("AINA_Rain_Key", key_location, 620, 2.4, target)
    create_light("AINA_Rain_Fill", fill_location, 270, 2.8, target)
    create_light("AINA_Rain_Rim", rim_location, 410, 2.1, target)
    camera_data = bpy.data.cameras.new("AINA_Rain_Probe_Camera")
    camera = bpy.data.objects.new("AINA_Rain_Probe_Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.data.lens = 92
    scene.camera = camera
    return camera


def head_target_and_scale(meshes, armatures):
    lo, hi, points = bounds(meshes)
    center = (lo + hi) * 0.5
    body_height = float(hi[2] - lo[2])
    head_bone = find_head_bone(armatures)
    if head_bone:
        _, _, armature, bone, target = head_bone
        bone_info = {"armature": armature.name, "bone": bone.name}
    else:
        target = np.array([center[0], center[1], lo[2] + 0.84 * body_height])
        bone_info = {"armature": None, "bone": None}
    high = points[:, 2] > target[2] - max(0.14, 0.10 * body_height)
    near = np.linalg.norm(points[:, :2] - target[:2], axis=1) < max(0.30, 0.22 * body_height)
    head_points = points[high & near]
    if len(head_points) < 100:
        head_points = points[points[:, 2] > lo[2] + 0.70 * body_height]
    head_lo, head_hi = head_points.min(axis=0), head_points.max(axis=0)
    head_center = (head_lo + head_hi) * 0.5
    head_size = head_hi - head_lo
    target = np.array([head_center[0], head_center[1], target[2] + 0.015 * body_height])
    distance = max(float(head_size[2]) * 2.45, float(head_size[0]) * 2.75, 0.70)
    return lo, hi, center, target, head_lo, head_hi, head_size, distance, bone_info


def view_locations(center, target, distance):
    # Render both possible forward directions. The real model orientation is then
    # selected from actual images rather than assumed from naming conventions.
    return {
        "Y_POS": np.array([target[0], center[1] + distance, target[2]]),
        "Y_NEG": np.array([target[0], center[1] - distance, target[2]]),
        "X_POS": np.array([center[0] + distance, target[1], target[2]]),
        "X_NEG": np.array([center[0] - distance, target[1], target[2]]),
        "Y_POS_X_POS_3Q": np.array([target[0] + 0.43 * distance, center[1] + 0.90 * distance, target[2]]),
        "Y_POS_X_NEG_3Q": np.array([target[0] - 0.43 * distance, center[1] + 0.90 * distance, target[2]]),
        "Y_NEG_X_POS_3Q": np.array([target[0] + 0.43 * distance, center[1] - 0.90 * distance, target[2]]),
        "Y_NEG_X_NEG_3Q": np.array([target[0] - 0.43 * distance, center[1] - 0.90 * distance, target[2]]),
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    preview = args.out / "Preview"
    qa = args.out / "QA"
    preview.mkdir(exist_ok=True)
    qa.mkdir(exist_ok=True)

    scene = bpy.context.scene
    reveal_scene(scene)
    meshes = production_meshes(scene)
    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    if not meshes:
        raise RuntimeError("Rain source contains no production mesh objects")

    lo, hi, center, target, head_lo, head_hi, head_size, distance, bone_info = head_target_and_scale(meshes, armatures)
    locations = view_locations(center, target, distance)
    camera = setup_render(scene, center, target, head_size, distance)

    beauty = {}
    for name, location in locations.items():
        path = preview / f"AINA_RAIN_BEAUTY_{name}.png"
        render_view(scene, camera, location, target, path)
        beauty[name] = str(path)

    original_materials = {obj.name: list(obj.data.materials) for obj in meshes}
    original_visibility = {obj.name: obj.hide_render for obj in meshes}
    clay = make_material("AINA_Rain_Clay", (0.27, 0.30, 0.36, 1.0), 0.58)
    eye_material = make_material("AINA_Rain_Clay_Eye", (0.030, 0.080, 0.11, 1.0), 0.28)
    hidden_hair = []
    for obj in meshes:
        text = object_text(obj)
        if "hair" in text or "ponytail" in text or "bang" in text:
            obj.hide_render = True
            hidden_hair.append(obj.name)
            continue
        assign_single_material(obj, eye_material if any(token in text for token in ("eye", "iris", "pupil")) else clay)

    clay_outputs = {}
    for name, location in locations.items():
        path = preview / f"AINA_RAIN_CLAY_{name}.png"
        render_view(scene, camera, location, target, path)
        clay_outputs[name] = str(path)

    for obj in meshes:
        obj.hide_render = original_visibility[obj.name]
        obj.data.materials.clear()
        for material in original_materials[obj.name]:
            obj.data.materials.append(material)

    blend_path = args.out / "AINA_RAIN_OFFICIAL_RIG_PROBE.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    inventory = []
    shape_keys = []
    total_vertices = 0
    total_triangles = 0
    for obj in meshes:
        keys = []
        if obj.data.shape_keys:
            keys = [key.name for key in obj.data.shape_keys.key_blocks]
            shape_keys.extend(keys[1:] if keys and keys[0] == "Basis" else keys)
        obj.data.calc_loop_triangles()
        total_vertices += len(obj.data.vertices)
        total_triangles += len(obj.data.loop_triangles)
        inventory.append({
            "name": obj.name,
            "vertices": len(obj.data.vertices),
            "triangles": len(obj.data.loop_triangles),
            "shape_keys": keys,
            "materials": [material.name for material in obj.data.materials if material],
            "modifiers": [modifier.type for modifier in obj.modifiers],
        })

    armature_inventory = []
    for armature in armatures:
        armature_inventory.append({
            "name": armature.name,
            "bone_count": len(armature.data.bones),
            "bones": [bone.name for bone in armature.data.bones],
        })

    attribution = args.attribution.read_text(encoding="utf-8", errors="replace")
    source_digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    report = {
        "product": "AINA Rain Official Rig Suitability Probe",
        "source": str(args.source),
        "source_sha256": source_digest,
        "source_origin": "Blender Studio Rain v3 / mirror copy used for reproducible CI download",
        "source_attribution": attribution,
        "source_license": "CC BY 4.0",
        "required_credit": "Rain Rig (CC) Blender Foundation | studio.blender.org",
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "mesh_modified": False,
        "new_vrm_exported": False,
        "mesh_object_count": len(meshes),
        "armature_count": len(armatures),
        "total_vertices": total_vertices,
        "total_triangles": total_triangles,
        "shape_key_count": len(shape_keys),
        "unique_shape_keys": sorted(set(shape_keys)),
        "head_reference": bone_info,
        "bounds_min": lo.tolist(),
        "bounds_max": hi.tolist(),
        "head_bounds_min": head_lo.tolist(),
        "head_bounds_max": head_hi.tolist(),
        "hidden_hair_for_clay": hidden_hair,
        "inventory": inventory,
        "armatures": armature_inventory,
        "identity_lock": False,
        "visual_identity_lock": False,
        "candidate": False,
        "stage": "base_suitability_probe",
        "next_gate": "Inspect real beauty and hair-hidden clay views. If the stylized facial topology is materially closer to approved AINA, begin topology-preserving neutral identity fit before any VRM work.",
        "files": {
            "blend": str(blend_path),
            "beauty_renders": beauty,
            "clay_renders": clay_outputs,
        },
    }
    (qa / "AINA_RAIN_OFFICIAL_RIG_PROBE_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
