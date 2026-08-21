#!/usr/bin/env python3
"""Probe the official AliciaSolid VRM as a more suitable AINA identity base.

This stage does not generate replacement effect art and does not export a new
VRM. It imports the official VRM Consortium sample as real geometry, inventories
its embedded VRM licence metadata, humanoid rig, materials and morph targets,
then renders actual original-material and clay head views for suitability review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def parse_glb_json(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise RuntimeError(f"Not a binary glTF/VRM file: {path}")
    version, declared_length = struct.unpack_from("<II", data, 4)
    if declared_length != len(data):
        raise RuntimeError(f"GLB length mismatch: {declared_length} != {len(data)}")
    offset = 12
    json_chunk = None
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            json_chunk = chunk
            break
    if json_chunk is None:
        raise RuntimeError("GLB contains no JSON chunk")
    return json.loads(json_chunk.rstrip(b"\x00 \t\r\n").decode("utf-8"))


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def local_array(obj) -> np.ndarray:
    values = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", values)
    return values.reshape(-1, 3)


def world_array(obj) -> np.ndarray:
    local = local_array(obj)
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    homogeneous = np.c_[local, np.ones(len(local))]
    return (homogeneous @ matrix.T)[:, :3]


def scene_bounds(meshes):
    points = np.concatenate([world_array(obj) for obj in meshes if len(obj.data.vertices)], axis=0)
    return points.min(axis=0), points.max(axis=0), points


def shape_key_names(obj):
    if not obj.data.shape_keys:
        return []
    return [key.name for key in obj.data.shape_keys.key_blocks]


def vrm_meta(document: dict) -> dict:
    extensions = document.get("extensions", {})
    if "VRM" in extensions:
        vrm = extensions["VRM"]
        meta = vrm.get("meta", {})
        humanoid = vrm.get("humanoid", {})
        bones = {item.get("bone"): item.get("node") for item in humanoid.get("humanBones", [])}
        return {"spec": "VRM-0.x", "meta": meta, "human_bones": bones}
    if "VRMC_vrm" in extensions:
        vrm = extensions["VRMC_vrm"]
        meta = vrm.get("meta", {})
        bones = {
            name: item.get("node")
            for name, item in vrm.get("humanoid", {}).get("humanBones", {}).items()
        }
        return {"spec": "VRM-1.0", "meta": meta, "human_bones": bones}
    return {"spec": "none", "meta": {}, "human_bones": {}}


def find_head_target(document: dict, metadata: dict, armatures, lo, hi):
    node_index = metadata.get("human_bones", {}).get("head")
    node_name = None
    if isinstance(node_index, int):
        nodes = document.get("nodes", [])
        if 0 <= node_index < len(nodes):
            node_name = nodes[node_index].get("name")
    for armature in armatures:
        candidates = [node_name, "head", "Head", "J_Bip_C_Head"]
        for name in candidates:
            if not name:
                continue
            bone = armature.data.bones.get(name)
            if bone:
                world = armature.matrix_world @ bone.head_local
                return np.array(world[:], dtype=np.float64), node_name, armature.name
    center = (lo + hi) * 0.5
    height = float(hi[2] - lo[2])
    return np.array([center[0], center[1], lo[2] + 0.83 * height]), node_name, None


def eye_objects(meshes):
    result = []
    for obj in meshes:
        text = " ".join(
            [obj.name] + [material.name for material in obj.data.materials if material]
        ).lower()
        if any(token in text for token in ("eye", "iris", "pupil", "sclera", "hitomi")):
            result.append(obj)
    return result


def material(name, color, roughness=0.50, metallic=0.0):
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


def render_view(scene, camera, location, target, output):
    camera.location = tuple(location)
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    preview = args.out / "Preview"
    qa = args.out / "QA"
    preview.mkdir(exist_ok=True)
    qa.mkdir(exist_ok=True)

    document = parse_glb_json(args.model)
    metadata = vrm_meta(document)
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()

    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(args.model))
    bpy.context.view_layer.update()
    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    if not meshes:
        raise RuntimeError("Official Alicia model imported no mesh objects")

    for obj in meshes:
        if obj.data.shape_keys:
            for key in obj.data.shape_keys.key_blocks:
                key.value = 0.0

    lo, hi, all_points = scene_bounds(meshes)
    center = (lo + hi) * 0.5
    size = hi - lo
    head_target, head_node_name, head_armature = find_head_target(
        document, metadata, armatures, lo, hi
    )
    high_mask = all_points[:, 2] > head_target[2] - max(0.13, 0.095 * float(size[2]))
    head_points = all_points[high_mask] if np.any(high_mask) else all_points
    head_lo, head_hi = head_points.min(axis=0), head_points.max(axis=0)
    head_size = head_hi - head_lo
    target = np.array([head_target[0], head_target[1], head_target[2] + 0.025 * float(size[2])])

    eyes = eye_objects(meshes)
    if eyes:
        eye_center = np.mean([world_array(obj).mean(axis=0) for obj in eyes], axis=0)
        forward_sign = -1.0 if eye_center[1] < center[1] else 1.0
        target[0] = eye_center[0]
        target[2] = eye_center[2] - 0.020
    else:
        forward_sign = -1.0

    distance = max(float(head_size[2]) * 2.55, float(head_size[0]) * 2.80, 0.70)
    front = np.array([target[0], center[1] + forward_sign * distance, target[2]])
    locations = {
        "FRONT": front,
        "THREE_QUARTER": front + np.array([0.43 * distance, -forward_sign * 0.10 * distance, 0.0]),
        "SIDE": center + np.array([distance, 0.0, target[2] - center[2]]),
        "OPPOSITE": np.array([target[0], center[1] - forward_sign * distance, target[2]]),
    }

    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.world.color = (0.035, 0.040, 0.055)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = -0.15
    except Exception:
        pass

    for obj in list(scene.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    create_light("AINA_Alicia_Key", tuple(front + np.array([0.55 * head_size[0], 0.0, 0.45 * head_size[2]])), 520, 2.1, target)
    create_light("AINA_Alicia_Fill", tuple(front + np.array([-0.72 * head_size[0], 0.18 * distance, 0.08 * head_size[2]])), 250, 2.6, target)
    create_light("AINA_Alicia_Rim", tuple(center + np.array([0.0, -forward_sign * distance * 0.72, 0.42 * head_size[2]])), 330, 1.8, target)
    camera_data = bpy.data.cameras.new("AINA_Alicia_Probe_Camera")
    camera = bpy.data.objects.new("AINA_Alicia_Probe_Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.data.lens = 88
    scene.camera = camera

    original_materials = {obj.name: list(obj.data.materials) for obj in meshes}
    renders = {}
    for name, location in locations.items():
        path = preview / f"AINA_ALICIA_ORIGINAL_{name}.png"
        render_view(scene, camera, location, target, path)
        renders[f"original_{name.lower()}"] = str(path)

    clay = material("AINA_Alicia_Clay", (0.29, 0.32, 0.38, 1.0), 0.58)
    eye_mat = material("AINA_Alicia_Clay_Eye", (0.035, 0.090, 0.12, 1.0), 0.30)
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(eye_mat if obj in eyes else clay)
        for polygon in obj.data.polygons:
            polygon.material_index = 0
            polygon.use_smooth = True
    for name in ("FRONT", "THREE_QUARTER", "SIDE"):
        path = preview / f"AINA_ALICIA_CLAY_{name}.png"
        render_view(scene, camera, locations[name], target, path)
        renders[f"clay_{name.lower()}"] = str(path)

    for obj in meshes:
        obj.data.materials.clear()
        for mat in original_materials[obj.name]:
            obj.data.materials.append(mat)

    blend_path = args.out / "AINA_ALICIA_OFFICIAL_BASE_PROBE.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    inventory = []
    all_shape_keys = []
    for obj in meshes:
        keys = shape_key_names(obj)
        all_shape_keys.extend(keys[1:] if keys and keys[0] == "Basis" else keys)
        inventory.append({
            "name": obj.name,
            "vertices": len(obj.data.vertices),
            "triangles": len(obj.data.loop_triangles),
            "shape_keys": keys,
            "materials": [mat.name for mat in obj.data.materials if mat],
        })

    report = {
        "product": "AINA Official Alicia Base Suitability Probe",
        "source_repository": "vrm-c/UniVRMTest",
        "source_path": "Models/Alicia_vrm-0.40/AliciaSolid_vrm-0.40.vrm",
        "source_sha256": digest,
        "source_repository_license": "MIT",
        "embedded_vrm_spec": metadata["spec"],
        "embedded_vrm_meta": metadata["meta"],
        "embedded_human_bones": metadata["human_bones"],
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "new_vrm_exported": False,
        "mesh_object_count": len(meshes),
        "armature_count": len(armatures),
        "total_vertices": int(sum(len(obj.data.vertices) for obj in meshes)),
        "total_triangles": int(sum(len(obj.data.loop_triangles) for obj in meshes)),
        "shape_key_count": len(all_shape_keys),
        "shape_keys": sorted(set(all_shape_keys)),
        "head_node_name": head_node_name,
        "head_armature": head_armature,
        "eye_objects": [obj.name for obj in eyes],
        "bounds_min": lo.tolist(),
        "bounds_max": hi.tolist(),
        "forward_sign_y": forward_sign,
        "inventory": inventory,
        "identity_lock": False,
        "visual_identity_lock": False,
        "candidate": False,
        "stage": "base_suitability_probe",
        "next_gate": "Review actual original-material and clay head views. Use this topology for AINA only if its anatomy, licence metadata, rig and morph inventory are suitable.",
        "files": {"blend": str(blend_path), "renders": renders},
    }
    (qa / "AINA_ALICIA_OFFICIAL_BASE_PROBE_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
