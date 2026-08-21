#!/usr/bin/env python3
"""Probe a CC0 Vitruvian/CharMorph head as AINA's next identity base.

Loads the real GLB with its FACS shape keys, renders neutral clay from multiple
views, records topology and expression inventory, and saves an editable Blender
file. This stage creates no replacement effect art and performs no VRM export.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def world_vertices(obj):
    matrix = obj.matrix_world
    return np.array([(matrix @ vertex.co)[:] for vertex in obj.data.vertices], dtype=np.float64)


def bounds(objects):
    points = np.concatenate([world_vertices(obj) for obj in objects], axis=0)
    return points.min(axis=0), points.max(axis=0)


def material(name, color, roughness=0.48, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = tuple(color)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
    if shader:
        shader.inputs["Base Color"].default_value = tuple(color)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metallic
    return mat


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
        polygon.use_smooth = True


def shape_key_names(obj):
    if not obj.data.shape_keys:
        return []
    return [key.name for key in obj.data.shape_keys.key_blocks]


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


def render(scene, camera, output, location, target, lens=82):
    camera.location = location
    camera.data.lens = lens
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    preview = args.out / "Preview"
    qa = args.out / "QA"
    preview.mkdir(exist_ok=True)
    qa.mkdir(exist_ok=True)

    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(args.head))
    bpy.context.view_layer.update()

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("Vitruvian GLB imported no mesh objects")

    inventory = []
    for obj in meshes:
        keys = shape_key_names(obj)
        inventory.append(
            {
                "name": obj.name,
                "vertices": len(obj.data.vertices),
                "triangles": len(obj.data.loop_triangles),
                "shape_key_count": max(0, len(keys) - (1 if keys and keys[0] == "Basis" else 0)),
                "shape_keys": keys,
                "materials": [mat.name for mat in obj.data.materials if mat],
            }
        )
        if obj.data.shape_keys:
            for key in obj.data.shape_keys.key_blocks:
                key.value = 0.0

    def is_eye(obj):
        name = obj.name.lower()
        return "eye" in name or "iris" in name or "pupil" in name or "sclera" in name

    def is_mouth(obj):
        name = obj.name.lower()
        return "mouth" in name or "teeth" in name or "tongue" in name or "gum" in name

    expression_candidates = [obj for obj in meshes if len(shape_key_names(obj)) > 8]
    skin = max(expression_candidates or meshes, key=lambda obj: len(obj.data.vertices))
    eye_objects = [obj for obj in meshes if is_eye(obj)]
    mouth_objects = [obj for obj in meshes if is_mouth(obj)]

    clay = material("AINA_Probe_Clay", (0.62, 0.65, 0.70, 1.0), 0.52)
    eye_white = material("AINA_Probe_EyeWhite", (0.92, 0.95, 0.98, 1.0), 0.26)
    mouth_dark = material("AINA_Probe_Mouth", (0.18, 0.055, 0.07, 1.0), 0.44)
    for obj in meshes:
        if obj in eye_objects:
            assign(obj, eye_white)
        elif obj in mouth_objects:
            assign(obj, mouth_dark)
        else:
            assign(obj, clay)

    lo, hi = bounds(meshes)
    center = (lo + hi) * 0.5
    size = hi - lo
    head_height = max(float(size[2]), 0.1)

    eye_centers = []
    for obj in eye_objects:
        verts = world_vertices(obj)
        if len(verts):
            eye_centers.append(verts.mean(axis=0))
    if eye_centers:
        eye_center = np.mean(eye_centers, axis=0)
        forward_sign = -1.0 if eye_center[1] < center[1] else 1.0
        target = np.array([eye_center[0], center[1], eye_center[2] - 0.018])
    else:
        forward_sign = -1.0
        target = center.copy()
        target[2] += 0.08 * head_height

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.world.color = (0.94, 0.95, 0.98)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = 0.10
    except Exception:
        pass

    for obj in list(scene.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    distance = max(head_height * 2.65, float(size[0]) * 2.8, 0.85)
    front_location = np.array([target[0], center[1] + forward_sign * distance, target[2]])
    create_light("Key", tuple(front_location + np.array([0.65 * size[0], 0.0, 0.55 * head_height])), 550, 2.4, target)
    create_light("Fill", tuple(front_location + np.array([-0.75 * size[0], 0.2 * distance, 0.15 * head_height])), 280, 2.7, target)
    create_light("Rim", tuple(center + np.array([0.0, -forward_sign * distance * 0.65, 0.55 * head_height])), 360, 2.2, target)

    camera_data = bpy.data.cameras.new("AINA_Probe_Camera")
    camera = bpy.data.objects.new("AINA_Probe_Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    front = preview / "AINA_VITRUVIAN_NEUTRAL_FRONT.png"
    q3_left = preview / "AINA_VITRUVIAN_NEUTRAL_3Q_LEFT.png"
    q3_right = preview / "AINA_VITRUVIAN_NEUTRAL_3Q_RIGHT.png"
    side = preview / "AINA_VITRUVIAN_NEUTRAL_SIDE.png"
    back = preview / "AINA_VITRUVIAN_OPPOSITE_DIRECTION.png"

    render(scene, camera, front, front_location, target)
    render(
        scene,
        camera,
        q3_left,
        tuple(front_location + np.array([0.42 * distance, -forward_sign * 0.10 * distance, 0.0])),
        target,
    )
    render(
        scene,
        camera,
        q3_right,
        tuple(front_location + np.array([-0.42 * distance, -forward_sign * 0.10 * distance, 0.0])),
        target,
    )
    render(scene, camera, side, tuple(center + np.array([distance, 0.0, target[2] - center[2]])), target)
    render(scene, camera, back, tuple(center + np.array([0.0, -forward_sign * distance, target[2] - center[2]])), target)

    blend_path = args.out / "AINA_VITRUVIAN_HEAD_PROBE.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    report = {
        "product": "AINA Vitruvian Identity Base Probe",
        "source": str(args.head),
        "source_license": "Vitruvian/Antonia character CC0; repository code MIT",
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "vrm_exported": False,
        "mesh_object_count": len(meshes),
        "skin_object": skin.name,
        "skin_vertices": len(skin.data.vertices),
        "skin_shape_key_count": max(0, len(shape_key_names(skin)) - 1),
        "eye_objects": [obj.name for obj in eye_objects],
        "mouth_objects": [obj.name for obj in mouth_objects],
        "forward_sign_y": forward_sign,
        "bounds_min": lo.tolist(),
        "bounds_max": hi.tolist(),
        "inventory": inventory,
        "identity_lock": False,
        "candidate": True,
        "next_gate": "Inspect neutral front/3Q/side. If anatomically suitable, fit this existing FACS topology to the approved AINA identity.",
        "files": {
            "blend": str(blend_path),
            "previews": [str(front), str(q3_left), str(q3_right), str(side), str(back)],
        },
    }
    (qa / "AINA_VITRUVIAN_HEAD_PROBE_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
