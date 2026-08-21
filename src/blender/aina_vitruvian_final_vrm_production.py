#!/usr/bin/env python3
"""Assemble the real AINA Identity Master into a full VRM 1.0 character.

Input: the visually converged editable Vitruvian/CharMorph AINA head.  The
script imports the matching CC0 rigged body, preserves the real FACS keys,
builds 52 non-zero ARKit controls, creates real silver updo geometry and hair
spring bones, restores production materials, exports AINA_MASTER.blend and
AINA.vrm, then performs a clean Blender glTF reimport and VRM-binary audit.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

import aina_vitruvian_final_visual_lock as lock
import aina_vitruvian_arkit52 as arkit
from aina_vitruvian_vrm_patch import ARKIT_52, inspect_vrm, patch_vrm


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-glb", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--visual-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def normalized(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def local_vertices(obj) -> np.ndarray:
    return np.asarray([vertex.co[:] for vertex in obj.data.vertices], dtype=np.float64)


def world_vertices(obj) -> np.ndarray:
    return lock.world_vertices(obj, local_vertices(obj))


def object_center(obj) -> np.ndarray:
    points = world_vertices(obj)
    return points.mean(axis=0) if len(points) else np.asarray(obj.matrix_world.translation[:], dtype=np.float64)


def preserve_bone_parent(obj, armature, bone_name: str) -> None:
    world = obj.matrix_world.copy()
    obj.parent = armature
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_world = world


def choose_armature(objects):
    armatures = [obj for obj in objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("The Vitruvian body contains no armature")
    return max(armatures, key=lambda obj: len(obj.data.bones))


def find_bone(armature, aliases):
    alias_values = [normalized(alias) for alias in aliases]
    bones = list(armature.data.bones)
    for alias in alias_values:
        for bone in bones:
            if normalized(bone.name) == alias:
                return bone
    for alias in alias_values:
        for bone in bones:
            value = normalized(bone.name)
            if value.endswith(alias) or alias in value:
                return bone
    return None


def add_edit_bone(armature, name: str, head_world, tail_world, parent_name: str | None):
    inverse = armature.matrix_world.inverted()
    bone = armature.data.edit_bones.get(name)
    if bone is None:
        bone = armature.data.edit_bones.new(name)
    bone.head = inverse @ Vector(head_world)
    bone.tail = inverse @ Vector(tail_world)
    if (bone.tail - bone.head).length < 0.004:
        bone.tail.z += 0.018
    if parent_name and armature.data.edit_bones.get(parent_name):
        bone.parent = armature.data.edit_bones[parent_name]
    return bone


def ensure_eye_and_hair_bones(armature, meshes, skin, setup):
    head_bone = find_bone(armature, ["Head", "mixamorig:Head"])
    if head_bone is None:
        raise RuntimeError("Required head bone was not found")
    eye_objects = [obj for obj in meshes if obj != skin and lock.is_eye_name(obj.name)]
    centers = [object_center(obj) for obj in eye_objects]
    if len(centers) >= 2:
        unique = sorted(centers, key=lambda value: value[0])
        right_center = np.mean(unique[: max(1, len(unique) // 2)], axis=0)
        left_center = np.mean(unique[max(1, len(unique) // 2) :], axis=0)
    else:
        center = np.asarray(setup["target"], dtype=np.float64)
        width = float(setup["size"][0])
        right_center = center + np.asarray([-0.18 * width, setup["forward_sign"] * 0.025, 0.018])
        left_center = center + np.asarray([0.18 * width, setup["forward_sign"] * 0.025, 0.018])
    outward = np.asarray(setup["locations"]["front"] - setup["target"], dtype=np.float64)
    outward /= max(float(np.linalg.norm(outward)), 1e-9)

    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    add_edit_bone(armature, "rightEye", right_center, right_center + outward * 0.025, head_bone.name)
    add_edit_bone(armature, "leftEye", left_center, left_center + outward * 0.025, head_bone.name)

    size = setup["size"]
    center = setup["center"]
    top = float(setup["hi"][2])
    front_sign = float(setup["forward_sign"])
    chains = {
        "Hair_L": [
            np.asarray([center[0] + 0.30 * size[0], center[1] + front_sign * 0.05 * size[1], top - 0.20 * size[2]]),
            np.asarray([center[0] + 0.34 * size[0], center[1] + front_sign * 0.08 * size[1], top - 0.40 * size[2]]),
            np.asarray([center[0] + 0.30 * size[0], center[1] + front_sign * 0.09 * size[1], top - 0.62 * size[2]]),
        ],
        "Hair_R": [
            np.asarray([center[0] - 0.30 * size[0], center[1] + front_sign * 0.05 * size[1], top - 0.20 * size[2]]),
            np.asarray([center[0] - 0.34 * size[0], center[1] + front_sign * 0.08 * size[1], top - 0.40 * size[2]]),
            np.asarray([center[0] - 0.30 * size[0], center[1] + front_sign * 0.09 * size[1], top - 0.62 * size[2]]),
        ],
        "Hair_Back": [
            np.asarray([center[0], center[1] - front_sign * 0.26 * size[1], top - 0.12 * size[2]]),
            np.asarray([center[0], center[1] - front_sign * 0.34 * size[1], top - 0.33 * size[2]]),
            np.asarray([center[0], center[1] - front_sign * 0.30 * size[1], top - 0.52 * size[2]]),
        ],
    }
    hair_names = []
    for chain_name, points in chains.items():
        previous = head_bone.name
        chain_bones = []
        for index in range(2):
            name = f"AINA_{chain_name}_{index + 1}"
            add_edit_bone(armature, name, points[index], points[index + 1], previous)
            previous = name
            chain_bones.append(name)
        hair_names.append(chain_bones)
    bpy.ops.object.mode_set(mode="OBJECT")
    return {
        "head_bone": head_bone.name,
        "eye_bones": {"left": "leftEye", "right": "rightEye"},
        "eye_centers": {"left": left_center, "right": right_center},
        "hair_chains": hair_names,
    }


def material(name, color, roughness=0.45, metallic=0.0, transmission=0.0, alpha=1.0):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = tuple(color[:3]) + (alpha,)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
    if shader:
        if shader.inputs.get("Base Color"):
            shader.inputs["Base Color"].default_value = tuple(color[:3]) + (alpha,)
        if shader.inputs.get("Roughness"):
            shader.inputs["Roughness"].default_value = roughness
        if shader.inputs.get("Metallic"):
            shader.inputs["Metallic"].default_value = metallic
        if shader.inputs.get("Transmission Weight"):
            shader.inputs["Transmission Weight"].default_value = transmission
        if shader.inputs.get("Alpha"):
            shader.inputs["Alpha"].default_value = alpha
        if name == "AINA_Skin_Final" and shader.inputs.get("Subsurface Weight"):
            shader.inputs["Subsurface Weight"].default_value = 0.065
            if shader.inputs.get("Subsurface Radius"):
                shader.inputs["Subsurface Radius"].default_value = (1.0, 0.45, 0.28)
    if alpha < 1.0:
        mat.surface_render_method = "DITHERED"
    return mat


def assign_single(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def restore_materials(meshes, skin, body_objects):
    mats = {
        "skin": material("AINA_Skin_Final", (0.70, 0.48, 0.44, 1.0), 0.43),
        "lip": material("AINA_Lips_Final", (0.42, 0.10, 0.14, 1.0), 0.34),
        "sclera": material("AINA_Sclera_Final", (0.78, 0.82, 0.88, 1.0), 0.22),
        "iris": material("AINA_Iris_IceBlue", (0.16, 0.34, 0.54, 1.0), 0.25),
        "pupil": material("AINA_Pupil_Final", (0.004, 0.006, 0.010, 1.0), 0.18),
        "cornea": material("AINA_Cornea_Final", (0.90, 0.96, 1.0, 1.0), 0.06, transmission=0.20, alpha=0.32),
        "lash": material("AINA_Lash_Brow_Final", (0.012, 0.010, 0.018, 1.0), 0.28),
        "mouth": material("AINA_Mouth_Final", (0.20, 0.025, 0.035, 1.0), 0.42),
        "teeth": material("AINA_Teeth_Final", (0.88, 0.83, 0.72, 1.0), 0.30),
        "hair": material("AINA_Hair_Silver_Final", (0.52, 0.61, 0.76, 1.0), 0.32, metallic=0.05),
        "suit": material("AINA_Suit_Pearl_Final", (0.62, 0.70, 0.82, 1.0), 0.38, metallic=0.04),
        "accent": material("AINA_Suit_Accent_Final", (0.22, 0.38, 0.68, 1.0), 0.31, metallic=0.12),
    }

    # Preserve material slots and replace them semantically so lip/mouth regions
    # remain separate when the source GLB provides those assignments.
    for index, slot in enumerate(skin.material_slots):
        value = normalized(slot.name if slot else "")
        if "lip" in value:
            skin.data.materials[index] = mats["lip"]
        elif "mouth" in value or "gum" in value:
            skin.data.materials[index] = mats["mouth"]
        else:
            skin.data.materials[index] = mats["skin"]
    if not skin.material_slots:
        assign_single(skin, mats["skin"])

    for obj in meshes:
        if obj == skin:
            continue
        value = normalized(obj.name)
        if "pupil" in value:
            assign_single(obj, mats["pupil"])
        elif "iris" in value:
            assign_single(obj, mats["iris"])
        elif any(token in value for token in ("cornea", "tear", "caruncle")):
            assign_single(obj, mats["cornea"])
        elif any(token in value for token in ("sclera", "eyeball", "eye")):
            assign_single(obj, mats["sclera"])
        elif any(token in value for token in ("brow", "lash")):
            assign_single(obj, mats["lash"])
        elif "teeth" in value:
            assign_single(obj, mats["teeth"])
        elif any(token in value for token in ("mouth", "tongue", "gum")):
            assign_single(obj, mats["mouth"])

    for obj in body_objects:
        if obj.type != "MESH":
            continue
        value = normalized(obj.name + " " + " ".join(slot.name for slot in obj.material_slots if slot))
        if any(token in value for token in ("shirt", "cloth", "pant", "dress", "suit", "tee")):
            assign_single(obj, mats["suit"])
        elif any(token in value for token in ("skin", "body", "torso", "leg", "arm")):
            assign_single(obj, mats["skin"])
    return mats


def create_uv_sphere(name, location, scale, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign_single(obj, mat)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def create_curve_mesh(name, points, radius, mat):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 5
    curve.bevel_depth = radius
    curve.bevel_resolution = 4
    curve.resolution_u = 5
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for point, value in zip(spline.bezier_points, points):
        point.co = value
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    curve.materials.append(mat)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
    obj = bpy.context.object
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def create_silver_updo(skin, armature, rig_info, setup, hair_mat):
    center = setup["center"].copy()
    size = setup["size"]
    top = float(setup["hi"][2])
    face_sign = float(setup["forward_sign"])
    hair_center = np.asarray([center[0], center[1] - face_sign * 0.08 * size[1], top - 0.25 * size[2]])
    rx, ry, rz = 0.58 * size[0], 0.56 * size[1], 0.36 * size[2]
    vertices = []
    faces = []
    nphi, ntheta = 112, 30
    for i in range(nphi):
        phi = 2.0 * math.pi * i / nphi
        frontness = face_sign * math.sin(phi)
        side = abs(math.cos(phi))
        theta_max = 1.05 + 0.36 * side if frontness > -0.20 else 2.04
        for j in range(ntheta):
            theta = theta_max * j / (ntheta - 1)
            vertices.append((
                hair_center[0] + rx * math.sin(theta) * math.cos(phi),
                hair_center[1] + ry * math.sin(theta) * math.sin(phi),
                hair_center[2] + rz * math.cos(theta),
            ))
    for i in range(nphi):
        next_i = (i + 1) % nphi
        for j in range(ntheta - 1):
            a = i * ntheta + j
            b = next_i * ntheta + j
            c = next_i * ntheta + j + 1
            d = i * ntheta + j + 1
            faces.extend(((a, b, c), (a, c, d)))
    mesh = bpy.data.meshes.new("AINA_Silver_Updo_Cap_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    cap = bpy.data.objects.new("AINA_Silver_Updo_Cap", mesh)
    bpy.context.collection.objects.link(cap)
    assign_single(cap, hair_mat)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    preserve_bone_parent(cap, armature, rig_info["head_bone"])

    back = -face_sign
    bun_location = (
        hair_center[0],
        hair_center[1] + back * 0.80 * ry,
        top - 0.02 * size[2],
    )
    bun = create_uv_sphere("AINA_Silver_Updo_Bun", bun_location, (0.34 * size[0], 0.30 * size[1], 0.22 * size[2]), hair_mat)
    preserve_bone_parent(bun, armature, rig_info["head_bone"])

    objects = [cap, bun]
    part_root = np.asarray([hair_center[0], hair_center[1] + face_sign * 0.42 * ry, top + 0.015 * size[2]])
    # Broad crown sweeps break the cap into visible hair flow instead of a helmet.
    for index, value in enumerate(np.linspace(-1.0, 1.0, 19)):
        end = np.asarray([
            hair_center[0] + value * 0.82 * rx,
            hair_center[1] - face_sign * (0.08 + 0.22 * abs(value)) * ry,
            top - (0.12 + 0.10 * abs(value)) * size[2],
        ])
        middle = 0.5 * (part_root + end) + np.asarray([value * 0.06 * rx, -face_sign * 0.12 * ry, 0.08 * size[2]])
        obj = create_curve_mesh(f"AINA_Crown_Sweep_{index + 1}", [part_root, middle, end], 0.0060 * max(size[0] / 0.18, 0.7), hair_mat)
        preserve_bone_parent(obj, armature, rig_info["head_bone"])
        objects.append(obj)

    # Face-framing locks are real mesh curves.  The lower pair is attached to
    # spring bones so the final VRM has visible dynamic hair rather than metadata only.
    for side_name, sign, chain in (
        ("L", 1.0, rig_info["hair_chains"][0]),
        ("R", -1.0, rig_info["hair_chains"][1]),
    ):
        for index in range(5):
            root = np.asarray([
                hair_center[0] + sign * (0.10 + 0.06 * index) * rx,
                hair_center[1] + face_sign * 0.40 * ry,
                top - (0.08 + 0.025 * index) * size[2],
            ])
            end = np.asarray([
                hair_center[0] + sign * (0.44 + 0.045 * index) * rx,
                hair_center[1] + face_sign * 0.56 * ry,
                top - (0.45 + 0.055 * index) * size[2],
            ])
            middle = 0.5 * (root + end) + np.asarray([sign * 0.10 * rx, face_sign * 0.08 * ry, 0.03 * size[2]])
            obj = create_curve_mesh(f"AINA_Side_Lock_{side_name}_{index + 1}", [root, middle, end], 0.0040 * max(size[0] / 0.18, 0.7), hair_mat)
            preserve_bone_parent(obj, armature, chain[1 if index >= 2 else 0])
            objects.append(obj)

    for index, value in enumerate(np.linspace(-0.72, 0.72, 9)):
        root = np.asarray([hair_center[0] + value * rx, hair_center[1] - face_sign * 0.20 * ry, top - 0.10 * size[2]])
        end = np.asarray([hair_center[0] + value * 0.76 * rx, hair_center[1] - face_sign * 0.82 * ry, top - 0.43 * size[2]])
        middle = 0.5 * (root + end) + np.asarray([0.0, -face_sign * 0.12 * ry, 0.03 * size[2]])
        obj = create_curve_mesh(f"AINA_Back_Flow_{index + 1}", [root, middle, end], 0.0042 * max(size[0] / 0.18, 0.7), hair_mat)
        preserve_bone_parent(obj, armature, rig_info["hair_chains"][2][1 if abs(value) < 0.35 else 0])
        objects.append(obj)
    return objects


def align_and_parent_head(head_objects, armature, rig_info, setup):
    head_bone = armature.data.bones[rig_info["head_bone"]]
    bone_world = np.asarray((armature.matrix_world @ head_bone.head_local)[:], dtype=np.float64)
    head_center = np.asarray(setup["target"], dtype=np.float64)
    distance = float(np.linalg.norm(head_center - bone_world))
    correction = np.zeros(3, dtype=np.float64)
    if distance > 0.30:
        expected = bone_world + np.asarray([0.0, setup["forward_sign"] * 0.018, 0.095])
        correction = expected - head_center
        for obj in head_objects:
            obj.matrix_world.translation = obj.matrix_world.translation + Vector(correction.tolist())
    for obj in head_objects:
        for modifier in list(obj.modifiers) if obj.type == "MESH" else []:
            if modifier.type == "ARMATURE" and modifier.object != armature:
                obj.modifiers.remove(modifier)
        preserve_bone_parent(obj, armature, rig_info["head_bone"])
    return {"initial_head_to_bone_distance_m": distance, "world_alignment_correction_m": correction.tolist()}


def parent_eye_anatomy(meshes, skin, armature, rig_info):
    moved = []
    centers = rig_info["eye_centers"]
    for obj in meshes:
        if obj == skin or not lock.is_eye_name(obj.name):
            continue
        center = object_center(obj)
        side = min(centers, key=lambda key: np.linalg.norm(center - centers[key]))
        preserve_bone_parent(obj, armature, rig_info["eye_bones"][side])
        moved.append({"object": obj.name, "bone": rig_info["eye_bones"][side]})
    return moved


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


def setup_render(scene, bounds, output: Path):
    for obj in list(scene.objects):
        if obj.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    lo, hi = bounds
    center = (lo + hi) * 0.5
    size = hi - lo
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 720
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.world.color = (0.018, 0.024, 0.040)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = -0.30
    except Exception:
        pass
    target = np.asarray([center[0], center[1], lo[2] + 0.64 * size[2]])
    create_light("AINA_Production_Key", tuple(target + np.asarray([0.70 * size[0], -1.25 * size[1], 0.55 * size[2]])), 780, 3.0, target)
    create_light("AINA_Production_Fill", tuple(target + np.asarray([-0.75 * size[0], -0.85 * size[1], 0.12 * size[2]])), 310, 3.6, target)
    create_light("AINA_Production_Rim", tuple(target + np.asarray([0.0, 1.10 * size[1], 0.65 * size[2]])), 480, 2.6, target)
    (output / "Preview").mkdir(parents=True, exist_ok=True)
    return center, size


def scene_mesh_bounds():
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    return lock.mesh_bounds(meshes)


def render_camera(scene, location, target, lens, path):
    data = bpy.data.cameras.new("AINA_Production_Camera")
    camera = bpy.data.objects.new("AINA_Production_Camera", data)
    bpy.context.collection.objects.link(camera)
    camera.data.lens = lens
    camera.location = location
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(camera, do_unlink=True)
    return path


def clear_arkit(skin):
    if not skin.data.shape_keys:
        return
    for name in ARKIT_52:
        key = skin.data.shape_keys.key_blocks.get(name)
        if key:
            key.value = 0.0


def activate(skin, values):
    clear_arkit(skin)
    activated = []
    for name, value in values.items():
        key = skin.data.shape_keys.key_blocks.get(name)
        if key:
            key.value = value
            activated.append(name)
    return activated


def render_qa(scene, skin, output: Path, setup, full_bounds):
    preview = output / "Preview"
    lo, hi = full_bounds
    full_center = (lo + hi) * 0.5
    full_size = hi - lo
    full_target = np.asarray([full_center[0], full_center[1], lo[2] + 0.55 * full_size[2]])
    full_front = full_target + np.asarray([0.0, -max(3.0 * full_size[1], 1.9), 0.05 * full_size[2]])
    full_q3 = full_front + np.asarray([0.65 * full_size[0], 0.16 * full_size[1], 0.0])
    clear_arkit(skin)
    renders = {
        "full_front": str(render_camera(scene, full_front, full_target, 70, preview / "AINA_FINAL_FULL_FRONT.png")),
        "full_three_quarter": str(render_camera(scene, full_q3, full_target, 72, preview / "AINA_FINAL_FULL_THREE_QUARTER.png")),
    }
    portrait_target = np.asarray(setup["target"], dtype=np.float64)
    portrait_front = np.asarray(setup["locations"]["front"], dtype=np.float64)
    portrait_q3 = np.asarray(setup["locations"]["three_quarter"], dtype=np.float64)
    renders["neutral_front"] = str(render_camera(scene, portrait_front, portrait_target, 86, preview / "AINA_FINAL_PORTRAIT_NEUTRAL.png"))
    renders["neutral_three_quarter"] = str(render_camera(scene, portrait_q3, portrait_target, 86, preview / "AINA_FINAL_PORTRAIT_NEUTRAL_3Q.png"))
    cases = {
        "happy": {"mouthSmileLeft": 0.85, "mouthSmileRight": 0.85, "cheekSquintLeft": 0.30, "cheekSquintRight": 0.30},
        "sad": {"browInnerUp": 0.70, "mouthFrownLeft": 0.72, "mouthFrownRight": 0.72},
        "angry": {"browDownLeft": 0.78, "browDownRight": 0.78, "mouthFrownLeft": 0.32, "mouthFrownRight": 0.32},
        "surprised": {"browInnerUp": 0.72, "eyeWideLeft": 0.70, "eyeWideRight": 0.70, "jawOpen": 0.58},
        "blink": {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0},
        "aa": {"jawOpen": 0.62, "mouthFunnel": 0.16},
        "ou": {"mouthPucker": 0.72, "mouthFunnel": 0.50},
    }
    activated = {}
    for name, values in cases.items():
        activated[name] = activate(skin, values)
        renders[name] = str(render_camera(scene, portrait_front, portrait_target, 86, preview / f"AINA_FINAL_PORTRAIT_{name.upper()}.png"))
    clear_arkit(skin)
    return renders, activated


def export_glb(path: Path):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        export_morph=True,
        export_morph_normal=True,
        export_apply=False,
        export_animations=False,
    )


def clean_reimport_qa(vrm_path: Path, expected_skin_name: str, output: Path):
    binary = inspect_vrm(vrm_path)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(vrm_path))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.data.shape_keys]
    candidate = max(meshes, key=lambda obj: len(obj.data.shape_keys.key_blocks), default=None)
    imported_names = []
    if candidate:
        imported_names = [key.name for key in candidate.data.shape_keys.key_blocks]
    exact = [name for name in ARKIT_52 if name in imported_names]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    bone_names = [bone.name for armature in armatures for bone in armature.data.bones]
    reimport_blend = output / "QA" / "AINA_VRM_CLEAN_REIMPORT.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(reimport_blend))
    result = {
        "binary": binary,
        "imported_shape_key_count": max(0, len(imported_names) - 1),
        "imported_arkit_52_count": len(exact),
        "imported_arkit_52_missing": sorted(set(ARKIT_52) - set(exact)),
        "imported_arkit_52_names": exact,
        "imported_armature_count": len(armatures),
        "imported_bone_count": len(bone_names),
        "imported_mesh_count": len([obj for obj in bpy.context.scene.objects if obj.type == "MESH"]),
        "candidate_skin": candidate.name if candidate else None,
        "clean_reimport_blend": str(reimport_blend),
    }
    result["pass"] = (
        binary.get("vrm_spec_version") == "1.0"
        and binary.get("arkit_52_count") == 52
        and binary.get("preset_expression_count") == 18
        and not binary.get("missing_required_humanoid")
        and len(exact) == 52
        and len(armatures) >= 1
    )
    return result


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)
    landmark_data = json.loads(args.landmarks.read_text())
    visual_report = json.loads(args.visual_report.read_text())

    scene = bpy.context.scene
    head_objects = [obj for obj in scene.objects if obj.type not in {"CAMERA", "LIGHT"}]
    head_meshes = [obj for obj in head_objects if obj.type == "MESH"]
    skin = lock.identify_skin(head_meshes)
    setup_before = lock.build_setup(scene, head_meshes)

    before_import = set(scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(args.body_glb))
    body_objects = [obj for obj in scene.objects if obj not in before_import]
    armature = choose_armature(body_objects)
    body_meshes = [obj for obj in body_objects if obj.type == "MESH"]

    rig_info = ensure_eye_and_hair_bones(armature, head_meshes, skin, setup_before)
    alignment = align_and_parent_head(head_objects, armature, rig_info, setup_before)
    eye_parenting = parent_eye_anatomy(head_meshes, skin, armature, rig_info)
    setup = lock.build_setup(scene, head_meshes)
    materials = restore_materials(head_meshes, skin, body_objects)
    hair_objects = create_silver_updo(skin, armature, rig_info, setup, materials["hair"])

    arkit_report = arkit.build_arkit52(scene, skin, head_meshes, landmark_data)
    shape_key_names = [key.name for key in skin.data.shape_keys.key_blocks]

    full_bounds = scene_mesh_bounds()
    setup_render(scene, full_bounds, args.out)
    renders, activated = render_qa(scene, skin, args.out, setup, full_bounds)

    master_path = args.out / "AINA_MASTER.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(master_path))
    glb_path = args.out / "AINA_MASTER.glb"
    export_glb(glb_path)
    vrm_path = args.out / "AINA.vrm"
    patch_report = patch_vrm(glb_path, vrm_path, skin.name, rig_info["hair_chains"])

    # All information needed after factory reset is copied to plain Python data.
    pre_reimport = {
        "skin_name": skin.name,
        "skin_vertices": len(skin.data.vertices),
        "shape_key_names": shape_key_names,
        "body_object_count": len(body_objects),
        "head_object_count": len(head_objects),
        "hair_object_count": len(hair_objects),
        "armature": armature.name,
        "bone_count": len(armature.data.bones),
        "rig_info": rig_info,
        "alignment": alignment,
        "eye_parenting": eye_parenting,
        "arkit": arkit_report,
        "renders": renders,
        "activated": activated,
        "patch": patch_report,
    }
    reimport = clean_reimport_qa(vrm_path, pre_reimport["skin_name"], args.out)

    # The technical release gate is automatic.  Visual identity remains explicit:
    # it may be locked only if the preceding real-render quantitative gate passed
    # and the new full-character review sheet is inspected after this build.
    technical_pass = (
        pre_reimport["skin_vertices"] == 17161
        and pre_reimport["arkit"]["created"] == 52
        and not pre_reimport["arkit"]["missing"]
        and not pre_reimport["arkit"]["zero_or_placeholder"]
        and pre_reimport["patch"]["arkit_52_count"] == 52
        and pre_reimport["patch"]["preset_expression_count"] == 18
        and not pre_reimport["patch"]["missing_required_humanoid"]
        and reimport["pass"]
    )
    input_visual_gate = bool(visual_report.get("quantitative_identity_gate"))
    report = {
        "product": "AINA Full VRM Production Candidate",
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "source_identity": "AINA real visual-lock Vitruvian FACS head",
        "source_body": "CC0 Vitruvian rigged body",
        "master_blend": str(master_path),
        "intermediate_glb": str(glb_path),
        "vrm": str(vrm_path),
        "vrm_bytes": vrm_path.stat().st_size,
        "pre_reimport": pre_reimport,
        "reimport": reimport,
        "input_visual_quantitative_gate": input_visual_gate,
        "technical_release_gate": technical_pass,
        "manual_full_character_visual_gate_required": True,
        "identity_lock": False,
        "visual_identity_lock": False,
        "production_release": False,
        "candidate": True,
        "next_gate": "Inspect the actual full-character neutral/front/3Q and expression renders. If the same AINA identity is preserved, set identity_lock and visual_identity_lock true, retain this exact BLEND/VRM, and publish the final release package without further face versioning.",
    }
    (qa / "AINA_FINAL_VRM_PRODUCTION_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
