#!/usr/bin/env python3
"""AINA Rain visual-identity reconstruction on the same production topology.

This pass starts from the verified Rain Identity Master v2 BLEND.  It does not
replace the character or generate effect art.  It applies one topology-preserving
neutral displacement to the real GEO-rain-head Basis and every existing shape
key, restores bilateral eye/brow/lash/cornea anatomy that was lost in the GLB
conversion, builds real iris geometry, and renders actual Blender QA views.

Identity and VRM locks deliberately remain false until the real renders pass.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aina_rain_identity_master as base


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def smoothstep01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def ellipsoid(points: np.ndarray, centre, radii, outer: float = 1.0) -> np.ndarray:
    centre = np.asarray(centre, dtype=np.float64)
    radii = np.maximum(np.asarray(radii, dtype=np.float64), 1.0e-7)
    q = np.sqrt(np.sum(((points - centre) / radii) ** 2, axis=1))
    weight = np.zeros(len(points), dtype=np.float64)
    mask = q < outer
    if np.any(mask):
        t = q[mask] / outer
        weight[mask] = 0.5 * (1.0 + np.cos(np.pi * t))
    return weight


def semantic_magnitude(skin, basis: np.ndarray, tokens: tuple[str, ...]) -> np.ndarray:
    result = np.zeros(len(basis), dtype=np.float64)
    if not skin.data.shape_keys:
        return result
    for key in skin.data.shape_keys.key_blocks:
        lower = key.name.lower().replace("_", "")
        if key.name == "Basis" or not any(token.replace("_", "") in lower for token in tokens):
            continue
        result = np.maximum(result, np.linalg.norm(base.key_array(key) - basis, axis=1))
    return result


def normalized_semantic(magnitude: np.ndarray, quantile: float = 0.45) -> np.ndarray:
    positive = magnitude[magnitude > 1.0e-8]
    if not len(positive):
        return np.zeros_like(magnitude)
    low = max(float(np.quantile(positive, quantile)), 1.0e-5)
    high = max(float(np.quantile(positive, 0.985)), low * 1.4)
    return smoothstep01((magnitude - low) / max(high - low, 1.0e-9))


def weighted_centre(points: np.ndarray, weight: np.ndarray, fallback) -> np.ndarray:
    total = float(weight.sum())
    if total < 1.0e-9:
        return np.asarray(fallback, dtype=np.float64)
    return np.sum(points * weight[:, None], axis=0) / total


def true_eye_centres(scene, face_x: float) -> tuple[list[np.ndarray], str | None]:
    preferred = scene.objects.get("GEO-rain-eyes")
    candidates = [preferred] if preferred and preferred.type == "MESH" else []
    candidates.extend(
        obj for obj in scene.objects
        if obj.type == "MESH" and obj not in candidates and "eyes" in obj.name.lower()
    )
    for obj in candidates:
        points = base.world_vertices(obj)
        left = points[points[:, 0] < face_x]
        right = points[points[:, 0] >= face_x]
        if len(left) > 80 and len(right) > 80:
            centres = [left.mean(axis=0), right.mean(axis=0)]
            centres.sort(key=lambda point: point[0])
            return centres, obj.name
    return [], None


def apply_same_delta(skin, world_delta: np.ndarray) -> None:
    base.apply_world_delta(skin, world_delta)


def art_directed_head(skin, head_ids: np.ndarray, eyes: list[np.ndarray], forward_sign: float) -> dict:
    basis_key = skin.data.shape_keys.key_blocks.get("Basis") if skin.data.shape_keys else None
    basis_local = base.key_array(basis_key) if basis_key else base.mesh_local_array(skin)
    points = base.to_world(skin, basis_local)
    region = points[head_ids]
    lo, hi = region.min(axis=0), region.max(axis=0)
    head_centre = (lo + hi) * 0.5
    face_x = float(np.mean(eyes, axis=0)[0]) if eyes else float(head_centre[0])
    eye_z = float(np.mean(eyes, axis=0)[2]) if eyes else float(lo[2] + 0.57 * (hi[2] - lo[2]))

    frontness = forward_sign * (region[:, 1] - head_centre[1])
    front_low = float(np.quantile(frontness, 0.34))
    front_high = float(np.quantile(frontness, 0.94))
    face_front = smoothstep01((frontness - front_low) / max(front_high - front_low, 1.0e-9))
    face_side = smoothstep01((0.145 - np.abs(region[:, 0] - face_x)) / 0.055)
    face_weight = face_front * face_side

    lip_sem_full = normalized_semantic(
        semantic_magnitude(skin, basis_local, ("lips", "smile", "cheekpuff")), 0.38
    )
    eye_sem_full = normalized_semantic(
        semantic_magnitude(skin, basis_local, ("eyelidsclose", "eyebrows")), 0.34
    )
    lip_sem = lip_sem_full[head_ids]
    eye_sem = eye_sem_full[head_ids]

    mouth_fallback = np.array([
        face_x,
        head_centre[1] + forward_sign * 0.30 * (hi[2] - lo[2]),
        eye_z - 0.075,
    ])
    below_eye = smoothstep01((eye_z - 0.010 - region[:, 2]) / 0.040)
    mouth = weighted_centre(region, lip_sem * face_weight * below_eye, mouth_fallback)
    mouth_z = float(mouth[2])

    lower_candidates = (
        (region[:, 2] < mouth_z)
        & (np.abs(region[:, 0] - face_x) < 0.055)
        & (face_weight > 0.38)
    )
    chin_z = float(np.quantile(region[lower_candidates, 2], 0.045)) if np.any(lower_candidates) else mouth_z - 0.055

    central_nose = (
        (np.abs(region[:, 0] - face_x) < 0.026)
        & (region[:, 2] > mouth_z + 0.012)
        & (region[:, 2] < eye_z - 0.004)
        & (face_weight > 0.35)
    )
    if np.any(central_nose):
        local_index = np.where(central_nose)[0][np.argmax(frontness[central_nose])]
        nose_tip = region[local_index].copy()
    else:
        nose_tip = np.array([face_x, head_centre[1] + forward_sign * 0.11, mouth_z + 0.036])

    delta = np.zeros_like(region)
    preserve = np.zeros(len(region), dtype=np.float64)

    # Shorter youthful lower third and a softer V jaw.
    lower = np.clip(
        (eye_z + 0.006 - region[:, 2]) / max(eye_z + 0.006 - (chin_z - 0.012), 1.0e-6),
        0.0,
        1.0,
    )
    lower_weight = face_weight * smoothstep01(lower)
    compressed_z = eye_z + (region[:, 2] - eye_z) * 0.91
    delta[:, 2] += (compressed_z - region[:, 2]) * lower_weight
    taper = 1.0 - 0.14 * np.power(lower, 1.22)
    delta[:, 0] += (
        (region[:, 0] - face_x) * taper - (region[:, 0] - face_x)
    ) * lower_weight

    # Reduce the oversized bald cranium without flattening the face.
    upper_origin = eye_z + 0.024
    upper = np.clip((region[:, 2] - upper_origin) / max(hi[2] - upper_origin, 1.0e-6), 0.0, 1.0)
    upper_weight = smoothstep01(upper) * smoothstep01((0.155 - np.abs(region[:, 0] - face_x)) / 0.050)
    delta[:, 0] += -0.075 * (region[:, 0] - face_x) * upper_weight
    upper_target_z = upper_origin + (region[:, 2] - upper_origin) * 0.90
    delta[:, 2] += (upper_target_z - region[:, 2]) * upper_weight

    # Natural almond eyelid apertures and gentle outer-corner lift.
    eye_reports = []
    for eye in eyes:
        side = -1.0 if eye[0] < face_x else 1.0
        spatial = ellipsoid(region, eye, (0.050, 0.052, 0.035), 1.20)
        weight = spatial * np.maximum(eye_sem, 0.16 * face_weight)
        target_x = eye[0] + (region[:, 0] - eye[0]) * 1.085
        target_z = eye[2] + (region[:, 2] - eye[2]) * 1.105
        delta[:, 0] += (target_x - region[:, 0]) * weight
        delta[:, 2] += (target_z - region[:, 2]) * weight
        delta[:, 2] += 0.0010 * weight
        outer = eye + np.array([side * 0.025, 0.0, 0.001])
        outer_weight = ellipsoid(region, outer, (0.023, 0.030, 0.019), 1.08) * weight
        delta[:, 0] += side * 0.0011 * outer_weight
        delta[:, 2] += 0.0010 * outer_weight
        under = eye + np.array([0.0, -forward_sign * 0.004, -0.020])
        under_weight = ellipsoid(region, under, (0.054, 0.050, 0.034), 1.12) * face_weight
        delta[:, 1] += forward_sign * 0.0018 * under_weight
        delta[:, 2] += 0.0008 * under_weight
        preserve = np.maximum(preserve, np.clip(weight + outer_weight, 0.0, 1.0))
        eye_reports.append({"centre": eye.tolist(), "active_vertices": int(np.sum(weight > 0.06))})

    # Delicate narrow nose, distinct bridge and small lifted tip.
    bridge = np.array([face_x, nose_tip[1], mouth_z + 0.62 * (eye_z - mouth_z)])
    bridge_weight = ellipsoid(region, bridge, (0.030, 0.044, 0.058), 1.14) * face_weight
    bridge_target_x = face_x + (region[:, 0] - face_x) * 0.82
    delta[:, 0] += (bridge_target_x - region[:, 0]) * bridge_weight
    delta[:, 2] += 0.0012 * bridge_weight
    delta[:, 1] += forward_sign * 0.0008 * bridge_weight
    tip_weight = ellipsoid(region, nose_tip, (0.024, 0.032, 0.026), 1.10) * face_weight
    delta[:, 0] += -(region[:, 0] - face_x) * 0.12 * tip_weight
    delta[:, 1] += forward_sign * 0.0016 * tip_weight
    delta[:, 2] += 0.0015 * tip_weight
    nose_base = np.array([face_x, nose_tip[1], mouth_z + 0.022])
    base_weight = ellipsoid(region, nose_base, (0.037, 0.040, 0.030), 1.10) * face_weight
    delta[:, 0] += -(region[:, 0] - face_x) * 0.18 * base_weight
    delta[:, 2] += 0.0012 * base_weight
    preserve = np.maximum(preserve, np.clip(bridge_weight + tip_weight + base_weight, 0.0, 1.0))

    # Compact lips integrated into the face instead of a wide floating oval.
    lip_spatial = ellipsoid(region, mouth, (0.056, 0.046, 0.030), 1.14)
    lip_weight = lip_spatial * np.maximum(lip_sem, 0.22 * face_weight)
    target_x = mouth[0] + (region[:, 0] - mouth[0]) * 0.84
    target_z = mouth[2] + (region[:, 2] - mouth[2]) * 0.80
    delta[:, 0] += (target_x - region[:, 0]) * lip_weight
    delta[:, 2] += (target_z - region[:, 2]) * lip_weight + 0.0011 * lip_weight
    delta[:, 1] += forward_sign * 0.0015 * lip_weight
    preserve = np.maximum(preserve, np.clip(lip_weight, 0.0, 1.0))

    # High apple-cheek support and continuous eye-to-cheek surface.
    for eye in eyes:
        side = -1.0 if eye[0] < face_x else 1.0
        cheek = np.array([
            eye[0] + side * 0.002,
            nose_tip[1] - forward_sign * 0.012,
            mouth_z + 0.53 * (eye_z - mouth_z),
        ])
        cheek_weight = ellipsoid(region, cheek, (0.060, 0.060, 0.055), 1.16) * face_weight
        delta[:, 1] += forward_sign * 0.0024 * cheek_weight
        delta[:, 2] += 0.0012 * cheek_weight
        delta[:, 0] += -side * 0.0006 * cheek_weight

    # Small rounded chin.
    chin = np.array([face_x, mouth[1] - forward_sign * 0.004, chin_z + 0.006])
    chin_weight = ellipsoid(region, chin, (0.052, 0.055, 0.048), 1.12) * face_weight
    delta[:, 0] += -(region[:, 0] - face_x) * 0.13 * chin_weight
    delta[:, 2] += 0.0030 * chin_weight
    delta[:, 1] += -forward_sign * 0.0007 * chin_weight
    preserve = np.maximum(preserve, np.clip(chin_weight, 0.0, 1.0))

    # Reduce ears and narrow the visible neck column.
    ear_weight = (
        smoothstep01((np.abs(region[:, 0] - face_x) - 0.102) / 0.040)
        * smoothstep01((region[:, 2] - (mouth_z - 0.030)) / 0.035)
        * smoothstep01(((eye_z + 0.040) - region[:, 2]) / 0.035)
        * smoothstep01((front_high - frontness) / max(front_high - front_low, 1.0e-9))
    )
    delta[:, 0] += -(region[:, 0] - face_x) * 0.13 * ear_weight
    ear_mid = 0.5 * (eye_z + mouth_z)
    delta[:, 2] += (ear_mid - region[:, 2]) * 0.10 * ear_weight
    delta[:, 1] += -forward_sign * 0.0015 * ear_weight
    neck_weight = (
        smoothstep01(((chin_z + 0.006) - region[:, 2]) / 0.040)
        * smoothstep01((0.100 - np.abs(region[:, 0] - face_x)) / 0.040)
    )
    delta[:, 0] += -(region[:, 0] - face_x) * 0.12 * neck_weight

    adjacency = base.adjacency_for_region(skin, head_ids)
    smoothed = base.smooth_region_delta(delta, adjacency, preserve, passes=2)
    lengths = np.linalg.norm(smoothed, axis=1)
    smoothed *= np.minimum(1.0, 0.018 / np.maximum(lengths, 1.0e-9))[:, None]
    full = np.zeros_like(points)
    full[head_ids] = smoothed
    apply_same_delta(skin, full)
    return {
        "face_x": face_x,
        "forward_sign_y": forward_sign,
        "eye_z": eye_z,
        "mouth_centre": mouth.tolist(),
        "chin_z": chin_z,
        "nose_tip": nose_tip.tolist(),
        "eye_regions": eye_reports,
        "max_displacement_m": float(np.linalg.norm(smoothed, axis=1).max()),
        "rms_displacement_m": float(np.sqrt(np.mean(np.sum(smoothed * smoothed, axis=1)))),
        "moved_vertices_over_0_5mm": int(np.sum(np.linalg.norm(smoothed, axis=1) > 0.0005)),
    }


def mirror_mesh_object(obj, face_x: float, suffix: str) -> bpy.types.Object:
    duplicate = obj.copy()
    duplicate.data = obj.data.copy()
    duplicate.name = f"{obj.name}_{suffix}"
    bpy.context.collection.objects.link(duplicate)
    for modifier in list(duplicate.modifiers):
        if modifier.type == "MIRROR":
            duplicate.modifiers.remove(modifier)
    if duplicate.data.shape_keys:
        for key in duplicate.data.shape_keys.key_blocks:
            world = base.to_world(duplicate, base.key_array(key))
            world[:, 0] = 2.0 * face_x - world[:, 0]
            base.set_key_array(key, base.to_local(duplicate, world))
    else:
        world = base.to_world(duplicate, base.mesh_local_array(duplicate))
        world[:, 0] = 2.0 * face_x - world[:, 0]
        base.set_mesh_local_array(duplicate, base.to_local(duplicate, world))
    for polygon in duplicate.data.polygons:
        try:
            polygon.flip()
        except Exception:
            pass
    duplicate.data.update()
    return duplicate


def scale_world_geometry(obj, factors, shift=(0.0, 0.0, 0.0)) -> None:
    factors = np.asarray(factors, dtype=np.float64)
    shift = np.asarray(shift, dtype=np.float64)
    if obj.data.shape_keys:
        for key in obj.data.shape_keys.key_blocks:
            world = base.to_world(obj, base.key_array(key))
            centre = world.mean(axis=0)
            world = centre + (world - centre) * factors + shift
            base.set_key_array(key, base.to_local(obj, world))
    else:
        world = base.to_world(obj, base.mesh_local_array(obj))
        centre = world.mean(axis=0)
        world = centre + (world - centre) * factors + shift
        base.set_mesh_local_array(obj, base.to_local(obj, world))
    obj.data.update()


def make_material(name: str, color, roughness: float = 0.28):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = tuple(color)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF") if material.node_tree else None
    if shader:
        shader.inputs["Base Color"].default_value = tuple(color)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = 0.0
    return material


def repair_bilateral_anatomy(scene, face_x: float, forward_sign: float) -> dict:
    targets = (
        "GEO-rain-eye_cornea",
        "GEO-rain-eye_dots",
        "GEO-rain-eyebrows",
        "GEO-rain-eyelashes",
    )
    mirrored = []
    for name in targets:
        obj = scene.objects.get(name)
        if not obj or obj.type != "MESH" or not len(obj.data.vertices):
            continue
        world = base.world_vertices(obj)
        left_count = int(np.sum(world[:, 0] < face_x - 0.002))
        right_count = int(np.sum(world[:, 0] > face_x + 0.002))
        if min(left_count, right_count) > max(4, int(0.08 * len(world))):
            continue
        duplicate = mirror_mesh_object(obj, face_x, "AINA_Mirror")
        mirrored.append({"source": obj.name, "mirror": duplicate.name, "vertices": len(duplicate.data.vertices)})

    # Thin and lower both eyebrows after mirroring.
    brow_names = [obj.name for obj in scene.objects if obj.type == "MESH" and "eyebrows" in obj.name.lower()]
    for name in brow_names:
        obj = scene.objects.get(name)
        if obj:
            scale_world_geometry(obj, (0.96, 1.0, 0.58), (0.0, forward_sign * 0.0008, -0.0080))

    # Make both pupils readable and build a larger teal iris shell behind each.
    pupil_objects = [obj for obj in scene.objects if obj.type == "MESH" and "eye_dots" in obj.name.lower()]
    iris_material = make_material("AINA_Rain_Iris_Teal", (0.025, 0.16, 0.19, 1.0), 0.20)
    pupil_material = make_material("AINA_Rain_Pupil_Dark", (0.002, 0.004, 0.006, 1.0), 0.18)
    iris_objects = []
    for pupil in pupil_objects:
        scale_world_geometry(pupil, (1.55, 1.0, 1.55))
        pupil.data.materials.clear()
        pupil.data.materials.append(pupil_material)
        iris = pupil.copy()
        iris.data = pupil.data.copy()
        iris.name = f"{pupil.name}_AINA_Iris"
        bpy.context.collection.objects.link(iris)
        scale_world_geometry(iris, (2.35, 1.0, 2.35), (0.0, -forward_sign * 0.00035, 0.0))
        iris.data.materials.clear()
        iris.data.materials.append(iris_material)
        iris_objects.append(iris.name)

    return {
        "mirrored_components": mirrored,
        "brow_objects": brow_names,
        "pupil_objects": [obj.name for obj in pupil_objects],
        "iris_objects": iris_objects,
    }


def render_blink(scene, cameras, skin, out: Path) -> str | None:
    if not skin.data.shape_keys:
        return None
    keys = skin.data.shape_keys.key_blocks
    left = keys.get("EyelidsClose.L")
    right = keys.get("EyelidsClose.R")
    if not left and not right:
        return None
    base.reset_shape_keys([skin])
    if left:
        left.value = 1.0
    if right:
        right.value = 1.0
    previous = base.set_hair_visible(scene, False)
    path = out / "Preview" / "AINA_RAIN_IDENTITY_VISUAL_LOCK_BLINK.png"
    base.render(scene, cameras["front"], path)
    base.restore_visibility(scene, previous)
    base.reset_shape_keys([skin])
    return str(path)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)

    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH" and len(obj.data.vertices)]
    base.reset_shape_keys(meshes)
    armature = base.find_armature(scene)
    head_bone = base.find_head_bone(armature)
    head_point = base.bone_world_point(armature, head_bone)
    skin, skin_report = base.identify_skin(scene, head_point)
    original_shape_deltas = base.capture_shape_deltas(skin)
    character_height = skin_report["character_height_m"]

    skin_world_before = base.world_vertices(skin)
    skin_bounds = skin_world_before.min(axis=0), skin_world_before.max(axis=0)
    initial_face_x = float(0.5 * (skin_bounds[0][0] + skin_bounds[1][0]))
    eyes, eye_source = true_eye_centres(scene, initial_face_x)
    if len(eyes) != 2:
        eyes = base.eye_centres(base.eye_objects(scene, head_point, character_height))
    if len(eyes) != 2:
        raise RuntimeError(f"Expected two real Rain eye centres, got {len(eyes)}")
    face_x = float(np.mean(eyes, axis=0)[0])
    skin_centre = skin_world_before.mean(axis=0)
    forward_sign = -1.0 if np.mean(eyes, axis=0)[1] < skin_centre[1] else 1.0
    head_ids, _, _, _ = base.head_region(skin, head_point, eyes, character_height)

    sculpt = art_directed_head(skin, head_ids, eyes, forward_sign)
    skin_world_after = base.world_vertices(skin)
    related = base.move_related_objects(scene, skin, skin_world_before, skin_world_after, eyes)
    bpy.context.view_layer.update()

    anatomy = repair_bilateral_anatomy(scene, face_x, forward_sign)
    bpy.context.view_layer.update()
    final_eyes, _ = true_eye_centres(scene, face_x)
    if len(final_eyes) != 2:
        final_eyes = eyes

    preservation = base.validate_shape_deltas(skin, original_shape_deltas)
    cameras, camera_report = base.setup_cameras(
        scene, skin, head_ids, final_eyes, head_point, character_height
    )
    renders = base.render_final_suite(scene, cameras, skin, args.out)
    blink = render_blink(scene, cameras, skin, args.out)
    if blink:
        renders.setdefault("expressions", {})["blink"] = {"file": blink, "shape_key": "EyelidsClose.L+R"}

    blend_path = args.out / "AINA_RAIN_IDENTITY_VISUAL_LOCK_CANDIDATE.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_RAIN_IDENTITY_VISUAL_LOCK_CANDIDATE.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    final_world = base.world_vertices(skin)
    report = {
        "product": "AINA Rain Real-Mesh Visual Identity Candidate",
        "source": "AINA Rain Identity Master v2",
        "source_character": "Blender Studio Rain v3",
        "source_license": "CC BY 4.0",
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "same_skin_topology": True,
        "topology_changed": False,
        "armature_preserved": True,
        "skin_weights_preserved": True,
        "uvs_preserved": True,
        "skin_object": skin.name,
        "vertices": len(skin.data.vertices),
        "triangles": sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons),
        "head_region_vertices": len(head_ids),
        "eye_source_object": eye_source,
        "eye_centres": [point.tolist() for point in final_eyes],
        "camera": camera_report,
        "sculpt": sculpt,
        "related_anatomy": related,
        "bilateral_anatomy_recovery": anatomy,
        "expression_preservation": preservation,
        "total_skin_displacement_max_m": float(np.linalg.norm(final_world - skin_world_before, axis=1).max()),
        "total_skin_displacement_rms_m": float(np.sqrt(np.mean(np.sum((final_world - skin_world_before) ** 2, axis=1)))),
        "renders": renders,
        "identity_lock": False,
        "visual_identity_lock": False,
        "candidate": True,
        "vrm_exported": False,
        "next_gate": "Inspect actual front, 3Q, side, clay and blink renders. Keep editing this same real Mesh until the approved AINA identity is visually stable before 52-control and VRM assembly.",
        "files": {"blend": str(blend_path), "glb": str(glb_path)},
    }
    (qa / "AINA_RAIN_IDENTITY_VISUAL_LOCK_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
