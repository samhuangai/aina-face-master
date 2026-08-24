#!/usr/bin/env python3
"""Balanced AINA identity correction on the verified Rain v3 production Mesh.

Precision v4 proved that a strong central-face warp over-sharpened the nose,
lips and chin. This pass deliberately returns to the successful v3 source and
uses a lower-amplitude, whole-surface correction: moderately smaller adult
cranium, narrower oval face, reduced eye whites/apertures, compact but natural
nose and lips, a slightly longer lower third, smaller ears and narrower neck.

Topology, UVs, weights, armature and every relative source shape-key delta are
preserved. No replacement effect art and no VRM export are produced here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aina_rain_identity_master as base
import aina_rain_appearance_candidate as appearance
import aina_rain_identity_reconstruction_v3 as v3
import aina_rain_identity_precision_v4 as v4


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def balanced_skin_pass(skin, head_ids: np.ndarray, eyes: list[np.ndarray], forward_sign: float) -> dict:
    frame = v4.feature_frame(skin, head_ids, eyes, forward_sign)
    full = frame["full"]
    points = frame["points"]
    hi = frame["hi"]
    face_x = frame["face_x"]
    eye_z = frame["eye_z"]
    mouth = frame["mouth"]
    mouth_z = frame["mouth_z"]
    chin_z = frame["chin_z"]
    nose_tip = frame["nose_tip"]
    frontness = frame["frontness"]
    f_lo = frame["front_low"]
    f_hi = frame["front_high"]
    eye_sem = frame["eye_sem"]
    lip_sem = frame["lip_sem"]

    shell = v3.smoothstep01((frontness - (f_lo - 0.015)) / max(f_hi - f_lo + 0.020, 1.0e-9))
    lateral = v3.smoothstep01((0.175 - np.abs(points[:, 0] - face_x)) / 0.050)
    face = shell * lateral
    delta = np.zeros_like(points)
    preserve = np.zeros(len(points), dtype=np.float64)

    top_origin = eye_z + 0.012
    top = np.clip((points[:, 2] - top_origin) / max(hi[2] - top_origin, 1.0e-6), 0.0, 1.0)
    top_w = v3.smoothstep01(top) * v3.smoothstep01(
        (0.185 - np.abs(points[:, 0] - face_x)) / 0.050
    )
    delta[:, 0] += -(points[:, 0] - face_x) * (0.08 * top) * top_w
    delta[:, 2] += -(points[:, 2] - top_origin) * (0.10 * top) * top_w
    delta[:, 1] += -forward_sign * 0.0015 * top_w * shell

    vertical = np.clip((eye_z + 0.018 - points[:, 2]) / max(eye_z + 0.018 - chin_z, 1.0e-6), 0.0, 1.0)
    x_scale = 0.96 - 0.06 * np.power(vertical, 1.25)
    oval_w = face * v3.smoothstep01(vertical + 0.22)
    delta[:, 0] += (
        (points[:, 0] - face_x) * x_scale - (points[:, 0] - face_x)
    ) * oval_w

    eye_targets = []
    for eye in sorted(eyes, key=lambda p: p[0]):
        target = eye.copy()
        target[0] = face_x + (eye[0] - face_x) * 0.97
        target[2] -= 0.0010
        spatial = v3.ellipsoid(points, eye, (0.058, 0.058, 0.044), 1.22)
        weight = spatial * np.maximum(eye_sem, 0.18 * face)
        target_x = target[0] + (points[:, 0] - eye[0]) * 0.88
        target_z = target[2] + (points[:, 2] - eye[2]) * 0.78
        delta[:, 0] += (target_x - points[:, 0]) * weight
        delta[:, 2] += (target_z - points[:, 2]) * weight
        preserve = np.maximum(preserve, np.clip(weight, 0.0, 1.0))
        eye_targets.append(target)

    bridge = np.array([face_x, nose_tip[1], mouth_z + 0.62 * (eye_z - mouth_z)])
    bridge_w = v3.ellipsoid(points, bridge, (0.033, 0.046, 0.061), 1.15) * face
    delta[:, 0] += -(points[:, 0] - face_x) * 0.12 * bridge_w
    tip_w = v3.ellipsoid(points, nose_tip, (0.026, 0.034, 0.028), 1.10) * face
    delta[:, 0] += -(points[:, 0] - face_x) * 0.14 * tip_w
    delta[:, 1] += forward_sign * 0.0010 * tip_w
    nose_base = np.array([face_x, nose_tip[1], mouth_z + 0.020])
    base_w = v3.ellipsoid(points, nose_base, (0.041, 0.044, 0.032), 1.10) * face
    delta[:, 0] += -(points[:, 0] - face_x) * 0.10 * base_w
    preserve = np.maximum(preserve, np.clip(bridge_w + tip_w + base_w, 0.0, 1.0))

    lip_spatial = v3.ellipsoid(points, mouth, (0.061, 0.050, 0.033), 1.15)
    lip_w = lip_spatial * np.maximum(lip_sem, 0.20 * face)
    target_x = mouth[0] + (points[:, 0] - mouth[0]) * 0.90
    target_z = mouth[2] + (points[:, 2] - mouth[2]) * 0.82
    delta[:, 0] += (target_x - points[:, 0]) * lip_w
    delta[:, 2] += (target_z - points[:, 2]) * lip_w
    delta[:, 1] += -forward_sign * 0.0006 * lip_w
    delta[:, 2] -= 0.0006 * lip_w
    preserve = np.maximum(preserve, np.clip(lip_w, 0.0, 1.0))

    nose_z = float(mouth_z + 0.44 * (eye_z - mouth_z))
    lower = np.clip((nose_z - points[:, 2]) / max(nose_z - chin_z, 1.0e-6), 0.0, 1.0)
    lower_w = face * v3.smoothstep01(lower)
    target_z = nose_z + (points[:, 2] - nose_z) * 1.06
    delta[:, 2] += (target_z - points[:, 2]) * lower_w * 0.82
    jaw_scale = 1.0 - 0.08 * np.power(lower, 1.15)
    delta[:, 0] += (
        (points[:, 0] - face_x) * jaw_scale - (points[:, 0] - face_x)
    ) * lower_w
    chin = np.array([face_x, mouth[1] - forward_sign * 0.003, chin_z])
    chin_w = v3.ellipsoid(points, chin, (0.055, 0.060, 0.052), 1.12) * face
    delta[:, 0] += -(points[:, 0] - face_x) * 0.12 * chin_w
    delta[:, 2] -= 0.0012 * chin_w
    preserve = np.maximum(preserve, np.clip(chin_w, 0.0, 1.0))

    mid_z = mouth_z + 0.58 * (eye_z - mouth_z)
    for eye in eyes:
        side = -1.0 if eye[0] < face_x else 1.0
        cheek = np.array([eye[0], nose_tip[1] - forward_sign * 0.006, mid_z])
        weight = v3.ellipsoid(points, cheek, (0.062, 0.060, 0.054), 1.14) * face
        delta[:, 0] += -side * 0.00045 * weight
        delta[:, 1] += forward_sign * 0.00085 * weight
        delta[:, 2] += 0.00045 * weight

    ear_w = (
        v3.smoothstep01((np.abs(points[:, 0] - face_x) - 0.098) / 0.045)
        * v3.smoothstep01((points[:, 2] - (mouth_z - 0.034)) / 0.040)
        * v3.smoothstep01(((eye_z + 0.034) - points[:, 2]) / 0.042)
        * v3.smoothstep01((f_hi - frontness) / max(f_hi - f_lo, 1.0e-9))
    )
    delta[:, 0] += -(points[:, 0] - face_x) * 0.10 * ear_w
    neck_w = (
        v3.smoothstep01(((chin_z + 0.006) - points[:, 2]) / 0.050)
        * v3.smoothstep01((0.112 - np.abs(points[:, 0] - face_x)) / 0.048)
    )
    delta[:, 0] += -(points[:, 0] - face_x) * 0.14 * neck_w
    delta[:, 2] -= 0.0010 * neck_w

    adjacency = base.adjacency_for_region(skin, head_ids)
    smoothed = base.smooth_region_delta(delta, adjacency, preserve, passes=3)
    length = np.linalg.norm(smoothed, axis=1)
    smoothed *= np.minimum(1.0, 0.012 / np.maximum(length, 1.0e-9))[:, None]
    full_delta = np.zeros_like(full)
    full_delta[head_ids] = smoothed
    base.apply_world_delta(skin, full_delta)
    return {
        "face_x": face_x,
        "eye_z": eye_z,
        "mouth_centre": mouth.tolist(),
        "nose_tip": nose_tip.tolist(),
        "chin_z": chin_z,
        "eye_targets": [point.tolist() for point in eye_targets],
        "max_displacement_m": float(np.linalg.norm(smoothed, axis=1).max()),
        "rms_displacement_m": float(np.sqrt(np.mean(np.sum(smoothed * smoothed, axis=1)))),
        "moved_vertices_over_0_5mm": int(np.sum(np.linalg.norm(smoothed, axis=1) > 0.0005)),
    }


def transform_eye_system(scene, eyes_before, eye_targets, forward_sign: float) -> dict:
    source = sorted(eyes_before, key=lambda p: p[0])
    target = sorted(eye_targets, key=lambda p: p[0])
    if len(source) != 2 or len(target) != 2:
        return {"objects": [], "max_displacement_m": 0.0}
    middle = 0.5 * (source[0][0] + source[1][0])
    moved = []
    maximum = 0.0
    for obj in scene.objects:
        if obj.type != "MESH" or not len(obj.data.vertices):
            continue
        text = (obj.name + " " + " ".join(mat.name for mat in obj.data.materials if mat)).lower()
        if not any(token in text for token in ("eye", "iris", "pupil", "cornea", "lash", "brow", "highlight")):
            continue
        world = base.world_vertices(obj)
        original = world.copy()
        side = 0 if float(world[:, 0].mean()) < middle else 1
        relative = world - source[side]
        if any(token in text for token in ("iris", "pupil", "highlight", "eye_dots")):
            scale = np.array([0.98, 0.96, 0.98])
            offset = np.array([0.0, forward_sign * 0.0007, -0.0002])
            kind = "iris"
        elif "brow" in text:
            scale = np.array([0.90, 0.95, 0.82])
            offset = np.array([0.0, -forward_sign * 0.0004, -0.0035])
            kind = "brow"
        elif "lash" in text:
            scale = np.array([0.84, 0.95, 0.70])
            offset = np.array([0.0, forward_sign * 0.0006, -0.0006])
            kind = "lash"
        else:
            scale = np.array([0.82, 0.95, 0.66])
            offset = np.zeros(3)
            kind = "globe"
        result = target[side] + relative * scale + offset
        base.set_mesh_local_array(obj, base.to_local(obj, result))
        displacement = np.linalg.norm(result - original, axis=1)
        current = float(displacement.max()) if len(displacement) else 0.0
        maximum = max(maximum, current)
        moved.append({"object": obj.name, "kind": kind, "max_m": current})
    return {"objects": moved, "max_displacement_m": maximum}


def style_materials(scene) -> dict:
    hair_main = appearance.material("AINA_Balanced_Silver_Hair", (0.18, 0.24, 0.35, 1.0), 0.38, 0.05)
    hair_strand = appearance.material("AINA_Balanced_Silver_Strands", (0.30, 0.39, 0.54, 1.0), 0.31, 0.04)
    hairband = appearance.material("AINA_Balanced_Pearl", (0.42, 0.54, 0.72, 1.0), 0.25, 0.10)
    brow = appearance.material("AINA_Balanced_Brow", (0.10, 0.12, 0.18, 1.0), 0.44)
    lash = appearance.material("AINA_Balanced_Lash", (0.008, 0.010, 0.018, 1.0), 0.34)
    iris = appearance.material("AINA_Balanced_Iris", (0.025, 0.17, 0.20, 1.0), 0.20)
    pupil = appearance.material("AINA_Balanced_Pupil", (0.001, 0.003, 0.007, 1.0), 0.16)
    sclera = appearance.material("AINA_Balanced_Sclera", (0.54, 0.58, 0.66, 1.0), 0.32)
    styled = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        lower = obj.name.lower()
        if "hair_ponytail" in lower:
            obj.hide_render = True
            obj.hide_viewport = True
        elif "hairband" in lower:
            appearance.set_all_materials(obj, hairband); styled.append(obj.name)
        elif "hair_strand" in lower or ("aina_mirror" in lower and "hair" in lower):
            appearance.set_all_materials(obj, hair_strand); styled.append(obj.name)
        elif "hair" in lower:
            appearance.set_all_materials(obj, hair_main); styled.append(obj.name)
        elif "eyebrow" in lower:
            appearance.set_all_materials(obj, brow); styled.append(obj.name)
        elif "eyelash" in lower:
            appearance.set_all_materials(obj, lash); styled.append(obj.name)
        elif "iris" in lower:
            appearance.set_all_materials(obj, iris); styled.append(obj.name)
        elif "pupil" in lower or "eye_dots" in lower:
            appearance.set_all_materials(obj, pupil); styled.append(obj.name)
        elif lower == "geo-rain-eyes":
            appearance.set_all_materials(obj, sclera); styled.append(obj.name)
    return {"styled_objects": styled}


def transform_hair(scene, face_x: float, eye_z: float, forward_sign: float) -> dict:
    moved = []
    maximum = 0.0
    for obj in scene.objects:
        if obj.type != "MESH" or not len(obj.data.vertices) or not base.is_hair(obj):
            continue
        world = base.world_vertices(obj)
        original = world.copy()
        pivot = np.array([face_x, float(np.median(world[:, 1])), eye_z + 0.012])
        relative = world - pivot
        top = v3.smoothstep01((world[:, 2] - (eye_z - 0.005)) / 0.135)
        relative[:, 0] *= 1.0 - 0.07 * top
        relative[:, 2] *= 1.0 - 0.08 * top
        relative[:, 1] *= 0.96
        result = pivot + relative
        result[:, 1] += -forward_sign * 0.0010 * top
        base.set_mesh_local_array(obj, base.to_local(obj, result))
        displacement = np.linalg.norm(result - original, axis=1)
        current = float(displacement.max()) if len(displacement) else 0.0
        maximum = max(maximum, current)
        moved.append(obj.name)
    return {"objects": moved, "max_displacement_m": maximum}


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)
    source_report = {}
    if args.source_report and args.source_report.exists():
        source_report = json.loads(args.source_report.read_text(encoding="utf-8"))

    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH" and len(obj.data.vertices)]
    base.reset_shape_keys(meshes)
    armature = base.find_armature(scene)
    head_bone = base.find_head_bone(armature)
    head_point = base.bone_world_point(armature, head_bone)
    skin, skin_report = base.identify_skin(scene, head_point)
    original_deltas = base.capture_shape_deltas(skin)
    original_vertices = len(skin.data.vertices)
    original_triangles = sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons)
    initial_world = base.world_vertices(skin)
    face_x = float(0.5 * (initial_world[:, 0].min() + initial_world[:, 0].max()))
    eyes_before = v3.true_eye_centres(scene, face_x)
    if len(eyes_before) != 2:
        raise RuntimeError(f"Expected two true eye centres, got {len(eyes_before)}")
    character_height = skin_report["character_height_m"]
    head_ids, _, _, _ = base.head_region(skin, head_point, eyes_before, character_height)
    forward_sign = -1.0 if np.mean(eyes_before, axis=0)[1] < initial_world.mean(axis=0)[1] else 1.0

    before = base.world_vertices(skin)
    balanced = balanced_skin_pass(skin, head_ids, eyes_before, forward_sign)
    after = base.world_vertices(skin)
    related = base.move_related_objects(scene, skin, before, after, eyes_before)
    eye_targets = [np.asarray(point, dtype=np.float64) for point in balanced["eye_targets"]]
    eyes_report = transform_eye_system(scene, eyes_before, eye_targets, forward_sign)
    hair_report = transform_hair(scene, face_x, float(np.mean(eye_targets, axis=0)[2]), forward_sign)
    material_report = style_materials(scene)
    bpy.context.view_layer.update()

    preservation = base.validate_shape_deltas(skin, original_deltas)
    triangles = sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons)
    if len(skin.data.vertices) != original_vertices or triangles != original_triangles:
        raise RuntimeError("Skin topology changed during balanced pass")

    final_world = base.world_vertices(skin)
    final_face_x = float(0.5 * (final_world[:, 0].min() + final_world[:, 0].max()))
    final_eyes = v3.true_eye_centres(scene, final_face_x)
    if len(final_eyes) != 2:
        final_eyes = eye_targets
    cameras, camera_report = base.setup_cameras(
        scene, skin, head_ids, final_eyes, head_point, character_height
    )
    appearance.soften_lighting(scene)
    try:
        scene.view_settings.exposure = -0.42
    except Exception:
        pass
    renders = appearance.render_full_suite(scene, cameras, skin, args.out)

    blend_path = args.out / "AINA_RAIN_IDENTITY_BALANCED_V5.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_RAIN_IDENTITY_BALANCED_V5.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    report = {
        "product": "AINA Rain Identity Balanced v5",
        "source": "AINA Rain Identity Reconstruction v3",
        "source_artifact_product": source_report.get("product"),
        "source_character": "Blender Studio Rain v3",
        "source_license": "CC BY 4.0",
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "skin_topology_changed": False,
        "armature_preserved": True,
        "skin_weights_preserved": True,
        "uvs_preserved": True,
        "skin_object": skin.name,
        "vertices": len(skin.data.vertices),
        "triangles": triangles,
        "source_vertices": original_vertices,
        "source_triangles": original_triangles,
        "head_region_vertices": len(head_ids),
        "balanced_identity_pass": balanced,
        "related_anatomy": related,
        "eye_system_transform": eyes_report,
        "hair_transform": hair_report,
        "material_restoration": material_report,
        "shape_key_preservation": preservation,
        "source_shape_key_count": len(original_deltas),
        "camera": camera_report,
        "total_skin_displacement_max_m": float(np.linalg.norm(final_world - initial_world, axis=1).max()),
        "total_skin_displacement_rms_m": float(np.sqrt(np.mean(np.sum((final_world - initial_world) ** 2, axis=1)))),
        "renders": renders,
        "identity_lock": False,
        "visual_identity_lock": False,
        "production_release": False,
        "candidate": True,
        "vrm_exported": False,
        "next_gate": "Directly inspect the actual balanced v5 front, 3Q, side, expressions and naked clay. Continue on this exact topology if the approved AINA identity is not yet stable.",
        "files": {"blend": str(blend_path), "glb": str(glb_path)},
    }
    (qa / "AINA_RAIN_IDENTITY_BALANCED_V5_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
