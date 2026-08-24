#!/usr/bin/env python3
"""Bounded multi-view residual pass for AINA on the verified Rain Mesh.

The source is AINA Rain Identity Balanced v5. This stage uses only the already
approved AINA front, 3/4 and profile references plus fresh renders of the real
Blender model. It applies a small topology-preserving residual to the same
6,868-vertex skin, adds the identical neutral movement to every source shape
key, moves separate eye/mouth/hair anatomy coherently, and renders actual model
QA. No replacement effect art is generated and no VRM is exported here.
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


MAX_RESIDUAL_M = 0.0035
FIT_STRENGTH = 0.30
ART_STRENGTH = 0.08


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def basis_local(skin) -> np.ndarray:
    if skin.data.shape_keys:
        key = skin.data.shape_keys.key_blocks.get("Basis") or skin.data.shape_keys.key_blocks[0]
        return base.key_array(key)
    return base.mesh_local_array(skin)


def feature_preserve(skin, head_ids: np.ndarray, eyes: list[np.ndarray]) -> np.ndarray:
    local = basis_local(skin)
    world = base.to_world(skin, local)
    points = world[head_ids]
    eye_sem = v3.normalized_semantic(
        v3.semantic_magnitude(
            skin,
            local,
            ("eyelid", "eyelidsclose", "blink", "brow", "eyebrow"),
        ),
        0.27,
    )[head_ids]
    lip_sem = v3.normalized_semantic(
        v3.semantic_magnitude(
            skin,
            local,
            ("lip", "lips", "smile", "jaw", "mouth", "cheekpuff"),
        ),
        0.31,
    )[head_ids]
    preserve = np.maximum(eye_sem, lip_sem)
    if len(eyes) == 2:
        face_x = float(np.mean(eyes, axis=0)[0])
        eye_z = float(np.mean(eyes, axis=0)[2])
        centre_y = float(np.mean(eyes, axis=0)[1])
        nose = np.array([face_x, centre_y, eye_z - 0.045])
        nose_w = v3.ellipsoid(points, nose, (0.033, 0.048, 0.060), 1.15)
        preserve = np.maximum(preserve, 0.75 * nose_w)
    return np.clip(preserve, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)
    data = json.loads(args.landmarks.read_text(encoding="utf-8"))
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
        raise RuntimeError(f"Expected two real eye centres, got {len(eyes_before)}")

    character_height = skin_report["character_height_m"]
    head_ids, _, _, _ = base.head_region(
        skin, head_point, eyes_before, character_height
    )
    cameras = {
        view: bpy.data.objects.get(f"AINA_Rain_Camera_{view}")
        for view in ("front", "three_quarter", "side", "left_45", "right_45")
    }
    if not all(cameras.values()):
        cameras, _ = base.setup_cameras(
            scene, skin, head_ids, eyes_before, head_point, character_height
        )
    forward_sign = (
        -1.0
        if np.mean(eyes_before, axis=0)[1] < initial_world.mean(axis=0)[1]
        else 1.0
    )

    preserve = feature_preserve(skin, head_ids, eyes_before)
    original_art = base.art_directed_residual

    def restrained_art(points, anchors_world, model_front, desired_front, sign):
        delta, report = original_art(
            points, anchors_world, model_front, desired_front, sign
        )
        delta *= ART_STRENGTH
        report = dict(report)
        report["v6_art_strength"] = ART_STRENGTH
        report["max_residual_m"] = float(np.linalg.norm(delta, axis=1).max())
        report["rms_residual_m"] = float(
            np.sqrt(np.mean(np.sum(delta * delta, axis=1)))
        )
        return delta, report

    base.art_directed_residual = restrained_art
    try:
        fitting = base.fit_identity(
            scene,
            skin,
            head_ids,
            cameras,
            data,
            forward_sign,
        )
    finally:
        base.art_directed_residual = original_art

    raw_initial = fitting.pop("initial_world")
    raw_final = fitting.pop("final_world")
    raw_delta = raw_final[head_ids] - raw_initial[head_ids]
    adjacency = base.adjacency_for_region(skin, head_ids)
    desired_delta = raw_delta * FIT_STRENGTH
    smoothed = base.smooth_region_delta(
        desired_delta,
        adjacency,
        preserve,
        passes=3,
    )
    lengths = np.linalg.norm(smoothed, axis=1)
    smoothed *= np.minimum(
        1.0,
        MAX_RESIDUAL_M / np.maximum(lengths, 1.0e-9),
    )[:, None]

    correction = np.zeros_like(raw_final)
    correction[head_ids] = smoothed - raw_delta
    base.apply_world_delta(skin, correction)
    final_world = base.world_vertices(skin)
    related = base.move_related_objects(
        scene,
        skin,
        raw_initial,
        final_world,
        eyes_before,
    )
    bpy.context.view_layer.update()

    preservation = base.validate_shape_deltas(skin, original_deltas)
    triangles = sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons)
    if len(skin.data.vertices) != original_vertices or triangles != original_triangles:
        raise RuntimeError("Skin topology changed during v6 residual convergence")

    final_face_x = float(0.5 * (final_world[:, 0].min() + final_world[:, 0].max()))
    eyes_after = v3.true_eye_centres(scene, final_face_x)
    if len(eyes_after) != 2:
        eyes_after = eyes_before
    cameras, camera_report = base.setup_cameras(
        scene,
        skin,
        head_ids,
        eyes_after,
        head_point,
        character_height,
    )
    appearance.soften_lighting(scene)
    try:
        scene.view_settings.exposure = -0.42
    except Exception:
        pass
    renders = appearance.render_full_suite(scene, cameras, skin, args.out)

    blend_path = args.out / "AINA_RAIN_IDENTITY_RESIDUAL_V6.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_RAIN_IDENTITY_RESIDUAL_V6.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    actual_delta = final_world - initial_world
    report = {
        "product": "AINA Rain Identity Residual v6",
        "source": "AINA Rain Identity Balanced v5",
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
        "landmark_views": data.get("available_views", []),
        "fit_strength": FIT_STRENGTH,
        "art_strength": ART_STRENGTH,
        "max_residual_cap_m": MAX_RESIDUAL_M,
        "multi_view_fitting": fitting,
        "raw_fit_delta_max_m": float(np.linalg.norm(raw_delta, axis=1).max()),
        "applied_residual_max_m": float(np.linalg.norm(smoothed, axis=1).max()),
        "applied_residual_rms_m": float(
            np.sqrt(np.mean(np.sum(smoothed * smoothed, axis=1)))
        ),
        "related_anatomy": related,
        "shape_key_preservation": preservation,
        "source_shape_key_count": len(original_deltas),
        "eye_centres_before": [point.tolist() for point in eyes_before],
        "eye_centres_after": [point.tolist() for point in eyes_after],
        "camera": camera_report,
        "total_skin_displacement_max_m": float(np.linalg.norm(actual_delta, axis=1).max()),
        "total_skin_displacement_rms_m": float(
            np.sqrt(np.mean(np.sum(actual_delta * actual_delta, axis=1)))
        ),
        "renders": renders,
        "identity_lock": False,
        "visual_identity_lock": False,
        "production_release": False,
        "candidate": True,
        "vrm_exported": False,
        "next_gate": "Inspect the actual v6 front, 3Q, profile, expressions and naked clay. Proceed to VRM production only when the approved AINA identity is visually stable.",
        "files": {"blend": str(blend_path), "glb": str(glb_path)},
    }
    (qa / "AINA_RAIN_IDENTITY_RESIDUAL_V6_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
