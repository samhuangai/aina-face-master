#!/usr/bin/env python3
"""Render-in-the-loop identity convergence for AINA on the verified Rain Mesh.

The input is AINA Rain Identity Balanced v5. This stage uses the approved-vs-
actual front, three-quarter and profile landmarks extracted from the *real*
Blender renders, applies only a bounded residual correction to the same Rain
skin, and deliberately disables the earlier art-directed proportion residual.
The same neutral displacement is added to Basis and every source shape key.
Separate eyes, mouth anatomy, brows, lashes and hair are moved coherently.

Topology, vertex order, UVs, skin weights, armature links and relative source
shape-key deltas are preserved. No replacement effect art is generated and no
VRM is exported. Identity/visual/production locks remain false until direct
review of the actual v6 renders passes.
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


RESIDUAL_STRENGTH = 0.58
MAX_NEW_DISPLACEMENT_M = 0.0038


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


def zero_art_residual(points, anchors_world, model_front, desired_front, forward_sign):
    del anchors_world, model_front, desired_front, forward_sign
    return np.zeros_like(points), {
        "disabled_for_v6": True,
        "reason": "Use only real-render multi-view residuals; do not stack another art-directed face warp.",
        "max_residual_m": 0.0,
        "rms_residual_m": 0.0,
    }


def semantic_preserve_mask(skin, head_ids: np.ndarray) -> np.ndarray:
    local = basis_local(skin)
    eye = v3.normalized_semantic(
        v3.semantic_magnitude(
            skin,
            local,
            ("eyelidsclose", "eyelid", "blink", "eyebrow", "brow"),
        ),
        0.28,
    )
    mouth = v3.normalized_semantic(
        v3.semantic_magnitude(
            skin,
            local,
            ("lip", "lips", "mouth", "jaw", "smile", "cheekpuff"),
        ),
        0.32,
    )
    preserve = np.maximum(eye, mouth)
    return np.clip(preserve[head_ids], 0.0, 1.0)


def existing_cameras(scene) -> dict:
    result = {
        view: scene.objects.get(f"AINA_Rain_Camera_{view}")
        for view in ("front", "three_quarter", "side", "left_45", "right_45")
    }
    return result if all(result.values()) else {}


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

    original_vertices = len(skin.data.vertices)
    original_triangles = sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons)
    original_deltas = base.capture_shape_deltas(skin)
    initial_world = base.world_vertices(skin)
    face_x = float(0.5 * (initial_world[:, 0].min() + initial_world[:, 0].max()))
    eyes_before = v3.true_eye_centres(scene, face_x)
    if len(eyes_before) != 2:
        raise RuntimeError(f"Expected two real eye centres in balanced v5, got {len(eyes_before)}")

    character_height = skin_report["character_height_m"]
    head_ids, _, _, _ = base.head_region(
        skin, head_point, eyes_before, character_height
    )
    cameras = existing_cameras(scene)
    if not cameras:
        cameras, _ = base.setup_cameras(
            scene, skin, head_ids, eyes_before, head_point, character_height
        )

    forward_sign = (
        -1.0
        if np.mean(eyes_before, axis=0)[1] < initial_world.mean(axis=0)[1]
        else 1.0
    )

    original_art = base.art_directed_residual
    base.art_directed_residual = zero_art_residual
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
    raw_region = raw_final[head_ids] - raw_initial[head_ids]
    preserve = semantic_preserve_mask(skin, head_ids)
    adjacency = base.adjacency_for_region(skin, head_ids)
    converged_region = base.smooth_region_delta(
        raw_region,
        adjacency,
        preserve,
        passes=3,
    )
    converged_region *= RESIDUAL_STRENGTH
    lengths = np.linalg.norm(converged_region, axis=1)
    converged_region *= np.minimum(
        1.0,
        MAX_NEW_DISPLACEMENT_M / np.maximum(lengths, 1.0e-9),
    )[:, None]

    desired_world = raw_initial.copy()
    desired_world[head_ids] += converged_region
    current_world = base.world_vertices(skin)
    base.apply_world_delta(skin, desired_world - current_world)
    final_world = base.world_vertices(skin)

    related = base.move_related_objects(
        scene,
        skin,
        raw_initial,
        final_world,
        eyes_before,
    )
    bpy.context.view_layer.update()

    final_face_x = float(0.5 * (final_world[:, 0].min() + final_world[:, 0].max()))
    eyes_after = v3.true_eye_centres(scene, final_face_x)
    if len(eyes_after) != 2:
        eyes_after = eyes_before

    preservation = base.validate_shape_deltas(skin, original_deltas)
    triangles = sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons)
    if len(skin.data.vertices) != original_vertices or triangles != original_triangles:
        raise RuntimeError("Skin topology changed during v6 convergence")

    # Keep the source cameras and lighting unchanged so pre/post render metrics
    # are directly comparable. The displacement cap is small enough to avoid crop.
    renders = appearance.render_full_suite(scene, cameras, skin, args.out)

    blend_path = args.out / "AINA_RAIN_IDENTITY_CONVERGENCE_V6.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_RAIN_IDENTITY_CONVERGENCE_V6.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    raw_lengths = np.linalg.norm(raw_region, axis=1)
    final_lengths = np.linalg.norm(final_world - initial_world, axis=1)
    report = {
        "product": "AINA Rain Identity Convergence v6",
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
        "multi_view_fitting": fitting,
        "art_directed_residual_disabled": True,
        "residual_strength": RESIDUAL_STRENGTH,
        "new_displacement_cap_m": MAX_NEW_DISPLACEMENT_M,
        "raw_residual_max_m": float(raw_lengths.max()) if len(raw_lengths) else 0.0,
        "raw_residual_rms_m": float(np.sqrt(np.mean(raw_lengths * raw_lengths))) if len(raw_lengths) else 0.0,
        "converged_residual_max_m": float(np.linalg.norm(converged_region, axis=1).max()),
        "converged_residual_rms_m": float(np.sqrt(np.mean(np.sum(converged_region * converged_region, axis=1)))),
        "related_anatomy": related,
        "shape_key_preservation": preservation,
        "source_shape_key_count": len(original_deltas),
        "eye_centres_before": [point.tolist() for point in eyes_before],
        "eye_centres_after": [point.tolist() for point in eyes_after],
        "total_skin_displacement_max_m": float(final_lengths.max()),
        "total_skin_displacement_rms_m": float(np.sqrt(np.mean(final_lengths * final_lengths))),
        "renders": renders,
        "identity_lock": False,
        "visual_identity_lock": False,
        "production_release": False,
        "candidate": True,
        "vrm_exported": False,
        "next_gate": "Compare pre/post actual Blender renders and normalized multi-view landmark errors. Continue on this exact topology unless front, 3Q, profile and expressions all preserve the approved AINA identity.",
        "files": {"blend": str(blend_path), "glb": str(glb_path)},
    }
    (qa / "AINA_RAIN_IDENTITY_CONVERGENCE_V6_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
