#!/usr/bin/env python3
"""Build one regularized residual candidate for AINA Rain identity search v7.

Each invocation starts from the exact same successful convergence-v6 BLEND,
uses approved-versus-actual front/3Q/profile landmarks, disables art-directed
warps, applies a configurable fraction of the real-render residual, and caps
the added displacement. It renders only the three neutral views needed for
candidate scoring and saves an editable BLEND. Topology, UVs, weights, armature
and every relative source shape-key delta are preserved.
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
import aina_rain_identity_reconstruction_v3 as v3
import aina_rain_identity_convergence_v6 as v6


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--cap", type=float, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def render_neutral_views(scene, cameras: dict, out: Path) -> dict:
    preview = out / "Preview"
    preview.mkdir(parents=True, exist_ok=True)
    base.reset_shape_keys([obj for obj in scene.objects if obj.type == "MESH"])
    result = {}
    for view in ("front", "three_quarter", "side"):
        path = preview / f"AINA_RAIN_SEARCH_{view.upper()}.png"
        base.render(scene, cameras[view], path)
        result[view] = str(path)
    return result


def main() -> None:
    args = parse_args()
    if not (0.0 < args.strength <= 1.0):
        raise ValueError("--strength must be in (0, 1]")
    if not (0.0 < args.cap <= 0.0040):
        raise ValueError("--cap must be in (0, 0.004]")

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
        raise RuntimeError(f"Expected two real eyes, got {len(eyes_before)}")
    character_height = skin_report["character_height_m"]
    head_ids, _, _, _ = base.head_region(skin, head_point, eyes_before, character_height)
    cameras = v6.existing_cameras(scene)
    if not cameras:
        cameras, _ = base.setup_cameras(
            scene, skin, head_ids, eyes_before, head_point, character_height
        )
    forward_sign = -1.0 if np.mean(eyes_before, axis=0)[1] < initial_world.mean(axis=0)[1] else 1.0

    original_art = base.art_directed_residual
    base.art_directed_residual = v6.zero_art_residual
    try:
        fitting = base.fit_identity(
            scene, skin, head_ids, cameras, data, forward_sign
        )
    finally:
        base.art_directed_residual = original_art

    raw_initial = fitting.pop("initial_world")
    raw_final = fitting.pop("final_world")
    raw_region = raw_final[head_ids] - raw_initial[head_ids]
    preserve = v6.semantic_preserve_mask(skin, head_ids)
    adjacency = base.adjacency_for_region(skin, head_ids)
    candidate_region = base.smooth_region_delta(
        raw_region, adjacency, preserve, passes=3
    )
    candidate_region *= args.strength
    lengths = np.linalg.norm(candidate_region, axis=1)
    candidate_region *= np.minimum(
        1.0, args.cap / np.maximum(lengths, 1.0e-9)
    )[:, None]

    desired_world = raw_initial.copy()
    desired_world[head_ids] += candidate_region
    current_world = base.world_vertices(skin)
    base.apply_world_delta(skin, desired_world - current_world)
    final_world = base.world_vertices(skin)
    related = base.move_related_objects(
        scene, skin, raw_initial, final_world, eyes_before
    )
    bpy.context.view_layer.update()

    preservation = base.validate_shape_deltas(skin, original_deltas)
    triangles = sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons)
    if len(skin.data.vertices) != original_vertices or triangles != original_triangles:
        raise RuntimeError("Skin topology changed in v7 search candidate")

    renders = render_neutral_views(scene, cameras, args.out)
    blend_path = args.out / f"AINA_RAIN_SEARCH_{args.label}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    total = final_world - initial_world
    report = {
        "product": f"AINA Rain Identity Search v7 Candidate {args.label}",
        "source": "AINA Rain Identity Convergence v6",
        "source_artifact_product": source_report.get("product"),
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "skin_topology_changed": False,
        "armature_preserved": True,
        "skin_weights_preserved": True,
        "uvs_preserved": True,
        "vertices": len(skin.data.vertices),
        "triangles": triangles,
        "source_vertices": original_vertices,
        "source_triangles": original_triangles,
        "head_region_vertices": len(head_ids),
        "source_shape_key_count": len(original_deltas),
        "shape_key_preservation": preservation,
        "strength": args.strength,
        "cap_m": args.cap,
        "candidate_residual_max_m": float(np.linalg.norm(candidate_region, axis=1).max()),
        "candidate_residual_rms_m": float(np.sqrt(np.mean(np.sum(candidate_region * candidate_region, axis=1)))),
        "total_skin_displacement_max_m": float(np.linalg.norm(total, axis=1).max()),
        "total_skin_displacement_rms_m": float(np.sqrt(np.mean(np.sum(total * total, axis=1)))),
        "multi_view_fitting": fitting,
        "related_anatomy": related,
        "renders": renders,
        "identity_lock": False,
        "visual_identity_lock": False,
        "production_release": False,
        "candidate": True,
        "vrm_exported": False,
        "files": {"blend": str(blend_path)},
    }
    report_path = qa / f"AINA_RAIN_SEARCH_{args.label}_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
