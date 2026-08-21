#!/usr/bin/env python3
"""Second bounded convergence pass on the same real Rain AINA topology.

The first Rain Identity Master candidate is used as the only source. This pass
re-detects its actual front/3Q/profile landmarks, applies a smaller multi-view
residual, smooths only the newly added displacement, and preserves topology,
UVs, weights, armature relationships and all existing shape-key deltas.
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


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def gentle_art(points, anchors_world, model_front, desired_front, forward_sign):
    delta, report = ORIGINAL_ART(
        points, anchors_world, model_front, desired_front, forward_sign
    )
    delta *= 0.55
    report = dict(report)
    report["v2_strength"] = 0.55
    report["max_residual_m"] = float(np.linalg.norm(delta, axis=1).max())
    report["rms_residual_m"] = float(
        np.sqrt(np.mean(np.sum(delta * delta, axis=1)))
    )
    return delta, report


ORIGINAL_ART = base.art_directed_residual


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)
    data = json.loads(args.landmarks.read_text(encoding="utf-8"))

    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH" and len(obj.data.vertices)]
    base.reset_shape_keys(meshes)
    armature = base.find_armature(scene)
    head_bone = base.find_head_bone(armature)
    head_point = base.bone_world_point(armature, head_bone)
    skin, skin_report = base.identify_skin(scene, head_point)
    character_height = skin_report["character_height_m"]
    eye_objects = base.eye_objects(scene, head_point, character_height)
    eyes_before = base.eye_centres(eye_objects)
    head_ids, _, _, _ = base.head_region(
        skin, head_point, eyes_before, character_height
    )
    cameras = {
        view: bpy.data.objects.get(f"AINA_Rain_Camera_{view}")
        for view in ("front", "three_quarter", "side", "left_45", "right_45")
    }
    if not all(cameras.values()):
        cameras, camera_report = base.setup_cameras(
            scene, skin, head_ids, eyes_before, head_point, character_height
        )
    else:
        eye_average = np.mean(eyes_before, axis=0) if eyes_before else head_point
        skin_points = base.world_vertices(skin)[head_ids]
        centre = skin_points.mean(axis=0)
        camera_report = {
            "forward_sign_y": -1.0 if eye_average[1] < centre[1] else 1.0,
            "reused_candidate_cameras": True,
        }

    # Capture the exact candidate before this residual so only the new movement
    # is smoothed. Existing identity detail is never globally relaxed.
    before_local = (
        base.key_array(skin.data.shape_keys.key_blocks.get("Basis"))
        if skin.data.shape_keys
        else base.mesh_local_array(skin)
    )
    before_world = base.to_world(skin, before_local)

    base.art_directed_residual = gentle_art
    fitting = base.fit_identity(
        scene,
        skin,
        head_ids,
        cameras,
        data,
        float(camera_report["forward_sign_y"]),
    )
    base.art_directed_residual = ORIGINAL_ART

    raw_initial = fitting.pop("initial_world")
    raw_final = fitting.pop("final_world")
    new_delta = raw_final[head_ids] - raw_initial[head_ids]
    adjacency = base.adjacency_for_region(skin, head_ids)
    smoothed = base.smooth_region_delta(
        new_delta,
        adjacency,
        np.zeros(len(new_delta), dtype=np.float64),
        passes=2,
    )
    lengths = np.linalg.norm(smoothed, axis=1)
    smoothed *= np.minimum(1.0, 0.0040 / np.maximum(lengths, 1e-9))[:, None]
    correction = np.zeros_like(raw_final)
    correction[head_ids] = smoothed - new_delta
    base.apply_world_delta(skin, correction)

    after_local = (
        base.key_array(skin.data.shape_keys.key_blocks.get("Basis"))
        if skin.data.shape_keys
        else base.mesh_local_array(skin)
    )
    after_world = base.to_world(skin, after_local)
    related = base.move_related_objects(
        scene, skin, before_world, after_world, eyes_before
    )
    bpy.context.view_layer.update()
    eyes_after = base.eye_centres(
        base.eye_objects(scene, head_point, character_height)
    )
    renders = base.render_final_suite(scene, cameras, skin, args.out)

    blend_path = args.out / "AINA_RAIN_IDENTITY_MASTER_V2.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_RAIN_IDENTITY_MASTER_V2.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    final_total = after_world - before_world
    report = {
        "product": "AINA Rain Identity Master v2",
        "source": "AINA Rain Identity Master Candidate",
        "source_character": "Blender Studio Rain v3",
        "source_license": "CC BY 4.0",
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "same_topology_as_candidate": True,
        "topology_changed": False,
        "armature_preserved": True,
        "skin_weights_preserved": True,
        "uvs_preserved": True,
        "skin_object": skin.name,
        "vertices": len(skin.data.vertices),
        "triangles": sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons),
        "head_region_vertices": len(head_ids),
        "fitting": fitting,
        "v2_new_displacement_max_m": float(np.linalg.norm(final_total, axis=1).max()),
        "v2_new_displacement_rms_m": float(
            np.sqrt(np.mean(np.sum(final_total * final_total, axis=1)))
        ),
        "related_anatomy": related,
        "eye_centres_before": [point.tolist() for point in eyes_before],
        "eye_centres_after": [point.tolist() for point in eyes_after],
        "renders": renders,
        "identity_lock": False,
        "visual_identity_lock": False,
        "candidate": True,
        "vrm_exported": False,
        "next_gate": "Inspect the actual v2 front, 3Q, profile, clay and available expression renders. Do not export VRM until the real face is visually accepted.",
        "files": {"blend": str(blend_path), "glb": str(glb_path)},
    }
    (qa / "AINA_RAIN_IDENTITY_MASTER_V2_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
