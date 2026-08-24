#!/usr/bin/env python3
"""Finalize the objectively selected AINA Rain identity-search v7 candidate.

This script makes no facial deformation. It opens the selected candidate BLEND,
renders the complete actual Blender beauty/expression/hair-hidden-clay suite,
exports a morph-preserving GLB and records the selection evidence. Locks remain
false until direct visual review accepts the selected front/3Q/profile and
expressions.
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
import aina_rain_identity_convergence_v6 as v6


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selected-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selected_report = json.loads(args.selected_report.read_text(encoding="utf-8"))

    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH" and len(obj.data.vertices)]
    base.reset_shape_keys(meshes)
    armature = base.find_armature(scene)
    head_bone = base.find_head_bone(armature)
    head_point = base.bone_world_point(armature, head_bone)
    skin, skin_report = base.identify_skin(scene, head_point)
    original_deltas = base.capture_shape_deltas(skin)
    vertices = len(skin.data.vertices)
    triangles = sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons)
    world = base.world_vertices(skin)
    face_x = float(0.5 * (world[:, 0].min() + world[:, 0].max()))
    eyes = v3.true_eye_centres(scene, face_x)
    if len(eyes) != 2:
        raise RuntimeError(f"Expected two true eyes in selected v7 candidate, got {len(eyes)}")
    character_height = skin_report["character_height_m"]
    head_ids, _, _, _ = base.head_region(skin, head_point, eyes, character_height)
    cameras = v6.existing_cameras(scene)
    if not cameras:
        cameras, _ = base.setup_cameras(
            scene, skin, head_ids, eyes, head_point, character_height
        )

    preservation = base.validate_shape_deltas(skin, original_deltas)
    renders = appearance.render_full_suite(scene, cameras, skin, args.out)

    blend_path = args.out / "AINA_RAIN_IDENTITY_SELECTED_V7.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_RAIN_IDENTITY_SELECTED_V7.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    report = {
        "product": "AINA Rain Identity Selected v7",
        "source": "AINA Rain Identity Convergence v6 plus regularized candidate search",
        "source_character": "Blender Studio Rain v3",
        "source_license": "CC BY 4.0",
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "skin_topology_changed": False,
        "armature_preserved": True,
        "skin_weights_preserved": True,
        "uvs_preserved": True,
        "skin_object": skin.name,
        "vertices": vertices,
        "triangles": triangles,
        "head_region_vertices": len(head_ids),
        "source_shape_key_count": len(original_deltas),
        "shape_key_preservation": preservation,
        "selection": selection,
        "selected_candidate_report": selected_report,
        "renders": renders,
        "identity_lock": False,
        "visual_identity_lock": False,
        "production_release": False,
        "candidate": True,
        "vrm_exported": False,
        "next_gate": "Directly review the selected actual Blender front, 3Q, profile, happy, sad, blink and clay views. Proceed to 52-control VRM production only if the approved AINA identity remains stable in every view.",
        "files": {"blend": str(blend_path), "glb": str(glb_path)},
    }
    (qa / "AINA_RAIN_IDENTITY_SELECTED_V7_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
