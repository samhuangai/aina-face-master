#!/usr/bin/env python3
"""Clean Blender reimport verification for the exact AINA VRM v7 bytes."""
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
import aina_rain_vrm_production_v7 as production


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--production-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes, bpy.data.curves, bpy.data.materials,
        bpy.data.cameras, bpy.data.lights,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def find_arkit_skin(scene):
    best = None
    best_count = -1
    expected = set(production.ARKIT_52)
    for obj in scene.objects:
        if obj.type != "MESH" or not obj.data.shape_keys:
            continue
        names = {key.name for key in obj.data.shape_keys.key_blocks}
        count = len(expected.intersection(names))
        if count > best_count:
            best = obj
            best_count = count
    if best is None or best_count < 52:
        raise RuntimeError(f"No reimported Mesh contains 52 ARKit controls; best={best_count}")
    return best


def shape_key_status(skin) -> tuple[list[str], dict[str, float]]:
    keys = skin.data.shape_keys.key_blocks
    basis = base.key_array(keys.get("Basis") or keys[0])
    missing = []
    maxima = {}
    for name in production.ARKIT_52:
        key = keys.get(name)
        if key is None:
            missing.append(name)
            continue
        delta = base.key_array(key) - basis
        maxima[name] = float(np.linalg.norm(delta, axis=1).max())
    return missing, maxima


def set_keys(skin, values: dict[str, float]) -> None:
    keys = skin.data.shape_keys.key_blocks
    for key in keys:
        key.value = 0.0
    for name, value in values.items():
        key = keys.get(name)
        if key:
            key.value = value
    bpy.context.view_layer.update()


def render_suite(scene, skin, cameras, out: Path) -> dict:
    preview = out / "Preview"
    preview.mkdir(parents=True, exist_ok=True)
    outputs = {"neutral": {}, "expressions": {}}
    set_keys(skin, {})
    for view in ("front", "three_quarter", "side"):
        path = preview / f"AINA_VRM_REIMPORT_NEUTRAL_{view.upper()}.png"
        base.render(scene, cameras[view], path)
        outputs["neutral"][view] = str(path)

    cases = {
        "happy": {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0, "cheekSquintLeft": 0.3, "cheekSquintRight": 0.3},
        "angry": {"browDownLeft": 1.0, "browDownRight": 1.0, "mouthFrownLeft": 0.55, "mouthFrownRight": 0.55},
        "sad": {"browInnerUp": 0.8, "mouthFrownLeft": 1.0, "mouthFrownRight": 1.0},
        "surprised": {"eyeWideLeft": 0.9, "eyeWideRight": 0.9, "jawOpen": 0.72},
        "blink": {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0},
        "aa": {"jawOpen": 1.0, "mouthFunnel": 0.22},
        "ou": {"mouthPucker": 1.0, "mouthFunnel": 0.85},
    }
    for label, values in cases.items():
        set_keys(skin, values)
        path = preview / f"AINA_VRM_REIMPORT_{label.upper()}.png"
        base.render(scene, cameras["front"], path)
        outputs["expressions"][label] = {"file": str(path), "shape_keys": values}
    set_keys(skin, {})
    return outputs


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)
    production_report = json.loads(args.production_report.read_text(encoding="utf-8"))

    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(args.source))
    bpy.context.view_layer.update()
    scene = bpy.context.scene
    skin = find_arkit_skin(scene)
    missing, maxima = shape_key_status(skin)
    nonzero = [name for name, value in maxima.items() if value > 1.0e-8]

    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("Clean reimport contains no armature")
    armature = max(armatures, key=lambda obj: len(obj.data.bones))
    head_bone = base.find_head_bone(armature)
    head_point = base.bone_world_point(armature, head_bone)
    world = base.world_vertices(skin)
    face_x = float(0.5 * (world[:, 0].min() + world[:, 0].max()))
    eyes = v3.true_eye_centres(scene, face_x)
    if len(eyes) != 2:
        eye_z = float(np.quantile(world[:, 2], 0.73))
        y = float(np.quantile(world[:, 1], 0.10))
        spread = 0.033
        eyes = [
            np.array([face_x - spread, y, eye_z]),
            np.array([face_x + spread, y, eye_z]),
        ]
    bounds = np.concatenate([base.world_vertices(obj) for obj in scene.objects if obj.type == "MESH" and len(obj.data.vertices)], axis=0)
    character_height = max(float(bounds[:, 2].max() - bounds[:, 2].min()), 0.5)
    head_ids, _, _, _ = base.head_region(skin, head_point, eyes, character_height)
    cameras, camera_report = base.setup_cameras(
        scene, skin, head_ids, eyes, head_point, character_height
    )
    appearance.soften_lighting(scene)
    renders = render_suite(scene, skin, cameras, args.out)

    reimport_blend = args.out / "AINA_VRM_REIMPORT_VERIFIED.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(reimport_blend))
    source_binary = production_report.get("binary_qa", {})
    preset_counts = source_binary.get("preset_counts", {})
    expected_presets = set(production.PRESET_ORDER)
    preset_ok = set(preset_counts) == expected_presets and all(
        count > 0 for name, count in preset_counts.items() if name != "neutral"
    ) and preset_counts.get("neutral") == 0
    humanoid_count = len(source_binary.get("humanoid_bones", {}))
    spring_nodes = len(source_binary.get("hair_nodes", []))
    pass_gate = (
        not missing
        and len(nonzero) == 52
        and preset_ok
        and humanoid_count >= 17
        and spring_nodes >= 1
        and len(armature.data.bones) >= 17
    )
    report = {
        "product": "AINA Rain VRM v7 Clean Reimport QA",
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "source": str(args.source),
        "skin_object": skin.name,
        "vertices": len(skin.data.vertices),
        "triangles": sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons),
        "arkit_expected": 52,
        "arkit_present": 52 - len(missing),
        "arkit_nonzero": len(nonzero),
        "arkit_missing": missing,
        "arkit_max_delta_m": maxima,
        "preset_expected": 18,
        "preset_present": len(preset_counts),
        "preset_counts": preset_counts,
        "preset_gate": preset_ok,
        "humanoid_bone_mappings": humanoid_count,
        "reimport_armature_bones": len(armature.data.bones),
        "spring_hair_nodes": spring_nodes,
        "camera": camera_report,
        "renders": renders,
        "pass": pass_gate,
        "stage": "reimport_passed" if pass_gate else "reimport_failed",
        "identity_lock": False,
        "visual_identity_lock": False,
        "production_release": False,
        "candidate": True,
        "technical_release_gate": pass_gate,
        "files": {"blend": str(reimport_blend)},
    }
    (qa / "AINA_VRM_REIMPORT_V7_QA.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not pass_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
