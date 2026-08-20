#!/usr/bin/env python3
"""AINA surface-refined real-mesh expression QA.

This stage consumes the real refined FaceVerse OBJ, creates the existing 52
production facial controls on the same topology, rebuilds the visible real 3D
eye system, and renders the actual model under neutral, emotional and viseme
poses. It does not generate replacement effect art and does not export VRM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aina_final_vrm_assembly as base
import aina_visual_eye_system as eye_system


CASES = {
    "neutral": {},
    "happy": dict(base.PRESET_BINDS["happy"]),
    "angry": dict(base.PRESET_BINDS["angry"]),
    "sad": dict(base.PRESET_BINDS["sad"]),
    "surprised": dict(base.PRESET_BINDS["surprised"]),
    "blink": dict(base.PRESET_BINDS["blink"]),
    "aa": dict(base.PRESET_BINDS["aa"]),
    "ih": dict(base.PRESET_BINDS["ih"]),
    "ou": dict(base.PRESET_BINDS["ou"]),
    "ee": dict(base.PRESET_BINDS["ee"]),
    "oh": dict(base.PRESET_BINDS["oh"]),
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--height", type=float, default=1.72)
    return ap.parse_args(argv)


def clear_scene() -> None:
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for blocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(blocks):
            if block.users == 0:
                blocks.remove(block)


def material(name: str, color, roughness: float, metallic: float = 0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = tuple(color)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = tuple(color)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    return mat


def build_character(face_path: Path, height: float):
    raw, faces = base.read_obj(face_path)
    mapped = base.map_face_vertices(raw, height)
    roots, groups = base.component_data(len(raw), faces)
    head_root = max(groups, key=lambda root: len(groups[root]))
    eye_roots = sorted(
        [root for root, ids in groups.items() if 650 < len(ids) < 900],
        key=lambda root: float(mapped[groups[root], 0].mean()),
    )
    if len(eye_roots) != 2:
        raise RuntimeError(f"Expected two source-eye components, got {len(eye_roots)}")
    oral_roots = sorted(
        [root for root in groups if root != head_root and root not in eye_roots],
        key=lambda root: len(groups[root]),
        reverse=True,
    )

    skin = material("AINA_Skin", (0.86, 0.73, 0.71, 1.0), 0.47)
    lip = material("AINA_Lip", (0.60, 0.27, 0.31, 1.0), 0.38)
    tooth = material("AINA_Teeth", (0.91, 0.88, 0.83, 1.0), 0.34)
    mouth = material("AINA_MouthInner", (0.27, 0.045, 0.060, 1.0), 0.50)
    eye_white = material("AINA_EyeWhite", (0.96, 0.97, 0.99, 1.0), 0.22)
    iris = material("AINA_Iris", (0.19, 0.42, 0.56, 1.0), 0.17)
    pupil = material("AINA_Pupil", (0.006, 0.010, 0.018, 1.0), 0.18)

    keep = np.array([roots[int(face[0])] not in set(eye_roots) for face in faces], dtype=bool)
    head_faces = faces[keep]
    head = base.mesh_object("AINA_SurfaceRefined_Head", mapped, head_faces)
    head.data.materials.append(skin)
    head.data.materials.append(tooth)
    head.data.materials.append(mouth)
    head.data.materials.append(lip)
    face_roots = [roots[int(face[0])] for face in faces[keep]]
    oral_big = set(oral_roots[:2])
    for polygon, root in zip(head.data.polygons, face_roots):
        polygon.material_index = 0 if root == head_root else (1 if root in oral_big else 2)
        polygon.use_smooth = True

    lm = mapped[base.K]
    mouth_center = lm[48:60].mean(0)
    for polygon in head.data.polygons:
        if polygon.material_index != 0:
            continue
        center = np.mean([np.asarray(head.data.vertices[i].co) for i in polygon.vertices], axis=0)
        q = ((center[0] - mouth_center[0]) / 0.0245) ** 2 + ((center[2] - mouth_center[2]) / 0.0085) ** 2
        if q < 1.0 and center[1] < mouth_center[1] + 0.012:
            polygon.material_index = 3

    tongue_ids = groups[oral_roots[-1]] if oral_roots else np.array([], dtype=np.int32)
    shape_stats = base.create_shape_keys(head, mapped, tongue_ids)

    visible_objects = [head]
    centers = {"R": lm[36:42].mean(0), "L": lm[42:48].mean(0)}
    for side in ("R", "L"):
        center = centers[side].copy()
        center[1] = -0.00035
        sclera = eye_system._almond(f"AINA_Eye_{side}", center, eye_white, side)
        iris_center = center.copy()
        iris_center[1] = -0.01185
        iris_obj = eye_system._disc(f"AINA_Iris_{side}", iris_center, 0.00565, iris, side, pupil=False)
        pupil_center = center.copy()
        pupil_center[1] = -0.01245
        pupil_obj = eye_system._disc(f"AINA_Pupil_{side}", pupil_center, 0.00220, pupil, side, pupil=True)
        visible_objects.extend((sclera, iris_obj, pupil_obj))

    return head, visible_objects, mapped, shape_stats


def reset_shapes(objects) -> None:
    for obj in objects:
        if not getattr(obj.data, "shape_keys", None):
            continue
        for key in obj.data.shape_keys.key_blocks:
            key.value = 0.0


def apply_case(objects, values) -> None:
    reset_shapes(objects)
    for obj in objects:
        keys = getattr(obj.data, "shape_keys", None)
        if not keys:
            continue
        for key_name, value in values.items():
            if key_name in keys.key_blocks:
                keys.key_blocks[key_name].value = float(value)


def setup_scene(out: Path):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.92, 0.94, 0.97)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = 0.15
    except Exception:
        pass

    def area(name, location, energy, size, target=(0, 0, 1.61)):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()

    area("AINA_Key", (1.35, -1.75, 2.25), 650, 2.5)
    area("AINA_Fill", (-1.45, -1.55, 1.90), 320, 2.7)
    area("AINA_Rim", (0, 1.65, 2.25), 390, 2.3)
    area("AINA_FaceSoft", (0, -2.10, 1.60), 90, 3.0)

    camera_data = bpy.data.cameras.new("AINA_Expression_Camera")
    camera = bpy.data.objects.new("AINA_Expression_Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera.data.lens = 86

    preview = out / "Preview"
    preview.mkdir(parents=True, exist_ok=True)
    return scene, camera, preview


def render_case(scene, camera, preview: Path, objects, case_name: str, values, three_q=False):
    apply_case(objects, values)
    if three_q:
        camera.location = (0.34, -0.93, 1.62)
        target = (0, 0, 1.605)
        suffix = "3Q"
    else:
        camera.location = (0, -0.99, 1.615)
        target = (0, 0, 1.610)
        suffix = "FRONT"
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    path = preview / f"AINA_EXPR_{case_name.upper()}_{suffix}.png"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    return path


def expression_metrics(head, mapped):
    key_blocks = head.data.shape_keys.key_blocks
    basis = np.asarray([point.co[:] for point in key_blocks["Basis"].data], dtype=np.float64)
    result = {}
    eye_anchor = base.K[36:48]
    nose_anchor = base.K[27:36]
    jaw_anchor = base.K[:17]
    mouth_anchor = base.K[48:68]

    for case_name, values in CASES.items():
        posed = basis.copy()
        for key_name, weight in values.items():
            if key_name not in key_blocks:
                continue
            coords = np.asarray([point.co[:] for point in key_blocks[key_name].data], dtype=np.float64)
            posed += float(weight) * (coords - basis)
        displacement = np.linalg.norm(posed - basis, axis=1)
        result[case_name] = {
            "max_m": float(displacement.max()),
            "rms_m": float(np.sqrt(np.mean(displacement * displacement))),
            "moved_vertices": int(np.sum(displacement > 1e-5)),
            "eye_anchor_max_m": float(displacement[eye_anchor].max()),
            "nose_anchor_max_m": float(displacement[nose_anchor].max()),
            "jaw_anchor_max_m": float(displacement[jaw_anchor].max()),
            "mouth_anchor_max_m": float(displacement[mouth_anchor].max()),
        }
    return result


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "QA").mkdir(exist_ok=True)
    clear_scene()

    head, visible_objects, mapped, shape_stats = build_character(args.face, args.height)
    scene, camera, preview = setup_scene(out)
    rendered = []
    for case_name, values in CASES.items():
        rendered.append(render_case(scene, camera, preview, visible_objects, case_name, values, False))
    rendered.append(render_case(scene, camera, preview, visible_objects, "neutral", CASES["neutral"], True))
    rendered.append(render_case(scene, camera, preview, visible_objects, "happy", CASES["happy"], True))
    reset_shapes(visible_objects)

    blend_path = out / "AINA_SURFACE_EXPRESSION_QA.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    key_names = [key.name for key in head.data.shape_keys.key_blocks if key.name != "Basis"]
    missing = sorted(set(base.SHAPE_KEYS) - set(key_names))
    extra = sorted(set(key_names) - set(base.SHAPE_KEYS))
    metrics = expression_metrics(head, mapped)
    render_missing = [str(path) for path in rendered if not path.exists() or path.stat().st_size < 5000]

    viseme_names = {"aa", "ih", "ou", "ee", "oh"}
    automated_pass = (
        len(key_names) == 52
        and not missing
        and not extra
        and not render_missing
        and all(np.isfinite(item["max_m"]) and item["max_m"] < 0.035 for item in metrics.values())
        and all(metrics[name]["eye_anchor_max_m"] < 0.0015 for name in viseme_names)
        and all(metrics[name]["nose_anchor_max_m"] < 0.0015 for name in viseme_names)
        and metrics["blink"]["mouth_anchor_max_m"] < 0.0010
    )

    qa = {
        "product": "AINA Surface-Refined Expression QA",
        "face_source": str(args.face),
        "real_mesh": True,
        "replacement_effect_art_generated": False,
        "topology_changed": False,
        "shape_control_count": len(key_names),
        "shape_controls_expected": len(base.SHAPE_KEYS),
        "missing_shape_controls": missing,
        "extra_shape_controls": extra,
        "shape_control_stats": shape_stats,
        "rendered_cases": [path.name for path in rendered],
        "missing_or_small_renders": render_missing,
        "expression_metrics": metrics,
        "automated_expression_gate": automated_pass,
        "visual_identity_lock": False,
        "next_gate": "Manually inspect every real Blender expression render. Only then permit VRM packaging.",
        "files": {
            "blend": str(blend_path),
            "blend_bytes": blend_path.stat().st_size,
            "preview_dir": str(preview),
        },
    }
    report = out / "QA" / "AINA_SURFACE_EXPRESSION_QA.json"
    report.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps({
        "shape_controls": len(key_names),
        "render_count": len(rendered),
        "automated_expression_gate": automated_pass,
        "report": str(report),
    }, indent=2))
    if not automated_pass:
        raise SystemExit("AINA surface expression automated gate failed")


if __name__ == "__main__":
    main()
