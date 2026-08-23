#!/usr/bin/env python3
"""Build an AINA approved-appearance candidate on the verified Rain v2 Mesh.

The previous over-sculpted visual candidate is intentionally not used. This
stage starts from the successful Rain Identity Master v2 BLEND, preserves its
head topology, UVs, weights, armature and 23 source shape-key deltas, repairs the
bilateral eye/brow/lash/cornea geometry, builds explicit visible iris/pupil
geometry, restores a silver production updo from Rain's existing real hair
meshes, and fixes the neutral mouth-interior intersection. It generates no
replacement effect art and exports no VRM.
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
import aina_rain_identity_visual_lock as visual


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def material(name: str, color, roughness=0.35, metallic=0.0):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = tuple(color)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
    if shader:
        shader.inputs["Base Color"].default_value = tuple(color)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metallic
    try:
        mat.use_backface_culling = False
    except Exception:
        pass
    return mat


def set_all_materials(obj, mat) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
        polygon.use_smooth = True


def one_sided(obj, face_x: float) -> bool:
    points = base.world_vertices(obj)
    left = int(np.sum(points[:, 0] < face_x - 0.002))
    right = int(np.sum(points[:, 0] > face_x + 0.002))
    return min(left, right) <= max(4, int(0.08 * len(points)))


def repair_mirrors(scene, face_x: float, forward_sign: float) -> dict:
    mirrored = []
    for name in (
        "GEO-rain-eye_cornea",
        "GEO-rain-eye_dots",
        "GEO-rain-eyebrows",
        "GEO-rain-eyelashes",
        "GEO-rain-hair_strand",
    ):
        obj = scene.objects.get(name)
        if not obj or obj.type != "MESH" or not len(obj.data.vertices) or not one_sided(obj, face_x):
            continue
        duplicate = visual.mirror_mesh_object(obj, face_x, "AINA_Mirror")
        mirrored.append({"source": obj.name, "mirror": duplicate.name, "vertices": len(duplicate.data.vertices)})

    brow_objects = [obj for obj in scene.objects if obj.type == "MESH" and "eyebrows" in obj.name.lower()]
    lash_objects = [obj for obj in scene.objects if obj.type == "MESH" and "eyelashes" in obj.name.lower()]
    brow_mat = material("AINA_Appearance_Brow", (0.055, 0.060, 0.075, 1.0), 0.42)
    lash_mat = material("AINA_Appearance_Lash", (0.008, 0.010, 0.016, 1.0), 0.34)
    for obj in brow_objects:
        visual.scale_world_geometry(obj, (0.96, 1.0, 0.58), (0.0, forward_sign * 0.0010, -0.0065))
        set_all_materials(obj, brow_mat)
    for obj in lash_objects:
        visual.scale_world_geometry(obj, (1.03, 1.0, 0.92), (0.0, forward_sign * 0.0013, -0.0005))
        set_all_materials(obj, lash_mat)
    return {
        "mirrored_components": mirrored,
        "brow_objects": [obj.name for obj in brow_objects],
        "lash_objects": [obj.name for obj in lash_objects],
    }


def actual_eye_centres(scene, face_x: float) -> list[np.ndarray]:
    centres, _ = visual.true_eye_centres(scene, face_x)
    if len(centres) == 2:
        return centres
    eye = scene.objects.get("GEO-rain-eyes")
    if not eye:
        return []
    points = base.world_vertices(eye)
    left = points[points[:, 0] < face_x]
    right = points[points[:, 0] >= face_x]
    if not len(left) or not len(right):
        return []
    return [left.mean(axis=0), right.mean(axis=0)]


def create_eye_discs(scene, face_x: float, forward_sign: float) -> dict:
    # The lossy Rain GLB conversion carries only one tiny eye-dot plane. Hide the
    # legacy planes and place two explicit flattened spheres at their real depth.
    legacy = [obj for obj in scene.objects if obj.type == "MESH" and "eye_dots" in obj.name.lower()]
    source_centres = [base.world_vertices(obj).mean(axis=0) for obj in legacy if len(obj.data.vertices)]
    source_centres.sort(key=lambda point: point[0])
    for obj in legacy:
        obj.hide_render = True
        obj.hide_viewport = True

    if len(source_centres) >= 2:
        centres = [source_centres[0], source_centres[-1]]
    elif source_centres:
        right = source_centres[0]
        centres = [right.copy(), right.copy()]
        centres[0][0] = 2.0 * face_x - right[0]
        centres.sort(key=lambda point: point[0])
    else:
        eye_centres = actual_eye_centres(scene, face_x)
        if len(eye_centres) != 2:
            raise RuntimeError("Could not determine two Rain eye centres")
        centres = [point.copy() for point in eye_centres]
        for point in centres:
            point[1] += forward_sign * 0.030

    iris_mat = material("AINA_Appearance_Iris", (0.025, 0.145, 0.175, 1.0), 0.20)
    pupil_mat = material("AINA_Appearance_Pupil", (0.001, 0.003, 0.006, 1.0), 0.16)
    highlight_mat = material("AINA_Appearance_EyeHighlight", (0.95, 0.98, 1.0, 1.0), 0.10)
    created = []
    for index, centre in enumerate(centres):
        side = "L" if centre[0] < face_x else "R"
        iris_centre = centre + np.array([0.0, forward_sign * 0.0035, 0.0])
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, location=tuple(iris_centre))
        iris = bpy.context.object
        iris.name = f"AINA_Appearance_Iris_{side}"
        iris.scale = (0.0102, 0.0018, 0.0102)
        set_all_materials(iris, iris_mat)

        pupil_centre = centre + np.array([0.0, forward_sign * 0.0050, 0.0])
        bpy.ops.mesh.primitive_uv_sphere_add(segments=40, ring_count=20, location=tuple(pupil_centre))
        pupil = bpy.context.object
        pupil.name = f"AINA_Appearance_Pupil_{side}"
        pupil.scale = (0.0040, 0.00125, 0.0040)
        set_all_materials(pupil, pupil_mat)

        highlight_centre = centre + np.array([-0.0020, forward_sign * 0.0060, 0.0030])
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, location=tuple(highlight_centre))
        highlight = bpy.context.object
        highlight.name = f"AINA_Appearance_Highlight_{side}"
        highlight.scale = (0.00125, 0.00070, 0.00125)
        set_all_materials(highlight, highlight_mat)
        created.append({"side": side, "centre": centre.tolist(), "iris": iris.name, "pupil": pupil.name, "highlight": highlight.name})
    return {"legacy_hidden": [obj.name for obj in legacy], "created": created}


def fix_mouth_interior(scene, forward_sign: float) -> dict:
    moved = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        lower = obj.name.lower()
        if not any(token in lower for token in ("gum", "tongue", "teeth", "mouth")):
            continue
        before = base.world_vertices(obj).mean(axis=0)
        visual.scale_world_geometry(obj, (1.0, 1.0, 1.0), (0.0, -forward_sign * 0.014, 0.0))
        after = base.world_vertices(obj).mean(axis=0)
        moved.append({"object": obj.name, "shift": (after - before).tolist()})
    return {"objects": moved}


def style_hair(scene, face_x: float) -> dict:
    silver = material("AINA_Silver_Hair", (0.63, 0.68, 0.78, 1.0), 0.30, 0.04)
    pearl = material("AINA_Pearl_Hairband", (0.58, 0.68, 0.82, 1.0), 0.24, 0.10)
    visible = []
    hidden = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        lower = obj.name.lower()
        if "hair_ponytail" in lower:
            obj.hide_render = True
            obj.hide_viewport = True
            hidden.append(obj.name)
        elif "hairband" in lower:
            obj.hide_render = False
            obj.hide_viewport = False
            set_all_materials(obj, pearl)
            visible.append(obj.name)
        elif "hair" in lower or "strand" in lower:
            obj.hide_render = False
            obj.hide_viewport = False
            set_all_materials(obj, silver)
            visible.append(obj.name)
    return {"visible": visible, "hidden": hidden}


def soften_lighting(scene) -> None:
    scene.world.color = (0.028, 0.034, 0.050)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = -0.85
    except Exception:
        pass
    for obj in scene.objects:
        if obj.type == "LIGHT" and obj.name.startswith("AINA_Rain_"):
            obj.data.energy *= 0.38


def render_full_suite(scene, cameras, skin, out: Path) -> dict:
    preview = out / "Preview"
    preview.mkdir(exist_ok=True)
    outputs = {"full_beauty": {}, "naked_clay": {}, "expressions": {}}
    base.reset_shape_keys([obj for obj in scene.objects if obj.type == "MESH"])
    for view in ("front", "three_quarter", "side", "left_45", "right_45"):
        path = preview / f"AINA_RAIN_APPEARANCE_{view.upper()}.png"
        base.render(scene, cameras[view], path)
        outputs["full_beauty"][view] = str(path)

    if skin.data.shape_keys:
        keys = skin.data.shape_keys.key_blocks
        cases = {
            "happy": ("Smile.L", "Smile.R"),
            "sad": ("LipsAdjust",),
            "blink": ("EyelidsClose.L", "EyelidsClose.R"),
        }
        for label, names in cases.items():
            base.reset_shape_keys([skin])
            activated = []
            for name in names:
                key = keys.get(name)
                if key:
                    key.value = 1.0
                    activated.append(name)
            if activated:
                path = preview / f"AINA_RAIN_APPEARANCE_{label.upper()}.png"
                base.render(scene, cameras["front"], path)
                outputs["expressions"][label] = {"file": str(path), "shape_keys": activated}
        base.reset_shape_keys([skin])

    # Hair-hidden clay checks use balanced dark clay rather than a blown-out white material.
    visibility = {}
    for obj in scene.objects:
        if obj.type == "MESH" and base.is_hair(obj):
            visibility[obj.name] = (obj.hide_render, obj.hide_viewport)
            obj.hide_render = True
            obj.hide_viewport = True
    clay = material("AINA_Appearance_Clay", (0.24, 0.28, 0.36, 1.0), 0.58)
    eye = material("AINA_Appearance_ClayEye", (0.52, 0.60, 0.72, 1.0), 0.26)
    mouth = material("AINA_Appearance_ClayMouth", (0.07, 0.012, 0.020, 1.0), 0.42)
    old_materials = {}
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        old_materials[obj.name] = [slot.material for slot in obj.material_slots]
        set_all_materials(obj, eye if base.is_eye(obj) else mouth if base.is_mouth(obj) else clay)
    for view in ("front", "three_quarter", "side"):
        path = preview / f"AINA_RAIN_APPEARANCE_CLAY_{view.upper()}.png"
        base.render(scene, cameras[view], path)
        outputs["naked_clay"][view] = str(path)
    for name, mats in old_materials.items():
        obj = scene.objects.get(name)
        if not obj:
            continue
        obj.data.materials.clear()
        for mat in mats:
            if mat:
                obj.data.materials.append(mat)
    for name, state in visibility.items():
        obj = scene.objects.get(name)
        if obj:
            obj.hide_render, obj.hide_viewport = state
    return outputs


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
    original_deltas = base.capture_shape_deltas(skin)
    original_vertices = len(skin.data.vertices)
    original_triangles = sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons)
    world = base.world_vertices(skin)
    face_x = float(0.5 * (world[:, 0].min() + world[:, 0].max()))
    eyes = actual_eye_centres(scene, face_x)
    if len(eyes) != 2:
        raise RuntimeError(f"Expected two Rain eye centres, got {len(eyes)}")
    forward_sign = -1.0 if np.mean(eyes, axis=0)[1] < world.mean(axis=0)[1] else 1.0

    mirror_report = repair_mirrors(scene, face_x, forward_sign)
    bpy.context.view_layer.update()
    eye_report = create_eye_discs(scene, face_x, forward_sign)
    mouth_report = fix_mouth_interior(scene, forward_sign)
    hair_report = style_hair(scene, face_x)
    bpy.context.view_layer.update()

    head_ids, _, _, _ = base.head_region(skin, head_point, eyes, skin_report["character_height_m"])
    cameras, camera_report = base.setup_cameras(
        scene, skin, head_ids, eyes, head_point, skin_report["character_height_m"]
    )
    soften_lighting(scene)
    renders = render_full_suite(scene, cameras, skin, args.out)
    preservation = base.validate_shape_deltas(skin, original_deltas)

    blend_path = args.out / "AINA_RAIN_APPROVED_APPEARANCE_CANDIDATE.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_RAIN_APPROVED_APPEARANCE_CANDIDATE.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    report = {
        "product": "AINA Rain Approved-Appearance Candidate",
        "source": "AINA Rain Identity Master v2",
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
        "triangles": sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons),
        "source_vertices": original_vertices,
        "source_triangles": original_triangles,
        "shape_key_preservation": preservation,
        "face_x": face_x,
        "forward_sign_y": forward_sign,
        "eye_centres": [point.tolist() for point in eyes],
        "bilateral_repair": mirror_report,
        "eye_geometry": eye_report,
        "mouth_interior_fix": mouth_report,
        "silver_updo": hair_report,
        "camera": camera_report,
        "renders": renders,
        "identity_lock": False,
        "visual_identity_lock": False,
        "candidate": True,
        "vrm_exported": False,
        "next_gate": "Inspect full silver-hair beauty and naked clay front/3Q/side. Keep locks false until the real face, eyes, hair and expressions visually match approved AINA.",
        "files": {"blend": str(blend_path), "glb": str(glb_path)},
    }
    (qa / "AINA_RAIN_APPROVED_APPEARANCE_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
