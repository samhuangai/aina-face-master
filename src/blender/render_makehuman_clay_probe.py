#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> dict[str, str]:
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return {raw[i].lstrip("-"): raw[i + 1] for i in range(0, len(raw), 2)}


def bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not points:
        raise RuntimeError("No imported MakeHuman mesh bounds")
    return (
        Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
        Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
    )


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def normalize_vertical(meshes: list[bpy.types.Object]) -> str:
    minimum, maximum = bounds(meshes)
    span = maximum - minimum
    largest = max(range(3), key=lambda index: span[index])
    axis = "XYZ"[largest]
    if axis == "Y":
        for obj in meshes:
            obj.rotation_euler.rotate_axis("X", math.radians(90))
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    elif axis == "X":
        for obj in meshes:
            obj.rotation_euler.rotate_axis("Y", math.radians(-90))
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    for obj in meshes:
        obj.select_set(False)
    return axis


def render(scene: bpy.types.Scene, camera: bpy.types.Object, target: Vector, location: Vector, path: Path) -> None:
    camera.location = location
    look_at(camera, target)
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    source = Path(args["input"]).resolve()
    output = Path(args["out"]).resolve()
    preview = output / "Preview"
    qa = output / "QA"
    preview.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=str(source), forward_axis="NEGATIVE_Z", up_axis="Y")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("MakeHuman OBJ imported without mesh objects")

    source_axis = normalize_vertical(meshes)
    minimum, maximum = bounds(meshes)
    height = maximum.z - minimum.z
    center = (minimum + maximum) * 0.5

    clay = bpy.data.materials.new("AINA_IDENTITY_CLAY")
    clay.diffuse_color = (0.57, 0.59, 0.63, 1.0)
    clay.use_nodes = True
    bsdf = clay.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.57, 0.59, 0.63, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.68
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.20
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(clay)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True

    world = bpy.data.worlds.new("AINA_CLAY_WORLD")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.026, 0.031, 0.043, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.38
    bpy.context.scene.world = world

    head_target = Vector((center.x, center.y, maximum.z - height * 0.115))
    for name, location, energy, size in (
        ("KEY", (-0.55, 0.85, maximum.z + height * 0.02), 900, height * 0.44),
        ("FILL", (0.65, 0.45, maximum.z - height * 0.11), 460, height * 0.38),
        ("RIM", (0.0, -0.65, maximum.z + height * 0.01), 650, height * 0.34),
    ):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.name = "LGT_" + name
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        look_at(light, head_target)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "CAM_MAKEHUMAN_CLAY_PROBE"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = height * 0.30
    camera.data.lens = 85
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    try:
        scene.eevee.taa_render_samples = 8
        scene.eevee.taa_samples = 8
    except Exception:
        pass
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass

    distance = height * 2.4
    views = {
        "POS_Y": Vector((head_target.x, head_target.y + distance, head_target.z)),
        "NEG_Y": Vector((head_target.x, head_target.y - distance, head_target.z)),
        "POS_X": Vector((head_target.x + distance, head_target.y, head_target.z)),
        "NEG_X": Vector((head_target.x - distance, head_target.y, head_target.z)),
        "POS_Y_3Q": Vector((head_target.x + distance * 0.58, head_target.y + distance * 0.82, head_target.z)),
        "NEG_Y_3Q": Vector((head_target.x - distance * 0.58, head_target.y - distance * 0.82, head_target.z)),
    }
    for label, location in views.items():
        render(scene, camera, head_target, location, preview / f"MAKEHUMAN_CLAY_{label}.png")

    camera.data.ortho_scale = height * 1.10
    render(scene, camera, center, Vector((center.x, center.y + distance, center.z)), preview / "MAKEHUMAN_FULL_POS_Y.png")

    blend = output / "MAKEHUMAN_ASIAN_FEMALE_YOUNG_CLAY_PROBE.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    bpy.ops.export_scene.gltf(
        filepath=str(output / "MAKEHUMAN_ASIAN_FEMALE_YOUNG_CLAY_PROBE.glb"),
        export_format="GLB",
        export_cameras=False,
        export_lights=False,
    )
    report = {
        "source": str(source),
        "source_largest_axis": source_axis,
        "mesh_objects": [obj.name for obj in meshes],
        "vertex_count": sum(len(obj.data.vertices) for obj in meshes),
        "polygon_count": sum(len(obj.data.polygons) for obj in meshes),
        "bounds_min": list(minimum),
        "bounds_max": list(maximum),
        "height": height,
        "head_target": list(head_target),
        "render_samples_requested": 8,
        "identity_lock": False,
        "purpose": "axis/orientation and neutral adult Asian female clay baseline probe",
    }
    (qa / "MAKEHUMAN_CLAY_PROBE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            args = parse_args()
            qa = Path(args.get("out", ".")) / "QA"
            qa.mkdir(parents=True, exist_ok=True)
            (qa / "MAKEHUMAN_CLAY_PROBE_ERROR.log").write_text(error, encoding="utf-8")
        except Exception:
            pass
        raise
