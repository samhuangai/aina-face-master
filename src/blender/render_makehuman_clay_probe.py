#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

TARGET_HEIGHT_M = 1.68


def parse_args() -> dict[str, str]:
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(raw) % 2:
        raise ValueError(f"Expected --key value pairs, got: {raw}")
    return {raw[index].lstrip("-"): raw[index + 1] for index in range(0, len(raw), 2)}


def select_only(objects: list[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def remove_loose_vertices(meshes: list[bpy.types.Object]) -> int:
    removed = 0
    for obj in meshes:
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        loose = [vertex for vertex in bm.verts if not vertex.link_faces]
        removed += len(loose)
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
    return removed


def bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not points:
        raise RuntimeError("No imported MakeHuman mesh bounds")
    return (
        Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points))),
        Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points))),
    )


def normalize_vertical(meshes: list[bpy.types.Object]) -> str:
    minimum, maximum = bounds(meshes)
    span = maximum - minimum
    largest = max(range(3), key=lambda index: span[index])
    axis = "XYZ"[largest]
    if axis == "Y":
        for obj in meshes:
            obj.rotation_euler.rotate_axis("X", math.radians(90.0))
        select_only(meshes)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    elif axis == "X":
        for obj in meshes:
            obj.rotation_euler.rotate_axis("Y", math.radians(-90.0))
        select_only(meshes)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    bpy.ops.object.select_all(action="DESELECT")
    return axis


def scale_and_center(meshes: list[bpy.types.Object], target_height: float) -> tuple[float, Vector, Vector]:
    minimum, maximum = bounds(meshes)
    source_height = maximum.z - minimum.z
    if source_height <= 0:
        raise RuntimeError("Invalid MakeHuman height")
    scale = target_height / source_height
    for obj in meshes:
        obj.scale = tuple(component * scale for component in obj.scale)
    select_only(meshes)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    minimum, maximum = bounds(meshes)
    center_x = (minimum.x + maximum.x) * 0.5
    center_y = (minimum.y + maximum.y) * 0.5
    translation = Vector((-center_x, -center_y, -minimum.z))
    for obj in meshes:
        obj.location += translation
    select_only(meshes)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    bpy.ops.object.select_all(action="DESELECT")
    final_minimum, final_maximum = bounds(meshes)
    return scale, final_minimum, final_maximum


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def configure_material(meshes: list[bpy.types.Object]) -> None:
    clay = bpy.data.materials.new("AINA_IDENTITY_CLAY")
    clay.diffuse_color = (0.58, 0.61, 0.67, 1.0)
    clay.roughness = 0.72
    clay.metallic = 0.0
    for index, obj in enumerate(meshes):
        obj.name = "AINA_MAKEHUMAN_CLEAN_BASE" if index == 0 else f"AINA_MAKEHUMAN_PART_{index:02d}"
        obj.data.name = obj.name + "_MESH"
        obj.data.materials.clear()
        obj.data.materials.append(clay)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True


def configure_scene(scene: bpy.types.Scene) -> None:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass

    shading = scene.display.shading
    for attr, value in (
        ("light", "STUDIO"),
        ("color_type", "MATERIAL"),
        ("background_type", "VIEWPORT"),
        ("background_color", (0.018, 0.023, 0.034)),
        ("show_shadows", True),
        ("show_cavity", True),
        ("cavity_type", "BOTH"),
        ("curvature_ridge_factor", 1.6),
        ("curvature_valley_factor", 1.2),
        ("show_specular_highlight", True),
    ):
        try:
            setattr(shading, attr, value)
        except Exception:
            pass


def render_view(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    target: Vector,
    location: Vector,
    path: Path,
) -> None:
    camera.location = location
    look_at(camera, target)
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Render was not written: {path}")


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

    removed_loose = remove_loose_vertices(meshes)
    source_axis = normalize_vertical(meshes)
    raw_minimum, raw_maximum = bounds(meshes)
    scale_factor, minimum, maximum = scale_and_center(meshes, TARGET_HEIGHT_M)
    height = maximum.z - minimum.z
    configure_material(meshes)

    scene = bpy.context.scene
    configure_scene(scene)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "CAM_AINA_MAKEHUMAN_CLEAN_BASE"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = height * 0.335
    camera.data.lens = 85
    scene.camera = camera

    head_target = Vector((0.0, 0.0, maximum.z - height * 0.115))
    distance = height * 2.4
    front_y = minimum.y - distance
    back_y = maximum.y + distance
    views = {
        "AINA_CLEAN_FRONT": Vector((0.0, front_y, head_target.z)),
        "AINA_CLEAN_FRONT_3Q_LEFT": Vector((-distance * 0.56, front_y * 0.82, head_target.z)),
        "AINA_CLEAN_FRONT_3Q_RIGHT": Vector((distance * 0.56, front_y * 0.82, head_target.z)),
        "AINA_CLEAN_PROFILE_LEFT": Vector((-distance, 0.0, head_target.z)),
        "AINA_CLEAN_PROFILE_RIGHT": Vector((distance, 0.0, head_target.z)),
        "AINA_CLEAN_BACK": Vector((0.0, back_y, head_target.z)),
    }
    for label, location in views.items():
        render_view(scene, camera, head_target, location, preview / f"{label}.png")

    camera.data.ortho_scale = height * 1.08
    full_target = Vector((0.0, 0.0, height * 0.50))
    render_view(
        scene,
        camera,
        full_target,
        Vector((0.0, front_y, full_target.z)),
        preview / "AINA_CLEAN_FULL_FRONT.png",
    )

    camera.data.ortho_scale = height * 0.335
    camera.location = views["AINA_CLEAN_FRONT"]
    look_at(camera, head_target)

    blend = output / "AINA_MAKEHUMAN_CLEAN_BASE.blend"
    glb = output / "AINA_MAKEHUMAN_CLEAN_BASE.glb"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    bpy.ops.export_scene.gltf(
        filepath=str(glb),
        export_format="GLB",
        export_cameras=False,
        export_lights=False,
    )
    report = {
        "source": str(source),
        "source_largest_axis_after_import": source_axis,
        "removed_loose_vertices": removed_loose,
        "mesh_objects": [obj.name for obj in meshes],
        "vertex_count": sum(len(obj.data.vertices) for obj in meshes),
        "polygon_count": sum(len(obj.data.polygons) for obj in meshes),
        "raw_bounds_min": list(raw_minimum),
        "raw_bounds_max": list(raw_maximum),
        "final_bounds_min": list(minimum),
        "final_bounds_max": list(maximum),
        "target_height_m": TARGET_HEIGHT_M,
        "scale_factor": scale_factor,
        "head_target": list(head_target),
        "render_engine": scene.render.engine,
        "views": sorted(views),
        "identity_lock": False,
        "purpose": "clean adult Asian female production basemesh before AINA identity sculpt",
    }
    (qa / "AINA_MAKEHUMAN_CLEAN_BASE_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
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
            (qa / "AINA_MAKEHUMAN_CLEAN_BASE_ERROR.log").write_text(error, encoding="utf-8")
        except Exception:
            pass
        raise
