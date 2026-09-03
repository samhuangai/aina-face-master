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
        raise ValueError(f"Expected --key value pairs, got {raw}")
    return {raw[index].lstrip("-"): raw[index + 1] for index in range(0, len(raw), 2)}


def select_only(objects: list[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not points:
        raise RuntimeError("No mesh bounds")
    return (
        Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points))),
        Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points))),
    )


def remove_loose_vertices(meshes: list[bpy.types.Object]) -> int:
    removed = 0
    for obj in meshes:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        loose = [vertex for vertex in bm.verts if not vertex.link_faces]
        removed += len(loose)
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
    return removed


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


def scale_and_center(meshes: list[bpy.types.Object]) -> tuple[float, Vector, Vector]:
    minimum, maximum = bounds(meshes)
    source_height = maximum.z - minimum.z
    scale = TARGET_HEIGHT_M / source_height
    for obj in meshes:
        obj.scale = tuple(value * scale for value in obj.scale)
    select_only(meshes)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    minimum, maximum = bounds(meshes)
    translation = Vector((-(minimum.x + maximum.x) * 0.5, -(minimum.y + maximum.y) * 0.5, -minimum.z))
    for obj in meshes:
        obj.location += translation
    select_only(meshes)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    bpy.ops.object.select_all(action="DESELECT")
    final_minimum, final_maximum = bounds(meshes)
    return scale, final_minimum, final_maximum


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def configure_scene(scene: bpy.types.Scene) -> None:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
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
        ("background_color", (0.017, 0.021, 0.031)),
        ("show_shadows", True),
        ("show_cavity", True),
        ("cavity_type", "BOTH"),
        ("curvature_ridge_factor", 1.75),
        ("curvature_valley_factor", 1.35),
        ("show_specular_highlight", True),
    ):
        try:
            setattr(shading, attr, value)
        except Exception:
            pass


def make_clay(label: str) -> bpy.types.Material:
    material = bpy.data.materials.new(f"AINA_VARIANT_CLAY_{label}")
    material.diffuse_color = (0.61, 0.64, 0.70, 1.0)
    material.roughness = 0.72
    material.metallic = 0.0
    return material


def render_view(scene: bpy.types.Scene, camera: bpy.types.Object, target: Vector, location: Vector, path: Path) -> None:
    camera.location = location
    look_at(camera, target)
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Render was not written: {path}")


def remove_candidate(meshes: list[bpy.types.Object], material: bpy.types.Material) -> None:
    for obj in meshes:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data.users == 0:
            bpy.data.meshes.remove(data)
    if material.users == 0:
        bpy.data.materials.remove(material)


def main() -> None:
    args = parse_args()
    input_dir = Path(args["input-dir"]).resolve()
    output = Path(args["out"]).resolve()
    preview = output / "Preview"
    qa = output / "QA"
    preview.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)

    sources = sorted(input_dir.glob("*.obj"))
    if not sources:
        raise RuntimeError(f"No OBJ variants in {input_dir}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_scene(scene)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "CAM_AINA_VARIANT_REVIEW"
    camera.data.type = "ORTHO"
    camera.data.lens = 85
    scene.camera = camera

    reports: list[dict[str, object]] = []
    for source in sources:
        label = source.stem
        before = set(bpy.context.scene.objects)
        bpy.ops.wm.obj_import(filepath=str(source), forward_axis="NEGATIVE_Z", up_axis="Y")
        imported = [obj for obj in bpy.context.scene.objects if obj not in before]
        meshes = [obj for obj in imported if obj.type == "MESH"]
        if not meshes:
            raise RuntimeError(f"No mesh imported for {source}")

        removed = remove_loose_vertices(meshes)
        source_axis = normalize_vertical(meshes)
        scale, minimum, maximum = scale_and_center(meshes)
        height = maximum.z - minimum.z
        material = make_clay(label)
        for index, obj in enumerate(meshes):
            obj.name = f"AINA_{label}_{index:02d}"
            obj.data.materials.clear()
            obj.data.materials.append(material)
            for polygon in obj.data.polygons:
                polygon.use_smooth = True

        camera.data.ortho_scale = height * 0.305
        target = Vector((0.0, 0.0, maximum.z - height * 0.115))
        distance = height * 2.35
        front_y = minimum.y - distance
        views = {
            "FRONT": Vector((0.0, front_y, target.z)),
            "THREEQ": Vector((distance * 0.56, front_y * 0.82, target.z)),
            "PROFILE": Vector((distance, 0.0, target.z)),
        }
        for view, location in views.items():
            render_view(scene, camera, target, location, preview / f"{label}_{view}.png")

        reports.append(
            {
                "label": label,
                "source": str(source),
                "source_axis": source_axis,
                "removed_loose_vertices": removed,
                "scale_factor": scale,
                "bounds_min": list(minimum),
                "bounds_max": list(maximum),
                "vertex_count": sum(len(obj.data.vertices) for obj in meshes),
                "polygon_count": sum(len(obj.data.polygons) for obj in meshes),
                "renders": [f"{label}_{view}.png" for view in views],
            }
        )
        remove_candidate(meshes, material)

    report = {
        "variant_count": len(reports),
        "render_engine": scene.render.engine,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "variants": reports,
    }
    (qa / "AINA_VARIANT_RENDER_REPORT.json").write_text(
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
            (qa / "AINA_VARIANT_RENDER_ERROR.log").write_text(error, encoding="utf-8")
        except Exception:
            pass
        raise
