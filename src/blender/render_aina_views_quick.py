#!/usr/bin/env python3
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> dict[str, str]:
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(raw) % 2:
        raise ValueError(f"Expected key/value arguments, got {raw}")
    return {raw[index].lstrip("-"): raw[index + 1] for index in range(0, len(raw), 2)}


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects if obj.type == "MESH" for corner in obj.bound_box]
    if not points:
        raise RuntimeError("No AINA mesh bounds for quick multi-view rendering")
    minimum = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
    maximum = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
    return minimum, maximum


def tune_scene(target: Vector, resolution: int) -> tuple[bpy.types.Scene, bpy.types.Object]:
    world = bpy.data.worlds.new("AINA_QUICK_MULTIVIEW_WORLD")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.012, 0.018, 0.032, 1.0)
    background.inputs["Strength"].default_value = 0.20
    bpy.context.scene.world = world
    for name, location, energy, size in (
        ("KEY", (-0.45, 0.52, target.z + 0.32), 430.0, 0.68),
        ("FILL", (0.48, 0.42, target.z + 0.08), 190.0, 0.75),
        ("RIM", (0.0, -0.45, target.z + 0.28), 300.0, 0.55),
    ):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.name = "LGT_QUICK_" + name
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        look_at(light, target)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "CAM_AINA_QUICK_MULTIVIEW"
    camera.data.lens = 86
    camera.data.sensor_width = 36
    bpy.context.scene.camera = camera
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = -0.65
    except Exception:
        pass
    return scene, camera


def render(scene: bpy.types.Scene, camera: bpy.types.Object, target: Vector, location: tuple[float, float, float], path: Path) -> None:
    camera.location = Vector(location)
    look_at(camera, target)
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"AINA quick view was not produced: {path}")


def main() -> None:
    p = parse_args()
    source = Path(p["input"]).resolve()
    output = Path(p["output-dir"]).resolve()
    prefix = p.get("prefix", "AINA")
    requested = [value.strip().lower() for value in p.get("views", "front,3q,profile").split(",") if value.strip()]
    resolution = int(p.get("resolution", "512"))
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    for obj in list(bpy.data.objects):
        if obj.type == "MESH" and obj.name.lower().startswith(("icosphere", "sphere", "cube")) and max(obj.dimensions) > 0.42:
            bpy.data.objects.remove(obj, do_unlink=True)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    head = [obj for obj in meshes if any(token in obj.name.lower() for token in ("face", "hair", "updo"))] or meshes
    minimum, maximum = bounds(head)
    target = (minimum + maximum) * 0.5
    target.z = minimum.z + (maximum.z - minimum.z) * 0.48
    size = max(maximum.x - minimum.x, maximum.z - minimum.z)
    distance = max(0.58, size * 2.50)
    scene, camera = tune_scene(target, resolution)
    views = {
        "front": ((target.x, target.y + distance, target.z), "FRONT"),
        "3q": ((target.x + distance * 0.56, target.y + distance * 0.83, target.z), "3Q"),
        "profile": ((target.x + distance, target.y + 0.01, target.z), "PROFILE"),
    }
    produced = {}
    for view in requested:
        if view not in views:
            raise ValueError(f"Unsupported AINA quick view: {view}")
        location, suffix = views[view]
        path = output / f"{prefix}_{suffix}.png"
        render(scene, camera, target, location, path)
        produced[view] = {"path": str(path), "bytes": path.stat().st_size}
    print({"source": str(source), "views": produced, "head_min": list(minimum), "head_max": list(maximum)}, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise
