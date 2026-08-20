#!/usr/bin/env python3
"""Render the directly refined AINA OBJ without applying another sculpt pass.

The input vertices are used exactly as supplied.  This script only maps the OBJ
into Blender coordinates, adds real convex eye/iris/pupil geometry plus simple
brows/lashes, and renders neutral front / calibrated 20-degree 3Q beauty and clay
views.  It intentionally skips body, shape generation and VRM export so the
actual mesh changes can be judged without downstream noise.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

K = np.array([
    1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,
    10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,
    10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,
    4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,
    6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,
    6817,6550,13695,12973,13422,6543,6537,
], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    av = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--face", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(av)


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            ids = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
            for i in range(1, len(ids) - 1):
                faces.append([ids[0], ids[i], ids[i + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def component_roots(n: int, faces: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    parent = np.arange(n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, c in faces:
        union(int(a), int(b)); union(int(b), int(c)); union(int(c), int(a))
    roots = np.array([find(i) for i in range(n)], dtype=np.int32)
    groups: dict[int, list[int]] = {}
    for i, root in enumerate(roots):
        groups.setdefault(int(root), []).append(i)
    return roots, {root: np.asarray(ids, dtype=np.int32) for root, ids in groups.items()}


def map_vertices(v: np.ndarray, height=1.72) -> np.ndarray:
    out = np.empty_like(v)
    scale = 1.08
    out[:, 0] = v[:, 0] * scale
    out[:, 1] = v[:, 2] * scale
    out[:, 2] = -v[:, 1] * scale
    out[:, 2] += height - float(out[:, 2].max())
    return out


def material(name: str, color, roughness=0.42, metallic=0.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        if bsdf.inputs.get("Specular IOR Level"):
            bsdf.inputs["Specular IOR Level"].default_value = 0.36
    return mat


def mesh_object(name: str, vertices: np.ndarray, faces: np.ndarray, mat) -> bpy.types.Object:
    data = bpy.data.meshes.new(name + "_Mesh")
    data.from_pydata([tuple(x) for x in vertices], [], [tuple(map(int, f)) for f in faces])
    data.update()
    data.materials.append(mat)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    for polygon in data.polygons:
        polygon.use_smooth = True
    return obj


def convex_almond(name: str, center, side: str, mat) -> bpy.types.Object:
    c = np.asarray(center, dtype=np.float64)
    rx, rz = 0.0182, 0.00655
    segments, rings = 64, 5
    vertices = [(c[0], c[1] - 0.0088, c[2])]
    ring_ids: list[list[int]] = []
    tail_sign = 1.0 if side == "L" else -1.0
    for ring_index in range(1, rings + 1):
        radial = ring_index / rings
        ids = []
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            x = rx * radial * math.cos(a)
            sine = math.sin(a)
            z = (rz if sine >= 0 else rz * 0.68) * radial * sine
            z += tail_sign * 0.00052 * (x / rx)
            y = c[1] - 0.00545 - 0.00335 * (1.0 - radial * radial)
            vertices.append((c[0] + x, y, c[2] + z))
            ids.append(1 + (ring_index - 1) * segments + i)
        ring_ids.append(ids)
    faces = []
    for i in range(segments):
        faces.append((0, ring_ids[0][i], ring_ids[0][(i + 1) % segments]))
    for r in range(1, rings):
        inner, outer = ring_ids[r - 1], ring_ids[r]
        for i in range(segments):
            j = (i + 1) % segments
            faces.extend(((inner[i], outer[i], outer[j]), (inner[i], outer[j], inner[j])))
    return mesh_object(name, np.asarray(vertices), np.asarray(faces), mat)


def convex_disc(name: str, center, radius: float, mat, oval=1.04) -> bpy.types.Object:
    c = np.asarray(center, dtype=np.float64)
    segments, rings = 64, 4
    vertices = [(c[0], c[1] - 0.00072, c[2])]
    ring_ids: list[list[int]] = []
    for ring_index in range(1, rings + 1):
        radial = ring_index / rings
        ids = []
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            x = radius * radial * math.cos(a)
            z = radius * radial * oval * math.sin(a)
            y = c[1] - 0.00072 * (1.0 - radial * radial)
            vertices.append((c[0] + x, y, c[2] + z))
            ids.append(1 + (ring_index - 1) * segments + i)
        ring_ids.append(ids)
    faces = []
    for i in range(segments):
        faces.append((0, ring_ids[0][i], ring_ids[0][(i + 1) % segments]))
    for r in range(1, rings):
        inner, outer = ring_ids[r - 1], ring_ids[r]
        for i in range(segments):
            j = (i + 1) % segments
            faces.extend(((inner[i], outer[i], outer[j]), (inner[i], outer[j], inner[j])))
    return mesh_object(name, np.asarray(vertices), np.asarray(faces), mat)


def curve(name: str, points, radius: float, mat) -> bpy.types.Object:
    data = bpy.data.curves.new(name + "_Curve", "CURVE")
    data.dimensions = "3D"
    data.resolution_u = 5
    data.bevel_depth = radius
    data.bevel_resolution = 3
    spline = data.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for bp, p in zip(spline.bezier_points, points):
        bp.co = p
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    data.materials.append(mat)
    return obj


def create_scene(face_path: Path, out: Path) -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    raw, faces = read_obj(face_path)
    if len(raw) <= int(K.max()):
        raise RuntimeError("The refined AINA OBJ does not preserve semantic vertex order")
    roots, groups = component_roots(len(raw), faces)
    head_root = max(groups, key=lambda root: len(groups[root]))
    mapped = map_vertices(raw)
    head_faces = faces[roots[faces[:, 0]] == head_root]

    skin = material("AINA_Skin", (0.82, 0.69, 0.67), 0.46)
    eye_white = material("AINA_EyeWhite", (0.965, 0.975, 0.995), 0.20)
    iris = material("AINA_Iris", (0.18, 0.40, 0.54), 0.17)
    pupil = material("AINA_Pupil", (0.006, 0.010, 0.018), 0.15)
    dark = material("AINA_LashBrow", (0.095, 0.085, 0.115), 0.32)
    lip = material("AINA_Lip", (0.60, 0.28, 0.31), 0.36)
    clay = material("AINA_Clay", (0.70, 0.72, 0.76), 0.58)

    head = mesh_object("AINA_DIRECT_REFINED_HEAD", mapped, head_faces, skin)
    head.data.materials.append(lip)
    lm = mapped[K]
    mouth = lm[48:60].mean(axis=0)
    for polygon in head.data.polygons:
        center = np.mean([np.asarray(head.data.vertices[i].co) for i in polygon.vertices], axis=0)
        q = ((center[0] - mouth[0]) / 0.0235) ** 2 + ((center[2] - mouth[2]) / 0.0082) ** 2
        if q < 1.0 and center[1] < 0.006:
            polygon.material_index = 1

    details: list[bpy.types.Object] = []
    for side, eye_ids in (("R", np.arange(36, 42)), ("L", np.arange(42, 48))):
        c = lm[eye_ids].mean(axis=0)
        c[1] = -0.00035
        eye = convex_almond("AINA_Eye_" + side, c, side, eye_white)
        iris_center = c.copy(); iris_center[1] = -0.01185
        pupil_center = c.copy(); pupil_center[1] = -0.01245
        iris_obj = convex_disc("AINA_Iris_" + side, iris_center, 0.00565, iris, oval=1.045)
        pupil_obj = convex_disc("AINA_Pupil_" + side, pupil_center, 0.00220, pupil, oval=1.0)
        details.extend((eye, iris_obj, pupil_obj))

        rx = 0.0182
        points = [
            (c[0] - rx, -0.01295, c[2] + (0.0009 if side == "R" else 0.0002)),
            (c[0] - rx * 0.52, -0.01308, c[2] + 0.0044),
            (c[0], -0.01315, c[2] + 0.0055),
            (c[0] + rx * 0.52, -0.01308, c[2] + 0.0044),
            (c[0] + rx, -0.01295, c[2] + (0.0002 if side == "R" else 0.0009)),
        ]
        details.append(curve("AINA_Lash_" + side, points, 0.00056, dark))

    for side, ids in (("R", np.arange(17, 22)), ("L", np.arange(22, 27))):
        points = [(float(p[0]), -0.0121, float(p[2] + 0.00025)) for p in lm[ids]]
        details.append(curve("AINA_Brow_" + side, points, 0.00086, dark))

    # Keep clay available as a temporary override material for QA renders.
    head["qa_clay_material"] = clay.name
    out.mkdir(parents=True, exist_ok=True)
    return head, details


def setup_lighting() -> bpy.types.Object:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 800
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.world.color = (0.93, 0.94, 0.97)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = 0.20
    except Exception:
        pass

    def area(name, location, energy, size):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector((0, 0, 1.60)) - obj.location).to_track_quat("-Z", "Y").to_euler()

    area("Key", (1.0, -1.5, 2.1), 520, 2.4)
    area("Fill", (-1.2, -1.4, 1.8), 290, 2.3)
    area("Rim", (0, 1.2, 2.0), 270, 2.0)
    area("FaceSoft", (0, -1.85, 1.62), 90, 2.8)

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera.data.lens = 85
    return camera


def render(scene, camera, path: Path, position, target=(0, 0, 1.61)) -> None:
    camera.location = position
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    head, details = create_scene(args.face, args.out)
    camera = setup_lighting()
    scene = bpy.context.scene

    render(scene, camera, args.out / "AINA_DIRECT_REFINED_FRONT.png", (0, -0.78, 1.61))
    render(scene, camera, args.out / "AINA_DIRECT_REFINED_Q3_20.png", (0.275, -0.755, 1.61))

    # Clay renders expose actual mesh continuity independent of cosmetic materials.
    original_slots = [slot.material for slot in head.material_slots]
    clay = bpy.data.materials.get(head["qa_clay_material"])
    for slot in head.material_slots:
        slot.material = clay
    for obj in details:
        obj.hide_render = True
    render(scene, camera, args.out / "AINA_DIRECT_REFINED_CLAY_FRONT.png", (0, -0.78, 1.61))
    render(scene, camera, args.out / "AINA_DIRECT_REFINED_CLAY_Q3_20.png", (0.275, -0.755, 1.61))
    for slot, mat in zip(head.material_slots, original_slots):
        slot.material = mat
    for obj in details:
        obj.hide_render = False

    bpy.ops.wm.save_as_mainfile(filepath=str(args.out / "AINA_DIRECT_REFINED_HEAD_QA.blend"))
    for name in (
        "AINA_DIRECT_REFINED_FRONT.png", "AINA_DIRECT_REFINED_Q3_20.png",
        "AINA_DIRECT_REFINED_CLAY_FRONT.png", "AINA_DIRECT_REFINED_CLAY_Q3_20.png",
    ):
        path = args.out / name
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing actual direct-mesh QA render: {path}")


if __name__ == "__main__":
    main()
