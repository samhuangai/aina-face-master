#!/usr/bin/env python3
"""Fast Blender renderer for the directly refined AINA real head mesh.

Loads the refined OBJ exactly as produced, uses only its external head component,
adds the current production convex eye geometry plus real brow/lash curves, and
renders front and shallow three-quarter views. No body, rig, Shape Keys, VRM
export or replacement reference art is created in this QA stage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aina_visual_eye_system as eye_system

K = np.array([
    1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,
    3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,
    5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,
    2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,
    13695,12973,13422,6543,6537,
], dtype=np.int64)


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def read_obj(path: Path):
    verts = []
    faces = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            q = line.split()
            verts.append((float(q[1]), float(q[2]), float(q[3])))
        elif line.startswith("f "):
            ids = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
            for i in range(1, len(ids) - 1):
                faces.append((ids[0], ids[i], ids[i + 1]))
    return np.asarray(verts, np.float64), np.asarray(faces, np.int32)


def components(n: int, faces: np.ndarray):
    parent = np.arange(n, dtype=np.int32)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x
    def union(a, b):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    for a, b, c in faces:
        union(a, b); union(b, c); union(c, a)
    roots = np.array([find(i) for i in range(n)], dtype=np.int32)
    return roots


def mapped(raw: np.ndarray):
    scale = 1.08
    out = np.empty_like(raw)
    out[:, 0] = raw[:, 0] * scale
    out[:, 1] = raw[:, 2] * scale
    out[:, 2] = -raw[:, 1] * scale
    out[:, 2] += 1.72 - float(out[:, 2].max())
    return out


def material(name, color, roughness=0.4, metallic=0.0, subsurface=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        if bsdf.inputs.get("Base Color"):
            bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        if bsdf.inputs.get("Roughness"):
            bsdf.inputs["Roughness"].default_value = roughness
        if bsdf.inputs.get("Metallic"):
            bsdf.inputs["Metallic"].default_value = metallic
        if bsdf.inputs.get("Subsurface Weight"):
            bsdf.inputs["Subsurface Weight"].default_value = subsurface
        if bsdf.inputs.get("Specular IOR Level"):
            bsdf.inputs["Specular IOR Level"].default_value = 0.34
        if bsdf.inputs.get("Coat Weight"):
            bsdf.inputs["Coat Weight"].default_value = 0.035
    return mat


def mesh_object(name, vertices, faces, mat):
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata([tuple(v) for v in vertices], [], [tuple(map(int, f)) for f in faces])
    mesh.update()
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return obj


def curve(name, points, radius, mat):
    data = bpy.data.curves.new(name + "_Curve", "CURVE")
    data.dimensions = "3D"
    data.resolution_u = 5
    data.bevel_depth = radius
    data.bevel_resolution = 3
    spline = data.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for point, value in zip(spline.bezier_points, points):
        point.co = value
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    data.materials.append(mat)
    return obj


def main():
    a = args()
    a.out.mkdir(parents=True, exist_ok=True)
    clear_scene()
    raw, faces = read_obj(a.face)
    if len(raw) <= int(K.max()):
        raise RuntimeError("Semantic AINA vertex order missing")
    roots = components(len(raw), faces)
    head_root = roots[int(K[0])]
    keep = roots[faces[:, 0]] == head_root
    vertices = mapped(raw)

    skin = material("AINA_QA_Skin", (0.82, 0.69, 0.67), 0.43, 0.0, 0.065)
    lip = material("AINA_QA_Lip", (0.56, 0.24, 0.28), 0.34)
    white = material("AINA_QA_EyeWhite", (0.96, 0.975, 0.995), 0.18)
    iris = material("AINA_QA_Iris", (0.20, 0.47, 0.62), 0.16)
    pupil = material("AINA_QA_Pupil", (0.006, 0.010, 0.018), 0.15)
    dark = material("AINA_QA_BrowLash", (0.08, 0.075, 0.10), 0.30)

    head = mesh_object("AINA_DIRECT_REFINED_HEAD", vertices, faces[keep], skin)
    head.data.materials.append(lip)
    lm = vertices[K]
    mouth_center = lm[48:60].mean(0)
    mouth_half_width = max(abs(float(lm[54, 0] - lm[48, 0])) * 0.54, 0.020)
    mouth_half_height = max(float(lm[48:60, 2].max() - lm[48:60, 2].min()) * 0.64, 0.0065)
    for polygon in head.data.polygons:
        center = np.mean([np.asarray(head.data.vertices[i].co) for i in polygon.vertices], axis=0)
        q = ((center[0] - mouth_center[0]) / mouth_half_width) ** 2 + ((center[2] - mouth_center[2]) / mouth_half_height) ** 2
        if q < 1.0 and center[1] < mouth_center[1] + 0.003:
            polygon.material_index = 1

    centers = {"R": lm[36:42].mean(0), "L": lm[42:48].mean(0)}
    for side in ("R", "L"):
        center = centers[side].copy()
        center[1] = -0.00035
        eye_system._almond("AINA_Eye_" + side, center, white, side)
        iris_location = center.copy(); iris_location[1] = -0.01185
        pupil_location = center.copy(); pupil_location[1] = -0.01245
        eye_system._disc("AINA_Iris_" + side, iris_location, 0.00565, iris, side, pupil=False)
        eye_system._disc("AINA_Pupil_" + side, pupil_location, 0.00220, pupil, side, pupil=True)

        radius_x = 0.0180
        points = [
            (center[0] - radius_x, -0.0126, center[2] + (0.0010 if side == "R" else 0.0003)),
            (center[0] - radius_x * 0.52, -0.0128, center[2] + 0.0044),
            (center[0], -0.0129, center[2] + 0.0055),
            (center[0] + radius_x * 0.52, -0.0128, center[2] + 0.0044),
            (center[0] + radius_x, -0.0126, center[2] + (0.0003 if side == "R" else 0.0010)),
        ]
        curve("AINA_Lash_" + side, points, 0.00052, dark)

    for side, indices in (("R", list(range(17, 22))), ("L", list(range(22, 27)))):
        points = [(float(p[0]), float(min(p[1] - 0.0017, -0.0105)), float(p[2] + 0.00025)) for p in lm[indices]]
        curve("AINA_Brow_" + side, points, 0.00082, dark)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.world.color = (0.92, 0.94, 0.97)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = 0.20
    except Exception:
        pass

    def area(name, location, energy, size, target=(0, 0, 1.595)):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()

    area("Key", (1.05, -1.40, 2.10), 560, 2.2)
    area("Fill", (-1.10, -1.25, 1.82), 260, 2.3)
    area("Rim", (0.0, 1.20, 2.05), 300, 2.0)
    area("FaceSoft", (0.0, -1.90, 1.58), 70, 2.8)

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera.data.lens = 92

    for filename, position in (
        ("AINA_DIRECT_REFINED_FRONT.png", (0.0, -0.72, 1.605)),
        ("AINA_DIRECT_REFINED_3Q.png", (0.255, -0.690, 1.608)),
    ):
        camera.location = position
        camera.rotation_euler = (Vector((0, 0, 1.600)) - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(a.out / filename)
        bpy.ops.render.render(write_still=True)

    bpy.ops.wm.save_as_mainfile(filepath=str(a.out / "AINA_DIRECT_REFINED_HEAD_QA.blend"))


if __name__ == "__main__":
    main()
