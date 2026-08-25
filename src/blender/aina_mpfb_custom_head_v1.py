#!/usr/bin/env python3
"""Build AINA on an adult MPFB2 topology instead of the rejected Rain head.

This is a direction reset, not another Rain refinement. The script creates a
new young adult Asian female from MPFB2's CC0 character data, applies a bounded
AINA-specific neutral sculpture to the real basemesh, adds a coherent eye,
brow, lash and silver-updo system, attaches the MPFB game-engine rig, renders
front/three-quarter/profile beauty and clay views, and exports editable BLEND
plus morph-preserving GLB.

The Rain head, Rain face proportions and Rain expression deltas are not used.
No replacement effect art is generated. Locks remain false until direct review
of the actual Blender model.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpfb-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def install_mpfb(root: Path):
    src = root.resolve() / "src"
    if not (src / "mpfb" / "__init__.py").exists():
        raise FileNotFoundError(f"MPFB2 source package not found under {src}")
    sys.path.insert(0, str(src))
    mpfb = importlib.import_module("mpfb")
    try:
        mpfb.register()
    except Exception as exc:
        # Registration can report already-registered classes if Blender loaded
        # the extension previously. Direct service imports are still valid.
        if "already registered" not in str(exc).lower():
            raise
    services = importlib.import_module("mpfb.services")
    return (
        getattr(services, "HumanService"),
        getattr(services, "TargetService"),
    )


def mesh_local_array(obj) -> np.ndarray:
    values = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", values)
    return values.reshape(-1, 3)


def set_mesh_local_array(obj, values: np.ndarray) -> None:
    obj.data.vertices.foreach_set("co", np.asarray(values, dtype=np.float32).ravel())
    obj.data.update()


def key_array(key) -> np.ndarray:
    values = np.empty(len(key.data) * 3, dtype=np.float64)
    key.data.foreach_get("co", values)
    return values.reshape(-1, 3)


def set_key_array(key, values: np.ndarray) -> None:
    key.data.foreach_set("co", np.asarray(values, dtype=np.float32).ravel())


def to_world(obj, local: np.ndarray) -> np.ndarray:
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    return (np.c_[local, np.ones(len(local))] @ matrix.T)[:, :3]


def to_local(obj, world: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(np.asarray(obj.matrix_world, dtype=np.float64))
    return (np.c_[world, np.ones(len(world))] @ inverse.T)[:, :3]


def world_vertices(obj) -> np.ndarray:
    if obj.data.shape_keys:
        keys = obj.data.shape_keys.key_blocks
        basis = keys.get("Basis") or keys[0]
        return to_world(obj, key_array(basis))
    return to_world(obj, mesh_local_array(obj))


def apply_world_delta_all_keys(obj, world_delta: np.ndarray) -> None:
    linear = np.asarray(obj.matrix_world, dtype=np.float64)[:3, :3]
    local_delta = world_delta @ np.linalg.inv(linear).T
    if obj.data.shape_keys:
        for key in obj.data.shape_keys.key_blocks:
            set_key_array(key, key_array(key) + local_delta)
        obj.data.update()
    else:
        set_mesh_local_array(obj, mesh_local_array(obj) + local_delta)


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def ellipsoid(points: np.ndarray, centre, radii, outer: float = 1.0) -> np.ndarray:
    centre = np.asarray(centre, dtype=np.float64)
    radii = np.maximum(np.asarray(radii, dtype=np.float64), 1.0e-8)
    q = np.sqrt(np.sum(((points - centre) / radii) ** 2, axis=1))
    result = np.zeros(len(points), dtype=np.float64)
    mask = q < outer
    if np.any(mask):
        t = q[mask] / outer
        result[mask] = 0.5 * (1.0 + np.cos(np.pi * t))
    return result


def adjacency(obj, ids: np.ndarray):
    lookup = {int(vertex): index for index, vertex in enumerate(ids)}
    output = [set() for _ in ids]
    for polygon in obj.data.polygons:
        vertices = list(polygon.vertices)
        for source_index, source in enumerate(vertices):
            a = lookup.get(int(source))
            if a is None:
                continue
            for target in vertices[source_index + 1 :] + vertices[:source_index]:
                b = lookup.get(int(target))
                if b is not None and a != b:
                    output[a].add(b)
                    output[b].add(a)
    return output


def smooth_delta(delta: np.ndarray, graph, preserve: np.ndarray, passes: int = 2) -> np.ndarray:
    result = delta.copy()
    for _ in range(passes):
        updated = result.copy()
        for index, neighbours in enumerate(graph):
            if not neighbours:
                continue
            average = np.mean(result[list(neighbours)], axis=0)
            strength = 0.28 * (1.0 - 0.82 * preserve[index])
            updated[index] = result[index] * (1.0 - strength) + average * strength
        result = updated
    return result


def create_character(HumanService, TargetService):
    macro = TargetService.get_default_macro_info_dict()
    if "race" in macro:
        for key in ("african", "asian", "caucasian"):
            if key in macro["race"]:
                macro["race"][key] = 0.0
        if "asian" in macro["race"]:
            macro["race"]["asian"] = 1.0
    values = {
        "gender": 0.0,
        "age": 0.46,
        "muscle": 0.38,
        "weight": 0.40,
        "proportions": 0.56,
        "height": 0.52,
        "cupsize": 0.42,
        "firmness": 0.62,
    }
    for key, value in values.items():
        if key in macro:
            macro[key] = value
    human = HumanService.create_human(
        mask_helpers=True,
        detailed_helpers=True,
        extra_vertex_groups=True,
        feet_on_ground=True,
        scale=0.1,
        macro_detail_dict=macro,
    )
    human.name = "AINA_MPFB_Body"
    for polygon in human.data.polygons:
        polygon.use_smooth = True
    return human, macro


def analyse_head(human) -> dict:
    points = world_vertices(human)
    lo, hi = points.min(axis=0), points.max(axis=0)
    height = float(hi[2] - lo[2])
    body_centre_x = float(0.5 * (lo[0] + hi[0]))
    preliminary = np.where(points[:, 2] > lo[2] + 0.775 * height)[0]
    if len(preliminary) < 700:
        preliminary = np.where(points[:, 2] > lo[2] + 0.74 * height)[0]
    cloud = points[preliminary]
    cloud_lo, cloud_hi = cloud.min(axis=0), cloud.max(axis=0)
    width = float(cloud_hi[0] - cloud_lo[0])
    depth = float(cloud_hi[1] - cloud_lo[1])
    centre = 0.5 * (cloud_lo + cloud_hi)
    ids = preliminary[
        (np.abs(cloud[:, 0] - body_centre_x) < max(0.62 * width, 0.10 * height))
        & (np.abs(cloud[:, 1] - centre[1]) < max(0.68 * depth, 0.11 * height))
    ]
    if len(ids) < 700:
        ids = preliminary
    head = points[ids]
    head_lo, head_hi = head.min(axis=0), head.max(axis=0)
    head_width = float(head_hi[0] - head_lo[0])
    head_depth = float(head_hi[1] - head_lo[1])
    head_height = float(head_hi[2] - head_lo[2])
    face_x = float(0.5 * (head_lo[0] + head_hi[0]))
    forward_sign = -1.0  # MakeHuman/MPFB faces -Y in its rest coordinate system.
    front_y = float(head_lo[1])
    eye_z = float(head_lo[2] + 0.615 * head_height)
    eye_y = float(front_y - forward_sign * 0.19 * head_depth)
    eye_sep = 0.205 * head_width
    eyes = [
        np.array([face_x - eye_sep, eye_y, eye_z]),
        np.array([face_x + eye_sep, eye_y, eye_z]),
    ]
    return {
        "body_bounds_min": lo,
        "body_bounds_max": hi,
        "character_height": height,
        "head_ids": ids.astype(np.int64),
        "head_bounds_min": head_lo,
        "head_bounds_max": head_hi,
        "head_width": head_width,
        "head_depth": head_depth,
        "head_height": head_height,
        "face_x": face_x,
        "forward_sign": forward_sign,
        "eyes": eyes,
    }


def sculpt_aina(human, analysis: dict) -> dict:
    full = world_vertices(human)
    ids = analysis["head_ids"]
    points = full[ids]
    lo = analysis["head_bounds_min"]
    hi = analysis["head_bounds_max"]
    width = analysis["head_width"]
    depth = analysis["head_depth"]
    height = analysis["head_height"]
    face_x = analysis["face_x"]
    forward = analysis["forward_sign"]
    eyes = analysis["eyes"]
    eye_z = float(np.mean(eyes, axis=0)[2])
    centre_y = float(0.5 * (lo[1] + hi[1]))
    frontness = forward * (points[:, 1] - centre_y)
    f_low = float(np.quantile(frontness, 0.34))
    f_high = float(np.quantile(frontness, 0.96))
    face_front = smoothstep((frontness - f_low) / max(f_high - f_low, 1.0e-9))
    face_side = smoothstep((0.58 * width - np.abs(points[:, 0] - face_x)) / (0.18 * width))
    face = face_front * face_side

    mouth_z = float(lo[2] + 0.305 * height)
    nose_z = float(lo[2] + 0.455 * height)
    chin_z = float(lo[2] + 0.125 * height)
    face_y = float(lo[1])
    nose = np.array([face_x, face_y - forward * 0.12 * depth, nose_z])
    mouth = np.array([face_x, face_y - forward * 0.07 * depth, mouth_z])
    chin = np.array([face_x, face_y - forward * 0.02 * depth, chin_z])

    delta = np.zeros_like(points)
    preserve = np.zeros(len(points), dtype=np.float64)

    # Adult semi-realistic skull: smaller upper cranium than the generic base,
    # but not the child/chibi collapse seen in Rain.
    top_origin = eye_z + 0.055 * height
    top = np.clip((points[:, 2] - top_origin) / max(hi[2] - top_origin, 1.0e-7), 0.0, 1.0)
    top_weight = smoothstep(top) * smoothstep((0.62 * width - np.abs(points[:, 0] - face_x)) / (0.20 * width))
    delta[:, 0] += -(points[:, 0] - face_x) * (0.085 * top) * top_weight
    delta[:, 2] += -(points[:, 2] - top_origin) * (0.075 * top) * top_weight
    delta[:, 1] += -forward * 0.015 * depth * top_weight * face_front

    # Narrow oval/V face and a compact lower third.
    vertical = np.clip((eye_z + 0.05 * height - points[:, 2]) / max(eye_z + 0.05 * height - chin_z, 1.0e-7), 0.0, 1.0)
    oval_scale = 0.975 - 0.105 * np.power(vertical, 1.25)
    oval_weight = face * smoothstep(vertical + 0.18)
    delta[:, 0] += ((points[:, 0] - face_x) * oval_scale - (points[:, 0] - face_x)) * oval_weight
    lower = np.clip((nose_z - points[:, 2]) / max(nose_z - chin_z, 1.0e-7), 0.0, 1.0)
    lower_weight = face * smoothstep(lower)
    target_z = nose_z + (points[:, 2] - nose_z) * 0.965
    delta[:, 2] += (target_z - points[:, 2]) * lower_weight

    # Almond eyes: wider than a photographic adult, but substantially more
    # restrained than Rain. The socket and lid skin are changed together.
    eye_targets = []
    for eye in eyes:
        side = -1.0 if eye[0] < face_x else 1.0
        target_eye = eye.copy()
        target_eye[0] = face_x + (eye[0] - face_x) * 0.98
        target_eye[2] += 0.006 * height
        weight = ellipsoid(points, eye, (0.20 * width, 0.28 * depth, 0.155 * height), 1.15) * face_front
        target_x = target_eye[0] + (points[:, 0] - eye[0]) * 1.075
        target_z = target_eye[2] + (points[:, 2] - eye[2]) * 1.045
        delta[:, 0] += (target_x - points[:, 0]) * weight
        delta[:, 2] += (target_z - points[:, 2]) * weight
        outer = eye + np.array([side * 0.18 * width, 0.0, 0.012 * height])
        outer_weight = ellipsoid(points, outer, (0.09 * width, 0.18 * depth, 0.08 * height), 1.05) * weight
        delta[:, 2] += 0.010 * height * outer_weight
        preserve = np.maximum(preserve, np.clip(weight + outer_weight, 0.0, 1.0))
        eye_targets.append(target_eye)

    # Delicate readable nose.
    bridge = np.array([face_x, nose[1], mouth_z + 0.63 * (eye_z - mouth_z)])
    bridge_weight = ellipsoid(points, bridge, (0.16 * width, 0.22 * depth, 0.23 * height), 1.10) * face
    delta[:, 0] += -(points[:, 0] - face_x) * 0.16 * bridge_weight
    tip_weight = ellipsoid(points, nose, (0.12 * width, 0.20 * depth, 0.11 * height), 1.08) * face
    delta[:, 0] += -(points[:, 0] - face_x) * 0.20 * tip_weight
    delta[:, 1] += -forward * 0.020 * depth * tip_weight
    delta[:, 2] += 0.010 * height * tip_weight
    base_weight = ellipsoid(points, np.array([face_x, nose[1], nose_z - 0.075 * height]), (0.20 * width, 0.20 * depth, 0.12 * height), 1.08) * face
    delta[:, 0] += -(points[:, 0] - face_x) * 0.12 * base_weight
    preserve = np.maximum(preserve, np.clip(bridge_weight + tip_weight + base_weight, 0.0, 1.0))

    # Compact integrated lips with a soft cupid bow, not a separate lip plate.
    lip_weight = ellipsoid(points, mouth, (0.30 * width, 0.20 * depth, 0.105 * height), 1.12) * face
    target_x = mouth[0] + (points[:, 0] - mouth[0]) * 0.88
    target_z = mouth[2] + (points[:, 2] - mouth[2]) * 1.055
    delta[:, 0] += (target_x - points[:, 0]) * lip_weight
    delta[:, 2] += (target_z - points[:, 2]) * lip_weight
    delta[:, 1] += forward * 0.010 * depth * lip_weight
    preserve = np.maximum(preserve, np.clip(lip_weight, 0.0, 1.0))

    # High apple-cheek support while keeping the side silhouette narrow.
    mid_z = mouth_z + 0.58 * (eye_z - mouth_z)
    for eye in eyes:
        side = -1.0 if eye[0] < face_x else 1.0
        cheek = np.array([eye[0] + side * 0.04 * width, nose[1] - forward * 0.04 * depth, mid_z])
        weight = ellipsoid(points, cheek, (0.27 * width, 0.26 * depth, 0.20 * height), 1.12) * face
        delta[:, 1] += forward * 0.016 * depth * weight
        delta[:, 0] += -side * 0.010 * width * weight
        delta[:, 2] += 0.006 * height * weight

    # Small round chin and soft V jaw.
    chin_weight = ellipsoid(points, chin, (0.24 * width, 0.26 * depth, 0.17 * height), 1.10) * face
    delta[:, 0] += -(points[:, 0] - face_x) * 0.15 * chin_weight
    delta[:, 2] += 0.010 * height * chin_weight
    delta[:, 1] += -forward * 0.004 * depth * chin_weight
    preserve = np.maximum(preserve, np.clip(chin_weight, 0.0, 1.0))

    graph = adjacency(human, ids)
    smoothed = smooth_delta(delta, graph, preserve, passes=3)
    lengths = np.linalg.norm(smoothed, axis=1)
    cap = 0.065 * height
    smoothed *= np.minimum(1.0, cap / np.maximum(lengths, 1.0e-9))[:, None]
    full_delta = np.zeros_like(full)
    full_delta[ids] = smoothed
    apply_world_delta_all_keys(human, full_delta)

    analysis["eyes"] = eye_targets
    return {
        "head_region_vertices": len(ids),
        "max_displacement_m": float(np.linalg.norm(smoothed, axis=1).max()),
        "rms_displacement_m": float(np.sqrt(np.mean(np.sum(smoothed * smoothed, axis=1)))),
        "moved_vertices_over_0_5mm": int(np.sum(np.linalg.norm(smoothed, axis=1) > 0.0005)),
        "eye_centres": [point.tolist() for point in eye_targets],
        "nose_centre": nose.tolist(),
        "mouth_centre": mouth.tolist(),
        "chin_centre": chin.tolist(),
    }


def material(name: str, color, roughness: float = 0.45, metallic: float = 0.0):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = tuple(color)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
    if shader:
        shader.inputs["Base Color"].default_value = tuple(color)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metallic
        if "Subsurface Weight" in shader.inputs and "Skin" in name:
            shader.inputs["Subsurface Weight"].default_value = 0.08
    return mat


def assign_material(obj, mat) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
        polygon.use_smooth = True


def create_sphere(name: str, location, scale, mat, segments=48, rings=24):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=rings,
        location=tuple(location),
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = tuple(scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign_material(obj, mat)
    return obj


def create_curve_mesh(name: str, points, bevel: float, mat):
    curve = bpy.data.curves.new(name + "_Curve", "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 4
    curve.bevel_depth = bevel
    curve.bevel_resolution = 3
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for point, value in zip(spline.bezier_points, points):
        point.co = tuple(value)
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
    obj.select_set(False)
    return obj


def create_face_assets(analysis: dict) -> dict:
    width = analysis["head_width"]
    depth = analysis["head_depth"]
    height = analysis["head_height"]
    forward = analysis["forward_sign"]
    eyes = analysis["eyes"]
    face_x = analysis["face_x"]

    skin = material("AINA_Mpfb_Skin", (0.72, 0.48, 0.39, 1.0), 0.48)
    sclera = material("AINA_Mpfb_Sclera", (0.82, 0.84, 0.88, 1.0), 0.24)
    iris_mat = material("AINA_Mpfb_Iris", (0.025, 0.19, 0.22, 1.0), 0.18)
    pupil_mat = material("AINA_Mpfb_Pupil", (0.002, 0.004, 0.008, 1.0), 0.14)
    highlight_mat = material("AINA_Mpfb_Highlight", (0.94, 0.98, 1.0, 1.0), 0.08)
    brow_mat = material("AINA_Mpfb_Brow", (0.10, 0.11, 0.14, 1.0), 0.44)
    lash_mat = material("AINA_Mpfb_Lash", (0.005, 0.006, 0.010, 1.0), 0.32)
    hair_main = material("AINA_Mpfb_Silver_Hair", (0.44, 0.50, 0.62, 1.0), 0.31, 0.04)
    hair_light = material("AINA_Mpfb_Silver_Strands", (0.68, 0.73, 0.84, 1.0), 0.27, 0.03)

    created = []
    eye_radius = 0.078 * width
    for index, centre in enumerate(eyes):
        side = "L" if centre[0] < face_x else "R"
        globe = create_sphere(
            f"AINA_Eye_{side}",
            centre,
            (eye_radius, 0.92 * eye_radius, eye_radius),
            sclera,
        )
        iris_centre = centre + np.array([0.0, forward * 0.91 * eye_radius, 0.0])
        iris = create_sphere(
            f"AINA_Iris_{side}",
            iris_centre,
            (0.49 * eye_radius, 0.08 * eye_radius, 0.49 * eye_radius),
            iris_mat,
            40,
            20,
        )
        pupil_centre = centre + np.array([0.0, forward * 0.97 * eye_radius, 0.0])
        pupil = create_sphere(
            f"AINA_Pupil_{side}",
            pupil_centre,
            (0.19 * eye_radius, 0.05 * eye_radius, 0.19 * eye_radius),
            pupil_mat,
            32,
            16,
        )
        highlight = create_sphere(
            f"AINA_EyeHighlight_{side}",
            centre + np.array([-0.15 * eye_radius, forward * eye_radius, 0.18 * eye_radius]),
            (0.07 * eye_radius, 0.035 * eye_radius, 0.07 * eye_radius),
            highlight_mat,
            20,
            10,
        )
        created.extend([globe, iris, pupil, highlight])

        side_sign = -1.0 if side == "L" else 1.0
        brow_points = [
            centre + np.array([-0.62 * eye_radius, forward * 1.04 * eye_radius, 0.72 * eye_radius]),
            centre + np.array([-0.12 * eye_radius, forward * 1.06 * eye_radius, 0.90 * eye_radius]),
            centre + np.array([0.42 * eye_radius, forward * 1.04 * eye_radius, 0.76 * eye_radius]),
            centre + np.array([0.72 * eye_radius, forward * 1.01 * eye_radius, 0.54 * eye_radius]),
        ]
        if side_sign > 0:
            brow_points = [
                centre + np.array([0.62 * eye_radius, forward * 1.04 * eye_radius, 0.72 * eye_radius]),
                centre + np.array([0.12 * eye_radius, forward * 1.06 * eye_radius, 0.90 * eye_radius]),
                centre + np.array([-0.42 * eye_radius, forward * 1.04 * eye_radius, 0.76 * eye_radius]),
                centre + np.array([-0.72 * eye_radius, forward * 1.01 * eye_radius, 0.54 * eye_radius]),
            ]
        brow = create_curve_mesh(f"AINA_Brow_{side}", brow_points, 0.08 * eye_radius, brow_mat)
        lash_points = [
            centre + np.array([-0.78 * eye_radius, forward * 1.02 * eye_radius, 0.07 * eye_radius]),
            centre + np.array([-0.30 * eye_radius, forward * 1.05 * eye_radius, 0.28 * eye_radius]),
            centre + np.array([0.28 * eye_radius, forward * 1.05 * eye_radius, 0.24 * eye_radius]),
            centre + np.array([0.82 * eye_radius, forward * 1.01 * eye_radius, 0.02 * eye_radius]),
        ]
        lash = create_curve_mesh(f"AINA_Lash_{side}", lash_points, 0.035 * eye_radius, lash_mat)
        created.extend([brow, lash])

    head_lo = analysis["head_bounds_min"]
    head_hi = analysis["head_bounds_max"]
    head_centre = 0.5 * (head_lo + head_hi)
    cap_centre = np.array([face_x, head_centre[1] - forward * 0.04 * depth, head_lo[2] + 0.72 * height])
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, location=tuple(cap_centre))
    cap = bpy.context.object
    cap.name = "AINA_Silver_Hair_Cap"
    cap.scale = (0.55 * width, 0.47 * depth, 0.46 * height)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    # Remove the lower/back underside so the face remains open.
    mesh = cap.data
    world = world_vertices(cap)
    keep = world[:, 2] > head_lo[2] + 0.48 * height
    if np.any(~keep):
        bpy.context.view_layer.objects.active = cap
        cap.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        for vertex, retain in zip(mesh.vertices, keep):
            vertex.select = not bool(retain)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.delete(type="VERT")
        bpy.ops.object.mode_set(mode="OBJECT")
        cap.select_set(False)
    assign_material(cap, hair_main)
    created.append(cap)

    bun_centre = np.array([
        face_x,
        head_centre[1] - forward * 0.34 * depth,
        head_lo[2] + 0.82 * height,
    ])
    bun = create_sphere(
        "AINA_Silver_Hair_Bun",
        bun_centre,
        (0.27 * width, 0.24 * depth, 0.22 * height),
        hair_main,
        48,
        24,
    )
    created.append(bun)

    # Swept centre-part and face-framing strands.
    strand_specs = []
    for side in (-1.0, 1.0):
        strand_specs.extend([
            [
                np.array([face_x + side * 0.03 * width, head_centre[1] + forward * 0.30 * depth, head_lo[2] + 0.92 * height]),
                np.array([face_x + side * 0.22 * width, head_centre[1] + forward * 0.36 * depth, head_lo[2] + 0.76 * height]),
                np.array([face_x + side * 0.37 * width, head_centre[1] + forward * 0.33 * depth, head_lo[2] + 0.57 * height]),
                np.array([face_x + side * 0.40 * width, head_centre[1] + forward * 0.29 * depth, head_lo[2] + 0.37 * height]),
            ],
            [
                np.array([face_x + side * 0.12 * width, head_centre[1] + forward * 0.28 * depth, head_lo[2] + 0.88 * height]),
                np.array([face_x + side * 0.30 * width, head_centre[1] + forward * 0.34 * depth, head_lo[2] + 0.70 * height]),
                np.array([face_x + side * 0.44 * width, head_centre[1] + forward * 0.30 * depth, head_lo[2] + 0.48 * height]),
                np.array([face_x + side * 0.38 * width, head_centre[1] + forward * 0.28 * depth, head_lo[2] + 0.30 * height]),
            ],
        ])
    for index, points in enumerate(strand_specs):
        strand = create_curve_mesh(
            f"AINA_Silver_Strand_{index:02d}",
            points,
            0.012 * width,
            hair_light,
        )
        created.append(strand)

    return {
        "skin_material": skin.name,
        "created_objects": [obj.name for obj in created],
    }


def assign_skin_material(human) -> None:
    skin = bpy.data.materials.get("AINA_Mpfb_Skin") or material(
        "AINA_Mpfb_Skin", (0.72, 0.48, 0.39, 1.0), 0.48
    )
    assign_material(human, skin)


def find_head_bone(rig):
    for name in ("head", "Head", "DEF-head", "DEF-Head"):
        bone = rig.data.bones.get(name)
        if bone:
            return bone
    candidates = [bone for bone in rig.data.bones if "head" in bone.name.lower()]
    return min(candidates, key=lambda bone: len(bone.name)) if candidates else None


def parent_face_assets_to_head(rig, names) -> dict:
    head = find_head_bone(rig)
    if not head:
        return {"head_bone": None, "objects": []}
    parented = []
    for name in names:
        obj = bpy.data.objects.get(name)
        if not obj or obj == rig:
            continue
        obj.parent = rig
        obj.parent_type = "BONE"
        obj.parent_bone = head.name
        obj.matrix_parent_inverse = rig.matrix_world.inverted()
        parented.append(name)
    return {"head_bone": head.name, "objects": parented}


def create_light(name: str, location, energy: float, size: float, target) -> None:
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = tuple(location)
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def setup_render(analysis: dict):
    width = analysis["head_width"]
    height = analysis["head_height"]
    depth = analysis["head_depth"]
    face_x = analysis["face_x"]
    forward = analysis["forward_sign"]
    eyes = analysis["eyes"]
    centre_y = float(0.5 * (analysis["head_bounds_min"][1] + analysis["head_bounds_max"][1]))
    eye_z = float(np.mean(eyes, axis=0)[2])
    target = np.array([face_x, centre_y, eye_z - 0.11 * height])
    distance = max(3.2 * height, 3.6 * width, 0.62)
    front = np.array([face_x, centre_y + forward * distance, target[2]])
    locations = {
        "front": front,
        "three_quarter": front + np.array([0.46 * distance, -forward * 0.10 * distance, 0.0]),
        "side": np.array([face_x + distance, centre_y, target[2]]),
        "left_45": front + np.array([-0.50 * distance, -forward * 0.12 * distance, 0.0]),
        "right_45": front + np.array([0.50 * distance, -forward * 0.12 * distance, 0.0]),
    }
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.world.color = (0.018, 0.022, 0.034)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = -0.35
    except Exception:
        pass
    create_light(
        "AINA_Mpfb_Key",
        front + np.array([0.72 * width, 0.0, 0.75 * height]),
        650,
        max(1.6 * width, 0.45),
        target,
    )
    create_light(
        "AINA_Mpfb_Fill",
        front + np.array([-0.85 * width, 0.18 * distance, 0.18 * height]),
        320,
        max(2.1 * width, 0.55),
        target,
    )
    create_light(
        "AINA_Mpfb_Rim",
        np.array([face_x, centre_y - forward * 0.72 * distance, target[2] + 0.55 * height]),
        430,
        max(1.8 * width, 0.50),
        target,
    )
    cameras = {}
    for name, location in locations.items():
        data = bpy.data.cameras.new(f"AINA_Mpfb_Camera_{name}")
        data.lens = 88
        camera = bpy.data.objects.new(f"AINA_Mpfb_Camera_{name}", data)
        bpy.context.collection.objects.link(camera)
        camera.location = tuple(location)
        camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
        cameras[name] = camera
    return scene, cameras, {
        "target": target.tolist(),
        "locations": {key: value.tolist() for key, value in locations.items()},
        "distance": distance,
    }


def render(scene, camera, path: Path) -> None:
    scene.camera = camera
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_suite(scene, cameras, human, out: Path) -> dict:
    preview = out / "Preview"
    preview.mkdir(exist_ok=True)
    outputs = {"beauty": {}, "clay": {}}
    for view in ("front", "three_quarter", "side", "left_45", "right_45"):
        path = preview / f"AINA_MPFB_CUSTOM_HEAD_{view.upper()}.png"
        render(scene, cameras[view], path)
        outputs["beauty"][view] = str(path)

    clay = material("AINA_Mpfb_Clay", (0.29, 0.33, 0.41, 1.0), 0.58)
    clay_eye = material("AINA_Mpfb_ClayEye", (0.60, 0.67, 0.78, 1.0), 0.28)
    old = {}
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        old[obj.name] = [slot.material for slot in obj.material_slots]
        lower = obj.name.lower()
        assign_material(obj, clay_eye if any(token in lower for token in ("eye", "iris", "pupil")) else clay)
    for view in ("front", "three_quarter", "side"):
        path = preview / f"AINA_MPFB_CUSTOM_HEAD_CLAY_{view.upper()}.png"
        render(scene, cameras[view], path)
        outputs["clay"][view] = str(path)
    for name, materials in old.items():
        obj = bpy.data.objects.get(name)
        if not obj:
            continue
        obj.data.materials.clear()
        for mat in materials:
            if mat:
                obj.data.materials.append(mat)
    return outputs


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)
    clear_scene()
    HumanService, TargetService = install_mpfb(args.mpfb_root)
    human, macro = create_character(HumanService, TargetService)
    analysis = analyse_head(human)
    original_vertices = len(human.data.vertices)
    original_polygons = len(human.data.polygons)
    original_shape_keys = (
        [key.name for key in human.data.shape_keys.key_blocks]
        if human.data.shape_keys else []
    )
    sculpt = sculpt_aina(human, analysis)
    assign_skin_material(human)
    face_assets = create_face_assets(analysis)

    rig = HumanService.add_builtin_rig(human, "game_engine")
    if rig is None:
        raise RuntimeError("MPFB game_engine rig could not be created")
    rig.name = "AINA_MPFB_Humanoid"
    parent_report = parent_face_assets_to_head(rig, face_assets["created_objects"])
    bpy.context.view_layer.update()

    if len(human.data.vertices) != original_vertices or len(human.data.polygons) != original_polygons:
        raise RuntimeError("MPFB body topology changed during AINA sculpture")
    scene, cameras, camera_report = setup_render(analysis)
    renders = render_suite(scene, cameras, human, args.out)

    blend_path = args.out / "AINA_MPFB_CUSTOM_HEAD_V1.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_MPFB_CUSTOM_HEAD_V1.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )

    final_shape_keys = (
        [key.name for key in human.data.shape_keys.key_blocks]
        if human.data.shape_keys else []
    )
    report = {
        "product": "AINA MPFB Custom Head v1",
        "direction_reset": True,
        "rejected_direction": "Blender Studio Rain chibi topology",
        "new_direction": "MPFB2 adult Asian female topology plus custom AINA head reconstruction",
        "source_project": "MakeHuman Community MPFB2",
        "source_assets_license": "CC0 1.0",
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "rain_mesh_used": False,
        "rain_shape_keys_used": False,
        "vertices": len(human.data.vertices),
        "polygons": len(human.data.polygons),
        "triangles": sum(max(1, len(poly.vertices) - 2) for poly in human.data.polygons),
        "topology_changed": False,
        "macro_details": macro,
        "original_shape_keys": original_shape_keys,
        "final_shape_keys": final_shape_keys,
        "rig": rig.name,
        "rig_bones": len(rig.data.bones),
        "head_parenting": parent_report,
        "head_analysis": {
            "head_region_vertices": len(analysis["head_ids"]),
            "head_width": analysis["head_width"],
            "head_depth": analysis["head_depth"],
            "head_height": analysis["head_height"],
            "eye_centres": [point.tolist() for point in analysis["eyes"]],
            "forward_sign": analysis["forward_sign"],
        },
        "sculpt": sculpt,
        "face_assets": face_assets,
        "camera": camera_report,
        "renders": renders,
        "identity_lock": False,
        "visual_identity_lock": False,
        "production_release": False,
        "candidate": True,
        "vrm_exported": False,
        "next_gate": "Compare the actual MPFB adult front/3Q/profile with approved AINA. Continue multi-view fitting only on this new adult topology; do not return to Rain.",
        "files": {"blend": str(blend_path), "glb": str(glb_path)},
    }
    (qa / "AINA_MPFB_CUSTOM_HEAD_V1_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
