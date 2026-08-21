#!/usr/bin/env python3
"""AINA final VRM production with approved-reference facial appearance.

This stage does not generate a replacement portrait.  It projects colour cues
from the already-approved AINA front/3Q/side references onto the real dense
17,161-vertex skin as a blendable vertex-colour attribute, preserving the 3D
Mesh, topology and every facial morph.  It also builds real eyebrow and upper
lash ribbon geometry from the fitted facial landmarks.  The layered ribbon-card
silver updo, full humanoid body, 52 controls, VRM 1.0 patch and clean reimport
pipeline remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

import aina_vitruvian_dense_identity_finalization as dense
import aina_vitruvian_final_visual_lock as lock
import aina_vitruvian_final_vrm_production_v5 as v5


base = v5.base
_ARGS = None
_MATERIALS = None
_VISUAL_SURFACE_REPORT = {}


def parse_args_v6() -> argparse.Namespace:
    global _ARGS
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-glb", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--dense-landmarks", type=Path, required=True)
    parser.add_argument("--approved-front", type=Path, required=True)
    parser.add_argument("--approved-q3", type=Path, required=True)
    parser.add_argument("--approved-side", type=Path, required=True)
    parser.add_argument("--visual-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    _ARGS = parser.parse_args(argv)
    return _ARGS


base.parse_args = parse_args_v6


def key_array(key) -> np.ndarray:
    values = np.empty(len(key.data) * 3, dtype=np.float64)
    key.data.foreach_get("co", values)
    return values.reshape(-1, 3)


def load_blender_image(path: Path):
    image = bpy.data.images.load(str(path.resolve()), check_existing=False)
    width, height = image.size
    values = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(values)
    # Blender's image buffer is bottom-to-top; dense landmark pixels are
    # conventional top-to-bottom.
    values = values.reshape(height, width, 4)[::-1].copy()
    return values[:, :, :3], width, height


def bilinear(image: np.ndarray, points: np.ndarray):
    height, width = image.shape[:2]
    x = np.clip(points[:, 0], 0.0, width - 1.001)
    y = np.clip(points[:, 1], 0.0, height - 1.001)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def apply_similarity(points: np.ndarray, transform: dict):
    scale = float(transform["scale"])
    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    source_center = np.asarray(transform["source_center"], dtype=np.float64)
    destination_center = np.asarray(transform["destination_center"], dtype=np.float64)
    return scale * ((points - source_center) @ rotation.T) + destination_center


def world_normals(obj) -> np.ndarray:
    transform = obj.matrix_world.to_3x3().inverted().transposed()
    values = np.asarray([(transform @ vertex.normal).normalized()[:] for vertex in obj.data.vertices], dtype=np.float64)
    return values


def projected_colors(scene, skin, meshes, dense_data: dict, image_paths: dict[str, Path]):
    basis = skin.data.shape_keys.key_blocks.get("Basis") or skin.data.shape_keys.key_blocks[0]
    local = key_array(basis)
    world = lock.world_vertices(skin, local)
    normals = world_normals(skin)
    setup = lock.build_setup(scene, meshes)
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    base_color = np.asarray([0.70, 0.48, 0.44], dtype=np.float64)
    accumulated = np.zeros((len(local), 3), dtype=np.float64)
    denominator = np.zeros(len(local), dtype=np.float64)
    per_view = {}
    nearest_feature_index = np.full(len(local), -1, dtype=np.int64)
    nearest_feature_distance = np.full(len(local), np.inf, dtype=np.float64)

    for view, priority in (("front", 1.0), ("three_quarter", 0.72), ("side", 0.34)):
        approved_item = dense_data.get("approved", {}).get(view)
        model_item = dense_data.get("model", {}).get(view)
        if not approved_item or not model_item:
            continue
        approved_all = np.asarray(approved_item["landmarks_xy"], dtype=np.float64)
        model_all = np.asarray(model_item["landmarks_xy"], dtype=np.float64)
        count = min(len(approved_all), len(model_all), 468)
        indices = dense.selected_indices(count)
        approved = approved_all[indices]
        model = model_all[indices]
        weights = np.asarray([dense.point_weight(int(index), view) for index in indices], dtype=np.float64)
        mapped_model, transform = lock.weighted_similarity(model, approved, weights)
        residual = approved - mapped_model

        camera = lock.camera_for_view(scene, view, setup)
        projected = lock.project_points(scene, camera, skin, local)
        projected_xy = projected[:, :2]
        distances2 = np.sum((projected_xy[:, None, :] - model[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(distances2, axis=1)
        nearest_distance = np.sqrt(distances2[np.arange(len(local)), nearest])
        k = min(12, len(indices))
        nearest_k = np.argpartition(distances2, k - 1, axis=1)[:, :k]
        distance_k = np.take_along_axis(distances2, nearest_k, axis=1)
        idw = 1.0 / np.maximum(distance_k + 16.0, 1e-6)
        idw /= np.maximum(idw.sum(axis=1, keepdims=True), 1e-9)
        local_residual = np.sum(residual[nearest_k] * idw[:, :, None], axis=1)
        approved_xy = apply_similarity(projected_xy, transform) + local_residual

        image, width, height = load_blender_image(image_paths[view])
        sampled = bilinear(image, approved_xy)
        inside = (
            (approved_xy[:, 0] >= 0.0)
            & (approved_xy[:, 0] < width - 1)
            & (approved_xy[:, 1] >= 0.0)
            & (approved_xy[:, 1] < height - 1)
            & (projected[:, 2] > 0.0)
        )
        camera_position = np.asarray(camera.location[:], dtype=np.float64)
        direction = camera_position[None, :] - world
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        facing = np.sum(normals * direction, axis=1)
        # Imported meshes occasionally carry globally reversed normals.  Resolve
        # the sign from the actual front-face anchor region, not by taking abs.
        anchor_vertices = lock.choose_anchor_vertices(projected, model, k=30)
        if float(np.median(facing[anchor_vertices])) < 0.0:
            facing *= -1.0
        facing = np.clip(facing, 0.0, 1.0) ** 1.65
        screen_support = np.exp(-0.5 * (nearest_distance / 34.0) ** 2)
        view_weight = priority * facing * screen_support * inside.astype(np.float64)

        dense_index = indices[nearest]
        is_lip = np.isin(dense_index, list(dense.LIPS))
        is_eye = np.isin(dense_index, list(dense.LEFT_EYE | dense.RIGHT_EYE))
        is_brow = np.isin(dense_index, list(dense.LEFT_BROW | dense.RIGHT_BROW))
        is_feature = is_lip | is_eye | is_brow | np.isin(dense_index, list(dense.NOSE))
        general = ~is_feature
        sampled[general] = sampled[general] * 0.78 + base_color * 0.22
        luminance = sampled @ np.asarray([0.2126, 0.7152, 0.0722])
        blue_hair_like = general & (sampled[:, 2] > sampled[:, 0] * 1.28) & (luminance < 0.22)
        view_weight[blue_hair_like] *= 0.18
        sampled = np.clip(sampled, 0.012, 0.96)

        accumulated += sampled * view_weight[:, None]
        denominator += view_weight
        closer = nearest_distance < nearest_feature_distance
        nearest_feature_distance[closer] = nearest_distance[closer]
        nearest_feature_index[closer] = dense_index[closer]
        per_view[view] = {
            "selected_dense_landmarks": int(len(indices)),
            "covered_vertices": int(np.sum(view_weight > 0.025)),
            "mean_weight": float(np.mean(view_weight)),
            "maximum_weight": float(np.max(view_weight)),
        }

    colors = np.tile(base_color, (len(local), 1))
    covered = denominator > 0.035
    colors[covered] = accumulated[covered] / denominator[covered, None]
    colors = np.clip(colors, 0.01, 0.98)
    report = {
        "covered_vertices": int(np.sum(covered)),
        "coverage_fraction": float(np.mean(covered)),
        "mean_accumulated_weight": float(np.mean(denominator)),
        "views": per_view,
        "minimum_color": colors.min(axis=0).tolist(),
        "maximum_color": colors.max(axis=0).tolist(),
    }
    return colors, report


def install_vertex_color_material(skin, colors: np.ndarray):
    name = "AINA_ApprovedFaceColor"
    existing = skin.data.color_attributes.get(name)
    if existing:
        skin.data.color_attributes.remove(existing)
    attribute = skin.data.color_attributes.new(name=name, type="FLOAT_COLOR", domain="POINT")
    rgba = np.c_[colors, np.ones(len(colors), dtype=np.float64)].astype(np.float32)
    attribute.data.foreach_set("color", rgba.ravel())
    skin.data.color_attributes.active_color = attribute
    skin.data.color_attributes.render_color_index = list(skin.data.color_attributes).index(attribute)
    modified = []
    for material in {slot.material for slot in skin.material_slots if slot.material}:
        material.use_nodes = True
        tree = material.node_tree
        shader = tree.nodes.get("Principled BSDF") if tree else None
        if not shader:
            continue
        for node in list(tree.nodes):
            if node.bl_idname == "ShaderNodeVertexColor" and node.layer_name == name:
                tree.nodes.remove(node)
        vertex = tree.nodes.new("ShaderNodeVertexColor")
        vertex.layer_name = name
        vertex.label = "Approved AINA facial appearance"
        tree.links.new(vertex.outputs["Color"], shader.inputs["Base Color"])
        if shader.inputs.get("Roughness"):
            shader.inputs["Roughness"].default_value = 0.42
        if shader.inputs.get("Subsurface Weight"):
            shader.inputs["Subsurface Weight"].default_value = 0.070
        modified.append(material.name)
    return {"attribute": name, "materials": modified}


_original_restore_materials = base.restore_materials


def restore_materials_v6(meshes, skin, body_objects):
    global _MATERIALS, _VISUAL_SURFACE_REPORT
    materials = _original_restore_materials(meshes, skin, body_objects)
    if _ARGS is None:
        raise RuntimeError("AINA v6 arguments were not initialized")
    dense_data = json.loads(_ARGS.dense_landmarks.read_text())
    colors, projection = projected_colors(
        bpy.context.scene,
        skin,
        meshes,
        dense_data,
        {
            "front": _ARGS.approved_front,
            "three_quarter": _ARGS.approved_q3,
            "side": _ARGS.approved_side,
        },
    )
    material_report = install_vertex_color_material(skin, colors)
    _VISUAL_SURFACE_REPORT["approved_face_projection"] = projection
    _VISUAL_SURFACE_REPORT["vertex_color_material"] = material_report
    _MATERIALS = materials
    return materials


base.restore_materials = restore_materials_v6


def front_dense_anchors(scene, skin, meshes, dense_data):
    item = dense_data["model"]["front"]
    model = np.asarray(item["landmarks_xy"], dtype=np.float64)[:468]
    setup = lock.build_setup(scene, meshes)
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    camera = lock.camera_for_view(scene, "front", setup)
    basis = skin.data.shape_keys.key_blocks.get("Basis") or skin.data.shape_keys.key_blocks[0]
    local = key_array(basis)
    projected = lock.project_points(scene, camera, skin, local)
    anchors = lock.choose_anchor_vertices(projected, model, k=36)
    world = lock.world_vertices(skin, local)
    return world[anchors], setup


def ribbon_from_indices(name, world_landmarks, indices, width, material, radial_center, outward, armature, bone):
    points = np.asarray([world_landmarks[index] for index in indices], dtype=np.float64)
    points = points + outward[None, :] * 0.00115
    obj = v5.create_ribbon(name, points, width, width * 0.55, material, radial_center, thickness=0.00035)
    base.preserve_bone_parent(obj, armature, bone)
    return obj


def create_brows_and_lashes(skin, meshes, armature, rig_info, setup, materials):
    dense_data = json.loads(_ARGS.dense_landmarks.read_text())
    world_landmarks, setup = front_dense_anchors(bpy.context.scene, skin, meshes, dense_data)
    outward = np.asarray(setup["locations"]["front"] - setup["target"], dtype=np.float64)
    outward /= max(float(np.linalg.norm(outward)), 1e-9)
    radial_center = np.asarray(setup["center"], dtype=np.float64)
    lash_material = materials["lash"]
    created = []

    brow_paths = [
        ("AINA_Brow_Left_Final", [70, 63, 105, 66, 107]),
        ("AINA_Brow_Right_Final", [336, 296, 334, 293, 300]),
    ]
    for name, indices in brow_paths:
        created.append(ribbon_from_indices(name, world_landmarks, indices, 0.0046, lash_material, radial_center, outward, armature, rig_info["head_bone"]))

    eye_paths = [
        ("L", [33, 160, 158, 133]),
        ("R", [362, 385, 387, 263]),
    ]
    for side, indices in eye_paths:
        upper = np.asarray([world_landmarks[index] for index in indices], dtype=np.float64) + outward[None, :] * 0.00105
        lash_band = v5.create_ribbon(f"AINA_Upper_Lash_Band_{side}", upper, 0.0022, 0.0012, lash_material, radial_center, 0.00025)
        base.preserve_bone_parent(lash_band, armature, rig_info["head_bone"])
        created.append(lash_band)
        # Tapered individual lashes follow the actual lid arc.
        samples = []
        for segment in range(len(upper) - 1):
            for t in np.linspace(0.12, 0.88, 3):
                samples.append(upper[segment] * (1.0 - t) + upper[segment + 1] * t)
        center = upper.mean(axis=0)
        for index, root in enumerate(samples):
            lateral = root - center
            lateral /= max(float(np.linalg.norm(lateral)), 1e-9)
            end = root + outward * 0.0042 + np.asarray([0.0, 0.0, 0.0024]) + lateral * 0.0010
            middle = 0.5 * (root + end) + outward * 0.0006
            lash = v5.create_ribbon(
                f"AINA_Upper_Lash_{side}_{index + 1}",
                np.asarray([root, middle, end]),
                0.00105,
                0.00010,
                lash_material,
                radial_center,
                0.00010,
            )
            base.preserve_bone_parent(lash, armature, rig_info["head_bone"])
            created.append(lash)
    return created


_original_create_silver_updo = base.create_silver_updo


def create_silver_updo_v6(skin, armature, rig_info, setup, hair_mat):
    global _VISUAL_SURFACE_REPORT
    objects = _original_create_silver_updo(skin, armature, rig_info, setup, hair_mat)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.name.startswith("AINA_Silver_") and not obj.name.startswith("AINA_Crown_")]
    extra = create_brows_and_lashes(skin, meshes, armature, rig_info, setup, _MATERIALS)
    _VISUAL_SURFACE_REPORT["real_brow_lash_geometry"] = {
        "object_count": len(extra),
        "objects": [obj.name for obj in extra],
    }
    return objects + extra


base.create_silver_updo = create_silver_updo_v6


_original_clean_reimport = base.clean_reimport_qa


def clean_reimport_v6(vrm_path, expected_skin_name, output):
    result = _original_clean_reimport(vrm_path, expected_skin_name, output)
    result["approved_visual_surface"] = _VISUAL_SURFACE_REPORT
    projection = _VISUAL_SURFACE_REPORT.get("approved_face_projection", {})
    result["approved_visual_surface_pass"] = (
        projection.get("coverage_fraction", 0.0) >= 0.26
        and _VISUAL_SURFACE_REPORT.get("real_brow_lash_geometry", {}).get("object_count", 0) >= 20
    )
    result["pass"] = bool(result.get("pass")) and result["approved_visual_surface_pass"]
    return result


base.clean_reimport_qa = clean_reimport_v6


if __name__ == "__main__":
    base.main()
