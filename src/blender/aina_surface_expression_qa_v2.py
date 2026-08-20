#!/usr/bin/env python3
"""AINA real-mesh expression convergence v2.

The script keeps the refined FaceVerse topology and 52 production controls, then
adds expression correctives that keep mouth and jaw poses away from the eyes and
nose. It also creates deforming real-geometry brows and a real mouth cavity so
emotions and visemes can be judged from Blender renders. No replacement effect
art is generated and no VRM is exported at this stage.
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
    for blocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
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


def key_array(key_block) -> np.ndarray:
    return np.asarray([point.co[:] for point in key_block.data], dtype=np.float64)


def set_key_array(key_block, coords: np.ndarray) -> None:
    try:
        key_block.data.foreach_set("co", coords.astype(np.float32).ravel())
    except Exception:
        for i, point in enumerate(coords):
            key_block.data[i].co = point


def post_correct_shape_keys(head, mapped: np.ndarray) -> dict:
    """Localize every control and reinforce readable emotion/viseme poses."""
    blocks = head.data.shape_keys.key_blocks
    basis = key_array(blocks["Basis"])
    lm = mapped[base.K]
    brow_left = lm[22:27].mean(0)
    brow_right = lm[17:22].mean(0)
    eye_left = lm[42:48].mean(0)
    eye_right = lm[36:42].mean(0)
    mouth = lm[48:60].mean(0)
    upper = lm[[49, 50, 51, 52, 53]].mean(0)
    lower = lm[[55, 56, 57, 58, 59]].mean(0)
    corner_left = lm[54]
    corner_right = lm[48]
    chin = lm[8]
    jaw = (mouth + chin) * 0.5
    cheek_left = (eye_left + lm[35] + corner_left) / 3.0
    cheek_right = (eye_right + lm[31] + corner_right) / 3.0

    eye_mask = np.maximum(
        base.weights(basis, eye_left, (0.044, 0.034, 0.027), 0.0, 1.08),
        base.weights(basis, eye_right, (0.044, 0.034, 0.027), 0.0, 1.08),
    )
    brow_mask = np.maximum(
        base.weights(basis, brow_left, (0.045, 0.035, 0.027), 0.0, 1.08),
        base.weights(basis, brow_right, (0.045, 0.035, 0.027), 0.0, 1.08),
    )
    mouth_mask = base.weights(basis, mouth, (0.055, 0.043, 0.035), 0.0, 1.05)
    cheek_mask = np.maximum(
        base.weights(basis, cheek_left, (0.047, 0.043, 0.040), 0.0, 1.06),
        base.weights(basis, cheek_right, (0.047, 0.043, 0.040), 0.0, 1.06),
    )
    nose_mask = base.weights(basis, lm[27:36].mean(0), (0.034, 0.035, 0.047), 0.0, 1.08)
    lower_z = np.clip(
        (mouth[2] + 0.010 - basis[:, 2]) / max(mouth[2] + 0.010 - (chin[2] - 0.025), 1e-6),
        0.0,
        1.0,
    )
    lower_front = np.exp(-0.5 * ((basis[:, 1] - mouth[1]) / 0.095) ** 4)
    lower_mask = lower_z * lower_front

    corrected = {}
    for name in base.SHAPE_KEYS:
        coords = key_array(blocks[name])
        delta = coords - basis

        if name.startswith("eye"):
            delta *= eye_mask[:, None]
            if name.startswith("eyeLook"):
                delta *= 0.15
        elif name.startswith("brow"):
            delta *= brow_mask[:, None]
        elif name.startswith("mouth"):
            delta *= mouth_mask[:, None]
        elif name.startswith("jaw"):
            delta *= lower_mask[:, None]
        elif name.startswith("cheek"):
            delta *= cheek_mask[:, None]
        elif name.startswith("noseSneer"):
            delta *= np.maximum(nose_mask, cheek_mask)[:, None]

        coords = basis + delta
        side = 1.0 if "Left" in name else (-1.0 if "Right" in name else 0.0)
        corner = corner_left if side > 0 else corner_right
        brow = brow_left if side > 0 else brow_right
        eye = eye_left if side > 0 else eye_right
        cheek = cheek_left if side > 0 else cheek_right

        if name.startswith("browDown"):
            base.shift_region(coords, brow, (0.037, 0.028, 0.022), (-0.0005 * side, 0.0, -0.0045), 0.05, 1.02)
        elif name == "browInnerUp":
            for center in (lm[21], lm[22]):
                base.shift_region(coords, center, (0.024, 0.024, 0.022), (0.0, 0.0, 0.0065), 0.04, 1.02)
        elif name.startswith("browOuterUp"):
            center = lm[[25, 26]].mean(0) if side > 0 else lm[[17, 18]].mean(0)
            base.shift_region(coords, center, (0.027, 0.023, 0.022), (0.0005 * side, 0.0, 0.0055), 0.04, 1.02)
        elif name.startswith("cheekSquint"):
            base.shift_region(coords, cheek, (0.036, 0.033, 0.029), (0.0, -0.0009, 0.0026), 0.03, 1.04)
        elif name == "jawForward":
            base.shift_region(coords, jaw, (0.058, 0.060, 0.052), (0.0, -0.0045, 0.0), 0.02, 1.02)
        elif name == "jawLeft":
            base.shift_region(coords, jaw, (0.060, 0.062, 0.052), (0.0045, 0.0, 0.0), 0.02, 1.02)
        elif name == "jawRight":
            base.shift_region(coords, jaw, (0.060, 0.062, 0.052), (-0.0045, 0.0, 0.0), 0.02, 1.02)
        elif name == "jawOpen":
            base.shift_region(coords, jaw, (0.060, 0.065, 0.055), (0.0, 0.0010, -0.0060), 0.02, 1.03)
            base.shift_region(coords, lower, (0.039, 0.030, 0.019), (0.0, -0.0008, -0.0048), 0.02, 1.02)
            base.shift_region(coords, upper, (0.037, 0.029, 0.017), (0.0, -0.0003, 0.0014), 0.02, 1.02)
        elif name == "mouthClose":
            base.scale_region(coords, mouth, (0.043, 0.030, 0.026), (1.0, 1.0, 0.12), 0.05, 1.02)
        elif name.startswith("mouthFrown"):
            base.shift_region(coords, corner, (0.027, 0.025, 0.021), (0.0006 * side, 0.0003, -0.0046), 0.04, 1.02)
        elif name == "mouthFunnel":
            base.scale_region(coords, mouth, (0.043, 0.032, 0.030), (0.76, 1.03, 1.18), 0.04, 1.03)
            base.shift_region(coords, mouth, (0.041, 0.031, 0.028), (0.0, -0.0024, 0.0), 0.04, 1.02)
        elif name.startswith("mouthLowerDown"):
            center = lm[[54, 55, 56]].mean(0) if side > 0 else lm[[48, 58, 59]].mean(0)
            base.shift_region(coords, center, (0.031, 0.026, 0.021), (0.0, -0.0004, -0.0042), 0.03, 1.02)
        elif name.startswith("mouthPress"):
            base.scale_region(coords, corner, (0.027, 0.025, 0.021), (0.98, 1.0, 0.55), 0.03, 1.02)
        elif name == "mouthPucker":
            base.scale_region(coords, mouth, (0.043, 0.032, 0.030), (0.70, 1.0, 1.20), 0.04, 1.03)
            base.shift_region(coords, mouth, (0.040, 0.031, 0.028), (0.0, -0.0038, 0.0), 0.04, 1.02)
        elif name.startswith("mouthSmile"):
            base.shift_region(coords, corner, (0.027, 0.025, 0.022), (0.0020 * side, -0.0005, 0.0048), 0.04, 1.02)
            base.shift_region(coords, cheek, (0.035, 0.032, 0.029), (0.0, -0.0007, 0.0012), 0.0, 1.02)
        elif name.startswith("mouthStretch"):
            base.shift_region(coords, corner, (0.028, 0.025, 0.022), (0.0048 * side, 0.0, 0.0002), 0.03, 1.02)
        elif name.startswith("mouthUpperUp"):
            center = lm[[52, 53, 54]].mean(0) if side > 0 else lm[[48, 49, 50]].mean(0)
            base.shift_region(coords, center, (0.030, 0.024, 0.020), (0.0, -0.0004, 0.0036), 0.03, 1.02)
        elif name.startswith("eyeBlink"):
            base.scale_region(coords, eye, (0.038, 0.028, 0.021), (1.0, 1.0, 0.04), 0.08, 1.02)
        elif name.startswith("eyeWide"):
            base.scale_region(coords, eye, (0.039, 0.029, 0.022), (1.0, 1.0, 1.22), 0.08, 1.02)
        elif name.startswith("eyeSquint"):
            base.scale_region(coords, eye, (0.039, 0.029, 0.022), (1.0, 1.0, 0.58), 0.08, 1.02)

        set_key_array(blocks[name], coords)
        moved = np.linalg.norm(coords - basis, axis=1)
        corrected[name] = {
            "max_m": float(moved.max()),
            "rms_m": float(np.sqrt(np.mean(moved * moved))),
            "moved_vertices": int(np.sum(moved > 1e-5)),
        }
    return corrected


def create_brow(name: str, points: np.ndarray, side_name: str, mat):
    points = np.asarray(points, dtype=np.float64).copy()
    points[:, 1] -= 0.0032
    points[:, 2] += 0.0005
    vertices = []
    for point in points:
        vertices.append((point[0], point[1], point[2] - 0.00075))
        vertices.append((point[0], point[1], point[2] + 0.00075))
    faces = []
    for i in range(len(points) - 1):
        a = 2 * i
        b = a + 1
        c = a + 2
        d = a + 3
        faces.extend(((a, c, d), (a, d, b)))
    obj = base.mesh_object(name, np.asarray(vertices), np.asarray(faces, dtype=np.int32))
    obj.data.materials.append(mat)
    obj.shape_key_add(name="Basis")
    basis = np.asarray(vertices, dtype=np.float64)
    inner_point = int(np.argmin(np.abs(points[:, 0])))
    outer_point = int(np.argmax(np.abs(points[:, 0])))
    index = np.arange(len(points), dtype=np.float64)
    span = max(abs(outer_point - inner_point), 1.0)
    inner_weight = np.clip(1.0 - np.abs(index - inner_point) / span, 0.0, 1.0)
    outer_weight = np.clip(1.0 - np.abs(index - outer_point) / span, 0.0, 1.0)

    def add_key(key_name: str, dz_per_point: np.ndarray, dx_per_point=None):
        coords = basis.copy()
        dx = np.zeros(len(points)) if dx_per_point is None else np.asarray(dx_per_point)
        for i in range(len(points)):
            coords[2 * i : 2 * i + 2, 0] += dx[i]
            coords[2 * i : 2 * i + 2, 2] += dz_per_point[i]
        key = obj.shape_key_add(name=key_name)
        set_key_array(key, coords)

    sign = 1.0 if side_name == "Left" else -1.0
    add_key(f"browDown{side_name}", -0.0045 * (0.75 + 0.25 * inner_weight), 0.0004 * sign * inner_weight)
    add_key(f"browOuterUp{side_name}", 0.0054 * outer_weight, 0.0004 * sign * outer_weight)
    add_key("browInnerUp", 0.0062 * inner_weight, -0.00025 * sign * inner_weight)
    return obj


def create_mouth_cavity(center: np.ndarray, mat):
    center = np.asarray(center, dtype=np.float64).copy()
    center[1] += 0.0040
    segments = 64

    def ellipse(rx: float, rz: float, z_shift: float = 0.0):
        vertices = [(center[0], center[1], center[2] + z_shift)]
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            vertices.append((center[0] + rx * np.cos(angle), center[1], center[2] + z_shift + rz * np.sin(angle)))
        return np.asarray(vertices, dtype=np.float64)

    basis = ellipse(0.0190, 0.00055)
    faces = [(0, 1 + i, 1 + ((i + 1) % segments)) for i in range(segments)]
    obj = base.mesh_object("AINA_Mouth_Cavity", basis, np.asarray(faces, dtype=np.int32))
    obj.data.materials.append(mat)
    obj.shape_key_add(name="Basis")
    shapes = {
        "jawOpen": ellipse(0.0185, 0.0068, -0.0018),
        "mouthFunnel": ellipse(0.0125, 0.0048),
        "mouthPucker": ellipse(0.0105, 0.0042),
        "mouthClose": ellipse(0.0185, 0.00020),
        "mouthStretchLeft": ellipse(0.0225, 0.0014),
        "mouthStretchRight": ellipse(0.0225, 0.0014),
        "mouthSmileLeft": ellipse(0.0205, 0.0018, 0.0008),
        "mouthSmileRight": ellipse(0.0205, 0.0018, 0.0008),
        "mouthFrownLeft": ellipse(0.0195, 0.0018, -0.0008),
        "mouthFrownRight": ellipse(0.0195, 0.0018, -0.0008),
    }
    for name, coords in shapes.items():
        key = obj.shape_key_add(name=name)
        set_key_array(key, coords)
    return obj


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

    skin = material("AINA_Skin", (0.52, 0.34, 0.31, 1.0), 0.48)
    lip = material("AINA_Lip", (0.42, 0.10, 0.12, 1.0), 0.36)
    tooth = material("AINA_Teeth", (0.72, 0.68, 0.61, 1.0), 0.34)
    mouth = material("AINA_MouthInner", (0.055, 0.008, 0.012, 1.0), 0.50)
    eye_white = material("AINA_EyeWhite", (0.78, 0.82, 0.88, 1.0), 0.20)
    iris = material("AINA_Iris", (0.08, 0.25, 0.36, 1.0), 0.16)
    pupil = material("AINA_Pupil", (0.003, 0.006, 0.012, 1.0), 0.16)
    brow_mat = material("AINA_Brow", (0.035, 0.028, 0.035, 1.0), 0.34)

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
    base.create_shape_keys(head, mapped, tongue_ids)
    shape_stats = post_correct_shape_keys(head, mapped)

    visible_objects = [head]
    centers = {"R": lm[36:42].mean(0), "L": lm[42:48].mean(0)}
    for side in ("R", "L"):
        center = centers[side].copy()
        center[1] = -0.00035
        sclera = eye_system._almond(f"AINA_Eye_{side}", center, eye_white, side)
        iris_center = center.copy(); iris_center[1] = -0.01185
        iris_obj = eye_system._disc(f"AINA_Iris_{side}", iris_center, 0.00565, iris, side, pupil=False)
        pupil_center = center.copy(); pupil_center[1] = -0.01245
        pupil_obj = eye_system._disc(f"AINA_Pupil_{side}", pupil_center, 0.00220, pupil, side, pupil=True)
        visible_objects.extend((sclera, iris_obj, pupil_obj))

    visible_objects.append(create_brow("AINA_Brow_Right", lm[17:22], "Right", brow_mat))
    visible_objects.append(create_brow("AINA_Brow_Left", lm[22:27], "Left", brow_mat))
    visible_objects.append(create_mouth_cavity(mouth_center, mouth))
    return head, visible_objects, mapped, shape_stats


def reset_shapes(objects) -> None:
    for obj in objects:
        keys = getattr(obj.data, "shape_keys", None)
        if keys:
            for key in keys.key_blocks:
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
    scene.world.color = (0.025, 0.030, 0.040)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
        scene.view_settings.exposure = 0.0
    except Exception:
        pass

    def area(name, location, energy, size, target=(0, 0, 1.61)):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy; data.shape = "DISK"; data.size = size
        obj = bpy.data.objects.new(name, data); bpy.context.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()

    area("AINA_Key", (1.25, -1.65, 2.20), 360, 2.3)
    area("AINA_Fill", (-1.35, -1.45, 1.88), 170, 2.6)
    area("AINA_Rim", (0, 1.50, 2.18), 240, 2.2)
    area("AINA_FaceSoft", (0, -1.95, 1.60), 50, 2.8)

    camera_data = bpy.data.cameras.new("AINA_Expression_Camera")
    camera = bpy.data.objects.new("AINA_Expression_Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera.data.lens = 86
    preview = out / "Preview"; preview.mkdir(parents=True, exist_ok=True)
    return scene, camera, preview


def render_case(scene, camera, preview: Path, objects, case_name: str, values, three_q=False):
    apply_case(objects, values)
    if three_q:
        camera.location = (0.34, -0.93, 1.62); target = (0, 0, 1.605); suffix = "3Q"
    else:
        camera.location = (0, -0.99, 1.615); target = (0, 0, 1.610); suffix = "FRONT"
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.resolution_x = 448; scene.render.resolution_y = 448; scene.render.resolution_percentage = 100
    path = preview / f"AINA_EXPR_{case_name.upper()}_{suffix}.png"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    return path


def expression_metrics(head):
    blocks = head.data.shape_keys.key_blocks
    basis = key_array(blocks["Basis"])
    result = {}
    eye_anchor = base.K[36:48]
    nose_anchor = base.K[27:36]
    jaw_anchor = base.K[:17]
    mouth_anchor = base.K[48:68]
    for case_name, values in CASES.items():
        posed = basis.copy()
        for key_name, weight in values.items():
            if key_name in blocks:
                posed += float(weight) * (key_array(blocks[key_name]) - basis)
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
    args = parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True); (out / "QA").mkdir(exist_ok=True)
    clear_scene()
    head, visible_objects, mapped, shape_stats = build_character(args.face, args.height)
    scene, camera, preview = setup_scene(out)
    rendered = [render_case(scene, camera, preview, visible_objects, name, values, False) for name, values in CASES.items()]
    rendered.append(render_case(scene, camera, preview, visible_objects, "neutral", CASES["neutral"], True))
    rendered.append(render_case(scene, camera, preview, visible_objects, "happy", CASES["happy"], True))
    reset_shapes(visible_objects)
    blend_path = out / "AINA_SURFACE_EXPRESSION_QA_V2.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    key_names = [key.name for key in head.data.shape_keys.key_blocks if key.name != "Basis"]
    missing = sorted(set(base.SHAPE_KEYS) - set(key_names)); extra = sorted(set(key_names) - set(base.SHAPE_KEYS))
    metrics = expression_metrics(head)
    render_missing = [str(path) for path in rendered if not path.exists() or path.stat().st_size < 5000]
    visemes = {"aa", "ih", "ou", "ee", "oh"}
    automated_pass = (
        len(key_names) == 52 and not missing and not extra and not render_missing
        and all(np.isfinite(item["max_m"]) and item["max_m"] < 0.035 for item in metrics.values())
        and all(metrics[name]["eye_anchor_max_m"] < 0.0012 for name in visemes)
        and all(metrics[name]["nose_anchor_max_m"] < 0.0015 for name in visemes)
        and metrics["blink"]["mouth_anchor_max_m"] < 0.0010
        and metrics["aa"]["mouth_anchor_max_m"] > 0.0080
        and metrics["happy"]["mouth_anchor_max_m"] > 0.0035
        and metrics["angry"]["eye_anchor_max_m"] > 0.0020
        and metrics["sad"]["eye_anchor_max_m"] > 0.0015
    )
    qa = {
        "product": "AINA Surface-Refined Expression QA v2",
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
        "next_gate": "Inspect every real Blender render; only visually accepted expressions may enter final VRM packaging.",
        "files": {"blend": str(blend_path), "blend_bytes": blend_path.stat().st_size, "preview_dir": str(preview)},
    }
    report = out / "QA" / "AINA_SURFACE_EXPRESSION_QA_V2.json"
    report.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps({"shape_controls": len(key_names), "render_count": len(rendered), "automated_expression_gate": automated_pass, "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
