#!/usr/bin/env python3
"""Real 3D AINA eye, neutral-mouth and collar-graft visual system."""
from __future__ import annotations

import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def _mesh(name, vertices, faces, material):
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata([tuple(v) for v in vertices], [], [tuple(f) for f in faces])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material:
        mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return obj


def _almond(name, center, material, side):
    """Build a shallow convex almond sclera instead of a flat eye card."""
    c = np.asarray(center, float)
    radius_x = 0.0182
    radius_z = 0.00655
    segments = 64
    ring_count = 5
    vertices = [(c[0], c[1] - 0.0088, c[2])]
    rings = []
    tail_sign = 1.0 if side == "L" else -1.0

    for ring_index in range(1, ring_count + 1):
        radial = ring_index / ring_count
        ring = []
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            cosine = math.cos(angle)
            sine = math.sin(angle)
            x = radius_x * radial * cosine
            vertical_scale = radius_z if sine >= 0.0 else radius_z * 0.68
            z = vertical_scale * radial * sine
            z += tail_sign * 0.00052 * (x / radius_x)
            # Edge sits close to the eyelid plane; centre bulges forward (-Y).
            y = c[1] - 0.00545 - 0.00335 * (1.0 - radial * radial)
            vertices.append((c[0] + x, y, c[2] + z))
            ring.append(1 + (ring_index - 1) * segments + i)
        rings.append(ring)

    faces = []
    first = rings[0]
    for i in range(segments):
        faces.append((0, first[i], first[(i + 1) % segments]))
    for ring_index in range(1, ring_count):
        inner = rings[ring_index - 1]
        outer = rings[ring_index]
        for i in range(segments):
            j = (i + 1) % segments
            faces.extend(((inner[i], outer[i], outer[j]), (inner[i], outer[j], inner[j])))

    obj = _mesh(name, vertices, faces, material)
    obj.shape_key_add(name="Basis")
    blink = "eyeBlinkLeft" if side == "L" else "eyeBlinkRight"
    wide = "eyeWideLeft" if side == "L" else "eyeWideRight"
    squint = "eyeSquintLeft" if side == "L" else "eyeSquintRight"

    key = obj.shape_key_add(name=blink)
    for point in key.data:
        point.co.z = c[2] + (point.co.z - c[2]) * 0.04
        point.co.y += 0.0055
    key = obj.shape_key_add(name=wide)
    for point in key.data:
        point.co.z = c[2] + (point.co.z - c[2]) * 1.13
    key = obj.shape_key_add(name=squint)
    for point in key.data:
        point.co.z = c[2] + (point.co.z - c[2]) * 0.56
    return obj


def _disc(name, center, radius, material, side, pupil=False):
    """Build a shallow convex iris/pupil cap with real geometry."""
    c = np.asarray(center, float)
    segments = 64
    ring_count = 4
    vertices = [(c[0], c[1] - 0.00072, c[2])]
    rings = []
    for ring_index in range(1, ring_count + 1):
        radial = ring_index / ring_count
        ring = []
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            x = radius * radial * math.cos(angle)
            z = radius * radial * (1.045 if not pupil else 1.0) * math.sin(angle)
            y = c[1] - 0.00072 * (1.0 - radial * radial)
            vertices.append((c[0] + x, y, c[2] + z))
            ring.append(1 + (ring_index - 1) * segments + i)
        rings.append(ring)

    faces = []
    for i in range(segments):
        faces.append((0, rings[0][i], rings[0][(i + 1) % segments]))
    for ring_index in range(1, ring_count):
        inner = rings[ring_index - 1]
        outer = rings[ring_index]
        for i in range(segments):
            j = (i + 1) % segments
            faces.extend(((inner[i], outer[i], outer[j]), (inner[i], outer[j], inner[j])))

    obj = _mesh(name, vertices, faces, material)
    obj.shape_key_add(name="Basis")
    blink = "eyeBlinkLeft" if side == "L" else "eyeBlinkRight"
    key = obj.shape_key_add(name=blink)
    for point in key.data:
        point.co.z = c[2] + (point.co.z - c[2]) * 0.04
        point.co.y += 0.0060

    directions = {
        ("L", "eyeLookUpLeft"): (0, 0, 0.0020),
        ("R", "eyeLookUpRight"): (0, 0, 0.0020),
        ("L", "eyeLookDownLeft"): (0, 0, -0.0018),
        ("R", "eyeLookDownRight"): (0, 0, -0.0018),
        ("L", "eyeLookInLeft"): (-0.0018, 0, 0),
        ("R", "eyeLookInRight"): (0.0018, 0, 0),
        ("L", "eyeLookOutLeft"): (0.0018, 0, 0),
        ("R", "eyeLookOutRight"): (-0.0018, 0, 0),
    }
    for (direction_side, key_name), delta in directions.items():
        if direction_side != side:
            continue
        key = obj.shape_key_add(name=key_name)
        for point in key.data:
            point.co += Vector(delta)
    return obj


def install(visual, release):
    base = visual.base
    original_uv_sphere = base.create_uv_sphere
    original_configure = base.configure_expressions
    original_collar = base.create_collar_and_accent

    def create_face_objects(face_path: Path, height, skin, eye_mat, teeth_mat, mouth_mat):
        raw, faces = base.read_obj(face_path)
        mapped = base.map_face_vertices(raw, height)
        roots, groups = base.component_data(len(raw), faces)
        head_root = max(groups, key=lambda root: len(groups[root]))
        eye_roots = [root for root, group in groups.items() if 650 < len(group) < 900]
        eye_roots = sorted(eye_roots, key=lambda root: float(mapped[groups[root], 0].mean()))
        if len(eye_roots) != 2:
            raise RuntimeError("Expected two FaceVerse eye components")
        oral_roots = sorted(
            [root for root in groups if root != head_root and root not in eye_roots],
            key=lambda root: len(groups[root]),
            reverse=True,
        )
        mapped = visual.polish_real_face(mapped, groups[head_root], [groups[root] for root in eye_roots])
        keep_mask = np.array([roots[int(face[0])] not in set(eye_roots) for face in faces], dtype=bool)
        head_faces = faces[keep_mask]
        head = base.mesh_object("AINA_Face_v15_5", mapped, head_faces)
        # Neutral teeth are deliberately darkened to prevent the old white block
        # from showing through the resting lip seam. Mouth expressions still use
        # the same topology and controls.
        head.data.materials.append(skin)
        head.data.materials.append(mouth_mat)
        head.data.materials.append(mouth_mat)
        face_roots = [roots[int(face[0])] for face in faces[keep_mask]]
        oral_big = set(oral_roots[:2])
        for polygon, root in zip(head.data.polygons, face_roots):
            polygon.material_index = 0 if root == head_root else (1 if root in oral_big else 2)
            polygon.use_smooth = True

        lm = mapped[visual.K]
        centers = {"R": lm[36:42].mean(0), "L": lm[42:48].mean(0)}
        eyes = []
        for side in ("R", "L"):
            center = centers[side].copy()
            center[1] = -0.00035
            eye = _almond("AINA_Eye_" + side, center, eye_mat, side)
            root = eye_roots[0] if side == "R" else eye_roots[1]
            eyes.append((eye, np.asarray(groups[root], np.int32), center))
        tongue_ids = groups[oral_roots[-1]] if oral_roots else np.array([], dtype=np.int32)
        return head, eyes, mapped, groups, head_root, oral_roots, tongue_ids

    def create_uv_sphere(name, location, scale, material, parent=None, rig=None):
        if name.startswith("AINA_Iris_") or name.startswith("AINA_Pupil_"):
            side = name.rsplit("_", 1)[-1]
            location = np.asarray(location, float)
            radius = 0.00565 if name.startswith("AINA_Iris_") else 0.00220
            location[1] = -0.01185 if name.startswith("AINA_Iris_") else -0.01245
            obj = _disc(name, location, radius, material, side, pupil=name.startswith("AINA_Pupil_"))
            if parent and rig:
                base.bone_parent_preserve(obj, rig, parent)
            return obj
        return original_uv_sphere(name, location, scale, material, parent, rig)

    def configure_expressions(rig, head):
        configured = original_configure(rig, head)
        from io_scene_vrm.editor.extension import get_armature_extension
        preset = get_armature_extension(rig.data).vrm1.expressions.preset
        for preset_name, items in base.PRESET_BINDS.items():
            expression = getattr(preset, preset_name)
            for key_name, weight in items:
                for object_name in (
                    "AINA_Eye_L", "AINA_Eye_R", "AINA_Iris_L", "AINA_Iris_R", "AINA_Pupil_L", "AINA_Pupil_R"
                ):
                    obj = bpy.data.objects.get(object_name)
                    if not obj or not obj.data.shape_keys or key_name not in obj.data.shape_keys.key_blocks:
                        continue
                    bind = expression.morph_target_binds.add()
                    bind.node.mesh_object_name = obj.name
                    bind.index = key_name
                    bind.weight = float(weight)
        return configured

    def create_collar_and_accent(rig, suit_mat, accent_mat):
        original_collar(rig, suit_mat, accent_mat)
        collar = bpy.data.objects.get("AINA_High_Collar")
        if collar:
            inverse = collar.matrix_world.inverted()
            for vertex in collar.data.vertices:
                world = collar.matrix_world @ vertex.co
                if world.z > 1.515:
                    world.z = 1.515
                    vertex.co = inverse @ world
            collar.data.update()

    def setup_render(out: Path):
        scene = bpy.context.scene
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.render.image_settings.file_format = "PNG"
        scene.render.film_transparent = False
        scene.world.color = (0.075, 0.085, 0.11)
        try:
            scene.view_settings.look = "AgX - Medium High Contrast"
            scene.view_settings.exposure = 0.25
        except Exception:
            pass
        for obj in scene.objects:
            if obj.type == "MESH":
                for polygon in obj.data.polygons:
                    polygon.use_smooth = True
        for obj in list(scene.objects):
            if obj.type in {"LIGHT", "CAMERA"}:
                bpy.data.objects.remove(obj, do_unlink=True)

        def area(name, location, energy, size, target=(0, 0, 1.59)):
            data = bpy.data.lights.new(name, "AREA")
            data.energy = energy
            data.shape = "DISK"
            data.size = size
            obj = bpy.data.objects.new(name, data)
            bpy.context.collection.objects.link(obj)
            obj.location = location
            obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()

        area("AINA_Key", (1.35, -1.75, 2.25), 680, 2.5)
        area("AINA_Fill", (-1.45, -1.55, 1.90), 330, 2.7)
        area("AINA_Rim", (0, 1.65, 2.25), 420, 2.3)
        area("AINA_FaceSoft", (0, -2.10, 1.60), 95, 3.0)

        camera_data = bpy.data.cameras.new("AINA_Camera")
        camera = bpy.data.objects.new("AINA_Camera", camera_data)
        bpy.context.collection.objects.link(camera)
        scene.camera = camera
        previews = out / "Preview"
        previews.mkdir(parents=True, exist_ok=True)

        def clear_all():
            for obj in scene.objects:
                if obj.type == "MESH" and obj.data.shape_keys:
                    for key in obj.data.shape_keys.key_blocks:
                        key.value = 0.0

        def apply(values):
            for obj in scene.objects:
                if obj.type != "MESH" or not obj.data.shape_keys:
                    continue
                for key_name, value in values.items():
                    if key_name in obj.data.shape_keys.key_blocks:
                        obj.data.shape_keys.key_blocks[key_name].value = float(value)

        def render(name, location, target, values, resolution=(768, 768)):
            clear_all()
            apply(values)
            camera.location = location
            camera.data.lens = 82
            camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
            scene.render.resolution_x = resolution[0]
            scene.render.resolution_y = resolution[1]
            scene.render.resolution_percentage = 100
            scene.render.filepath = str(previews / name)
            bpy.ops.render.render(write_still=True)

        cases = {
            "AINA_REAL_NEUTRAL_FRONT.png": {},
            "AINA_REAL_HAPPY_FRONT.png": {"mouthSmileLeft": 0.82, "mouthSmileRight": 0.82, "cheekSquintLeft": 0.30, "cheekSquintRight": 0.30},
            "AINA_REAL_SURPRISED_FRONT.png": {"browInnerUp": 0.55, "eyeWideLeft": 0.86, "eyeWideRight": 0.86, "jawOpen": 0.58},
            "AINA_REAL_BLINK_FRONT.png": {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0},
            "AINA_REAL_AA_FRONT.png": {"jawOpen": 0.72, "mouthFunnel": 0.20},
        }
        for name, values in cases.items():
            render(name, (0, -1.02, 1.615), (0, 0, 1.610), values)
        render("AINA_REAL_NEUTRAL_3Q.png", (0.36, -0.96, 1.62), (0, 0, 1.605), {})
        render("AINA_REAL_FULL_BODY_FRONT.png", (0, -4.7, 1.05), (0, 0, 0.98), {}, (1024, 1536))
        clear_all()
        return [str(path) for path in sorted(previews.glob("AINA_REAL_*.png"))]

    base.create_face_objects = create_face_objects
    base.create_uv_sphere = create_uv_sphere
    base.configure_expressions = configure_expressions
    base.create_collar_and_accent = create_collar_and_accent
    base.setup_render = setup_render
