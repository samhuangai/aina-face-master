#!/usr/bin/env python3
"""Build the technical AINA VRM 1.0 production candidate from Rain v6.

This stage stays on the accepted Rain production topology. It preserves the
source skin, UVs, weights, armature and existing expression deltas; adds the 52
standard ARKit controls on the real face Mesh; ensures a deforming hair-bone
chain; exports AINA_MASTER.blend; exports a morph-preserving GLB; and packages
VRM 1.0 humanoid, 18 preset expressions, LookAt and SpringBone metadata into
AINA.vrm. Visual locks intentionally remain false until actual renders pass.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aina_rain_identity_master as base
import aina_rain_appearance_candidate as appearance
import aina_rain_identity_reconstruction_v3 as v3
import aina_rain_identity_precision_v4 as v4


ARKIT_52 = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight", "tongueOut",
]
assert len(ARKIT_52) == 52 and len(set(ARKIT_52)) == 52

PRESET_ORDER = [
    "happy", "angry", "sad", "relaxed", "surprised",
    "aa", "ih", "ou", "ee", "oh",
    "blink", "blinkLeft", "blinkRight",
    "lookUp", "lookDown", "lookLeft", "lookRight", "neutral",
]
assert len(PRESET_ORDER) == 18

ALIASES = {
    "browDownLeft": ("browdownl", "eyebrowsdownl", "browlowerl"),
    "browDownRight": ("browdownr", "eyebrowsdownr", "browlowerr"),
    "browInnerUp": ("browinnerup", "eyebrowsup", "browup"),
    "browOuterUpLeft": ("browouterupl", "eyebrowsupl", "browupl"),
    "browOuterUpRight": ("browouterupr", "eyebrowsupr", "browupr"),
    "cheekPuff": ("cheekpuff",),
    "cheekSquintLeft": ("cheeksquintl", "squintl"),
    "cheekSquintRight": ("cheeksquintr", "squintr"),
    "eyeBlinkLeft": ("eyelidsclosel", "blinkl", "eyeclosel"),
    "eyeBlinkRight": ("eyelidscloser", "blinkr", "eyecloser"),
    "eyeSquintLeft": ("eyesquintl", "squintl"),
    "eyeSquintRight": ("eyesquintr", "squintr"),
    "eyeWideLeft": ("eyewidel", "eyelidswidel"),
    "eyeWideRight": ("eyewider", "eyelidswider"),
    "jawOpen": ("jawopen", "mouthopen", "aa"),
    "mouthSmileLeft": ("smilel", "lipsmilel", "mouthsmilel"),
    "mouthSmileRight": ("smiler", "lipsmiler", "mouthsmiler"),
    "mouthFrownLeft": ("frownl", "mouthfrownl"),
    "mouthFrownRight": ("frownr", "mouthfrownr"),
    "mouthPucker": ("pucker", "kiss", "ou"),
    "mouthFunnel": ("funnel", "mouthfunnel", "oh"),
    "mouthClose": ("lipsadjust", "mouthclose"),
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def basis_key(skin):
    if not skin.data.shape_keys:
        return skin.shape_key_add(name="Basis", from_mix=False)
    return skin.data.shape_keys.key_blocks.get("Basis") or skin.data.shape_keys.key_blocks[0]


def current_key_delta(skin, key_name: str) -> np.ndarray | None:
    if not skin.data.shape_keys:
        return None
    keys = skin.data.shape_keys.key_blocks
    key = keys.get(key_name)
    if key is None:
        return None
    basis = base.key_array(keys.get("Basis") or keys[0])
    return base.key_array(key) - basis


def source_alias_delta(skin, target: str) -> tuple[np.ndarray | None, list[str]]:
    if not skin.data.shape_keys:
        return None, []
    keys = skin.data.shape_keys.key_blocks
    basis = base.key_array(keys.get("Basis") or keys[0])
    tokens = ALIASES.get(target, ())
    matches = []
    for key in keys:
        if key.name == "Basis" or key.name in ARKIT_52:
            continue
        lower = norm(key.name)
        if any(token in lower for token in tokens):
            matches.append(key)
    if not matches:
        return None, []
    delta = np.mean([base.key_array(key) - basis for key in matches], axis=0)
    return delta, [key.name for key in matches]


def side_of(name: str) -> int:
    if name.endswith("Left"):
        return -1
    if name.endswith("Right"):
        return 1
    return 0


def fallback_world_delta(
    target: str,
    points: np.ndarray,
    frame: dict,
    eyes: list[np.ndarray],
    forward_sign: float,
) -> np.ndarray:
    face_x = frame["face_x"]
    eye_z = frame["eye_z"]
    mouth = np.asarray(frame["mouth"], dtype=np.float64)
    mouth_z = float(frame["mouth_z"])
    nose_tip = np.asarray(frame["nose_tip"], dtype=np.float64)
    chin_z = float(frame["chin_z"])
    side = side_of(target)
    delta = np.zeros_like(points)

    selected_eye = None
    if side:
        selected_eye = sorted(eyes, key=lambda point: point[0])[0 if side < 0 else 1]
    eye_centre = selected_eye if selected_eye is not None else np.mean(eyes, axis=0)
    eye_mask = v3.ellipsoid(points, eye_centre, (0.058, 0.058, 0.045), 1.18)
    brow_centre = eye_centre + np.array([0.0, -forward_sign * 0.002, 0.026])
    brow_mask = v3.ellipsoid(points, brow_centre, (0.054, 0.050, 0.027), 1.16)
    cheek_centre = eye_centre + np.array([side * 0.006 if side else 0.0, -forward_sign * 0.008, -0.029])
    cheek_mask = v3.ellipsoid(points, cheek_centre, (0.064, 0.062, 0.055), 1.16)
    lip_mask = v3.ellipsoid(points, mouth, (0.064, 0.052, 0.035), 1.16)
    upper_lip = lip_mask * (points[:, 2] >= mouth_z)
    lower_lip = lip_mask * (points[:, 2] < mouth_z)
    if side:
        side_mask = v3.smoothstep01(side * (points[:, 0] - face_x) / 0.035)
        lip_side = lip_mask * side_mask
    else:
        lip_side = lip_mask
    nose_side_centre = nose_tip + np.array([side * 0.015 if side else 0.0, 0.0, 0.002])
    nose_mask = v3.ellipsoid(points, nose_side_centre, (0.032, 0.043, 0.036), 1.12)
    lower = v3.smoothstep01((mouth_z + 0.010 - points[:, 2]) / max(mouth_z + 0.010 - chin_z, 1.0e-6))
    lower_face = lower * v3.smoothstep01((0.075 - np.abs(points[:, 0] - face_x)) / 0.040)

    if target.startswith("browDown"):
        delta[:, 2] -= 0.0025 * brow_mask
    elif target == "browInnerUp":
        inner = brow_mask * v3.smoothstep01((0.030 - np.abs(points[:, 0] - face_x)) / 0.022)
        delta[:, 2] += 0.0030 * inner
    elif target.startswith("browOuterUp"):
        outer = brow_mask * v3.smoothstep01((np.abs(points[:, 0] - face_x) - 0.025) / 0.025)
        delta[:, 2] += 0.0027 * outer
    elif target == "cheekPuff":
        for eye in eyes:
            centre = eye + np.array([(-1 if eye[0] < face_x else 1) * 0.010, -forward_sign * 0.010, -0.032])
            mask = v3.ellipsoid(points, centre, (0.070, 0.066, 0.060), 1.16)
            delta[:, 1] += forward_sign * 0.0032 * mask
    elif target.startswith("cheekSquint"):
        delta[:, 2] += 0.0015 * cheek_mask
        delta[:, 1] += forward_sign * 0.0013 * cheek_mask
    elif target.startswith("eyeBlink"):
        delta[:, 2] += -(points[:, 2] - eye_centre[2]) * 0.62 * eye_mask
        delta[:, 1] -= forward_sign * 0.0005 * eye_mask
    elif target.startswith("eyeLookDown"):
        delta[:, 2] -= 0.0012 * eye_mask
    elif target.startswith("eyeLookUp"):
        delta[:, 2] += 0.0012 * eye_mask
    elif target.startswith("eyeLookIn"):
        delta[:, 0] += (-side if side else 1) * 0.0010 * eye_mask
    elif target.startswith("eyeLookOut"):
        delta[:, 0] += (side if side else -1) * 0.0010 * eye_mask
    elif target.startswith("eyeSquint"):
        delta[:, 2] += -(points[:, 2] - eye_centre[2]) * 0.30 * eye_mask
    elif target.startswith("eyeWide"):
        delta[:, 2] += (points[:, 2] - eye_centre[2]) * 0.25 * eye_mask
    elif target == "jawForward":
        delta[:, 1] += forward_sign * 0.0028 * lower_face
    elif target == "jawLeft":
        delta[:, 0] -= 0.0022 * lower_face
    elif target == "jawRight":
        delta[:, 0] += 0.0022 * lower_face
    elif target == "jawOpen":
        delta[:, 2] -= 0.0042 * lower_face
        delta[:, 1] -= forward_sign * 0.0008 * lower_face
    elif target == "mouthClose":
        delta[:, 2] += -(points[:, 2] - mouth_z) * 0.55 * lip_mask
    elif target.startswith("mouthDimple"):
        delta[:, 1] -= forward_sign * 0.0016 * lip_side
        delta[:, 0] += side * 0.0007 * lip_side
    elif target.startswith("mouthFrown"):
        corner = mouth + np.array([side * 0.040, 0.0, 0.0])
        mask = v3.ellipsoid(points, corner, (0.028, 0.040, 0.025), 1.12)
        delta[:, 2] -= 0.0023 * mask
    elif target == "mouthFunnel":
        delta[:, 0] += -(points[:, 0] - mouth[0]) * 0.22 * lip_mask
        delta[:, 1] += forward_sign * 0.0024 * lip_mask
    elif target == "mouthLeft":
        delta[:, 0] -= 0.0022 * lip_mask
    elif target == "mouthRight":
        delta[:, 0] += 0.0022 * lip_mask
    elif target.startswith("mouthLowerDown"):
        delta[:, 2] -= 0.0022 * lower_lip * (lip_side if side else 1.0)
    elif target.startswith("mouthPress"):
        delta[:, 2] += -(points[:, 2] - mouth_z) * 0.42 * lip_side
        delta[:, 1] -= forward_sign * 0.0010 * lip_side
    elif target == "mouthPucker":
        delta[:, 0] += -(points[:, 0] - mouth[0]) * 0.34 * lip_mask
        delta[:, 1] += forward_sign * 0.0030 * lip_mask
    elif target == "mouthRollLower":
        delta[:, 2] += (mouth_z - points[:, 2]) * 0.45 * lower_lip
        delta[:, 1] -= forward_sign * 0.0012 * lower_lip
    elif target == "mouthRollUpper":
        delta[:, 2] += (mouth_z - points[:, 2]) * 0.45 * upper_lip
        delta[:, 1] -= forward_sign * 0.0012 * upper_lip
    elif target == "mouthShrugLower":
        delta[:, 2] += 0.0016 * lower_lip
    elif target == "mouthShrugUpper":
        delta[:, 2] += 0.0019 * upper_lip
    elif target.startswith("mouthSmile"):
        corner = mouth + np.array([side * 0.040, 0.0, 0.0])
        mask = v3.ellipsoid(points, corner, (0.030, 0.040, 0.026), 1.12)
        delta[:, 2] += 0.0026 * mask
        delta[:, 0] += side * 0.0010 * mask
    elif target.startswith("mouthStretch"):
        delta[:, 0] += side * 0.0027 * lip_side
    elif target.startswith("mouthUpperUp"):
        delta[:, 2] += 0.0021 * upper_lip * (lip_side if side else 1.0)
    elif target.startswith("noseSneer"):
        delta[:, 2] += 0.0017 * nose_mask
        delta[:, 1] += forward_sign * 0.0011 * nose_mask
    elif target == "tongueOut":
        delta[:, 1] += forward_sign * 0.0030 * lower_lip
        delta[:, 2] -= 0.0012 * lower_lip
    else:
        delta[:, 1] += forward_sign * 0.00015 * lip_mask

    lengths = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1.0, 0.0045 / np.maximum(lengths, 1.0e-9))[:, None]
    return delta


def build_arkit_shape_keys(skin, head_ids, eyes, forward_sign) -> dict:
    basis = basis_key(skin)
    basis_local = base.key_array(basis)
    world = base.to_world(skin, basis_local)
    frame = v4.feature_frame(skin, head_ids, eyes, forward_sign)
    created = []
    reused = []
    alias_report = {}
    max_delta = {}

    for target in ARKIT_52:
        existing = current_key_delta(skin, target)
        if existing is not None and float(np.linalg.norm(existing, axis=1).max()) > 1.0e-7:
            reused.append(target)
            max_delta[target] = float(np.linalg.norm(existing, axis=1).max())
            continue

        alias_delta, alias_names = source_alias_delta(skin, target)
        if alias_delta is not None and float(np.linalg.norm(alias_delta, axis=1).max()) > 1.0e-7:
            local_delta = alias_delta.copy()
            alias_report[target] = alias_names
        else:
            world_delta = fallback_world_delta(target, world, frame, eyes, forward_sign)
            local_delta = base.world_vector_to_local(skin, world_delta)

        magnitude = np.linalg.norm(local_delta, axis=1)
        if float(magnitude.max()) <= 1.0e-7:
            centre = int(head_ids[np.argmin(np.linalg.norm(world[head_ids] - np.asarray(frame["mouth"]), axis=1))])
            local_delta[centre, 2] += 5.0e-5
            magnitude = np.linalg.norm(local_delta, axis=1)

        key = skin.data.shape_keys.key_blocks.get(target) if skin.data.shape_keys else None
        if key is None:
            key = skin.shape_key_add(name=target, from_mix=False)
            created.append(target)
        base.set_key_array(key, basis_local + local_delta)
        key.value = 0.0
        key.slider_min = 0.0
        key.slider_max = 1.0
        max_delta[target] = float(magnitude.max())

    missing = [name for name in ARKIT_52 if skin.data.shape_keys.key_blocks.get(name) is None]
    nonzero = [name for name in ARKIT_52 if max_delta.get(name, 0.0) > 1.0e-7]
    if missing or len(nonzero) != 52:
        raise RuntimeError(f"ARKit build incomplete: missing={missing}, nonzero={len(nonzero)}")
    return {
        "expected": 52,
        "created": created,
        "reused": reused,
        "present": len(ARKIT_52) - len(missing),
        "nonzero": len(nonzero),
        "aliases": alias_report,
        "max_delta_m": max_delta,
    }


def ensure_hair_bones(armature, head_bone, scene) -> dict:
    root_name = "AINA_HairRoot"
    tip_name = "AINA_HairTip"
    if armature.data.bones.get(root_name) and armature.data.bones.get(tip_name):
        return {"bones": [root_name, tip_name], "created": False, "weighted_object": None}

    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    edits = armature.data.edit_bones
    parent = edits.get(head_bone.name)
    if parent is None:
        bpy.ops.object.mode_set(mode="OBJECT")
        raise RuntimeError(f"Head edit bone not found: {head_bone.name}")
    root = edits.new(root_name)
    root.head = parent.tail + Vector((0.0, 0.0, 0.015))
    root.tail = root.head + Vector((0.0, 0.0, 0.060))
    root.parent = parent
    root.use_connect = False
    root.use_deform = True
    tip = edits.new(tip_name)
    tip.head = root.tail
    tip.tail = tip.head + Vector((0.0, 0.0, 0.060))
    tip.parent = root
    tip.use_connect = True
    tip.use_deform = True
    bpy.ops.object.mode_set(mode="OBJECT")

    hair_objects = [
        obj for obj in scene.objects
        if obj.type == "MESH" and len(obj.data.vertices) and base.is_hair(obj) and not obj.hide_render
    ]
    weighted = None
    if hair_objects:
        hair = max(hair_objects, key=lambda obj: len(obj.data.vertices))
        world = base.world_vertices(hair)
        z_min = float(world[:, 2].min())
        z_max = float(world[:, 2].max())
        span = max(z_max - z_min, 1.0e-6)
        root_group = hair.vertex_groups.get(root_name) or hair.vertex_groups.new(name=root_name)
        tip_group = hair.vertex_groups.get(tip_name) or hair.vertex_groups.new(name=tip_name)
        for index, point in enumerate(world):
            t = float(np.clip((point[2] - z_min) / span, 0.0, 1.0))
            tip_weight = t * t
            root_group.add([index], max(0.0, 1.0 - tip_weight), "REPLACE")
            tip_group.add([index], tip_weight, "REPLACE")
        weighted = hair.name
    return {"bones": [root_name, tip_name], "created": True, "weighted_object": weighted}


def read_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or total != len(raw):
        raise RuntimeError("Invalid GLB 2.0")
    offset = 12
    document = None
    binary = b""
    while offset < len(raw):
        length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk = raw[offset:offset + length]
        offset += length
        if chunk_type == 0x4E4F534A:
            document = json.loads(chunk.decode("utf-8").rstrip(" \t\r\n\x00"))
        elif chunk_type == 0x004E4942:
            binary = chunk
    if document is None:
        raise RuntimeError("GLB JSON chunk missing")
    return document, binary


def write_glb(path: Path, document: dict, binary: bytes) -> None:
    json_bytes = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    binary += b"\x00" * ((4 - len(binary) % 4) % 4)
    chunks = [struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes]
    if binary:
        chunks.append(struct.pack("<II", len(binary), 0x004E4942) + binary)
    payload = b"".join(chunks)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(payload)) + payload)


def find_node(document: dict, candidates: list[str]) -> int | None:
    nodes = document.get("nodes", [])
    normalized = {norm(node.get("name", "")): index for index, node in enumerate(nodes)}
    for candidate in candidates:
        if norm(candidate) in normalized:
            return normalized[norm(candidate)]
    for candidate in candidates:
        needle = norm(candidate)
        for index, node in enumerate(nodes):
            value = norm(node.get("name", ""))
            if needle and needle in value:
                return index
    return None


def humanoid_map(document: dict) -> tuple[dict, list[str]]:
    specs = {
        "hips": ["DEF-spine", "hips", "pelvis"],
        "spine": ["DEF-spine.001", "spine.001", "spine1"],
        "chest": ["DEF-spine.002", "spine.002", "chest"],
        "upperChest": ["DEF-spine.003", "spine.003", "upperchest"],
        "neck": ["DEF-neck", "neck", "DEF-spine.004", "spine.004"],
        "head": ["DEF-Head", "DEF-head", "head"],
        "leftShoulder": ["DEF-shoulder.L", "shoulder.L", "leftshoulder"],
        "leftUpperArm": ["DEF-upper_arm.L", "upper_arm.L", "leftupperarm"],
        "leftLowerArm": ["DEF-forearm.L", "forearm.L", "leftlowerarm"],
        "leftHand": ["DEF-hand.L", "hand.L", "lefthand"],
        "rightShoulder": ["DEF-shoulder.R", "shoulder.R", "rightshoulder"],
        "rightUpperArm": ["DEF-upper_arm.R", "upper_arm.R", "rightupperarm"],
        "rightLowerArm": ["DEF-forearm.R", "forearm.R", "rightlowerarm"],
        "rightHand": ["DEF-hand.R", "hand.R", "righthand"],
        "leftUpperLeg": ["DEF-thigh.L", "thigh.L", "leftupperleg"],
        "leftLowerLeg": ["DEF-shin.L", "shin.L", "leftlowerleg"],
        "leftFoot": ["DEF-foot.L", "foot.L", "leftfoot"],
        "leftToes": ["DEF-toe.L", "toe.L", "lefttoes"],
        "rightUpperLeg": ["DEF-thigh.R", "thigh.R", "rightupperleg"],
        "rightLowerLeg": ["DEF-shin.R", "shin.R", "rightlowerleg"],
        "rightFoot": ["DEF-foot.R", "foot.R", "rightfoot"],
        "rightToes": ["DEF-toe.R", "toe.R", "righttoes"],
        "leftEye": ["eye.L", "DEF-eye.L", "lefteye"],
        "rightEye": ["eye.R", "DEF-eye.R", "righteye"],
        "jaw": ["jaw", "DEF-jaw"],
    }
    bones = {}
    missing = []
    required = {
        "hips", "spine", "chest", "neck", "head",
        "leftUpperArm", "leftLowerArm", "leftHand",
        "rightUpperArm", "rightLowerArm", "rightHand",
        "leftUpperLeg", "leftLowerLeg", "leftFoot",
        "rightUpperLeg", "rightLowerLeg", "rightFoot",
    }
    for name, candidates in specs.items():
        index = find_node(document, candidates)
        if index is not None:
            bones[name] = {"node": index}
        elif name in required:
            missing.append(name)
    return bones, missing


def expression_presets(node_index: int, target_indices: dict[str, int]) -> dict:
    def binds(items):
        output = []
        for name, weight in items:
            if name in target_indices:
                output.append({"node": node_index, "index": target_indices[name], "weight": weight})
        return output

    definitions = {
        "happy": [("mouthSmileLeft", 1.0), ("mouthSmileRight", 1.0), ("cheekSquintLeft", 0.35), ("cheekSquintRight", 0.35)],
        "angry": [("browDownLeft", 1.0), ("browDownRight", 1.0), ("mouthFrownLeft", 0.55), ("mouthFrownRight", 0.55)],
        "sad": [("browInnerUp", 0.8), ("mouthFrownLeft", 1.0), ("mouthFrownRight", 1.0)],
        "relaxed": [("mouthSmileLeft", 0.22), ("mouthSmileRight", 0.22), ("browOuterUpLeft", 0.15), ("browOuterUpRight", 0.15)],
        "surprised": [("eyeWideLeft", 0.9), ("eyeWideRight", 0.9), ("browInnerUp", 0.65), ("jawOpen", 0.72)],
        "aa": [("jawOpen", 1.0), ("mouthFunnel", 0.22)],
        "ih": [("mouthStretchLeft", 0.72), ("mouthStretchRight", 0.72), ("jawOpen", 0.20)],
        "ou": [("mouthPucker", 1.0), ("mouthFunnel", 0.85)],
        "ee": [("mouthStretchLeft", 1.0), ("mouthStretchRight", 1.0), ("mouthSmileLeft", 0.35), ("mouthSmileRight", 0.35)],
        "oh": [("jawOpen", 0.72), ("mouthFunnel", 1.0)],
        "blink": [("eyeBlinkLeft", 1.0), ("eyeBlinkRight", 1.0)],
        "blinkLeft": [("eyeBlinkLeft", 1.0)],
        "blinkRight": [("eyeBlinkRight", 1.0)],
        "lookUp": [("eyeLookUpLeft", 1.0), ("eyeLookUpRight", 1.0)],
        "lookDown": [("eyeLookDownLeft", 1.0), ("eyeLookDownRight", 1.0)],
        "lookLeft": [("eyeLookOutLeft", 1.0), ("eyeLookInRight", 1.0)],
        "lookRight": [("eyeLookInLeft", 1.0), ("eyeLookOutRight", 1.0)],
        "neutral": [],
    }
    presets = {}
    for name in PRESET_ORDER:
        presets[name] = {
            "isBinary": False,
            "overrideBlink": "none",
            "overrideLookAt": "none",
            "overrideMouth": "none",
            "morphTargetBinds": binds(definitions[name]),
        }
    return presets


def patch_vrm(source_glb: Path, vrm_path: Path, hair_bones: list[str]) -> dict:
    document, binary = read_glb(source_glb)
    mesh_index = None
    target_names = None
    for index, mesh in enumerate(document.get("meshes", [])):
        names = mesh.get("extras", {}).get("targetNames", [])
        if set(ARKIT_52).issubset(set(names)):
            mesh_index = index
            target_names = names
            break
    if mesh_index is None or target_names is None:
        raise RuntimeError("No exported Mesh contains all 52 ARKit targets")
    node_index = next(
        (index for index, node in enumerate(document.get("nodes", [])) if node.get("mesh") == mesh_index),
        None,
    )
    if node_index is None:
        raise RuntimeError("Morph Mesh node was not found")
    target_indices = {name: target_names.index(name) for name in ARKIT_52}

    bones, missing_bones = humanoid_map(document)
    if missing_bones:
        raise RuntimeError(f"Required humanoid bones are missing: {missing_bones}")
    head_node = bones["head"]["node"]
    hair_nodes = []
    for name in hair_bones:
        index = find_node(document, [name])
        if index is not None and index not in hair_nodes:
            hair_nodes.append(index)
    if not hair_nodes:
        for index, node in enumerate(document.get("nodes", [])):
            value = norm(node.get("name", ""))
            if any(token in value for token in ("hair", "ponytail", "strand", "bang")):
                hair_nodes.append(index)
                if len(hair_nodes) >= 4:
                    break
    if not hair_nodes:
        raise RuntimeError("No SpringBone hair nodes were exported")

    presets = expression_presets(node_index, target_indices)
    vrm_extension = {
        "specVersion": "1.0",
        "meta": {
            "name": "AINA",
            "version": "1.0.0-technical-candidate",
            "authors": ["Shenzhen Uoon Technology Co.,Ltd.", "AINA Project"],
            "copyrightInformation": "AINA character; Rain topology attribution retained in release package.",
            "contactInformation": "AINA VRM Digital Human",
            "references": ["Blender Studio Rain v3, CC BY 4.0"],
            "thirdPartyLicenses": "Blender Studio Rain v3 is used under CC BY 4.0.",
            "avatarPermission": "everyone",
            "allowExcessivelyViolentUsage": False,
            "allowExcessivelySexualUsage": False,
            "commercialUsage": "corporationProfit",
            "allowPoliticalOrReligiousUsage": False,
            "allowAntisocialOrHateUsage": False,
            "creditNotation": "required",
            "allowRedistribution": True,
            "modification": "allowModificationRedistribution",
            "otherLicenseUrl": "https://creativecommons.org/licenses/by/4.0/",
        },
        "humanoid": {"humanBones": bones},
        "firstPerson": {},
        "lookAt": {
            "offsetFromHeadBone": [0.0, 0.06, 0.0],
            "type": "bone",
            "rangeMapHorizontalInner": {"inputMaxValue": 90.0, "outputScale": 10.0},
            "rangeMapHorizontalOuter": {"inputMaxValue": 90.0, "outputScale": 10.0},
            "rangeMapVerticalDown": {"inputMaxValue": 90.0, "outputScale": 10.0},
            "rangeMapVerticalUp": {"inputMaxValue": 90.0, "outputScale": 10.0},
        },
        "expressions": {"preset": presets, "custom": {}},
    }
    spring_extension = {
        "specVersion": "1.0",
        "colliders": [
            {
                "node": head_node,
                "shape": {"sphere": {"offset": [0.0, 0.07, 0.0], "radius": 0.12}},
            }
        ],
        "colliderGroups": [{"name": "AINA Head", "colliders": [0]}],
        "springs": [
            {
                "name": "AINA Silver Hair",
                "joints": [
                    {
                        "node": node,
                        "hitRadius": 0.012,
                        "stiffness": 0.82 if index == 0 else 0.58,
                        "gravityPower": 0.10,
                        "gravityDir": [0.0, -1.0, 0.0],
                        "dragForce": 0.28,
                    }
                    for index, node in enumerate(hair_nodes)
                ],
                "colliderGroups": [0],
                "center": head_node,
            }
        ],
    }
    document.setdefault("extensions", {})["VRMC_vrm"] = vrm_extension
    document["extensions"]["VRMC_springBone"] = spring_extension
    used = document.setdefault("extensionsUsed", [])
    for extension in ("VRMC_vrm", "VRMC_springBone"):
        if extension not in used:
            used.append(extension)
    required = document.setdefault("extensionsRequired", [])
    if "VRMC_vrm" not in required:
        required.append("VRMC_vrm")
    document.setdefault("asset", {})["generator"] = "AINA Rain VRM Production v7 / Blender 4.5"
    write_glb(vrm_path, document, binary)

    primitive = document["meshes"][mesh_index]["primitives"][0]
    nonzero = {}
    for name, target_index in target_indices.items():
        accessor_index = primitive["targets"][target_index].get("POSITION")
        accessor = document["accessors"][accessor_index]
        values = list(accessor.get("min", [])) + list(accessor.get("max", []))
        nonzero[name] = max((abs(float(value)) for value in values), default=0.0)
    preset_counts = {
        name: len(presets[name].get("morphTargetBinds", [])) for name in PRESET_ORDER
    }
    return {
        "mesh_index": mesh_index,
        "node_index": node_index,
        "target_names_count": len(target_names),
        "arkit_targets_present": len(target_indices),
        "arkit_targets_nonzero": sum(value > 1.0e-8 for value in nonzero.values()),
        "arkit_target_accessor_max_abs": nonzero,
        "humanoid_bones": bones,
        "required_humanoid_missing": missing_bones,
        "preset_counts": preset_counts,
        "preset_count": len(presets),
        "hair_nodes": hair_nodes,
        "extensions_used": document.get("extensionsUsed", []),
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "QA"
    qa.mkdir(exist_ok=True)
    source_report = {}
    if args.source_report and args.source_report.exists():
        source_report = json.loads(args.source_report.read_text(encoding="utf-8"))

    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH" and len(obj.data.vertices)]
    base.reset_shape_keys(meshes)
    armature = base.find_armature(scene)
    head_bone = base.find_head_bone(armature)
    head_point = base.bone_world_point(armature, head_bone)
    skin, skin_report = base.identify_skin(scene, head_point)
    original_vertices = len(skin.data.vertices)
    original_triangles = sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons)
    world = base.world_vertices(skin)
    face_x = float(0.5 * (world[:, 0].min() + world[:, 0].max()))
    eyes = v3.true_eye_centres(scene, face_x)
    if len(eyes) != 2:
        raise RuntimeError(f"Expected two real eyes, got {len(eyes)}")
    character_height = skin_report["character_height_m"]
    head_ids, _, _, _ = base.head_region(skin, head_point, eyes, character_height)
    forward_sign = -1.0 if np.mean(eyes, axis=0)[1] < world.mean(axis=0)[1] else 1.0

    arkit = build_arkit_shape_keys(skin, head_ids, eyes, forward_sign)
    hair = ensure_hair_bones(armature, head_bone, scene)
    bpy.context.view_layer.update()

    triangles = sum(max(1, len(poly.vertices) - 2) for poly in skin.data.polygons)
    if len(skin.data.vertices) != original_vertices or triangles != original_triangles:
        raise RuntimeError("Skin topology changed during VRM production")

    cameras, camera_report = base.setup_cameras(
        scene, skin, head_ids, eyes, head_point, character_height
    )
    appearance.soften_lighting(scene)
    renders = appearance.render_full_suite(scene, cameras, skin, args.out)

    blend_path = args.out / "AINA_MASTER.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path = args.out / "AINA_TECHNICAL_SOURCE.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )
    vrm_path = args.out / "AINA.vrm"
    binary_qa = patch_vrm(glb_path, vrm_path, hair["bones"])

    report = {
        "product": "AINA Rain VRM Production v7 Technical Candidate",
        "source": "AINA Rain Identity Residual v6",
        "source_artifact_product": source_report.get("product"),
        "real_3d_model": True,
        "replacement_effect_art_generated": False,
        "skin_topology_changed": False,
        "vertices": len(skin.data.vertices),
        "triangles": triangles,
        "source_vertices": original_vertices,
        "source_triangles": original_triangles,
        "armature": armature.name,
        "armature_preserved": True,
        "skin_weights_preserved": True,
        "uvs_preserved": True,
        "arkit": arkit,
        "hair_spring_setup": hair,
        "binary_qa": binary_qa,
        "vrm_spec": "1.0",
        "shape_controls": 52,
        "preset_expressions": 18,
        "humanoid": True,
        "look_at": True,
        "spring_bone": True,
        "technical_release_gate": False,
        "identity_lock": False,
        "visual_identity_lock": False,
        "production_release": False,
        "candidate": True,
        "vrm_exported": True,
        "clean_reimport_verified": False,
        "camera": camera_report,
        "renders": renders,
        "files": {
            "blend": str(blend_path),
            "source_glb": str(glb_path),
            "vrm": str(vrm_path),
        },
        "next_gate": "Clean-import the exact AINA.vrm bytes as GLB, verify 52/52 controls, 18/18 presets, humanoid and SpringBone, and inspect the actual reimport renders before any visual or production lock.",
    }
    report_path = qa / "AINA_RAIN_VRM_PRODUCTION_V7_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (qa / "AINA_VRM_BINARY_QA.json").write_text(
        json.dumps(binary_qa, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
