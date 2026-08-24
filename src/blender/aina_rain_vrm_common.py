#!/usr/bin/env python3
"""Common ARKit, GLB and VRM helpers for the final AINA Rain release."""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path


ARKIT_52 = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut",
]

VRM_PRESETS_18 = [
    "happy", "angry", "sad", "relaxed", "surprised",
    "aa", "ih", "ou", "ee", "oh",
    "blink", "blinkLeft", "blinkRight",
    "lookUp", "lookDown", "lookLeft", "lookRight", "neutral",
]

REQUIRED_HUMANOID = [
    "hips", "spine", "head",
    "leftUpperArm", "leftLowerArm", "leftHand",
    "rightUpperArm", "rightLowerArm", "rightHand",
    "leftUpperLeg", "leftLowerLeg", "leftFoot",
    "rightUpperLeg", "rightLowerLeg", "rightFoot",
]


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def read_glb(path: Path):
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"GLB too small: {path}")
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or length != len(data):
        raise ValueError(f"Invalid GLB header: {path}")
    chunks = []
    offset = 12
    document = None
    json_index = None
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        payload = data[offset : offset + chunk_length]
        offset += chunk_length
        chunks.append([chunk_type, payload])
        if chunk_type == 0x4E4F534A:
            document = json.loads(payload.decode("utf-8").rstrip(" \t\r\n\x00"))
            json_index = len(chunks) - 1
    if document is None or json_index is None:
        raise ValueError("GLB has no JSON chunk")
    return document, chunks, json_index


def write_glb(path: Path, document: dict, chunks: list, json_index: int) -> None:
    encoded = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    output_chunks = [list(item) for item in chunks]
    output_chunks[json_index] = [0x4E4F534A, encoded]
    body = bytearray()
    for chunk_type, payload in output_chunks:
        payload = bytes(payload)
        if len(payload) % 4:
            pad = b" " if chunk_type == 0x4E4F534A else b"\x00"
            payload += pad * ((4 - len(payload) % 4) % 4)
        body += struct.pack("<II", len(payload), chunk_type)
        body += payload
    header = struct.pack("<4sII", b"glTF", 2, 12 + len(body))
    path.write_bytes(header + body)


def target_names(mesh: dict) -> list[str]:
    extras = mesh.get("extras") or {}
    names = extras.get("targetNames") or extras.get("target_names")
    if names:
        return [str(name) for name in names]
    for primitive in mesh.get("primitives", []):
        extras = primitive.get("extras") or {}
        names = extras.get("targetNames") or extras.get("target_names")
        if names:
            return [str(name) for name in names]
    return []


def find_arkit_mesh_node(document: dict):
    nodes = document.get("nodes", [])
    meshes = document.get("meshes", [])
    wanted = set(ARKIT_52)
    best = None
    for node_index, node in enumerate(nodes):
        mesh_index = node.get("mesh")
        if mesh_index is None or mesh_index >= len(meshes):
            continue
        names = target_names(meshes[mesh_index])
        matches = sum(name in wanted for name in names)
        candidate = (matches, len(names), node_index, mesh_index, names)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("Export contains no mesh node with morph targets")
    matches, _, node_index, mesh_index, names = best
    if matches < len(ARKIT_52):
        missing = [name for name in ARKIT_52 if name not in names]
        raise RuntimeError(f"Only {matches}/52 ARKit targets exported; missing {missing}")
    return node_index, mesh_index, {name: names.index(name) for name in ARKIT_52}, names


def joint_nodes(document: dict) -> set[int]:
    result = set()
    for skin in document.get("skins", []):
        result.update(int(index) for index in skin.get("joints", []))
    return result


def parent_map(document: dict) -> dict[int, int]:
    result = {}
    for parent, node in enumerate(document.get("nodes", [])):
        for child in node.get("children", []):
            result[int(child)] = parent
    return result


HUMANOID_ALIASES = {
    "hips": ["DEF-spine", "spine", "hips", "pelvis", "root"],
    "spine": ["DEF-spine.001", "spine.001", "spine1", "spine_01"],
    "chest": ["DEF-spine.002", "spine.002", "chest", "spine2", "spine_02"],
    "upperChest": ["DEF-spine.003", "spine.003", "upperchest", "spine3", "spine_03"],
    "neck": ["DEF-neck", "neck"],
    "head": ["DEF-Head", "DEF-head", "head"],
    "leftEye": ["DEF-eye.L", "eye.L", "eye_left", "lefteye"],
    "rightEye": ["DEF-eye.R", "eye.R", "eye_right", "righteye"],
    "jaw": ["DEF-jaw", "jaw"],
    "leftShoulder": ["DEF-shoulder.L", "shoulder.L", "leftshoulder"],
    "leftUpperArm": ["DEF-upper_arm.L", "upper_arm.L", "upperarm.L", "leftupperarm"],
    "leftLowerArm": ["DEF-forearm.L", "forearm.L", "lower_arm.L", "leftlowerarm"],
    "leftHand": ["DEF-hand.L", "hand.L", "lefthand"],
    "rightShoulder": ["DEF-shoulder.R", "shoulder.R", "rightshoulder"],
    "rightUpperArm": ["DEF-upper_arm.R", "upper_arm.R", "upperarm.R", "rightupperarm"],
    "rightLowerArm": ["DEF-forearm.R", "forearm.R", "lower_arm.R", "rightlowerarm"],
    "rightHand": ["DEF-hand.R", "hand.R", "righthand"],
    "leftUpperLeg": ["DEF-thigh.L", "thigh.L", "upper_leg.L", "leftupperleg"],
    "leftLowerLeg": ["DEF-shin.L", "shin.L", "calf.L", "leftlowerleg"],
    "leftFoot": ["DEF-foot.L", "foot.L", "leftfoot"],
    "leftToes": ["DEF-toe.L", "toe.L", "lefttoes"],
    "rightUpperLeg": ["DEF-thigh.R", "thigh.R", "upper_leg.R", "rightupperleg"],
    "rightLowerLeg": ["DEF-shin.R", "shin.R", "calf.R", "rightlowerleg"],
    "rightFoot": ["DEF-foot.R", "foot.R", "rightfoot"],
    "rightToes": ["DEF-toe.R", "toe.R", "righttoes"],
}


def map_humanoid_nodes(document: dict) -> dict[str, int]:
    nodes = document.get("nodes", [])
    joints = joint_nodes(document)
    candidates = joints or set(range(len(nodes)))
    normalized = {index: normalize_name(nodes[index].get("name", "")) for index in candidates}
    result = {}
    for human_bone, aliases in HUMANOID_ALIASES.items():
        exact = {normalize_name(alias) for alias in aliases}
        hit = next((index for index, name in normalized.items() if name in exact), None)
        if hit is None:
            tokens = [normalize_name(alias) for alias in aliases]
            scored = []
            for index, name in normalized.items():
                score = max((len(token) for token in tokens if token and token in name), default=0)
                if score:
                    penalty = 0
                    if any(bad in name for bad in ("mch", "org", "ctrl", "tweak", "mechanism")):
                        penalty += 20
                    if name.startswith("def"):
                        penalty -= 4
                    scored.append((score - penalty, -len(name), index))
            if scored:
                hit = max(scored)[2]
        if hit is not None:
            result[human_bone] = int(hit)
    missing = [name for name in REQUIRED_HUMANOID if name not in result]
    if missing:
        available = [nodes[index].get("name", "") for index in sorted(candidates)]
        raise RuntimeError(f"Missing required humanoid bones {missing}; available joints include {available[:120]}")
    return result


def expression(bindings: list[dict], *, binary=False, blink="none", look="none", mouth="none") -> dict:
    return {
        "isBinary": bool(binary),
        "overrideBlink": blink,
        "overrideLookAt": look,
        "overrideMouth": mouth,
        "morphTargetBinds": bindings,
        "materialColorBinds": [],
        "textureTransformBinds": [],
    }
