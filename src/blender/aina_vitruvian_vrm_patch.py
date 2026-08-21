#!/usr/bin/env python3
"""Patch a morph-preserving AINA GLB into a VRM 1.0 binary.

The Blender scene already contains the actual humanoid armature, 52 real facial
morph targets, eye bones and optional hair spring bones.  This module adds the
VRMC_vrm and VRMC_springBone extensions without touching any geometry or binary
accessor data.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path


JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


ARKIT_52 = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward",
    "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut",
]


HUMANOID_ALIASES = {
    "hips": ["hips", "mixamorighips", "pelvis"],
    "spine": ["spine", "mixamorigspine"],
    "chest": ["spine1", "mixamorigspine1", "chest"],
    "upperChest": ["spine2", "mixamorigspine2", "upperchest"],
    "neck": ["neck", "mixamorigneck"],
    "head": ["head", "mixamorighead"],
    "leftEye": ["lefteye", "eyel", "eyeleft"],
    "rightEye": ["righteye", "eyer", "eyeright"],
    "jaw": ["jaw", "mixamorigjaw"],
    "leftUpperLeg": ["leftupleg", "mixamorigleftupleg", "leftthigh", "thighl"],
    "leftLowerLeg": ["leftleg", "mixamorigleftleg", "leftlowerleg", "calfl"],
    "leftFoot": ["leftfoot", "mixamorigleftfoot", "footl"],
    "leftToes": ["lefttoebase", "mixamoriglefttoebase", "lefttoes", "toel"],
    "rightUpperLeg": ["rightupleg", "mixamorigrightupleg", "rightthigh", "thighr"],
    "rightLowerLeg": ["rightleg", "mixamorigrightleg", "rightlowerleg", "calfr"],
    "rightFoot": ["rightfoot", "mixamorigrightfoot", "footr"],
    "rightToes": ["righttoebase", "mixamorigrighttoebase", "righttoes", "toer"],
    "leftShoulder": ["leftshoulder", "mixamorigleftshoulder", "shoulderl"],
    "leftUpperArm": ["leftarm", "mixamorigleftarm", "leftupperarm", "upperarml"],
    "leftLowerArm": ["leftforearm", "mixamorigleftforearm", "leftlowerarm", "lowerarml"],
    "leftHand": ["lefthand", "mixamoriglefthand", "handl"],
    "rightShoulder": ["rightshoulder", "mixamorigrightshoulder", "shoulderr"],
    "rightUpperArm": ["rightarm", "mixamorigrightarm", "rightupperarm", "upperarmr"],
    "rightLowerArm": ["rightforearm", "mixamorigrightforearm", "rightlowerarm", "lowerarmr"],
    "rightHand": ["righthand", "mixamorigrighthand", "handr"],
    "leftThumbMetacarpal": ["lefthandthumb1", "mixamoriglefthandthumb1"],
    "leftThumbProximal": ["lefthandthumb2", "mixamoriglefthandthumb2"],
    "leftThumbDistal": ["lefthandthumb3", "mixamoriglefthandthumb3"],
    "leftIndexProximal": ["lefthandindex1", "mixamoriglefthandindex1"],
    "leftIndexIntermediate": ["lefthandindex2", "mixamoriglefthandindex2"],
    "leftIndexDistal": ["lefthandindex3", "mixamoriglefthandindex3"],
    "leftMiddleProximal": ["lefthandmiddle1", "mixamoriglefthandmiddle1"],
    "leftMiddleIntermediate": ["lefthandmiddle2", "mixamoriglefthandmiddle2"],
    "leftMiddleDistal": ["lefthandmiddle3", "mixamoriglefthandmiddle3"],
    "leftRingProximal": ["lefthandring1", "mixamoriglefthandring1"],
    "leftRingIntermediate": ["lefthandring2", "mixamoriglefthandring2"],
    "leftRingDistal": ["lefthandring3", "mixamoriglefthandring3"],
    "leftLittleProximal": ["lefthandpinky1", "mixamoriglefthandpinky1"],
    "leftLittleIntermediate": ["lefthandpinky2", "mixamoriglefthandpinky2"],
    "leftLittleDistal": ["lefthandpinky3", "mixamoriglefthandpinky3"],
    "rightThumbMetacarpal": ["righthandthumb1", "mixamorigrighthandthumb1"],
    "rightThumbProximal": ["righthandthumb2", "mixamorigrighthandthumb2"],
    "rightThumbDistal": ["righthandthumb3", "mixamorigrighthandthumb3"],
    "rightIndexProximal": ["righthandindex1", "mixamorigrighthandindex1"],
    "rightIndexIntermediate": ["righthandindex2", "mixamorigrighthandindex2"],
    "rightIndexDistal": ["righthandindex3", "mixamorigrighthandindex3"],
    "rightMiddleProximal": ["righthandmiddle1", "mixamorigrighthandmiddle1"],
    "rightMiddleIntermediate": ["righthandmiddle2", "mixamorigrighthandmiddle2"],
    "rightMiddleDistal": ["righthandmiddle3", "mixamorigrighthandmiddle3"],
    "rightRingProximal": ["righthandring1", "mixamorigrighthandring1"],
    "rightRingIntermediate": ["righthandring2", "mixamorigrighthandring2"],
    "rightRingDistal": ["righthandring3", "mixamorigrighthandring3"],
    "rightLittleProximal": ["righthandpinky1", "mixamorigrighthandpinky1"],
    "rightLittleIntermediate": ["righthandpinky2", "mixamorigrighthandpinky2"],
    "rightLittleDistal": ["righthandpinky3", "mixamorigrighthandpinky3"],
}


REQUIRED_HUMANOID = {
    "hips", "spine", "chest", "neck", "head",
    "leftUpperLeg", "leftLowerLeg", "leftFoot",
    "rightUpperLeg", "rightLowerLeg", "rightFoot",
    "leftUpperArm", "leftLowerArm", "leftHand",
    "rightUpperArm", "rightLowerArm", "rightHand",
}


def normalized(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def read_glb(path: Path):
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise RuntimeError(f"Not a GLB file: {path}")
    magic, version, total_length = struct.unpack_from("<4sII", data, 0)
    if version != 2 or total_length != len(data):
        raise RuntimeError("Unexpected GLB header")
    offset = 12
    chunks = []
    document = None
    while offset < len(data):
        length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        payload = data[offset : offset + length]
        offset += length
        if chunk_type == JSON_CHUNK:
            document = json.loads(payload.rstrip(b" \t\r\n\x00").decode("utf-8"))
        else:
            chunks.append((chunk_type, payload))
    if document is None:
        raise RuntimeError("GLB has no JSON chunk")
    return document, chunks


def write_glb(path: Path, document: dict, chunks) -> None:
    json_bytes = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    body = struct.pack("<II", len(json_bytes), JSON_CHUNK) + json_bytes
    for chunk_type, payload in chunks:
        padded = payload + b"\x00" * ((4 - len(payload) % 4) % 4)
        body += struct.pack("<II", len(padded), chunk_type) + padded
    header = struct.pack("<4sII", b"glTF", 2, 12 + len(body))
    path.write_bytes(header + body)


def node_index(document: dict, aliases) -> int | None:
    names = [normalized(node.get("name", "")) for node in document.get("nodes", [])]
    alias_values = [normalized(alias) for alias in aliases]
    for alias in alias_values:
        for index, name in enumerate(names):
            if name == alias:
                return index
    for alias in alias_values:
        for index, name in enumerate(names):
            if alias and (name.endswith(alias) or alias in name):
                return index
    return None


def target_names_for_mesh(mesh: dict) -> list[str]:
    extras = mesh.get("extras") or {}
    names = extras.get("targetNames") or extras.get("target_names")
    if names:
        return list(names)
    for primitive in mesh.get("primitives", []):
        extras = primitive.get("extras") or {}
        names = extras.get("targetNames") or extras.get("target_names")
        if names:
            return list(names)
    count = 0
    for primitive in mesh.get("primitives", []):
        count = max(count, len(primitive.get("targets", [])))
    return [f"target_{index}" for index in range(count)]


def find_morph_mesh(document: dict, preferred_name: str):
    nodes = document.get("nodes", [])
    meshes = document.get("meshes", [])
    preferred = normalized(preferred_name)
    candidates = []
    for node_index_value, node in enumerate(nodes):
        mesh_index = node.get("mesh")
        if mesh_index is None or mesh_index >= len(meshes):
            continue
        names = target_names_for_mesh(meshes[mesh_index])
        arkit_count = sum(name in names for name in ARKIT_52)
        score = arkit_count * 100
        if preferred and preferred in normalized(node.get("name", "")):
            score += 20
        candidates.append((score, len(names), node_index_value, mesh_index, names))
    if not candidates:
        raise RuntimeError("No morph mesh node found in GLB")
    score, _, node_value, mesh_value, names = max(candidates)
    return node_value, mesh_value, names, score


def expression_bind(node: int, target_indices: dict[str, int], controls: dict[str, float]):
    binds = []
    for name, weight in controls.items():
        index = target_indices.get(name)
        if index is not None:
            binds.append({"node": node, "index": index, "weight": float(weight)})
    return binds


def expression_payload(binds, binary=False, override="none"):
    return {
        "morphTargetBinds": binds,
        "materialColorBinds": [],
        "textureTransformBinds": [],
        "isBinary": bool(binary),
        "overrideBlink": override,
        "overrideLookAt": override,
        "overrideMouth": override,
    }


def patch_vrm(source_glb: Path, output_vrm: Path, skin_node_name: str, hair_bone_chains: list[list[str]]) -> dict:
    document, chunks = read_glb(source_glb)
    nodes = document.get("nodes", [])
    skin_node, skin_mesh, target_names, morph_score = find_morph_mesh(document, skin_node_name)
    target_indices = {name: index for index, name in enumerate(target_names)}
    arkit_present = [name for name in ARKIT_52 if name in target_indices]

    human_bones = {}
    missing_required = []
    for human_name, aliases in HUMANOID_ALIASES.items():
        index = node_index(document, aliases)
        if index is not None:
            human_bones[human_name] = {"node": index}
        elif human_name in REQUIRED_HUMANOID:
            missing_required.append(human_name)

    preset_controls = {
        "neutral": {},
        "happy": {"mouthSmileLeft": 0.85, "mouthSmileRight": 0.85, "cheekSquintLeft": 0.35, "cheekSquintRight": 0.35},
        "angry": {"browDownLeft": 0.78, "browDownRight": 0.78, "mouthFrownLeft": 0.35, "mouthFrownRight": 0.35},
        "sad": {"browInnerUp": 0.75, "mouthFrownLeft": 0.70, "mouthFrownRight": 0.70},
        "relaxed": {"eyeBlinkLeft": 0.12, "eyeBlinkRight": 0.12, "mouthSmileLeft": 0.18, "mouthSmileRight": 0.18},
        "surprised": {"browInnerUp": 0.75, "eyeWideLeft": 0.72, "eyeWideRight": 0.72, "jawOpen": 0.58},
        "aa": {"jawOpen": 0.62, "mouthFunnel": 0.18},
        "ih": {"mouthStretchLeft": 0.55, "mouthStretchRight": 0.55, "jawOpen": 0.16},
        "ou": {"mouthPucker": 0.72, "mouthFunnel": 0.52},
        "ee": {"mouthSmileLeft": 0.48, "mouthSmileRight": 0.48, "mouthStretchLeft": 0.42, "mouthStretchRight": 0.42},
        "oh": {"jawOpen": 0.52, "mouthFunnel": 0.62},
        "blink": {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0},
        "blinkLeft": {"eyeBlinkLeft": 1.0},
        "blinkRight": {"eyeBlinkRight": 1.0},
        "lookUp": {"eyeLookUpLeft": 0.55, "eyeLookUpRight": 0.55},
        "lookDown": {"eyeLookDownLeft": 0.55, "eyeLookDownRight": 0.55},
        "lookLeft": {"eyeLookOutLeft": 0.48, "eyeLookInRight": 0.48},
        "lookRight": {"eyeLookInLeft": 0.48, "eyeLookOutRight": 0.48},
    }
    preset = {
        name: expression_payload(expression_bind(skin_node, target_indices, controls), binary=name.startswith("blink"))
        for name, controls in preset_controls.items()
    }

    head_node = human_bones.get("head", {}).get("node")
    left_eye_node = human_bones.get("leftEye", {}).get("node")
    right_eye_node = human_bones.get("rightEye", {}).get("node")
    look_at = {
        "offsetFromHeadBone": [0.0, 0.06, 0.0],
        "type": "bone" if left_eye_node is not None and right_eye_node is not None else "expression",
        "rangeMapHorizontalInner": {"inputMaxValue": 90.0, "outputScale": 10.0},
        "rangeMapHorizontalOuter": {"inputMaxValue": 90.0, "outputScale": 10.0},
        "rangeMapVerticalDown": {"inputMaxValue": 90.0, "outputScale": 10.0},
        "rangeMapVerticalUp": {"inputMaxValue": 90.0, "outputScale": 10.0},
    }

    vrm = {
        "specVersion": "1.0",
        "meta": {
            "name": "AINA",
            "version": "1.0 Final Production",
            "authors": ["AINA Project"],
            "copyrightInformation": "Copyright 2026 AINA Project",
            "contactInformation": "",
            "references": [],
            "thirdPartyLicenses": "CC0 Vitruvian/Antonia base; AINA-specific sculpt, expressions, hair and materials are project work.",
            "avatarPermission": "everyone",
            "commercialUsage": "corporation",
            "creditNotation": "required",
            "modification": "allowModificationRedistribution",
            "allowRedistribution": True,
            "allowExcessivelyViolentUsage": False,
            "allowExcessivelySexualUsage": False,
            "politicalOrReligiousUsage": "disallow",
            "antisocialOrHateUsage": "disallow",
        },
        "humanoid": {"humanBones": human_bones},
        "firstPerson": {"meshAnnotations": []},
        "lookAt": look_at,
        "expressions": {"preset": preset, "custom": {}},
    }

    spring = None
    spring_nodes = []
    if hair_bone_chains:
        colliders = []
        collider_groups = []
        if head_node is not None:
            colliders.append({"node": head_node, "shape": {"sphere": {"offset": [0.0, 0.055, 0.0], "radius": 0.105}}})
            collider_groups.append({"name": "AINA Head", "colliders": [0]})
        springs = []
        for chain in hair_bone_chains:
            indices = [node_index(document, [name]) for name in chain]
            indices = [index for index in indices if index is not None]
            if not indices:
                continue
            spring_nodes.extend(indices)
            joints = [
                {
                    "node": index,
                    "hitRadius": 0.008,
                    "stiffness": 0.72 if joint_index == 0 else 0.56,
                    "gravityPower": 0.16,
                    "gravityDir": [0.0, -1.0, 0.0],
                    "dragForce": 0.34,
                }
                for joint_index, index in enumerate(indices)
            ]
            item = {"name": chain[0], "joints": joints}
            if collider_groups:
                item["colliderGroups"] = [0]
            springs.append(item)
        spring = {
            "specVersion": "1.0",
            "colliders": colliders,
            "colliderGroups": collider_groups,
            "springs": springs,
        }

    document.setdefault("extensions", {})["VRMC_vrm"] = vrm
    extensions_used = document.setdefault("extensionsUsed", [])
    if "VRMC_vrm" not in extensions_used:
        extensions_used.append("VRMC_vrm")
    extensions_required = document.setdefault("extensionsRequired", [])
    if "VRMC_vrm" not in extensions_required:
        extensions_required.append("VRMC_vrm")
    if spring is not None:
        document["extensions"]["VRMC_springBone"] = spring
        if "VRMC_springBone" not in extensions_used:
            extensions_used.append("VRMC_springBone")

    write_glb(output_vrm, document, chunks)
    return {
        "skin_node": skin_node,
        "skin_mesh": skin_mesh,
        "morph_mesh_score": morph_score,
        "target_name_count": len(target_names),
        "arkit_52_present": arkit_present,
        "arkit_52_count": len(arkit_present),
        "arkit_52_missing": sorted(set(ARKIT_52) - set(arkit_present)),
        "humanoid_bones": human_bones,
        "missing_required_humanoid": missing_required,
        "preset_expression_names": list(preset.keys()),
        "preset_expression_count": len(preset),
        "spring_bone_nodes": spring_nodes,
        "spring_count": len(spring.get("springs", [])) if spring else 0,
        "vrm_spec_version": vrm["specVersion"],
        "output_bytes": output_vrm.stat().st_size,
    }


def inspect_vrm(path: Path) -> dict:
    document, _ = read_glb(path)
    vrm = document.get("extensions", {}).get("VRMC_vrm", {})
    expressions = vrm.get("expressions", {}).get("preset", {})
    humanoid = vrm.get("humanoid", {}).get("humanBones", {})
    spring = document.get("extensions", {}).get("VRMC_springBone", {})
    candidates = []
    for node_index_value, node in enumerate(document.get("nodes", [])):
        mesh_index = node.get("mesh")
        if mesh_index is None:
            continue
        names = target_names_for_mesh(document.get("meshes", [])[mesh_index])
        candidates.append((sum(name in names for name in ARKIT_52), node_index_value, mesh_index, names))
    best = max(candidates, default=(0, None, None, []))
    return {
        "extensions_used": document.get("extensionsUsed", []),
        "vrm_spec_version": vrm.get("specVersion"),
        "humanoid_bone_count": len(humanoid),
        "missing_required_humanoid": sorted(REQUIRED_HUMANOID - set(humanoid)),
        "preset_expression_names": list(expressions),
        "preset_expression_count": len(expressions),
        "arkit_52_count": best[0],
        "arkit_52_missing": sorted(set(ARKIT_52) - set(best[3])),
        "spring_count": len(spring.get("springs", [])),
        "node_count": len(document.get("nodes", [])),
        "mesh_count": len(document.get("meshes", [])),
        "bytes": path.stat().st_size,
    }
