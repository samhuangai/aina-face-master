#!/usr/bin/env python3
"""Compatibility wrapper for AINA VRM 1.0 patching.

VRMC_vrm is advertised through extensionsUsed but not extensionsRequired so a
stock Blender glTF importer can perform the mandated clean morph/armature
reimport even when the VRM add-on is not installed.  VRM-aware applications
still read the complete extension normally.
"""
from __future__ import annotations

from pathlib import Path

import aina_vitruvian_vrm_patch as base

ARKIT_52 = base.ARKIT_52
inspect_vrm = base.inspect_vrm


def patch_vrm(source_glb: Path, output_vrm: Path, skin_node_name: str, hair_bone_chains: list[list[str]]) -> dict:
    report = base.patch_vrm(source_glb, output_vrm, skin_node_name, hair_bone_chains)
    document, chunks = base.read_glb(output_vrm)
    required = [name for name in document.get("extensionsRequired", []) if name not in {"VRMC_vrm", "VRMC_springBone"}]
    if required:
        document["extensionsRequired"] = required
    else:
        document.pop("extensionsRequired", None)
    base.write_glb(output_vrm, document, chunks)
    report["extensions_required_after_patch"] = document.get("extensionsRequired", [])
    report["stock_gltf_reimport_compatible"] = True
    report["output_bytes"] = output_vrm.stat().st_size
    return report
