#!/usr/bin/env python3
"""Release-gated entry point for approved-appearance AINA VRM production."""
from __future__ import annotations

import bpy

import aina_vitruvian_final_vrm_production_v6 as visual


base = visual.base
_original_clean_reimport = base.clean_reimport_qa


def clean_reimport_with_color_gate(vrm_path, expected_skin_name, output):
    result = _original_clean_reimport(vrm_path, expected_skin_name, output)
    candidates = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.data.shape_keys and len(obj.data.shape_keys.key_blocks) >= 53
    ]
    candidate = max(candidates, key=lambda obj: len(obj.data.vertices), default=None)
    attributes = [attribute.name for attribute in candidate.data.color_attributes] if candidate else []
    result["imported_color_attributes"] = attributes
    result["approved_face_color_imported"] = "AINA_ApprovedFaceColor" in attributes
    result["pass"] = bool(result.get("pass")) and result["approved_face_color_imported"]
    return result


base.clean_reimport_qa = clean_reimport_with_color_gate


if __name__ == "__main__":
    base.main()
