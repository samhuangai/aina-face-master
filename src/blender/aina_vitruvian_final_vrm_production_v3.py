#!/usr/bin/env python3
"""Hardened final AINA VRM production entry point."""
from __future__ import annotations

import numpy as np
import bpy

import aina_vitruvian_final_vrm_production as base
import aina_vitruvian_vrm_patch_v2 as patch_v2


base.patch_vrm = patch_v2.patch_vrm
base.inspect_vrm = patch_v2.inspect_vrm
base.ARKIT_52 = patch_v2.ARKIT_52


_original_create_curve_mesh = base.create_curve_mesh


def safe_create_curve_mesh(name, points, radius, mat):
    # Conversion must never include a previously selected body/head object.
    bpy.ops.object.select_all(action="DESELECT")
    obj = _original_create_curve_mesh(name, points, radius, mat)
    bpy.ops.object.select_all(action="DESELECT")
    return obj


base.create_curve_mesh = safe_create_curve_mesh


_original_render_qa = base.render_qa


def render_qa_with_side(scene, skin, output, setup, full_bounds):
    renders, activated = _original_render_qa(scene, skin, output, setup, full_bounds)
    base.clear_arkit(skin)
    target = np.asarray(setup["target"], dtype=np.float64)
    side = np.asarray(setup["locations"]["side"], dtype=np.float64)
    path = output / "Preview" / "AINA_FINAL_PORTRAIT_NEUTRAL_SIDE.png"
    renders["neutral_side"] = str(base.render_camera(scene, side, target, 86, path))
    base.clear_arkit(skin)
    return renders, activated


base.render_qa = render_qa_with_side


def export_glb_safe(path):
    bpy.ops.object.select_all(action="SELECT")
    # Blender 4.5 exports morph normals by default when morphs are enabled.  Keep
    # the call limited to stable operator properties used by stock Blender.
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        export_morph=True,
        export_apply=False,
        export_animations=False,
    )


base.export_glb = export_glb_safe


if __name__ == "__main__":
    base.main()
