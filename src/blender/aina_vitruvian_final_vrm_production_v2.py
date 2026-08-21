#!/usr/bin/env python3
"""QA-complete entry point for AINA final VRM production."""
from __future__ import annotations

import numpy as np

import aina_vitruvian_final_vrm_production as base


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


if __name__ == "__main__":
    base.main()
