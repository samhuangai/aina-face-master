#!/usr/bin/env python3
"""Corrected tonal probe for the real CC0 Vitruvian FACS head."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aina_vitruvian_head_probe as probe


_original_material = probe.material


def material_v2(name, color, roughness=0.48, metallic=0.0):
    overrides = {
        "AINA_Probe_Clay": ((0.20, 0.23, 0.28, 1.0), 0.58, 0.0),
        "AINA_Probe_EyeWhite": ((0.62, 0.69, 0.78, 1.0), 0.30, 0.0),
        "AINA_Probe_Mouth": ((0.055, 0.012, 0.018, 1.0), 0.48, 0.0),
    }
    if name in overrides:
        color, roughness, metallic = overrides[name]
    return _original_material(name, color, roughness, metallic)


_original_light = probe.create_light


def create_light_v2(name, location, energy, size, target):
    return _original_light(name, location, energy * 0.42, size, target)


_original_render = probe.render


def render_v2(scene, camera, output, location, target, lens=82):
    scene.world.color = (0.025, 0.030, 0.045)
    try:
        scene.view_settings.exposure = -0.80
    except Exception:
        pass
    return _original_render(scene, camera, output, location, target, lens)


probe.material = material_v2
probe.create_light = create_light_v2
probe.render = render_v2


if __name__ == "__main__":
    probe.main()
