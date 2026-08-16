#!/usr/bin/env python3
"""Final VRM clean-scene reimport QA with Blender-managed add-on enablement."""
from __future__ import annotations
import sys
from pathlib import Path
import bpy, addon_utils
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import aina_final_vrm_reimport_qa as qa


def register_vrm(root:Path):
    src=root/'vendor'/'VRM-Addon-for-Blender'/'src'
    if str(src) not in sys.path: sys.path.insert(0,str(src))
    addon_utils.enable('io_scene_vrm', default_set=True, persistent=True)
    import io_scene_vrm  # noqa: F401
    if bpy.context.preferences.addons.get('io_scene_vrm') is None:
        raise RuntimeError('VRM AddonPreferences missing during reimport QA')

qa.register_vrm=register_vrm

if __name__=='__main__':
    qa.main()
