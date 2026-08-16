#!/usr/bin/env python3
"""Final assembly entry point with Blender-managed VRM add-on enablement."""
from __future__ import annotations
import sys
from pathlib import Path
import bpy, addon_utils
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import aina_final_vrm_release as release


def enable_vrm_addon(root:Path):
    src=root/'vendor'/'VRM-Addon-for-Blender'/'src'
    if str(src) not in sys.path: sys.path.insert(0,str(src))
    # addon_utils.enable() creates the Addon preferences entry required by the
    # VRM validator/exporter. Direct module.register() does not.
    addon_utils.enable('io_scene_vrm', default_set=True, persistent=True)
    import io_scene_vrm  # noqa: F401
    pref=bpy.context.preferences.addons.get('io_scene_vrm')
    if pref is None:
        raise RuntimeError('VRM Addon was imported but Blender AddonPreferences were not created')
    release.core.log('VRM Addon enabled through Blender AddonPreferences')
    return None,None,None

release.core.enable_addons=enable_vrm_addon

if __name__=='__main__':
    release.core.main()
