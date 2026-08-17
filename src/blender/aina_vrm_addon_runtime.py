#!/usr/bin/env python3
"""Install and enable the VRM Addon through Blender's real add-on preferences.

Directly calling io_scene_vrm.register() registers classes but does not create
bpy.context.preferences.addons['io_scene_vrm'], which the VRM validator and
migration handlers require.  Final production jobs use this helper instead.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import bpy


def ensure_vrm_addon(root: Path) -> None:
    source = root / 'vendor' / 'VRM-Addon-for-Blender' / 'src' / 'io_scene_vrm'
    if not source.is_dir():
        raise RuntimeError(f'VRM Addon source is missing: {source}')

    user_addons = Path(bpy.utils.user_resource('SCRIPTS', path='addons', create=True))
    user_addons.mkdir(parents=True, exist_ok=True)
    installed = user_addons / 'io_scene_vrm'
    if installed.exists():
        shutil.rmtree(installed)
    shutil.copytree(source, installed)

    # Make Blender discover the copied package as a real add-on, not merely a
    # Python module. This is what creates AddonPreferences in context.preferences.
    if str(user_addons) not in sys.path:
        sys.path.insert(0, str(user_addons))
    bpy.utils.refresh_script_paths()
    try:
        bpy.ops.preferences.addon_refresh()
    except Exception:
        # Some Blender builds refresh automatically. The explicit enable below
        # remains the authoritative operation.
        pass

    result = bpy.ops.preferences.addon_enable(module='io_scene_vrm')
    if result != {'FINISHED'}:
        raise RuntimeError(f'VRM Addon enable failed: {result}')
    if 'io_scene_vrm' not in bpy.context.preferences.addons:
        raise RuntimeError('VRM Addon enabled without AddonPreferences entry')

    prefs = bpy.context.preferences.addons['io_scene_vrm'].preferences
    if prefs is None:
        raise RuntimeError('VRM Addon preferences object is unavailable')

    # Force a depsgraph update now so registration/migration problems fail
    # before the expensive final preview renders begin.
    bpy.context.view_layer.update()
    print('[AINA_FINAL] VRM Addon enabled through Blender preferences:', installed, flush=True)
