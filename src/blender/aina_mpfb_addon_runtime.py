#!/usr/bin/env python3
"""Enable MPFB2 as a real Blender add-on for headless production jobs."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import bpy


def ensure_mpfb_addon(root: Path) -> None:
    source = root / 'vendor' / 'mpfb2' / 'src' / 'mpfb'
    if not source.is_dir():
        raise RuntimeError(f'MPFB2 source missing: {source}')
    user_addons = Path(bpy.utils.user_resource('SCRIPTS', path='addons', create=True))
    user_addons.mkdir(parents=True, exist_ok=True)
    installed = user_addons / 'mpfb'
    if installed.exists():
        shutil.rmtree(installed)
    shutil.copytree(source, installed)
    if str(user_addons) not in sys.path:
        sys.path.insert(0, str(user_addons))
    bpy.utils.refresh_script_paths()
    try:
        bpy.ops.preferences.addon_refresh()
    except Exception:
        pass
    result = bpy.ops.preferences.addon_enable(module='mpfb')
    if result != {'FINISHED'}:
        raise RuntimeError(f'MPFB2 addon_enable failed: {result}')
    if 'mpfb' not in bpy.context.preferences.addons:
        raise RuntimeError('MPFB2 enabled without Blender AddonPreferences entry')
    bpy.context.view_layer.update()
    print('[AINA_BODY] MPFB2 enabled through Blender preferences:', installed, flush=True)
