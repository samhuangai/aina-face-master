#!/usr/bin/env python3
"""Production entrypoint for final AINA VRM assembly.

Enables the VRM Addon through Blender preferences before importing the existing
final release implementation. The face/body/52-controls logic remains in the
canonical assembly files; this wrapper only fixes Blender add-on lifecycle.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aina_vrm_addon_runtime import ensure_vrm_addon


def main() -> None:
    root = Path.cwd()
    ensure_vrm_addon(root)

    import aina_final_vrm_release as release

    def already_enabled(_root: Path):
        import bpy
        if 'io_scene_vrm' not in bpy.context.preferences.addons:
            raise RuntimeError('VRM Addon preferences disappeared before assembly')
        return None, None, None

    # Avoid the old direct io_scene_vrm.register() path. Keep the deterministic
    # Blender-native body factory selected by aina_final_vrm_release.py.
    release.core.enable_addons = already_enabled
    release.core.create_body = release.create_native_body
    release.core.main()


if __name__ == '__main__':
    main()
