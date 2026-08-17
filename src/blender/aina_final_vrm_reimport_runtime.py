#!/usr/bin/env python3
"""Production entrypoint for clean-scene AINA VRM reimport QA."""
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

    import aina_final_vrm_reimport_qa as qa

    def already_enabled(_root: Path) -> None:
        import bpy
        if 'io_scene_vrm' not in bpy.context.preferences.addons:
            raise RuntimeError('VRM Addon preferences disappeared before reimport QA')

    qa.register_vrm = already_enabled
    qa.main()


if __name__ == '__main__':
    main()
