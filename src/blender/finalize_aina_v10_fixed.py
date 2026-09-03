#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finalize_aina_v10 as base


# The imported face already contains preserved VRoid eyeline geometry. Keeping the
# accent curves out of V10 avoids double-transforming world-space curve points when
# bone parenting under the head.
base.add_eyelash_accents = lambda armature: []

_original_world_bounds = base.world_bounds
_bounds_call_count = 0


def world_bounds_fixed(objects):
    global _bounds_call_count
    _bounds_call_count += 1
    # First call is the portrait/head framing. Second call is the full-body frame;
    # replace the size-filtered list with all character mesh/curve objects after the
    # diagnostic sphere has already been deleted.
    if _bounds_call_count == 2:
        import bpy
        objects = [
            obj for obj in bpy.context.scene.objects
            if obj.type in {"MESH", "CURVE"}
            and not obj.name.startswith(("CAM_", "LGT_"))
            and not obj.name.lower().startswith(("icosphere", "sphere", "cube"))
        ]
    return _original_world_bounds(objects)


base.world_bounds = world_bounds_fixed


if __name__ == "__main__":
    base.main()
