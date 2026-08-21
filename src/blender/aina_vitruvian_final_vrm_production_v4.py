#!/usr/bin/env python3
"""Final hardened AINA production entry point with stock-glTF reimport alias."""
from __future__ import annotations

import shutil
from pathlib import Path

import aina_vitruvian_final_vrm_production_v3 as hardened

base = hardened.base
_original_clean_reimport = base.clean_reimport_qa


def clean_reimport_with_glb_alias(vrm_path: Path, expected_skin_name: str, output: Path):
    # Stock Blender recognizes the GLB container by .glb extension.  The bytes
    # remain exactly the final VRM 1.0 bytes; only the temporary QA filename is
    # changed for import dispatch.
    alias = output / "QA" / "AINA_VRM_CLEAN_REIMPORT_ALIAS.glb"
    shutil.copy2(vrm_path, alias)
    result = _original_clean_reimport(alias, expected_skin_name, output)
    result["source_vrm"] = str(vrm_path)
    result["temporary_glb_alias"] = str(alias)
    result["alias_bytes_equal"] = vrm_path.read_bytes() == alias.read_bytes()
    result["pass"] = bool(result.get("pass")) and result["alias_bytes_equal"]
    return result


base.clean_reimport_qa = clean_reimport_with_glb_alias


if __name__ == "__main__":
    base.main()
