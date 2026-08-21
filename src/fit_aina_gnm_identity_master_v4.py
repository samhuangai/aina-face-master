#!/usr/bin/env python3
"""AINA Identity Master v4 runtime correction.

Runs the real v3 reconstruction after correcting the GNM expression-basis API
name and publishing stable final mesh aliases required by the production gate.
No effect-art images are generated and VRM export remains frozen.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

import fit_aina_gnm_identity_master as v1
import fit_aina_gnm_identity_master_v3 as v3


def fixed_landmark_expression_basis(gnm, indices, weights):
    basis = np.asarray(gnm.expression_basis, dtype=np.float64)
    return (basis[:, indices, :] * weights[None, ..., None]).sum(-2)


_original_export = v3.export


def export_with_final_aliases(gnm, vertices, out, name):
    skin = _original_export(gnm, vertices, out, name)
    if name == "AINA_IDENTITY_MASTER_V3_FINAL":
        # Stable names consumed by downstream identity/rig/VRM stages.
        v1.export_meshes(gnm, vertices, out)
    return skin


def output_dir_from_argv() -> Path:
    if "--out" in sys.argv:
        index = sys.argv.index("--out")
        if index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1])
    return Path("output_identity_master")


def main():
    v3.lm_expression_basis = fixed_landmark_expression_basis
    v3.export = export_with_final_aliases
    v3.main()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        out = output_dir_from_argv()
        out.mkdir(parents=True, exist_ok=True)
        (out / "AINA_IDENTITY_MASTER_V4_ERROR.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        traceback.print_exc()
        raise
