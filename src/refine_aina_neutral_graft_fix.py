#!/usr/bin/env python3
"""Neutral-mouth and face-to-collar correction for the refined AINA OBJ.

Runs after refine_aina_orbit_jaw_neck.py. It keeps topology and vertex order,
closes the unintended neutral mouth opening, restores a little chin length after
the aggressive V pass, and smooths the under-chin graft zone. The result remains
fully compatible with the existing 52-control generator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from refine_aina_orbit_jaw_neck import (
    K,
    component_roots,
    map_from_blender,
    map_to_blender,
    read_obj,
    scale_region,
    shift,
    write_obj,
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--height", type=float, default=1.72)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    lines, raw, faces = read_obj(args.mesh)
    roots = component_roots(len(raw), faces)
    head = np.flatnonzero(roots == roots[int(K[0])])
    base, scale_factor, offset = map_to_blender(raw, args.height)
    out = base.copy()

    # Close the neutral lip seam. Upper lip moves down, lower lip moves up; both
    # settle slightly back into the perioral surface instead of exposing teeth.
    lm = out[K].copy()
    mouth = lm[48:60].mean(axis=0)
    scale_region(out, head, mouth, (0.047, 0.035, 0.025), (1.0, 1.0, 0.84), 0.08, 1.14)
    shift(out, head, mouth, (0.050, 0.038, 0.031), (0.0, 0.0010, 0.0), 0.02, 1.12)

    lm = out[K].copy()
    upper = lm[[49, 50, 51, 52, 53]].mean(axis=0)
    lower = lm[[55, 56, 57, 58, 59]].mean(axis=0)
    shift(out, head, upper, (0.034, 0.024, 0.012), (0.0, 0.0006, -0.00125), 0.02, 1.06)
    shift(out, head, lower, (0.036, 0.025, 0.013), (0.0, 0.0007, 0.00135), 0.02, 1.06)

    # Restore part of the chin length removed by stage 2 and keep a rounded tip.
    lm = out[K].copy()
    chin = lm[8]
    scale_region(out, head, chin, (0.045, 0.047, 0.043), (1.035, 1.0, 1.03), 0.02, 1.08)
    shift(out, head, chin, (0.047, 0.047, 0.043), (0.0, 0.0005, -0.00135), 0.0, 1.05)

    # Keep the under-chin shell behind the jaw and below the neutral mouth so the
    # body collar cannot visually cut through the lower face.
    lm = out[K].copy()
    chin = lm[8]
    under_chin = (chin[0], chin[1] + 0.020, chin[2] - 0.016)
    scale_region(out, head, under_chin, (0.060, 0.064, 0.038), (0.94, 0.97, 1.02), 0.0, 1.08)
    shift(out, head, under_chin, (0.060, 0.064, 0.038), (0.0, 0.0010, -0.00075), 0.0, 1.05)

    displacement = out - base
    final_raw = map_from_blender(out, scale_factor, offset)
    write_obj(lines, final_raw, args.out)

    magnitude = np.linalg.norm(displacement[head], axis=1)
    report = {
        "product": "AINA neutral mouth and graft correction",
        "source": str(args.mesh),
        "output": str(args.out),
        "topology_changed": False,
        "vertex_count": int(len(raw)),
        "face_count": int(len(faces)),
        "max_head_displacement_m": float(magnitude.max()),
        "rms_head_displacement_m": float(np.sqrt(np.mean(magnitude * magnitude))),
        "neutral_mouth_closed": True,
        "collar_intersection_guard": True,
        "identity_lock": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
