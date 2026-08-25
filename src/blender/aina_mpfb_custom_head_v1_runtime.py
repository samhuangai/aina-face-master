#!/usr/bin/env python3
"""Runtime wrapper for the installed MPFB2 Blender extension."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aina_mpfb_custom_head_v1 as pipeline


def installed_mpfb(_root: Path):
    package = "bl_ext.user_default.mpfb"
    try:
        importlib.import_module(package)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MPFB2 is not installed in Blender's user_default extension repository"
        ) from exc
    services = importlib.import_module(package + ".services")
    return (
        getattr(services, "HumanService"),
        getattr(services, "TargetService"),
    )


pipeline.install_mpfb = installed_mpfb
pipeline.main()
