"""Narrow runtime compatibility shim for the AINA custom-head fusion worker.

FaceVerseModel_torch.all_dims covers identity, expression and texture only, while
run() also slices 27 lighting, 3 angle, 3 translation and 4 eye values. The
custom neutral fusion intentionally has no pose or lighting, so pad those 37
slots with zeros before FaceVerse splits the coefficient tensor.

This shim activates only for fuse_aina_custom_head_v1.py and does not alter any
other repository worker.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _install() -> None:
    if Path(sys.argv[0]).name != "fuse_aina_custom_head_v1.py":
        return
    vendor = Path.cwd() / "vendor" / "faceverse-onnx"
    if not vendor.exists():
        return
    sys.path.insert(0, str(vendor))
    import torch
    from faceversev4 import FaceVerseModel_torch

    original_run = FaceVerseModel_torch.run
    if getattr(original_run, "_aina_coeff_padding", False):
        return

    def padded_run(self, coeffs, *args, **kwargs):
        required = int(self.all_dims) + 37
        if coeffs.ndim != 2:
            raise ValueError(f"FaceVerse coefficients must be rank 2, got {tuple(coeffs.shape)}")
        if coeffs.shape[1] < required:
            padding = coeffs.new_zeros((coeffs.shape[0], required - coeffs.shape[1]))
            coeffs = torch.cat((coeffs, padding), dim=1)
        elif coeffs.shape[1] > required:
            coeffs = coeffs[:, :required]
        return original_run(self, coeffs, *args, **kwargs)

    padded_run._aina_coeff_padding = True
    FaceVerseModel_torch.run = padded_run
    print(f"AINA FaceVerse coefficient compatibility active: {required} slots")


_install()
