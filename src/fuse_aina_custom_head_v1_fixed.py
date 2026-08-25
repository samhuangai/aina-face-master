#!/usr/bin/env python3
"""Compatibility entry point for AINA Custom Head v1.

FaceVerseModel_torch.all_dims covers identity, expression, and texture only.
The public FaceVerse run() API additionally requires 27 lighting, 3 rotation,
3 translation, and 4 eye coefficients. The first custom-head run allocated only
all_dims, leaving the angle slice empty. This wrapper pads the coefficient
vector to all_dims + 37 while leaving FaceVerse's internal dimension offsets
unchanged, then executes the unchanged graft pipeline.
"""
from __future__ import annotations

import torch

import fuse_aina_custom_head_v1 as pipeline

_ORIGINAL_RUN = pipeline.FaceVerseModel_torch.run


def _run_with_complete_coefficients(self, coeffs, only_lms=False, use_color=False, use_lighting=False):
    required = int(self.all_dims) + 37
    if coeffs.ndim != 2:
        raise ValueError(f"FaceVerse coefficients must be rank-2, got {tuple(coeffs.shape)}")
    if coeffs.shape[1] < required:
        padding = torch.zeros(
            (coeffs.shape[0], required - coeffs.shape[1]),
            dtype=coeffs.dtype,
            device=coeffs.device,
        )
        coeffs = torch.cat([coeffs, padding], dim=1)
    elif coeffs.shape[1] > required:
        coeffs = coeffs[:, :required]
    return _ORIGINAL_RUN(
        self,
        coeffs,
        only_lms=only_lms,
        use_color=use_color,
        use_lighting=use_lighting,
    )


pipeline.FaceVerseModel_torch.run = _run_with_complete_coefficients


if __name__ == "__main__":
    pipeline.main()
