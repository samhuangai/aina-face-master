#!/usr/bin/env python3
"""Corrected entry point for AINA Custom Head v1.

This wrapper fixes two public FaceVerse integration details while keeping the
core custom-head graft unchanged:

1. ``FaceVerseModel_torch.all_dims`` covers identity, expression and texture,
   but ``run()`` additionally reads 27 lighting, 3 rotation, 3 translation and
   4 eye coefficients. Neutral inputs are therefore padded to ``all_dims+37``.
2. The release's ``parsing['skin']`` mask is only a narrow auxiliary strip and
   is not the complete renderable head. Before the graft reads it, the mask is
   replaced by the union of FaceVerse's explicit front-face mask and its scalp
   skin mask. If that union is unexpectedly small, the largest connected mesh
   component is included as a safe outer-surface fallback.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import fuse_aina_custom_head_v1 as pipeline

_ORIGINAL_INIT = pipeline.FaceVerseModel_torch.__init__
_ORIGINAL_RUN = pipeline.FaceVerseModel_torch.run


def _largest_component_mask(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels, minlength=component_count)
    return labels == int(np.argmax(counts))


def _init_with_complete_surface(self, *args, **kwargs):
    _ORIGINAL_INIT(self, *args, **kwargs)
    vertex_count = int(self.meanshape.shape[1])
    faces = np.asarray(self.tri.cpu(), dtype=np.int64)
    if faces.min() == 1:
        faces = faces - 1

    parsing_skin = np.asarray(self.fvd["parsing"]["skin"]).reshape(-1) > 0
    front_face = np.asarray(self.fvd["face_mask"]).reshape(-1) > 0
    if len(parsing_skin) != vertex_count or len(front_face) != vertex_count:
        raise RuntimeError(
            f"FaceVerse mask length mismatch: skin={len(parsing_skin)}, "
            f"front={len(front_face)}, vertices={vertex_count}"
        )

    complete_surface = parsing_skin | front_face
    if int(complete_surface.sum()) < 5000:
        complete_surface |= _largest_component_mask(vertex_count, faces)
    if int(complete_surface.sum()) < 5000:
        raise RuntimeError(
            f"Recovered FaceVerse outer surface is unexpectedly small: "
            f"{int(complete_surface.sum())}"
        )

    self.fvd["parsing"]["skin"] = complete_surface.astype(np.uint8)
    print(
        {
            "faceverse_surface_mask": {
                "parsing_skin_vertices": int(parsing_skin.sum()),
                "front_face_vertices": int(front_face.sum()),
                "complete_surface_vertices": int(complete_surface.sum()),
            }
        }
    )


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


pipeline.FaceVerseModel_torch.__init__ = _init_with_complete_surface
pipeline.FaceVerseModel_torch.run = _run_with_complete_coefficients


if __name__ == "__main__":
    pipeline.main()
