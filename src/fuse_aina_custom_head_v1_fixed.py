#!/usr/bin/env python3
"""Corrected entry point for AINA Custom Head v1.

Two public FaceVerse compatibility details are handled here without changing the
core graft algorithm:

1. FaceVerseModel_torch.all_dims excludes the 37 lighting, rotation,
   translation and eye slots required by run(), so neutral inputs are padded.
2. The FaceVerse parsing['skin'] field is not a complete production-head mask
   for this release. The actual facial/head skin is the largest connected
   component of the 19,546-vertex topology, so it is recovered directly from
   mesh connectivity instead of using the incomplete parsing strip.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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


def _largest_component_mask(vertex_count: int, faces: np.ndarray) -> tuple[np.ndarray, dict]:
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
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
    label = int(np.argmax(counts))
    mask = labels == label
    return mask, {
        "component_count": int(component_count),
        "largest_component_label": label,
        "largest_component_vertices": int(mask.sum()),
        "component_sizes_desc": [int(value) for value in sorted(counts.tolist(), reverse=True)],
    }


def _load_faceverse_complete_head():
    identity_path = (
        pipeline.ROOT
        / "output_faceverse_v120"
        / "AINA_FACEVERSE_IDENTITY_156_v12.0.npy"
    )
    identity = np.load(identity_path).astype(np.float32)
    model = pipeline.FaceVerseModel_torch(
        device=torch.device("cpu"),
        facevrsepath=str(pipeline.FVROOT / "data/faceverse_v4_2.npy"),
        camera_distance=10,
        focal=1000,
        center=128,
    )
    neutral = np.zeros((int(model.all_dims),), dtype=np.float32)
    neutral[: int(model.id_dims)] = identity[: int(model.id_dims)]
    with torch.no_grad():
        result = model.run(
            torch.from_numpy(neutral[None]).float(),
            only_lms=False,
            use_color=False,
        )
    raw = np.asarray(result["vertices"][0].cpu(), dtype=np.float64)
    vertices, scale, centre = pipeline.normalize_metric(raw)
    faces = np.asarray(model.tri.cpu(), dtype=np.int64)
    if faces.min() == 1:
        faces -= 1

    skin_mask, component_report = _largest_component_mask(len(vertices), faces)
    if int(skin_mask.sum()) < 5000:
        raise RuntimeError(
            f"Largest FaceVerse surface component is unexpectedly small: {int(skin_mask.sum())}"
        )
    print({"faceverse_surface_components": component_report})
    return vertices, faces, skin_mask, identity, scale, centre


pipeline.FaceVerseModel_torch.run = _run_with_complete_coefficients
pipeline.load_faceverse = _load_faceverse_complete_head


if __name__ == "__main__":
    pipeline.main()
