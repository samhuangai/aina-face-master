#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_aina_v10 as base


def quaternion_matrix(rotation):
    x, y, z, w = rotation
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w, 0],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w, 0],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)


def local_matrix(node):
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = node.get("translation", [0, 0, 0])
    rotation = quaternion_matrix(node.get("rotation", [0, 0, 0, 1]))
    scale = np.diag(list(node.get("scale", [1, 1, 1])) + [1]).astype(np.float64)
    return translation @ rotation @ scale


def global_matrices(document):
    parents = {}
    for parent, node in enumerate(document.get("nodes", [])):
        for child in node.get("children", []):
            parents[child] = parent
    cache = {}
    def solve(index):
        if index not in cache:
            own = local_matrix(document["nodes"][index])
            cache[index] = solve(parents[index]) @ own if index in parents else own
        return cache[index]
    return [solve(index) for index in range(len(document.get("nodes", [])))]


_original_add_segmented_hairline = base.add_segmented_hairline


def add_segmented_hairline_fixed(document, binary, face_report):
    report = _original_add_segmented_hairline(document, binary, face_report)
    parent = report["parent_node"]
    primitive = document["meshes"][report["mesh_index"]]["primitives"][0]
    position_accessor = primitive["attributes"]["POSITION"]
    normal_accessor = primitive["attributes"]["NORMAL"]
    vertices = base.accessor_array(document, binary, position_accessor).astype(np.float64)
    normals = base.accessor_array(document, binary, normal_accessor).astype(np.float64)
    parent_global = global_matrices(document)[parent]
    inverse = np.linalg.inv(parent_global)
    local_vertices = (np.c_[vertices, np.ones(len(vertices))] @ inverse.T)[:, :3]
    local_normals = normals @ parent_global[:3, :3]
    local_normals /= np.maximum(np.linalg.norm(local_normals, axis=1, keepdims=True), 1e-8)
    base.write_accessor(document, binary, position_accessor, local_vertices.astype(np.float32))
    base.write_accessor(document, binary, normal_accessor, local_normals.astype(np.float32))
    report["head_local_transform_applied"] = True
    return report


base.add_segmented_hairline = add_segmented_hairline_fixed


if __name__ == "__main__":
    base.main()
