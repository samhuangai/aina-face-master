#!/usr/bin/env python3
"""AINA real-mesh expression convergence v3.

A focused corrective layer over v2. It protects the nose/upper midface from OU
and funnel/pucker spill, strengthens the real brow geometry, and exposes a
proper real mouth cavity for jaw-open and viseme QA. The underlying refined OBJ,
vertex order, 52 controls and expression preset weights are unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aina_final_vrm_assembly as base
import aina_surface_expression_qa_v2 as v2


_original_post = v2.post_correct_shape_keys


def post_correct_shape_keys_v3(head, mapped):
    stats = _original_post(head, mapped)
    blocks = head.data.shape_keys.key_blocks
    basis = v2.key_array(blocks["Basis"])
    lm = mapped[base.K]
    nose = lm[27:36].mean(0)
    mouth = lm[48:60].mean(0)
    upper = lm[[49, 50, 51, 52, 53]].mean(0)
    lower = lm[[55, 56, 57, 58, 59]].mean(0)
    chin = lm[8]
    jaw = (mouth + chin) * 0.5

    # The only automated v2 miss was OU pulling the lower nose. Suppress pucker
    # and funnel displacement inside a strict nasal protection envelope while
    # preserving all deformation on the lips and philtrum below the alar base.
    nose_protect = base.weights(basis, nose, (0.038, 0.040, 0.050), 0.0, 1.02)
    below_nose = np.clip((nose[2] - basis[:, 2]) / 0.040, 0.0, 1.0)
    protect = nose_protect * (1.0 - below_nose)
    for name in ("mouthFunnel", "mouthPucker"):
        coords = v2.key_array(blocks[name])
        delta = coords - basis
        delta *= (1.0 - 0.98 * protect)[:, None]
        v2.set_key_array(blocks[name], basis + delta)

    # Make jaw-open visibly readable without affecting the eye/nose identity.
    coords = v2.key_array(blocks["jawOpen"])
    base.shift_region(coords, jaw, (0.060, 0.066, 0.056), (0.0, 0.0008, -0.0022), 0.02, 1.01)
    base.shift_region(coords, lower, (0.040, 0.031, 0.020), (0.0, -0.0004, -0.0018), 0.02, 1.01)
    base.shift_region(coords, upper, (0.038, 0.030, 0.018), (0.0, -0.0002, 0.0007), 0.02, 1.01)
    v2.set_key_array(blocks["jawOpen"], coords)

    # Slightly strengthen smile/frown corner separation so emotion remains clear
    # at portrait distance without changing the neutral identity.
    for name, direction in (
        ("mouthSmileLeft", 1.0), ("mouthSmileRight", -1.0),
        ("mouthFrownLeft", 1.0), ("mouthFrownRight", -1.0),
    ):
        coords = v2.key_array(blocks[name])
        corner = lm[54] if "Left" in name else lm[48]
        if "Smile" in name:
            delta = (0.00055 * direction, -0.00015, 0.00085)
        else:
            delta = (0.00020 * direction, 0.00010, -0.00070)
        base.shift_region(coords, corner, (0.026, 0.024, 0.021), delta, 0.04, 1.01)
        v2.set_key_array(blocks[name], coords)

    # Refresh stats after v3 corrections.
    for name in base.SHAPE_KEYS:
        coords = v2.key_array(blocks[name])
        moved = np.linalg.norm(coords - basis, axis=1)
        stats[name] = {
            "max_m": float(moved.max()),
            "rms_m": float(np.sqrt(np.mean(moved * moved))),
            "moved_vertices": int(np.sum(moved > 1e-5)),
        }
    return stats


def create_brow_v3(name, points, side_name, mat):
    points = np.asarray(points, dtype=np.float64).copy()
    points[:, 1] -= 0.0044
    points[:, 2] += 0.0005
    half_thickness = 0.00120
    vertices = []
    for point in points:
        vertices.append((point[0], point[1], point[2] - half_thickness))
        vertices.append((point[0], point[1], point[2] + half_thickness))
    faces = []
    for i in range(len(points) - 1):
        a = 2 * i; b = a + 1; c = a + 2; d = a + 3
        faces.extend(((a, c, d), (a, d, b)))
    obj = base.mesh_object(name, np.asarray(vertices), np.asarray(faces, dtype=np.int32))
    obj.data.materials.append(mat)
    obj.shape_key_add(name="Basis")
    basis = np.asarray(vertices, dtype=np.float64)
    inner_point = int(np.argmin(np.abs(points[:, 0])))
    outer_point = int(np.argmax(np.abs(points[:, 0])))
    index = np.arange(len(points), dtype=np.float64)
    span = max(abs(outer_point - inner_point), 1.0)
    inner_weight = np.clip(1.0 - np.abs(index - inner_point) / span, 0.0, 1.0)
    outer_weight = np.clip(1.0 - np.abs(index - outer_point) / span, 0.0, 1.0)

    def add_key(key_name, dz_per_point, dx_per_point=None):
        coords = basis.copy()
        dx = np.zeros(len(points)) if dx_per_point is None else np.asarray(dx_per_point)
        for i in range(len(points)):
            coords[2 * i : 2 * i + 2, 0] += dx[i]
            coords[2 * i : 2 * i + 2, 2] += dz_per_point[i]
        key = obj.shape_key_add(name=key_name)
        v2.set_key_array(key, coords)

    sign = 1.0 if side_name == "Left" else -1.0
    add_key(f"browDown{side_name}", -0.0062 * (0.72 + 0.28 * inner_weight), 0.00065 * sign * inner_weight)
    add_key(f"browOuterUp{side_name}", 0.0066 * outer_weight, 0.00055 * sign * outer_weight)
    add_key("browInnerUp", 0.0072 * inner_weight, -0.00035 * sign * inner_weight)
    return obj


def create_mouth_cavity_v3(center, mat):
    center = np.asarray(center, dtype=np.float64).copy()
    # Move the cavity just behind the lip seam, rather than deep inside the head.
    center[1] += 0.0015
    segments = 64

    def ellipse(rx, rz, z_shift=0.0):
        vertices = [(center[0], center[1], center[2] + z_shift)]
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            vertices.append((
                center[0] + rx * np.cos(angle),
                center[1],
                center[2] + z_shift + rz * np.sin(angle),
            ))
        return np.asarray(vertices, dtype=np.float64)

    basis = ellipse(0.0190, 0.00045)
    faces = [(0, 1 + i, 1 + ((i + 1) % segments)) for i in range(segments)]
    obj = base.mesh_object("AINA_Mouth_Cavity", basis, np.asarray(faces, dtype=np.int32))
    obj.data.materials.append(mat)
    obj.shape_key_add(name="Basis")
    shapes = {
        "jawOpen": ellipse(0.0195, 0.0088, -0.0022),
        "mouthFunnel": ellipse(0.0130, 0.0058),
        "mouthPucker": ellipse(0.0105, 0.0052),
        "mouthClose": ellipse(0.0185, 0.00018),
        "mouthStretchLeft": ellipse(0.0230, 0.0015),
        "mouthStretchRight": ellipse(0.0230, 0.0015),
        "mouthSmileLeft": ellipse(0.0208, 0.0020, 0.0010),
        "mouthSmileRight": ellipse(0.0208, 0.0020, 0.0010),
        "mouthFrownLeft": ellipse(0.0198, 0.0020, -0.0010),
        "mouthFrownRight": ellipse(0.0198, 0.0020, -0.0010),
    }
    for name, coords in shapes.items():
        key = obj.shape_key_add(name=name)
        v2.set_key_array(key, coords)
    return obj


v2.post_correct_shape_keys = post_correct_shape_keys_v3
v2.create_brow = create_brow_v3
v2.create_mouth_cavity = create_mouth_cavity_v3


if __name__ == "__main__":
    v2.main()
