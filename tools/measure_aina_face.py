#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def detect(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.18,
    ) as detector:
        result = detector.process(rgb)
    if not result.multi_face_landmarks:
        raise RuntimeError(f"MediaPipe did not detect a face in {path}")
    return np.asarray(
        [(point.x, point.y, point.z) for point in result.multi_face_landmarks[0].landmark],
        dtype=np.float64,
    )


def distance(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a, :2] - points[b, :2]))


def metrics(points: np.ndarray) -> dict[str, float]:
    face_width = distance(points, 234, 454)
    face_height = distance(points, 10, 152)
    eye_left_width = distance(points, 33, 133)
    eye_right_width = distance(points, 362, 263)
    eye_left_height = distance(points, 159, 145)
    eye_right_height = distance(points, 386, 374)
    brow_left_gap = distance(points, 105, 159)
    brow_right_gap = distance(points, 334, 386)
    return {
        "face_aspect": face_height / face_width,
        "eye_width": (eye_left_width + eye_right_width) / (2.0 * face_width),
        "eye_height": (eye_left_height + eye_right_height) / (2.0 * face_height),
        "eye_spacing": distance(points, 133, 362) / face_width,
        "brow_eye_gap": (brow_left_gap + brow_right_gap) / (2.0 * face_height),
        "nose_length": distance(points, 168, 1) / face_height,
        "nose_width": distance(points, 129, 358) / face_width,
        "mouth_width": distance(points, 61, 291) / face_width,
        "nose_mouth": distance(points, 1, 13) / face_height,
        "lower_third": distance(points, 13, 152) / face_height,
        "upper_face": distance(points, 10, 168) / face_height,
    }


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def ratio(reference: dict[str, float], model: dict[str, float], key: str, limits=(0.82, 1.18)) -> float:
    return clamp(reference[key] / max(model[key], 1e-8), limits[0], limits[1])


def normalized_shape(points: np.ndarray) -> np.ndarray:
    indices = np.asarray([
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
        379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
        127, 162, 21, 54, 103, 67, 109, 33, 133, 362, 263, 168, 1, 61, 291, 13, 14,
    ], dtype=np.int64)
    shape = points[indices, :2].copy()
    shape -= shape.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(shape)
    if norm > 1e-8:
        shape /= norm
    return shape


def procrustes_rms(reference_points: np.ndarray, model_points: np.ndarray) -> float:
    reference = normalized_shape(reference_points)
    model = normalized_shape(model_points)
    u, _, vt = np.linalg.svd(model.T @ reference)
    rotation = u @ vt
    aligned = model @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=1))))


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("measure_aina_face.py REFERENCE_FRONT MODEL_FRONT OUTPUT_JSON")
    reference_path = Path(sys.argv[1])
    model_path = Path(sys.argv[2])
    output = Path(sys.argv[3])
    reference_points = detect(reference_path)
    model_points = detect(model_path)
    reference = metrics(reference_points)
    model = metrics(model_points)
    parameters = {
        "face_width_scale": clamp(model["face_aspect"] / reference["face_aspect"], 0.88, 1.12),
        "eye_width_scale": ratio(reference, model, "eye_width", (0.78, 1.15)),
        "eye_height_scale": ratio(reference, model, "eye_height", (0.74, 1.18)),
        "eye_spacing_scale": ratio(reference, model, "eye_spacing", (0.90, 1.10)),
        "brow_eye_gap_scale": ratio(reference, model, "brow_eye_gap", (0.84, 1.16)),
        "nose_length_scale": ratio(reference, model, "nose_length", (0.86, 1.16)),
        "nose_width_scale": ratio(reference, model, "nose_width", (0.88, 1.12)),
        "mouth_width_scale": ratio(reference, model, "mouth_width", (0.86, 1.14)),
        "nose_mouth_scale": ratio(reference, model, "nose_mouth", (0.86, 1.14)),
        "lower_third_scale": ratio(reference, model, "lower_third", (0.88, 1.14)),
        "upper_face_scale": ratio(reference, model, "upper_face", (0.88, 1.12)),
    }
    # Damp the second and later closed-loop pass to prevent oscillation.
    damping = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    if damping != 1.0:
        parameters = {key: 1.0 + (value - 1.0) * damping for key, value in parameters.items()}
    report = {
        "reference_image": str(reference_path),
        "model_image": str(model_path),
        "reference_metrics": reference,
        "model_metrics": model,
        "ratios": {key: reference[key] / max(model[key], 1e-8) for key in reference},
        "parameters": parameters,
        "procrustes_rms": procrustes_rms(reference_points, model_points),
        "detected_reference_landmarks": len(reference_points),
        "detected_model_landmarks": len(model_points),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
