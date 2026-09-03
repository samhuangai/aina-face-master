#!/usr/bin/env python3
from __future__ import annotations

import colorsys
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


LEFT_EYE = np.array([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], dtype=np.int64)
RIGHT_EYE = np.array([362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398], dtype=np.int64)


def detect(image_rgb: np.ndarray) -> np.ndarray:
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.16) as detector:
        result = detector.process(image_rgb)
    if not result.multi_face_landmarks:
        raise RuntimeError("MediaPipe could not detect the approved AINA face")
    return np.asarray([(point.x, point.y, point.z) for point in result.multi_face_landmarks[0].landmark], dtype=np.float64)


def disk_pixels(image: np.ndarray, center: tuple[float, float], radius: float) -> np.ndarray:
    height, width = image.shape[:2]
    cx, cy = int(round(center[0] * width)), int(round(center[1] * height))
    r = max(2, int(round(radius * min(width, height))))
    y0, y1 = max(0, cy - r), min(height, cy + r + 1)
    x0, x1 = max(0, cx - r), min(width, cx + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    return image[y0:y1, x0:x1][mask]


def median_rgb(samples: list[np.ndarray], fallback: tuple[int, int, int], predicate=None) -> list[int]:
    valid = [sample.reshape(-1, 3) for sample in samples if sample.size]
    if not valid:
        return list(fallback)
    pixels = np.concatenate(valid).astype(np.float64)
    if predicate is not None:
        keep = predicate(pixels)
        if np.count_nonzero(keep) >= 12:
            pixels = pixels[keep]
    if len(pixels) < 4:
        return list(fallback)
    return np.median(pixels, axis=0).round().clip(0, 255).astype(int).tolist()


def saturation(pixels: np.ndarray) -> np.ndarray:
    maximum = pixels.max(axis=1)
    minimum = pixels.min(axis=1)
    return (maximum - minimum) / np.maximum(maximum, 1.0)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("extract_aina_palette.py APPROVED_FRONT OUTPUT_JSON")
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to read {source}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    landmarks = detect(rgb)

    cheek_indices = [116, 123, 147, 213, 345, 352, 376, 433]
    cheek_samples = [disk_pixels(rgb, tuple(landmarks[index, :2]), 0.018) for index in cheek_indices]
    skin = median_rgb(
        cheek_samples,
        (205, 158, 150),
        lambda px: (px.mean(axis=1) > 55) & (px.mean(axis=1) < 244) & (saturation(px) < 0.58) & (px[:, 0] >= px[:, 2] * 0.90),
    )

    lip_centers = [tuple(landmarks[index, :2]) for index in (13, 14, 61, 291, 0, 17)]
    lip = median_rgb(
        [disk_pixels(rgb, center, 0.010) for center in lip_centers],
        (145, 66, 76),
        lambda px: (px[:, 0] > px[:, 1] * 1.04) & (px[:, 0] > px[:, 2] * 1.01) & (px.mean(axis=1) > 35) & (px.mean(axis=1) < 225),
    )

    eye_centers = [tuple(landmarks[LEFT_EYE, :2].mean(axis=0)), tuple(landmarks[RIGHT_EYE, :2].mean(axis=0))]
    iris = median_rgb(
        [disk_pixels(rgb, center, 0.010) for center in eye_centers],
        (45, 91, 112),
        lambda px: (px.mean(axis=1) > 16) & (px.mean(axis=1) < 145),
    )

    brow_centers = [tuple(landmarks[[46, 53, 52, 65, 55, 70, 63, 105, 66, 107], :2].mean(axis=0)), tuple(landmarks[[276, 283, 282, 295, 285, 300, 293, 334, 296, 336], :2].mean(axis=0))]
    brow = median_rgb(
        [disk_pixels(rgb, center, 0.018) for center in brow_centers],
        (52, 48, 61),
        lambda px: (px.mean(axis=1) > 12) & (px.mean(axis=1) < 125),
    )

    height, width = rgb.shape[:2]
    face_left = int(max(0, min(landmarks[234, 0], landmarks[454, 0]) * width))
    face_right = int(min(width, max(landmarks[234, 0], landmarks[454, 0]) * width))
    face_top = int(max(0, landmarks[10, 1] * height))
    face_width = max(face_right - face_left, 1)
    hair_region = rgb[max(0, face_top - int(face_width * 0.55)):min(height, face_top + int(face_width * 0.08)), max(0, face_left - int(face_width * 0.15)):min(width, face_right + int(face_width * 0.15))]
    hair = median_rgb(
        [hair_region],
        (165, 178, 205),
        lambda px: (px.mean(axis=1) > 65) & (px.mean(axis=1) < 246) & (saturation(px) < 0.42),
    )

    eye_white = [232, 230, 228]
    result = {
        "source": str(source),
        "skin_rgb": skin,
        "lip_rgb": lip,
        "iris_rgb": iris,
        "brow_rgb": brow,
        "hair_rgb": hair,
        "eye_white_rgb": eye_white,
        "uniform_rgb": [190, 205, 232],
        "core_rgb": [35, 144, 255],
        "sampling": {"cheek_landmarks": cheek_indices, "eye_centers": eye_centers, "face_bbox": [face_left, face_top, face_right, int(landmarks[152, 1] * height)]},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
