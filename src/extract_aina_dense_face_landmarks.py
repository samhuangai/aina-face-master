#!/usr/bin/env python3
"""Extract dense approved/model facial landmarks for the final AINA Mesh pass.

This script analyses only the already-approved AINA reference images and real
Blender renders.  It does not generate replacement character art.  MediaPipe
FaceMesh is run on several deterministic image scales/contrast variants so the
stylised approved portrait and the clay/beauty model renders are handled more
reliably.  Front and three-quarter detections are mandatory; side is retained
when available and otherwise the existing sparse side gate remains authoritative.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


VIEWS = ("front", "three_quarter", "side")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-front", type=Path, required=True)
    parser.add_argument("--approved-q3", type=Path, required=True)
    parser.add_argument("--approved-side", type=Path, required=True)
    parser.add_argument("--model-front", type=Path, required=True)
    parser.add_argument("--model-q3", type=Path, required=True)
    parser.add_argument("--model-side", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def candidates(image: Image.Image):
    """Yield deterministic detection variants and original-coordinate scales."""
    rgb = image.convert("RGB")
    for scale in (1.0, 1.75, 2.5):
        size = (max(64, int(round(rgb.width * scale))), max(64, int(round(rgb.height * scale))))
        resized = rgb.resize(size, Image.Resampling.LANCZOS)
        variants = [
            resized,
            ImageEnhance.Contrast(resized).enhance(1.18),
            ImageEnhance.Sharpness(ImageEnhance.Contrast(resized).enhance(1.10)).enhance(1.20),
        ]
        # A light autocontrast copy often helps the neutral clay renders without
        # changing landmark geometry.
        variants.append(ImageOps.autocontrast(resized, cutoff=0.5))
        for variant in variants:
            yield np.asarray(variant, dtype=np.uint8), scale


def face_area(landmarks) -> float:
    x = np.asarray([point.x for point in landmarks], dtype=np.float64)
    y = np.asarray([point.y for point in landmarks], dtype=np.float64)
    return float(max(x.max() - x.min(), 0.0) * max(y.max() - y.min(), 0.0))


def detect(path: Path, detector) -> dict | None:
    image = Image.open(path).convert("RGB")
    best = None
    for array, scale in candidates(image):
        result = detector.process(array)
        if not result.multi_face_landmarks:
            continue
        face = max(result.multi_face_landmarks, key=lambda item: face_area(item.landmark))
        points = np.asarray(
            [[point.x * array.shape[1] / scale, point.y * array.shape[0] / scale, point.z] for point in face.landmark],
            dtype=np.float64,
        )
        # Prefer complete refined detections and then the largest face box.
        area = face_area(face.landmark)
        score = (len(points), area)
        if best is None or score > best[0]:
            best = (score, points)
    if best is None:
        return None
    points = best[1]
    return {
        "image": str(path),
        "image_size": [image.width, image.height],
        "landmark_count": int(len(points)),
        "landmarks_xyz": points.tolist(),
        "landmarks_xy": points[:, :2].tolist(),
    }


def draw_overlay(path: Path, item: dict | None, output: Path, title: str) -> None:
    image = Image.open(path).convert("RGB")
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    if item:
        points = np.asarray(item["landmarks_xy"], dtype=np.float64)
        # Dense points remain readable at small sizes when every third point is
        # shown; feature/contour points are still present in the JSON in full.
        radius = max(1, int(round(max(image.size) / 420)))
        for index, (x, y) in enumerate(points):
            if index % 3:
                continue
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 255, 255), width=1)
    draw.rectangle((0, 0, image.width, max(22, image.height // 15)), fill=(0, 0, 0))
    draw.text((6, 4), title, fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def build_sheet(paths: list[tuple[Path, str]], output: Path) -> None:
    panels = []
    for path, label in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((430, 430))
        panel = Image.new("RGB", (450, 475), "white")
        panel.paste(image, ((450 - image.width) // 2, 5))
        ImageDraw.Draw(panel).text((8, 448), label, fill="black")
        panels.append(panel)
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (450 * columns, 475 * rows), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * 450, (index // columns) * 475))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = {
        "approved": {
            "front": args.approved_front,
            "three_quarter": args.approved_q3,
            "side": args.approved_side,
        },
        "model": {
            "front": args.model_front,
            "three_quarter": args.model_q3,
            "side": args.model_side,
        },
    }

    face_mesh = mp.solutions.face_mesh
    payload = {"detector": "MediaPipe FaceMesh refine_landmarks", "approved": {}, "model": {}}
    overlay_paths: list[tuple[Path, str]] = []
    with face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.30,
    ) as detector:
        for family, items in inputs.items():
            for view, path in items.items():
                item = detect(path, detector)
                payload[family][view] = item
                overlay = args.out / f"AINA_DENSE_{family.upper()}_{view.upper()}_OVERLAY.png"
                draw_overlay(path, item, overlay, f"{family.upper()} {view.upper()} — {'DETECTED' if item else 'NOT DETECTED'}")
                overlay_paths.append((overlay, f"{family.title()} {view.replace('_', ' ').title()}"))

    mandatory = [
        (family, view)
        for family in ("approved", "model")
        for view in ("front", "three_quarter")
        if payload[family][view] is None
    ]
    if mandatory:
        raise SystemExit(f"Mandatory dense detections failed: {mandatory}")

    payload["detected"] = {
        family: {view: payload[family][view] is not None for view in VIEWS}
        for family in ("approved", "model")
    }
    payload["dense_front_q3_ready"] = not mandatory
    payload["side_dense_ready"] = payload["approved"]["side"] is not None and payload["model"]["side"] is not None
    payload["note"] = "Side dense landmarks are optional; the approved sparse side profile gate remains active when absent."

    json_path = args.out / "AINA_DENSE_FACE_LANDMARKS.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    build_sheet(overlay_paths, args.out / "AINA_DENSE_LANDMARK_OVERLAYS.png")
    print(json.dumps({
        "dense_front_q3_ready": payload["dense_front_q3_ready"],
        "side_dense_ready": payload["side_dense_ready"],
        "detected": payload["detected"],
        "json": str(json_path),
    }, indent=2))


if __name__ == "__main__":
    main()
