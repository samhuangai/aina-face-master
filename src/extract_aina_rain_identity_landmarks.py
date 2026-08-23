#!/usr/bin/env python3
"""Extract approved AINA and Rain source landmarks for real-Mesh fitting.

The approved front uses the repository's manually approved 68-point target.
Three-quarter and side detection are best-effort; unavailable side landmarks do
not block the reconstruction because the side view is retained as a visual gate.
No replacement effect art is generated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import face_alignment
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-front", type=Path, required=True)
    parser.add_argument("--approved-q3", type=Path, required=True)
    parser.add_argument("--approved-side", type=Path, required=True)
    parser.add_argument("--model-front", type=Path, required=True)
    parser.add_argument("--model-q3", type=Path, required=True)
    parser.add_argument("--model-side", type=Path, required=True)
    parser.add_argument("--front-target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def detect(detector, path: Path):
    image = Image.open(path).convert("RGB")
    prediction = detector.get_landmarks_from_image(np.asarray(image))
    if not prediction:
        return image, None
    centre = np.array([image.width * 0.5, image.height * 0.5])
    points = min(
        prediction,
        key=lambda item: np.linalg.norm(np.asarray(item)[:, :2].mean(axis=0) - centre),
    )
    return image, np.asarray(points, dtype=np.float64)[:, :2]


def annotate(image: Image.Image, points: np.ndarray | None, path: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    if points is None:
        draw.rectangle((5, 5, canvas.width - 5, canvas.height - 5), outline=(255, 45, 45), width=4)
        draw.text((12, 12), "NO 68-POINT DETECTION", fill=(255, 45, 45))
    else:
        for index, (x, y) in enumerate(points):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 60, 60))
            if index % 5 == 0:
                draw.text((x + 3, y), str(index), fill=(30, 90, 255))
    canvas.save(path)


def item(image: Image.Image, points: np.ndarray):
    return {
        "image_size": [image.width, image.height],
        "landmarks_xy": points.tolist(),
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device="cpu",
        face_detector="sfd",
    )

    approved_paths = {
        "front": args.approved_front,
        "three_quarter": args.approved_q3,
        "side": args.approved_side,
    }
    model_paths = {
        "front": args.model_front,
        "three_quarter": args.model_q3,
        "side": args.model_side,
    }

    approved = {}
    model = {}
    detection = {"approved": {}, "model": {}}
    for group_name, mapping, destination in (
        ("approved", approved_paths, approved),
        ("model", model_paths, model),
    ):
        for view, path in mapping.items():
            image, points = detect(detector, path)
            detection[group_name][view] = points is not None
            annotate(image, points, args.out / f"AINA_RAIN_{group_name.upper()}_{view.upper()}_68.png")
            if points is not None:
                destination[view] = item(image, points)

    exact_front = json.loads(args.front_target.read_text(encoding="utf-8"))
    approved["front"] = exact_front
    detection["approved"]["front"] = True

    available = [
        view for view in ("front", "three_quarter", "side")
        if view in approved and view in model
    ]
    if "front" not in available:
        raise RuntimeError("Rain source front face landmarks were not detected")
    if len(available) < 2:
        raise RuntimeError(f"At least front plus one additional view are required; got {available}")

    output = {
        "approved": approved,
        "model": model,
        "available_views": available,
        "detection": detection,
        "landmark_order": "standard 68",
        "front_target_source": str(args.front_target),
        "replacement_effect_art_generated": False,
    }
    (args.out / "AINA_RAIN_IDENTITY_LANDMARKS.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
