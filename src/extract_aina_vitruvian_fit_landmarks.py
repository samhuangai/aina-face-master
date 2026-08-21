#!/usr/bin/env python3
"""Extract approved and rendered 68-point landmarks for Vitruvian fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import face_alignment
import numpy as np
from PIL import Image, ImageDraw


def parse_args():
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


def detect(fa, path: Path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image)
    predictions = fa.get_landmarks_from_image(array)
    if not predictions:
        raise RuntimeError(f"No face landmarks detected in {path}")
    center = np.array([image.width * 0.5, image.height * 0.5])
    points = min(
        predictions,
        key=lambda item: np.linalg.norm(np.asarray(item)[:, :2].mean(axis=0) - center),
    )
    points = np.asarray(points, dtype=np.float64)[:, :2]
    return image, points


def annotate(image: Image.Image, points: np.ndarray, output: Path):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for index, (x, y) in enumerate(points):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 60, 60))
        if index % 5 == 0:
            draw.text((x + 3, y), str(index), fill=(30, 80, 255))
    canvas.save(output)


def item(image: Image.Image, points: np.ndarray):
    return {
        "image_size": [image.width, image.height],
        "landmarks_xy": points.tolist(),
    }


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fa = face_alignment.FaceAlignment(
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
    for name, path in approved_paths.items():
        image, points = detect(fa, path)
        approved[name] = item(image, points)
        annotate(image, points, args.out / f"AINA_APPROVED_{name.upper()}_68.png")
    # Use the previously approved front landmark set as the exact front target.
    front_exact = json.loads(args.front_target.read_text())
    approved["front"] = front_exact

    for name, path in model_paths.items():
        image, points = detect(fa, path)
        model[name] = item(image, points)
        annotate(image, points, args.out / f"AINA_VITRUVIAN_MODEL_{name.upper()}_68.png")

    output = {
        "approved": approved,
        "model": model,
        "landmark_order": "standard 68",
        "replacement_effect_art_generated": False,
    }
    (args.out / "AINA_VITRUVIAN_FIT_LANDMARKS.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
