#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_aina_face as base


def main() -> None:
    if len(sys.argv) not in (4, 5):
        raise SystemExit("measure_aina_face_fixed.py REFERENCE_FRONT MODEL_FRONT OUTPUT_JSON [DAMPING]")
    reference_path = Path(sys.argv[1])
    model_path = Path(sys.argv[2])
    output = Path(sys.argv[3])
    damping = float(sys.argv[4]) if len(sys.argv) == 5 else 1.0
    reference_points = base.detect(reference_path)
    model_points = base.detect(model_path)
    reference = base.metrics(reference_points)
    model = base.metrics(model_points)
    parameters = {
        "face_width_scale": base.clamp(model["face_aspect"] / reference["face_aspect"], 0.88, 1.12),
        "eye_width_scale": base.ratio(reference, model, "eye_width", (0.78, 1.15)),
        "eye_height_scale": base.ratio(reference, model, "eye_height", (0.74, 1.18)),
        "eye_spacing_scale": base.ratio(reference, model, "eye_spacing", (0.90, 1.10)),
        "brow_eye_gap_scale": base.ratio(reference, model, "brow_eye_gap", (0.84, 1.16)),
        "nose_length_scale": base.ratio(reference, model, "nose_length", (0.86, 1.16)),
        "nose_width_scale": base.ratio(reference, model, "nose_width", (0.88, 1.12)),
        "mouth_width_scale": base.ratio(reference, model, "mouth_width", (0.86, 1.14)),
        "nose_mouth_scale": base.ratio(reference, model, "nose_mouth", (0.86, 1.14)),
        "lower_third_scale": base.ratio(reference, model, "lower_third", (0.88, 1.14)),
        "upper_face_scale": base.ratio(reference, model, "upper_face", (0.88, 1.12)),
    }
    parameters = {key: 1.0 + (value - 1.0) * damping for key, value in parameters.items()}
    report = {
        "reference_image": str(reference_path),
        "model_image": str(model_path),
        "reference_metrics": reference,
        "model_metrics": model,
        "ratios": {key: reference[key] / max(model[key], 1e-8) for key in reference},
        "parameters": parameters,
        "damping": damping,
        "procrustes_rms": base.procrustes_rms(reference_points, model_points),
        "detected_reference_landmarks": len(reference_points),
        "detected_model_landmarks": len(model_points),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
