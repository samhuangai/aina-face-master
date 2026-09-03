#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_aina_v12 as base


class OrientedLandmarkField(base.LandmarkField):
    orientation = "normal"

    def mesh_to_image(self, positions):
        uv = super().mesh_to_image(positions)
        if self.orientation == "flipped":
            uv[:, 0] = self.model_left_x + self.model_right_x - uv[:, 0]
        elif self.orientation != "normal":
            raise ValueError(f"Unsupported AINA V12 X orientation: {self.orientation}")
        return uv


def main() -> None:
    if len(sys.argv) != 9:
        raise SystemExit("build_aina_v12_oriented.py FORMAL.vrm BLENDER.glb MAPPING.json OUTPUT_DIR LABEL STEM STRENGTH ORIENTATION")
    formal_source = Path(sys.argv[1])
    safe_source = Path(sys.argv[2])
    mapping_path = Path(sys.argv[3])
    output = Path(sys.argv[4])
    label = sys.argv[5]
    stem = sys.argv[6]
    strength = float(sys.argv[7])
    orientation = sys.argv[8].lower()
    output.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(mapping_path.read_text())
    OrientedLandmarkField.orientation = orientation
    base.LandmarkField = OrientedLandmarkField
    formal_path = output / f"{stem}.vrm"
    safe_path = output / f"{stem}_BLENDER.glb"
    formal = base.process(formal_source, formal_path, mapping, strength, label)
    safe = base.process(safe_source, safe_path, mapping, strength, label)
    report = {
        "version": label,
        "method": "full 468-landmark Gaussian-IDW deformation field with camera-axis orientation search",
        "mapping": str(mapping_path),
        "strength": strength,
        "orientation": orientation,
        "formal": formal,
        "blender": safe,
        "preserved": {"vrm_1_0": True, "humanoid_bones": 54, "face_morphs": 57, "expression_presets": 14},
        "identity_lock": False,
        "visual_identity_lock": False,
        "manual_three_view_review_required": True,
    }
    (output / f"{stem}_BUILD_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
