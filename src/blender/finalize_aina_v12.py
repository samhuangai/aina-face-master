#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finalize_aina_v10_fixed as fixed


def parse_args():
    return fixed.base.parse_args()


def rename_outputs(out: Path) -> None:
    mapping = {
        "AINA_MASTER_V10.blend": "AINA_MASTER_V12.blend",
        "AINA_EXPORT_V10.glb": "AINA_EXPORT_V12.glb",
        "AINA_EXPORT_V10.fbx": "AINA_EXPORT_V12.fbx",
        "QA/AINA_V10_BLENDER_QA.json": "QA/AINA_V12_BLENDER_QA.json",
    }
    for source_name, destination_name in mapping.items():
        source = out / source_name
        destination = out / destination_name
        if source.is_file():
            source.replace(destination)
    preview = out / "Preview"
    for source in list(preview.glob("AINA_V10_*.png")):
        source.replace(preview / source.name.replace("AINA_V10_", "AINA_V12_", 1))


def patch_report(out: Path) -> None:
    report_path = out / "QA" / "AINA_V12_BLENDER_QA.json"
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text())
    report["version"] = "V12"
    report["identity_lock"] = False
    report["visual_identity_lock"] = False
    report["manual_review_required"] = True
    files = report.get("files", {})
    files["blend"] = (out / "AINA_MASTER_V12.blend").stat().st_size
    files["glb"] = (out / "AINA_EXPORT_V12.glb").stat().st_size
    files["fbx"] = (out / "AINA_EXPORT_V12.fbx").stat().st_size
    report["files"] = files
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    p = parse_args()
    out = Path(p["out"]).resolve()
    fixed.base.main()
    rename_outputs(out)
    patch_report(out)
    required = [
        out / "AINA_MASTER_V12.blend",
        out / "AINA_EXPORT_V12.glb",
        out / "AINA_EXPORT_V12.fbx",
        out / "QA" / "AINA_V12_BLENDER_QA.json",
        out / "Preview" / "AINA_V12_FRONT.png",
        out / "Preview" / "AINA_V12_3Q.png",
        out / "Preview" / "AINA_V12_PROFILE.png",
        out / "Preview" / "AINA_V12_FULLBODY_FRONT.png",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"AINA V12 finalizer did not produce: {missing}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            p = parse_args()
            out = Path(p.get("out", ".")).resolve()
            (out / "QA").mkdir(parents=True, exist_ok=True)
            (out / "QA" / "AINA_V12_FINALIZER_ERROR.log").write_text(error, encoding="utf-8")
        except Exception:
            pass
        raise
