#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finalize_aina_v10_fixed as fixed


def parse_args() -> dict[str, str]:
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(raw) % 2:
        raise ValueError(f"Expected key/value arguments, got {raw}")
    return {raw[index].lstrip("-"): raw[index + 1] for index in range(0, len(raw), 2)}


def rename_outputs(out: Path, version: str) -> None:
    version = version.upper()
    mapping = {
        "AINA_MASTER_V10.blend": f"AINA_MASTER_{version}.blend",
        "AINA_EXPORT_V10.glb": f"AINA_EXPORT_{version}.glb",
        "AINA_EXPORT_V10.fbx": f"AINA_EXPORT_{version}.fbx",
        "QA/AINA_V10_BLENDER_QA.json": f"QA/AINA_{version}_BLENDER_QA.json",
    }
    for source_name, destination_name in mapping.items():
        source = out / source_name
        destination = out / destination_name
        if source.is_file():
            source.replace(destination)
    preview = out / "Preview"
    for source in list(preview.glob("AINA_V10_*.png")):
        source.replace(preview / source.name.replace("AINA_V10_", f"AINA_{version}_", 1))


def patch_report(out: Path, version: str) -> None:
    version = version.upper()
    report_path = out / "QA" / f"AINA_{version}_BLENDER_QA.json"
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text())
    report["version"] = version
    report["identity_lock"] = False
    report["visual_identity_lock"] = False
    report["manual_review_required"] = True
    report["files"] = {
        "blend": (out / f"AINA_MASTER_{version}.blend").stat().st_size,
        "glb": (out / f"AINA_EXPORT_{version}.glb").stat().st_size,
        "fbx": (out / f"AINA_EXPORT_{version}.fbx").stat().st_size,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    p = parse_args()
    out = Path(p["out"]).resolve()
    version = p["version"].upper()
    fixed.base.main()
    rename_outputs(out, version)
    patch_report(out, version)
    required = [
        out / f"AINA_MASTER_{version}.blend",
        out / f"AINA_EXPORT_{version}.glb",
        out / f"AINA_EXPORT_{version}.fbx",
        out / "QA" / f"AINA_{version}_BLENDER_QA.json",
        out / "Preview" / f"AINA_{version}_FRONT.png",
        out / "Preview" / f"AINA_{version}_3Q.png",
        out / "Preview" / f"AINA_{version}_PROFILE.png",
        out / "Preview" / f"AINA_{version}_FULLBODY_FRONT.png",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"AINA {version} finalizer did not produce: {missing}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            p = parse_args()
            out = Path(p.get("out", ".")).resolve()
            version = p.get("version", "UNKNOWN").upper()
            (out / "QA").mkdir(parents=True, exist_ok=True)
            (out / "QA" / f"AINA_{version}_FINALIZER_ERROR.log").write_text(error, encoding="utf-8")
        except Exception:
            pass
        raise
