#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from makehuman_target_obj import apply_target, bounds, read_obj, write_filtered_obj

KEEP_GROUPS = {
    "body",
    "helper-l-eye",
    "helper-r-eye",
    "helper-upper-teeth",
    "helper-lower-teeth",
    "helper-tongue",
    "helper-l-eyelashes-1",
    "helper-r-eyelashes-1",
    "helper-l-eyelashes-2",
    "helper-r-eyelashes-2",
}


def symmetric(category: str, stem: str, weight: float) -> list[tuple[str, float]]:
    return [
        (f"{category}/l-{stem}.target", weight),
        (f"{category}/r-{stem}.target", weight),
    ]


def make_recipe(
    *,
    head_oval: float = 0.0,
    head_v: float = 0.0,
    head_narrow: float = 0.0,
    head_depth: float = 0.0,
    head_height: float = 0.0,
    forehead_height: float = 0.0,
    temple_softness: float = 0.0,
    eye_scale: float = 0.0,
    eye_h1: float = 0.0,
    eye_h2: float = 0.0,
    eye_h3: float = 0.0,
    eye_outer_up: float = 0.0,
    eye_inner_up: float = 0.0,
    eye_fold_up: float = 0.0,
    eye_bag_decr: float = 0.0,
    epicanthus: float = 0.0,
    cheek_volume: float = 0.0,
    cheek_inner: float = 0.0,
    cheek_bones: float = 0.0,
    cheek_up: float = 0.0,
    chin_width: float = 0.0,
    chin_bones: float = 0.0,
    chin_height: float = 0.0,
    chin_prominence: float = 0.0,
    nose_narrow: float = 0.0,
    nose_width1: float = 0.0,
    nose_width2: float = 0.0,
    nose_width3: float = 0.0,
    nose_alar: float = 0.0,
    nose_nostril: float = 0.0,
    nose_tip_width: float = 0.0,
    nose_point: float = 0.0,
    nose_tip_up: float = 0.0,
    nose_short: float = 0.0,
    mouth_width: float = 0.0,
    mouth_height: float = 0.0,
    mouth_up: float = 0.0,
    mouth_back: float = 0.0,
    upper_lip_volume: float = 0.0,
    lower_lip_volume: float = 0.0,
    upper_lip_height: float = 0.0,
    lower_lip_height: float = 0.0,
    cupids_bow: float = 0.0,
    mouth_corner_up: float = 0.0,
    ear_scale: float = 0.0,
    ear_wing: float = 0.0,
    neck_narrow: float = 0.0,
    neck_depth: float = 0.0,
) -> list[tuple[str, float]]:
    targets: list[tuple[str, float]] = []

    def add(path: str, weight: float) -> None:
        if abs(weight) > 1e-9:
            targets.append((path, float(weight)))

    add("head/head-oval.target", head_oval)
    add("head/head-invertedtriangular.target", head_v)
    add("head/head-scale-horiz-decr.target", head_narrow)
    add("head/head-scale-depth-decr.target", head_depth)
    add("head/head-scale-vert-incr.target", head_height)
    add("forehead/forehead-scale-vert-incr.target", forehead_height)
    add("forehead/forehead-temple-incr.target", temple_softness)

    for path, weight in symmetric("eyes", "eye-scale-incr", eye_scale):
        add(path, weight)
    for path, weight in symmetric("eyes", "eye-height1-incr", eye_h1):
        add(path, weight)
    for path, weight in symmetric("eyes", "eye-height2-incr", eye_h2):
        add(path, weight)
    for path, weight in symmetric("eyes", "eye-height3-incr", eye_h3):
        add(path, weight)
    for path, weight in symmetric("eyes", "eye-corner1-up", eye_outer_up):
        add(path, weight)
    for path, weight in symmetric("eyes", "eye-corner2-up", eye_inner_up):
        add(path, weight)
    for path, weight in symmetric("eyes", "eye-eyefold-angle-up", eye_fold_up):
        add(path, weight)
    for path, weight in symmetric("eyes", "eye-bag-decr", eye_bag_decr):
        add(path, weight)
    for path, weight in symmetric("eyes", "eye-epicanthus-in", epicanthus):
        add(path, weight)

    for path, weight in symmetric("cheek", "cheek-volume-incr", cheek_volume):
        add(path, weight)
    for path, weight in symmetric("cheek", "cheek-inner-incr", cheek_inner):
        add(path, weight)
    for path, weight in symmetric("cheek", "cheek-bones-incr", cheek_bones):
        add(path, weight)
    for path, weight in symmetric("cheek", "cheek-trans-up", cheek_up):
        add(path, weight)

    add("chin/chin-width-decr.target", chin_width)
    add("chin/chin-bones-decr.target", chin_bones)
    add("chin/chin-height-decr.target", chin_height)
    add("chin/chin-prominent-decr.target", chin_prominence)

    add("nose/nose-scale-horiz-decr.target", nose_narrow)
    add("nose/nose-width1-decr.target", nose_width1)
    add("nose/nose-width2-decr.target", nose_width2)
    add("nose/nose-width3-decr.target", nose_width3)
    add("nose/nose-flaring-decr.target", nose_alar)
    add("nose/nose-nostrils-width-decr.target", nose_nostril)
    add("nose/nose-point-width-decr.target", nose_tip_width)
    add("nose/nose-volume-incr.target", nose_point)
    add("nose/nose-point-up.target", nose_tip_up)
    add("nose/nose-scale-vert-decr.target", nose_short)

    add("mouth/mouth-scale-horiz-incr.target", mouth_width)
    add("mouth/mouth-scale-vert-incr.target", mouth_height)
    add("mouth/mouth-trans-up.target", mouth_up)
    add("mouth/mouth-trans-backward.target", mouth_back)
    add("mouth/mouth-upperlip-volume-incr.target", upper_lip_volume)
    add("mouth/mouth-lowerlip-volume-incr.target", lower_lip_volume)
    add("mouth/mouth-upperlip-height-incr.target", upper_lip_height)
    add("mouth/mouth-lowerlip-height-incr.target", lower_lip_height)
    add("mouth/mouth-cupidsbow-incr.target", cupids_bow)
    add("mouth/mouth-angles-up.target", mouth_corner_up)

    for path, weight in symmetric("ears", "ear-scale-decr", ear_scale):
        add(path, weight)
    for path, weight in symmetric("ears", "ear-wing-decr", ear_wing):
        add(path, weight)
    add("neck/neck-scale-horiz-decr.target", neck_narrow)
    add("neck/neck-scale-depth-decr.target", neck_depth)
    return targets


RECIPES: dict[str, list[tuple[str, float]]] = {
    "00_BASELINE": [],
    "01_FACE_SHAPE": make_recipe(
        head_oval=0.40,
        head_v=0.28,
        head_narrow=0.14,
        head_depth=0.08,
        forehead_height=0.08,
        temple_softness=0.12,
        cheek_volume=0.16,
        cheek_inner=0.08,
        cheek_bones=0.08,
        cheek_up=0.08,
        chin_width=0.24,
        chin_bones=0.20,
        chin_height=0.10,
        ear_scale=0.20,
        ear_wing=0.12,
        neck_narrow=0.15,
        neck_depth=0.08,
    ),
    "02_AINA_SOFT": make_recipe(
        head_oval=0.45,
        head_v=0.28,
        head_narrow=0.14,
        head_depth=0.08,
        forehead_height=0.08,
        temple_softness=0.12,
        eye_scale=0.28,
        eye_h1=0.10,
        eye_h2=0.14,
        eye_h3=0.10,
        eye_outer_up=0.08,
        eye_fold_up=0.08,
        eye_bag_decr=0.18,
        epicanthus=0.08,
        cheek_volume=0.18,
        cheek_inner=0.10,
        cheek_bones=0.08,
        cheek_up=0.08,
        chin_width=0.26,
        chin_bones=0.22,
        chin_height=0.12,
        chin_prominence=0.05,
        nose_narrow=0.28,
        nose_width1=0.18,
        nose_width2=0.26,
        nose_width3=0.24,
        nose_alar=0.24,
        nose_nostril=0.18,
        nose_tip_width=0.20,
        nose_point=0.16,
        nose_tip_up=0.10,
        nose_short=0.08,
        mouth_width=0.06,
        mouth_height=0.06,
        mouth_up=0.05,
        mouth_back=0.03,
        upper_lip_volume=0.16,
        lower_lip_volume=0.22,
        upper_lip_height=0.10,
        lower_lip_height=0.12,
        cupids_bow=0.18,
        mouth_corner_up=0.05,
        ear_scale=0.22,
        ear_wing=0.14,
        neck_narrow=0.16,
        neck_depth=0.10,
    ),
    "03_AINA_BALANCED": make_recipe(
        head_oval=0.55,
        head_v=0.40,
        head_narrow=0.18,
        head_depth=0.10,
        forehead_height=0.10,
        temple_softness=0.16,
        eye_scale=0.46,
        eye_h1=0.15,
        eye_h2=0.23,
        eye_h3=0.16,
        eye_outer_up=0.12,
        eye_inner_up=0.03,
        eye_fold_up=0.12,
        eye_bag_decr=0.28,
        epicanthus=0.12,
        cheek_volume=0.23,
        cheek_inner=0.13,
        cheek_bones=0.11,
        cheek_up=0.11,
        chin_width=0.36,
        chin_bones=0.30,
        chin_height=0.17,
        chin_prominence=0.08,
        nose_narrow=0.42,
        nose_width1=0.28,
        nose_width2=0.40,
        nose_width3=0.36,
        nose_alar=0.36,
        nose_nostril=0.28,
        nose_tip_width=0.32,
        nose_point=0.24,
        nose_tip_up=0.17,
        nose_short=0.12,
        mouth_width=0.10,
        mouth_height=0.09,
        mouth_up=0.08,
        mouth_back=0.04,
        upper_lip_volume=0.24,
        lower_lip_volume=0.34,
        upper_lip_height=0.14,
        lower_lip_height=0.18,
        cupids_bow=0.27,
        mouth_corner_up=0.08,
        ear_scale=0.28,
        ear_wing=0.18,
        neck_narrow=0.20,
        neck_depth=0.12,
    ),
    "04_AINA_STRONG": make_recipe(
        head_oval=0.65,
        head_v=0.55,
        head_narrow=0.22,
        head_depth=0.12,
        forehead_height=0.12,
        temple_softness=0.20,
        eye_scale=0.64,
        eye_h1=0.22,
        eye_h2=0.34,
        eye_h3=0.23,
        eye_outer_up=0.17,
        eye_inner_up=0.04,
        eye_fold_up=0.17,
        eye_bag_decr=0.38,
        epicanthus=0.16,
        cheek_volume=0.29,
        cheek_inner=0.16,
        cheek_bones=0.14,
        cheek_up=0.14,
        chin_width=0.47,
        chin_bones=0.39,
        chin_height=0.23,
        chin_prominence=0.10,
        nose_narrow=0.55,
        nose_width1=0.38,
        nose_width2=0.52,
        nose_width3=0.48,
        nose_alar=0.48,
        nose_nostril=0.38,
        nose_tip_width=0.43,
        nose_point=0.32,
        nose_tip_up=0.24,
        nose_short=0.18,
        mouth_width=0.13,
        mouth_height=0.13,
        mouth_up=0.12,
        mouth_back=0.05,
        upper_lip_volume=0.33,
        lower_lip_volume=0.46,
        upper_lip_height=0.20,
        lower_lip_height=0.25,
        cupids_bow=0.36,
        mouth_corner_up=0.11,
        ear_scale=0.34,
        ear_wing=0.23,
        neck_narrow=0.24,
        neck_depth=0.15,
    ),
    "05_AINA_DOLL": make_recipe(
        head_oval=0.75,
        head_v=0.68,
        head_narrow=0.25,
        head_depth=0.14,
        forehead_height=0.15,
        temple_softness=0.24,
        eye_scale=0.82,
        eye_h1=0.30,
        eye_h2=0.45,
        eye_h3=0.30,
        eye_outer_up=0.20,
        eye_inner_up=0.05,
        eye_fold_up=0.20,
        eye_bag_decr=0.48,
        epicanthus=0.18,
        cheek_volume=0.35,
        cheek_inner=0.20,
        cheek_bones=0.16,
        cheek_up=0.17,
        chin_width=0.58,
        chin_bones=0.48,
        chin_height=0.30,
        chin_prominence=0.12,
        nose_narrow=0.68,
        nose_width1=0.48,
        nose_width2=0.64,
        nose_width3=0.58,
        nose_alar=0.58,
        nose_nostril=0.48,
        nose_tip_width=0.54,
        nose_point=0.38,
        nose_tip_up=0.30,
        nose_short=0.24,
        mouth_width=0.16,
        mouth_height=0.16,
        mouth_up=0.16,
        mouth_back=0.06,
        upper_lip_volume=0.42,
        lower_lip_volume=0.58,
        upper_lip_height=0.26,
        lower_lip_height=0.32,
        cupids_bow=0.44,
        mouth_corner_up=0.14,
        ear_scale=0.40,
        ear_wing=0.28,
        neck_narrow=0.28,
        neck_depth=0.18,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "base": str(args.base),
        "target_root": str(args.target_root),
        "baseline_macro_target": "macrodetails/asian-female-young.target",
        "variants": {},
    }

    for label, recipe in RECIPES.items():
        lines, vertices = read_obj(args.base)
        original = [row[:] for row in vertices]
        full_recipe = [("macrodetails/asian-female-young.target", 1.0), *recipe]
        applied: list[dict[str, object]] = []
        for relative_path, weight in full_recipe:
            target = args.target_root / relative_path
            if not target.is_file():
                raise FileNotFoundError(f"Missing MakeHuman target: {target}")
            changed = apply_target(vertices, target, weight)
            applied.append(
                {
                    "target": relative_path,
                    "weight": weight,
                    "changed_vertices": changed,
                }
            )

        obj_path = args.out / f"{label}.obj"
        stats = write_filtered_obj(lines, vertices, obj_path, KEEP_GROUPS)
        used_indexes = stats.pop("used_vertex_indexes")
        selected_vertices = [vertices[index] for index in used_indexes]
        max_delta = max(
            (
                (vertices[index][0] - original[index][0]) ** 2
                + (vertices[index][1] - original[index][1]) ** 2
                + (vertices[index][2] - original[index][2]) ** 2
            )
            ** 0.5
            for index in range(len(vertices))
        )
        report = {
            "label": label,
            "object": obj_path.name,
            "recipe": applied,
            "output_bounds": bounds(selected_vertices),
            "max_vertex_delta": max_delta,
            **stats,
        }
        (args.out / f"{label}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        manifest["variants"][label] = report
        print(f"built {label}: {stats['output_vertex_count']} vertices, {stats['output_face_count']} faces")

    (args.out / "AINA_VARIANT_RECIPES.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"variant_count": len(RECIPES), "labels": list(RECIPES)}, indent=2))


if __name__ == "__main__":
    main()
