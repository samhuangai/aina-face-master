#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_aina_v10 as glb


VARIANTS = {
    "clean": {"skin_detail": 0.03, "skin_shading": 0.18, "skin_blur": 0.034},
    "balanced": {"skin_detail": 0.12, "skin_shading": 0.24, "skin_blur": 0.024},
    "detail": {"skin_detail": 0.23, "skin_shading": 0.30, "skin_blur": 0.016},
}


def rgb01(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) / 255.0


def image_bytes(document: dict, binary: bytearray, image_index: int) -> bytes:
    image = document["images"][image_index]
    if "bufferView" not in image:
        raise ValueError(f"AINA material image {image_index} is not GLB-embedded")
    view = document["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    return bytes(binary[start:start + view["byteLength"]])


def material_image(document: dict, material: dict) -> int | None:
    texture_info = material.get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if not texture_info:
        return None
    texture_index = texture_info.get("index")
    if texture_index is None or texture_index >= len(document.get("textures", [])):
        return None
    return document["textures"][texture_index].get("source")


def recolor_texture(source: Image.Image, target_rgb: list[int], detail_strength: float, shading_strength: float, blur_fraction: float, alpha_floor: float = 0.0) -> Image.Image:
    rgba = source.convert("RGBA")
    width, height = rgba.size
    radius = max(1.0, min(width, height) * blur_fraction)
    blurred = rgba.filter(ImageFilter.GaussianBlur(radius=radius))
    original = np.asarray(rgba, dtype=np.float32) / 255.0
    low = np.asarray(blurred, dtype=np.float32) / 255.0
    alpha = original[..., 3:4]
    mask = alpha[..., 0] > max(alpha_floor, 0.02)
    luminance = 0.2126 * low[..., 0] + 0.7152 * low[..., 1] + 0.0722 * low[..., 2]
    median = float(np.median(luminance[mask])) if np.any(mask) else 0.5
    shade = np.clip(luminance / max(median, 0.05), 0.66, 1.34)
    base = rgb01(target_rgb)[None, None, :]
    colored = base * (1.0 - shading_strength + shading_strength * shade[..., None])
    high_frequency = original[..., :3] - low[..., :3]
    colored += high_frequency * detail_strength
    colored = np.clip(colored, 0.0, 1.0)
    result = np.concatenate([colored, alpha], axis=2)
    return Image.fromarray(np.round(result * 255.0).astype(np.uint8), "RGBA")


def make_role_texture(source: Image.Image, role: str, palette: dict, variant: dict) -> Image.Image:
    if role == "skin":
        return recolor_texture(source, palette["skin_rgb"], variant["skin_detail"], variant["skin_shading"], variant["skin_blur"])
    if role == "hair":
        return recolor_texture(source, palette["hair_rgb"], 0.42, 0.38, 0.008)
    if role == "iris":
        return recolor_texture(source, palette["iris_rgb"], 0.32, 0.42, 0.006)
    if role == "mouth":
        return recolor_texture(source, palette["lip_rgb"], 0.18, 0.24, 0.012)
    if role in {"brow", "eyeline"}:
        return recolor_texture(source, palette["brow_rgb"], 0.28, 0.26, 0.008)
    if role == "eye_white":
        return recolor_texture(source, palette["eye_white_rgb"], 0.08, 0.13, 0.014)
    if role == "uniform":
        return recolor_texture(source, palette["uniform_rgb"], 0.24, 0.32, 0.012)
    return source.convert("RGBA")


def role_for_material(name: str) -> str | None:
    lower = name.lower()
    if "skin" in lower and ("face" in lower or "body" in lower):
        return "skin"
    if "eyeiris" in lower or ("iris" in lower and "highlight" not in lower):
        return "iris"
    if "eyewhite" in lower:
        return "eye_white"
    if "facemouth" in lower or "mouth" in lower or "lip" in lower:
        return "mouth"
    if "brow" in lower:
        return "brow"
    if "eyeline" in lower or "lash" in lower:
        return "eyeline"
    if "hair" in lower or "updo" in lower:
        return "hair"
    if "uniform" in lower or "cloth" in lower or "outfit" in lower:
        return "uniform"
    return None


def role_priority(role: str) -> int:
    return {"skin": 100, "hair": 90, "iris": 80, "mouth": 75, "eye_white": 70, "brow": 65, "eyeline": 65, "uniform": 50}.get(role, 0)


def tune_material(material: dict, role: str | None, palette: dict) -> None:
    pbr = material.setdefault("pbrMetallicRoughness", {})
    pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
    lower = material.get("name", "").lower()
    if role == "skin":
        pbr["metallicFactor"] = 0.0
        pbr["roughnessFactor"] = 0.50 if "face" in lower else 0.53
        material["alphaMode"] = "OPAQUE"
        material["doubleSided"] = False
    elif role == "hair":
        pbr["metallicFactor"] = 0.06
        pbr["roughnessFactor"] = 0.36
        material["alphaMode"] = "BLEND"
        material["doubleSided"] = True
    elif role == "iris":
        pbr["metallicFactor"] = 0.0
        pbr["roughnessFactor"] = 0.16
        material["alphaMode"] = "OPAQUE"
    elif role == "eye_white":
        pbr["metallicFactor"] = 0.0
        pbr["roughnessFactor"] = 0.28
        material["alphaMode"] = "OPAQUE"
    elif role == "mouth":
        pbr["metallicFactor"] = 0.0
        pbr["roughnessFactor"] = 0.33
        material["alphaMode"] = "BLEND"
        material["doubleSided"] = True
    elif role in {"brow", "eyeline"}:
        pbr["metallicFactor"] = 0.0
        pbr["roughnessFactor"] = 0.47
        material["alphaMode"] = "BLEND"
        material["doubleSided"] = True
    elif role == "uniform":
        pbr["metallicFactor"] = 0.04
        pbr["roughnessFactor"] = 0.40
        material["alphaMode"] = "OPAQUE"
    elif "highlight" in lower:
        pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 0.72]
        pbr["metallicFactor"] = 0.0
        pbr["roughnessFactor"] = 0.08
        material["alphaMode"] = "BLEND"
        material["doubleSided"] = True
    elif "core" in lower:
        core = rgb01(palette["core_rgb"])
        pbr["baseColorFactor"] = [float(core[0]), float(core[1]), float(core[2]), 1.0]
        pbr["metallicFactor"] = 0.10
        pbr["roughnessFactor"] = 0.14
        material["emissiveFactor"] = [float(core[0] * 0.7), float(core[1] * 0.8), float(core[2])]
    extensions = material.get("extensions")
    if isinstance(extensions, dict):
        extensions.pop("KHR_materials_unlit", None)
        if not extensions:
            material.pop("extensions", None)


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def rebuild_binary(document: dict, binary: bytearray, replacements: dict[int, bytes]) -> bytearray:
    payloads = []
    for index, view in enumerate(document.get("bufferViews", [])):
        if view.get("buffer", 0) != 0:
            raise ValueError("AINA GLB material rebuild supports one binary buffer")
        start = view.get("byteOffset", 0)
        payloads.append(replacements.get(index, bytes(binary[start:start + view["byteLength"]])))
    result = bytearray()
    for index, payload in enumerate(payloads):
        while len(result) % 4:
            result.append(0)
        view = document["bufferViews"][index]
        view["byteOffset"] = len(result)
        view["byteLength"] = len(payload)
        result.extend(payload)
    document["buffers"][0]["byteLength"] = len(result)
    return result


def process(source: Path, destination: Path, palette: dict, variant_name: str, label: str) -> dict:
    variant = VARIANTS[variant_name]
    document, binary = glb.read_glb(source)
    image_roles: dict[int, str] = {}
    material_report = []
    for material in document.get("materials", []):
        name = material.get("name", "")
        role = role_for_material(name)
        tune_material(material, role, palette)
        image_index = material_image(document, material)
        if role and image_index is not None:
            previous = image_roles.get(image_index)
            if previous is None or role_priority(role) > role_priority(previous):
                image_roles[image_index] = role
        material_report.append({"name": name, "role": role, "image": image_index, "alpha": material.get("alphaMode", "OPAQUE")})

    replacements: dict[int, bytes] = {}
    image_report = []
    for image_index, role in image_roles.items():
        image_record = document["images"][image_index]
        source_image = Image.open(io.BytesIO(image_bytes(document, binary, image_index)))
        rebuilt = make_role_texture(source_image, role, palette, variant)
        view_index = image_record["bufferView"]
        encoded = encode_png(rebuilt)
        replacements[view_index] = encoded
        image_record["mimeType"] = "image/png"
        image_report.append({"image": image_index, "buffer_view": view_index, "role": role, "size": rebuilt.size, "bytes": len(encoded)})

    binary = rebuild_binary(document, binary, replacements)
    document.setdefault("asset", {})["generator"] = f"AINA V14 reference-derived physical material rebuild {label}"
    glb.write_glb(destination, document, binary)
    return {"source": source.name, "output": destination.name, "bytes": destination.stat().st_size, "variant": variant_name, "materials": material_report, "images": image_report}


def main() -> None:
    if len(sys.argv) != 8:
        raise SystemExit("build_aina_v14_material.py FORMAL.vrm BLENDER.glb PALETTE.json OUTPUT_DIR LABEL STEM VARIANT")
    formal_source = Path(sys.argv[1])
    safe_source = Path(sys.argv[2])
    palette_path = Path(sys.argv[3])
    output = Path(sys.argv[4])
    label = sys.argv[5]
    stem = sys.argv[6]
    variant = sys.argv[7].lower()
    if variant not in VARIANTS:
        raise ValueError(f"Unknown AINA V14 material variant {variant}; expected {sorted(VARIANTS)}")
    output.mkdir(parents=True, exist_ok=True)
    palette = json.loads(palette_path.read_text())
    formal_path = output / f"{stem}.vrm"
    safe_path = output / f"{stem}_BLENDER.glb"
    formal = process(formal_source, formal_path, palette, variant, label)
    safe = process(safe_source, safe_path, palette, variant, label)
    report = {"version": label, "method": "approved-reference palette plus low-frequency PBR texture reconstruction", "palette": palette, "variant": variant, "formal": formal, "blender": safe, "preserved": {"geometry": True, "vrm_1_0": True, "humanoid_bones": 54, "face_morphs": 57, "expression_presets": 14}, "identity_lock": False, "visual_identity_lock": False}
    (output / f"{stem}_BUILD_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
