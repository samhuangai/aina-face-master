#!/usr/bin/env python3
"""Strict visual audit for the actual final AINA model renders.

The audit compares the approved/reference portrait halves with the real Blender
model halves using three independent signals:

1. MediaPipe dense landmark Procrustes geometry.
2. VGGFace2 FaceNet and DINOv2 visual embeddings on landmark-aligned crops.
3. A small multimodal language model asked to judge identity anatomy, hair and
   full-character integration from the supplied QA sheets.

Expression renders are also checked for face detectability, identity stability,
blink closure and mouth articulation.  The result is advisory but deliberately
strict: a final production lock is allowed only when all major gates agree.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import traceback
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from facenet_pytorch import InceptionResnetV1


IDENTITY_VIEWS = ("front", "three_quarter", "side")
EXPRESSION_NAMES = ("neutral", "happy", "sad", "angry", "surprised", "blink", "aa", "ou")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-sheet", type=Path, required=True)
    parser.add_argument("--expression-sheet", type=Path, required=True)
    parser.add_argument("--full-sheet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vlm-model", default="Qwen/Qwen2-VL-2B-Instruct")
    return parser.parse_args()


def image_variants(image: Image.Image):
    rgb = image.convert("RGB")
    for scale in (1.0, 1.5, 2.0):
        size = (max(96, int(rgb.width * scale)), max(96, int(rgb.height * scale)))
        resized = rgb.resize(size, Image.Resampling.LANCZOS)
        yield resized, scale
        yield ImageEnhance.Contrast(resized).enhance(1.15), scale
        yield ImageOps.autocontrast(resized, cutoff=0.5), scale


def face_area(landmarks) -> float:
    x = np.asarray([point.x for point in landmarks], dtype=np.float64)
    y = np.asarray([point.y for point in landmarks], dtype=np.float64)
    return float((x.max() - x.min()) * (y.max() - y.min()))


def detect_face(image: Image.Image, detector):
    best = None
    for variant, scale in image_variants(image):
        array = np.asarray(variant, dtype=np.uint8)
        result = detector.process(array)
        if not result.multi_face_landmarks:
            continue
        face = max(result.multi_face_landmarks, key=lambda item: face_area(item.landmark))
        points = np.asarray(
            [[point.x * variant.width / scale, point.y * variant.height / scale, point.z] for point in face.landmark],
            dtype=np.float64,
        )
        score = (len(points), face_area(face.landmark))
        if best is None or score > best[0]:
            best = (score, points)
    return None if best is None else best[1]


def crop_identity_sheet(path: Path, output: Path):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    row_height = height // 3
    split = width // 2
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for row, view in enumerate(IDENTITY_VIEWS):
        top = row * row_height
        bottom = min(height, top + row_height - max(26, row_height // 14))
        approved = image.crop((0, top, split, bottom))
        model = image.crop((split, top, width, bottom))
        approved_path = output / f"approved_{view}.png"
        model_path = output / f"model_{view}.png"
        approved.save(approved_path)
        model.save(model_path)
        result[view] = {"approved": approved, "model": model, "approved_path": approved_path, "model_path": model_path}
    return result


def crop_expression_sheet(path: Path, output: Path):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    columns, rows = 4, 2
    cell_width, cell_height = width // columns, height // rows
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for index, name in enumerate(EXPRESSION_NAMES):
        column = index % columns
        row = index // columns
        crop = image.crop((column * cell_width, row * cell_height, (column + 1) * cell_width, (row + 1) * cell_height - max(24, cell_height // 14)))
        crop_path = output / f"expression_{name}.png"
        crop.save(crop_path)
        result[name] = {"image": crop, "path": crop_path}
    return result


def procrustes_rmse(left: np.ndarray, right: np.ndarray) -> float:
    count = min(len(left), len(right), 468)
    x = np.asarray(left[:count, :2], dtype=np.float64)
    y = np.asarray(right[:count, :2], dtype=np.float64)
    # Normalise by each face box before the similarity solve.
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    sx = math.sqrt(float(np.mean(np.sum(x * x, axis=1))))
    sy = math.sqrt(float(np.mean(np.sum(y * y, axis=1))))
    x /= max(sx, 1e-9)
    y /= max(sy, 1e-9)
    covariance = x.T @ y
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    fitted = x @ rotation.T
    return float(np.sqrt(np.mean(np.sum((fitted - y) ** 2, axis=1))))


def canonical_align(image: Image.Image, landmarks: np.ndarray, size: int = 160) -> Image.Image:
    # MediaPipe landmark groups provide stable centres across stylised and real
    # renders.  The canonical template matches the FaceNet 160×160 convention.
    left_eye = landmarks[[33, 133, 159, 145], :2].mean(axis=0)
    right_eye = landmarks[[362, 263, 386, 374], :2].mean(axis=0)
    nose = landmarks[1, :2]
    mouth_left = landmarks[61, :2]
    mouth_right = landmarks[291, :2]
    source = np.asarray([left_eye, right_eye, nose, mouth_left, mouth_right], dtype=np.float32)
    destination = np.asarray(
        [[52.0, 60.0], [108.0, 60.0], [80.0, 91.0], [58.0, 119.0], [102.0, 119.0]],
        dtype=np.float32,
    ) * (size / 160.0)
    matrix, _ = cv2.estimateAffinePartial2D(source, destination, method=cv2.LMEDS)
    if matrix is None:
        raise RuntimeError("Could not align detected face")
    array = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    aligned = cv2.warpAffine(array, matrix, (size, size), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
    return Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))


def facenet_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.resize((160, 160), Image.Resampling.LANCZOS), dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return (tensor - 127.5) / 128.0


def dino_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((tensor - mean) / std).unsqueeze(0)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = torch.nn.functional.normalize(left.float().flatten(1), dim=1)
    right = torch.nn.functional.normalize(right.float().flatten(1), dim=1)
    return float((left * right).sum().item())


def eye_aspect(landmarks: np.ndarray) -> float:
    left_vertical = 0.5 * (np.linalg.norm(landmarks[159, :2] - landmarks[145, :2]) + np.linalg.norm(landmarks[158, :2] - landmarks[153, :2]))
    left_width = np.linalg.norm(landmarks[33, :2] - landmarks[133, :2])
    right_vertical = 0.5 * (np.linalg.norm(landmarks[386, :2] - landmarks[374, :2]) + np.linalg.norm(landmarks[385, :2] - landmarks[380, :2]))
    right_width = np.linalg.norm(landmarks[362, :2] - landmarks[263, :2])
    return float(0.5 * (left_vertical / max(left_width, 1e-9) + right_vertical / max(right_width, 1e-9)))


def mouth_metrics(landmarks: np.ndarray):
    width = np.linalg.norm(landmarks[61, :2] - landmarks[291, :2])
    height = 0.5 * (
        np.linalg.norm(landmarks[13, :2] - landmarks[14, :2])
        + np.linalg.norm(landmarks[0, :2] - landmarks[17, :2])
    )
    return {"width": float(width), "height": float(height), "openness": float(height / max(width, 1e-9))}


def vlm_judgement(identity_sheet: Path, expression_sheet: Path, full_sheet: Path, model_id: str):
    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(identity_sheet)},
                    {"type": "image", "image": str(expression_sheet)},
                    {"type": "image", "image": str(full_sheet)},
                    {
                        "type": "text",
                        "text": (
                            "You are the final visual QA reviewer for a digital-human production model. "
                            "Image 1 has three rows: approved AINA reference on the left and the actual real 3D model render on the right for front, 3/4 and side. "
                            "Image 2 contains the real model neutral and expression renders. Image 3 contains full-character front and 3/4 renders. "
                            "Judge whether the right-side model is recognisably the same designed AINA character, not merely a generic woman. "
                            "Inspect eye shape/spacing, nose bridge/tip/wings, lips, lower-face/jaw/chin, age and expression identity preservation, silver updo quality, head/body integration and whether hair looks like flat layered ribbons rather than tubes. "
                            "Return exactly one JSON object with integer scores 0-100 for identity_likeness, eyes, nose, lips, jaw_chin, expression_consistency, hair, materials, full_character_integration; a boolean same_character; a boolean visually_release_ready; and a short issues array. "
                            "Be strict: visually_release_ready should be true only if a production customer would accept the real model as the approved character without another face-sculpt pass."
                        ),
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        generated = model.generate(**inputs, max_new_tokens=320, do_sample=False)
        generated = generated[:, inputs.input_ids.shape[1] :]
        response = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise RuntimeError(f"VLM did not return JSON: {response}")
        parsed = json.loads(match.group(0))
        return {"available": True, "model": model_id, "raw": response, "parsed": parsed}
    except Exception as exc:
        return {
            "available": False,
            "model": model_id,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    crops = args.out / "crops"
    identity = crop_identity_sheet(args.identity_sheet, crops / "identity")
    expressions = crop_expression_sheet(args.expression_sheet, crops / "expressions")

    face_mesh = mp.solutions.face_mesh
    detected_identity = {}
    detected_expressions = {}
    with face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.28) as detector:
        for view, values in identity.items():
            detected_identity[view] = {
                "approved": detect_face(values["approved"], detector),
                "model": detect_face(values["model"], detector),
            }
        for name, values in expressions.items():
            detected_expressions[name] = detect_face(values["image"], detector)

    missing_identity = [f"{view}:{side}" for view, values in detected_identity.items() for side, points in values.items() if points is None]
    mandatory_missing = [item for item in missing_identity if item.startswith("front") or item.startswith("three_quarter")]
    if mandatory_missing:
        raise SystemExit(f"Mandatory identity faces were not detected: {mandatory_missing}")

    face_model = InceptionResnetV1(pretrained="vggface2").eval()
    dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).eval()
    identity_metrics = {}
    aligned_models = {}
    with torch.no_grad():
        for view, values in identity.items():
            approved_points = detected_identity[view]["approved"]
            model_points = detected_identity[view]["model"]
            if approved_points is None or model_points is None:
                identity_metrics[view] = {"detected": False}
                continue
            approved_aligned = canonical_align(values["approved"], approved_points)
            model_aligned = canonical_align(values["model"], model_points)
            approved_aligned.save(crops / f"aligned_approved_{view}.png")
            model_aligned.save(crops / f"aligned_model_{view}.png")
            aligned_models[view] = model_aligned
            face_approved = face_model(facenet_tensor(approved_aligned))
            face_real = face_model(facenet_tensor(model_aligned))
            dino_approved = dino_model(dino_tensor(approved_aligned))
            dino_real = dino_model(dino_tensor(model_aligned))
            identity_metrics[view] = {
                "detected": True,
                "dense_procrustes_rmse": procrustes_rmse(approved_points, model_points),
                "facenet_vggface2_cosine": cosine(face_approved, face_real),
                "dinov2_cosine": cosine(dino_approved, dino_real),
            }

        expression_metrics = {}
        neutral_points = detected_expressions.get("neutral")
        neutral_aligned = canonical_align(expressions["neutral"]["image"], neutral_points) if neutral_points is not None else None
        neutral_embedding = face_model(facenet_tensor(neutral_aligned)) if neutral_aligned is not None else None
        for name, values in expressions.items():
            points = detected_expressions[name]
            if points is None:
                expression_metrics[name] = {"detected": False}
                continue
            aligned = canonical_align(values["image"], points)
            aligned.save(crops / f"aligned_expression_{name}.png")
            embedding = face_model(facenet_tensor(aligned))
            expression_metrics[name] = {
                "detected": True,
                "identity_cosine_to_neutral": cosine(neutral_embedding, embedding) if neutral_embedding is not None else None,
                "eye_aspect": eye_aspect(points),
                "mouth": mouth_metrics(points),
            }

    vlm = vlm_judgement(args.identity_sheet, args.expression_sheet, args.full_sheet, args.vlm_model)

    front = identity_metrics.get("front", {})
    q3 = identity_metrics.get("three_quarter", {})
    side = identity_metrics.get("side", {})
    detected_expression_count = sum(bool(item.get("detected")) for item in expression_metrics.values())
    expression_cosines = [item.get("identity_cosine_to_neutral") for item in expression_metrics.values() if item.get("identity_cosine_to_neutral") is not None]
    neutral = expression_metrics.get("neutral", {})
    blink = expression_metrics.get("blink", {})
    aa = expression_metrics.get("aa", {})
    ou = expression_metrics.get("ou", {})

    geometry_gate = (
        front.get("dense_procrustes_rmse", 9.0) <= 0.105
        and q3.get("dense_procrustes_rmse", 9.0) <= 0.125
        and (not side.get("detected") or side.get("dense_procrustes_rmse", 9.0) <= 0.180)
    )
    embedding_gate = (
        front.get("facenet_vggface2_cosine", -1.0) >= 0.40
        and q3.get("facenet_vggface2_cosine", -1.0) >= 0.32
        and front.get("dinov2_cosine", -1.0) >= 0.64
        and q3.get("dinov2_cosine", -1.0) >= 0.57
    )
    expression_identity_gate = detected_expression_count >= 7 and (min(expression_cosines) if expression_cosines else -1.0) >= 0.67
    blink_gate = (
        neutral.get("detected")
        and blink.get("detected")
        and blink.get("eye_aspect", 9.0) <= neutral.get("eye_aspect", 0.0) * 0.72
    )
    aa_gate = (
        neutral.get("detected")
        and aa.get("detected")
        and aa.get("mouth", {}).get("openness", 0.0) >= neutral.get("mouth", {}).get("openness", 9.0) * 1.35
    )
    ou_gate = (
        neutral.get("detected")
        and ou.get("detected")
        and ou.get("mouth", {}).get("width", 9.0) <= neutral.get("mouth", {}).get("width", 0.0) * 1.02
        and ou.get("mouth", {}).get("openness", 0.0) >= neutral.get("mouth", {}).get("openness", 0.0) * 1.05
    )

    vlm_parsed = vlm.get("parsed", {}) if vlm.get("available") else {}
    score_fields = ("identity_likeness", "eyes", "nose", "lips", "jaw_chin", "expression_consistency", "hair", "materials", "full_character_integration")
    vlm_scores_valid = all(isinstance(vlm_parsed.get(name), (int, float)) for name in score_fields)
    vlm_gate = (
        vlm.get("available")
        and vlm_scores_valid
        and bool(vlm_parsed.get("same_character"))
        and bool(vlm_parsed.get("visually_release_ready"))
        and float(vlm_parsed.get("identity_likeness", 0)) >= 74
        and float(vlm_parsed.get("eyes", 0)) >= 68
        and float(vlm_parsed.get("nose", 0)) >= 65
        and float(vlm_parsed.get("lips", 0)) >= 65
        and float(vlm_parsed.get("jaw_chin", 0)) >= 68
        and float(vlm_parsed.get("expression_consistency", 0)) >= 72
        and float(vlm_parsed.get("hair", 0)) >= 68
        and float(vlm_parsed.get("full_character_integration", 0)) >= 68
    )

    pass_all = geometry_gate and embedding_gate and expression_identity_gate and blink_gate and aa_gate and ou_gate and vlm_gate
    report = {
        "product": "AINA Final Actual-Render Visual Audit",
        "real_model_renders_only": True,
        "identity_metrics": identity_metrics,
        "expression_metrics": expression_metrics,
        "vlm": vlm,
        "gates": {
            "geometry": geometry_gate,
            "cross_style_embeddings": embedding_gate,
            "expression_identity": expression_identity_gate,
            "blink": blink_gate,
            "aa_viseme": aa_gate,
            "ou_viseme": ou_gate,
            "multimodal_visual_review": vlm_gate,
        },
        "pass": pass_all,
        "identity_lock_recommended": pass_all,
        "visual_identity_lock_recommended": pass_all,
        "production_release_recommended": pass_all,
        "policy": "All gates must pass; no technical-only override is permitted.",
    }
    (args.out / "AINA_FINAL_VISUAL_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / ("VISUAL_PASS" if pass_all else "VISUAL_FAIL")).write_text(json.dumps(report["gates"], indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
