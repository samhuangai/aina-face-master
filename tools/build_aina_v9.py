#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
COMPONENTS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def read_glb(path: Path):
    data = path.read_bytes()
    magic, version, total = struct.unpack_from('<4sII', data, 0)
    if magic != b'glTF' or version != 2 or total != len(data):
        raise ValueError(f'Invalid GLB 2.0 file: {path}')
    offset = 12
    document = None
    binary = b''
    while offset < total:
        length, chunk_type = struct.unpack_from('<II', data, offset)
        offset += 8
        chunk = data[offset:offset + length]
        offset += length
        if chunk_type == 0x4E4F534A:
            document = json.loads(chunk.decode('utf-8').rstrip(' \0'))
        elif chunk_type == 0x004E4942:
            binary = chunk
    if document is None:
        raise ValueError(f'No JSON chunk in {path}')
    return document, bytearray(binary)


def write_glb(path: Path, document: dict, binary: bytes):
    binary = bytes(binary)
    document['buffers'][0]['byteLength'] = len(binary)
    encoded = json.dumps(document, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    encoded += b' ' * ((4 - len(encoded) % 4) % 4)
    binary += b'\0' * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    path.write_bytes(
        struct.pack('<4sII', b'glTF', 2, total)
        + struct.pack('<II', len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack('<II', len(binary), 0x004E4942)
        + binary
    )


def accessor(document: dict, binary: bytes, index: int):
    acc = document['accessors'][index]
    view = document['bufferViews'][acc['bufferView']]
    dtype = np.dtype(DTYPES[acc['componentType']]).newbyteorder('<')
    count = COMPONENTS[acc['type']]
    start = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
    stride = view.get('byteStride', dtype.itemsize * count)
    if stride == dtype.itemsize * count:
        return np.frombuffer(binary, dtype=dtype, count=acc['count'] * count, offset=start).reshape(acc['count'], count).copy()
    return np.ndarray(
        (acc['count'], count),
        dtype=dtype,
        buffer=binary,
        offset=start,
        strides=(stride, dtype.itemsize),
    ).copy()


def write_accessor(document: dict, binary: bytearray, index: int, values):
    acc = document['accessors'][index]
    view = document['bufferViews'][acc['bufferView']]
    dtype = np.dtype(DTYPES[acc['componentType']]).newbyteorder('<')
    count = COMPONENTS[acc['type']]
    values = np.asarray(values, dtype=dtype).reshape(acc['count'], count)
    start = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
    stride = view.get('byteStride', dtype.itemsize * count)
    if stride == dtype.itemsize * count:
        binary[start:start + values.nbytes] = values.tobytes()
    else:
        for row_index, row in enumerate(values):
            binary[start + row_index * stride:start + row_index * stride + row.nbytes] = row.tobytes()
    if np.issubdtype(dtype, np.floating):
        acc['min'] = values.min(axis=0).astype(float).tolist()
        acc['max'] = values.max(axis=0).astype(float).tolist()


def image_bytes(document: dict, binary: bytes, index: int):
    image = document['images'][index]
    view = document['bufferViews'][image['bufferView']]
    start = view.get('byteOffset', 0)
    return bytes(binary[start:start + view['byteLength']])


def png_bytes(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', optimize=True)
    return buffer.getvalue()


def open_rgba(document: dict, binary: bytes, index: int):
    return Image.open(io.BytesIO(image_bytes(document, binary, index))).convert('RGBA')


def gaussian_field(width, height, cx, cy, rx, ry):
    yy, xx = np.mgrid[0:height, 0:width]
    return np.exp(-0.5 * (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))


def recolor_reference_textures(base_document: dict, base_binary: bytes, v8_document: dict, v8_binary: bytes):
    replacements = {}

    # Mouth atlas: retain original VRoid mouth UVs, move toward AINA's soft rose lip palette.
    mouth = np.array(open_rgba(base_document, base_binary, 0), dtype=np.float32)
    luminance = mouth[..., :3].mean(axis=2, keepdims=True) / 255.0
    target = np.array([191, 91, 105], dtype=np.float32)
    mouth[..., :3] = target * (0.66 + 0.34 * luminance)
    mouth[..., 3] = 255
    replacements[0] = Image.fromarray(np.clip(mouth, 0, 255).astype(np.uint8), 'RGBA')

    # Iris: cool grey/ice-blue, dark limbal ring and pupil retained from source luminance.
    iris = np.array(open_rgba(base_document, base_binary, 4), dtype=np.float32)
    lum = (0.2126 * iris[..., 0] + 0.7152 * iris[..., 1] + 0.0722 * iris[..., 2]) / 255.0
    cool = np.zeros_like(iris)
    cool[..., 0] = 17 + 128 * lum
    cool[..., 1] = 25 + 155 * lum
    cool[..., 2] = 38 + 181 * lum
    cool[..., 3] = iris[..., 3]
    replacements[4] = Image.fromarray(np.clip(cool, 0, 255).astype(np.uint8), 'RGBA')

    # Highlights are deliberately softer than the anime base.
    highlights = np.array(open_rgba(base_document, base_binary, 5), dtype=np.float32)
    highlights[..., :3] = 232
    highlights[..., 3] *= 0.56
    replacements[5] = Image.fromarray(np.clip(highlights, 0, 255).astype(np.uint8), 'RGBA')

    # Restore the proper UV skin texture instead of projecting the concept portrait as a square.
    skin = np.array(open_rgba(base_document, base_binary, 6), dtype=np.float32)
    original = skin[..., :3]
    skin_tone = np.array([203, 164, 156], dtype=np.float32)
    skin[..., :3] = original * 0.48 + skin_tone * 0.52
    h, w = skin.shape[:2]
    blush = (
        gaussian_field(w, h, w * 0.30, h * 0.61, w * 0.095, h * 0.055)
        + gaussian_field(w, h, w * 0.70, h * 0.61, w * 0.095, h * 0.055)
    )[..., None]
    nose = gaussian_field(w, h, w * 0.50, h * 0.57, w * 0.045, h * 0.050)[..., None]
    rose = np.array([223, 126, 139], dtype=np.float32)
    skin[..., :3] = skin[..., :3] * (1.0 - 0.075 * blush) + rose * (0.075 * blush)
    skin[..., :3] = skin[..., :3] * (1.0 - 0.035 * nose) + rose * (0.035 * nose)
    skin[..., 3] = 255
    replacements[6] = Image.fromarray(np.clip(skin, 0, 255).astype(np.uint8), 'RGBA')

    # Eye whites: cool off-white, never pure white.
    whites = np.array(open_rgba(base_document, base_binary, 10), dtype=np.float32)
    wlum = whites[..., :3].mean(axis=2, keepdims=True) / 255.0
    whites[..., 0] = 178 + 54 * wlum[..., 0]
    whites[..., 1] = 183 + 54 * wlum[..., 0]
    whites[..., 2] = 190 + 54 * wlum[..., 0]
    whites[..., 3] = 255
    replacements[10] = Image.fromarray(np.clip(whites, 0, 255).astype(np.uint8), 'RGBA')

    # Brows: slightly fuller, cool charcoal-brown.
    brows_image = open_rgba(base_document, base_binary, 11)
    alpha = brows_image.getchannel('A').filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.35))
    brows = np.zeros((brows_image.height, brows_image.width, 4), dtype=np.uint8)
    brows[..., 0] = 55
    brows[..., 1] = 48
    brows[..., 2] = 52
    brows[..., 3] = np.array(alpha)
    replacements[11] = Image.fromarray(brows, 'RGBA')

    # Eyeline: narrower geometry is handled in mesh deformation; texture remains clean charcoal.
    line_image = open_rgba(base_document, base_binary, 12)
    line = np.zeros((line_image.height, line_image.width, 4), dtype=np.uint8)
    line[..., 0] = 42
    line[..., 1] = 36
    line[..., 2] = 43
    line[..., 3] = np.array(line_image.getchannel('A'))
    replacements[12] = Image.fromarray(line, 'RGBA')

    # Retain the V8 future uniform and silver hair atlases, but remove clipping-level whites.
    for image_index, tint, multiplier in (
        (13, np.array([10, 13, 22], dtype=np.float32), 0.74),
        (15, np.array([17, 19, 29], dtype=np.float32), 0.76),
    ):
        image = np.array(open_rgba(v8_document, v8_binary, image_index), dtype=np.float32)
        image[..., :3] = image[..., :3] * multiplier + tint
        replacements[image_index] = Image.fromarray(np.clip(image, 0, 242).astype(np.uint8), 'RGBA')

    return replacements


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def face_sets(document: dict, binary: bytes):
    primitives = document['meshes'][0]['primitives']
    if len(primitives) < 7:
        raise RuntimeError('AINA V9 expects the seven VRoid face primitives')
    return [
        np.unique(accessor(document, binary, primitive['indices']).reshape(-1).astype(np.int64))
        for primitive in primitives[:7]
    ]


def scale_group(x, y, indices, sx, sy, inward=0.0, vertical_shift=0.0):
    for sign in (-1.0, 1.0):
        group = indices[x[indices] * sign > 0]
        if len(group) == 0:
            continue
        cx = float(x[group].mean())
        cy = float(y[group].mean())
        x[group] = cx + (x[group] - cx) * sx - sign * inward
        y[group] = cy + (y[group] - cy) * sy + vertical_shift


def deform_face(document: dict, binary: bytearray):
    primitive = document['meshes'][0]['primitives'][0]
    position_accessor = primitive['attributes']['POSITION']
    vertices = accessor(document, binary, position_accessor).astype(np.float64)
    mouth, iris, highlight, skin, eye_white, brow, eyeline = face_sets(document, binary)
    all_face = np.unique(np.concatenate((mouth, iris, highlight, skin, eye_white, brow, eyeline)))
    x, y, z = vertices.T

    # Reduce the oversized upper cranium while preserving the jaw and all topology.
    upper = smoothstep((y - 1.485) / 0.135)
    x[all_face] *= 1.0 - 0.135 * upper[all_face]
    y[all_face] = 1.485 + (y[all_face] - 1.485) * (1.0 - 0.105 * upper[all_face])

    # Restore an oval lower face instead of the extreme anime taper.
    cheek = np.exp(-0.5 * ((y - 1.405) / 0.030) ** 2)
    x[skin] *= 1.0 + 0.024 * cheek[skin]
    lower = smoothstep((1.385 - y) / 0.047)
    x[skin] *= 1.0 + 0.035 * lower[skin]
    y[skin] += 0.0030 * smoothstep((1.370 - y[skin]) / 0.030)

    # Smaller almond eyes, closer to the approved AINA proportions.
    scale_group(x, y, eye_white, 0.895, 0.685, inward=0.0022, vertical_shift=-0.0007)
    scale_group(x, y, eyeline, 0.900, 0.700, inward=0.0022, vertical_shift=-0.0005)
    scale_group(x, y, iris, 0.785, 0.785, inward=0.0022, vertical_shift=-0.0006)
    scale_group(x, y, highlight, 0.785, 0.785, inward=0.0022, vertical_shift=-0.0006)
    scale_group(x, y, brow, 0.940, 0.820, inward=0.0014, vertical_shift=-0.0040)

    # Natural compact lips.
    mouth_center_x = float(x[mouth].mean())
    mouth_center_y = float(y[mouth].mean())
    x[mouth] = mouth_center_x + (x[mouth] - mouth_center_x) * 1.115
    y[mouth] = mouth_center_y + (y[mouth] - mouth_center_y) * 0.82 - 0.0008

    # Refine the small straight nose and soften the profile projection.
    nose_weight = np.exp(-0.5 * ((x / 0.018) ** 2 + ((y - 1.408) / 0.024) ** 2))
    nose_ids = skin[nose_weight[skin] > 0.025]
    z[nose_ids] += 0.0048 * nose_weight[nose_ids]
    x[nose_ids] *= 1.0 - 0.070 * nose_weight[nose_ids]
    bridge_weight = np.exp(-0.5 * ((x / 0.012) ** 2 + ((y - 1.442) / 0.032) ** 2))
    bridge_ids = skin[bridge_weight[skin] > 0.03]
    z[bridge_ids] -= 0.0014 * bridge_weight[bridge_ids]

    # Give the cheeks a gentle forward volume without making the face round.
    for center_x in (-0.043, 0.043):
        volume = np.exp(-0.5 * (((x - center_x) / 0.032) ** 2 + ((y - 1.405) / 0.027) ** 2))
        ids = skin[volume[skin] > 0.03]
        z[ids] -= 0.00135 * volume[ids]

    vertices[:, 0] = x
    vertices[:, 1] = y
    vertices[:, 2] = z
    write_accessor(document, binary, position_accessor, vertices.astype(np.float32))
    return {
        'face_vertices': len(vertices),
        'skin_vertices': len(skin),
        'eye_white_vertices': len(eye_white),
        'morph_targets': len(primitive.get('targets', [])),
    }


def deform_updo(document: dict, binary: bytearray):
    report = {'found': False}
    for mesh in document.get('meshes', []):
        if mesh.get('name') != 'AINA_Updo':
            continue
        report['found'] = True
        total = 0
        for primitive in mesh.get('primitives', []):
            position_index = primitive.get('attributes', {}).get('POSITION')
            if position_index is None:
                continue
            vertices = accessor(document, binary, position_index).astype(np.float64)
            center = vertices.mean(axis=0)
            vertices[:, 0] = center[0] + (vertices[:, 0] - center[0]) * 0.82
            vertices[:, 1] = center[1] + (vertices[:, 1] - center[1]) * 0.84 - 0.012
            vertices[:, 2] = center[2] + (vertices[:, 2] - center[2]) * 0.80 + 0.004
            write_accessor(document, binary, position_index, vertices.astype(np.float32))
            total += len(vertices)
        report['vertices'] = total
    return report


def patch_materials(document: dict):
    material_by_keyword = {}
    for material in document.get('materials', []):
        lower = material.get('name', '').lower()
        material_by_keyword[lower] = material
        pbr = material.setdefault('pbrMetallicRoughness', {})
        pbr.setdefault('metallicFactor', 0.0)
        pbr.setdefault('roughnessFactor', 0.48)

        if 'facemouth' in lower:
            pbr['baseColorFactor'] = [1.0, 1.0, 1.0, 1.0]
            pbr['roughnessFactor'] = 0.42
            material['alphaMode'] = 'OPAQUE'
        elif 'eyeiris' in lower:
            pbr['baseColorFactor'] = [1.0, 1.0, 1.0, 1.0]
            pbr['roughnessFactor'] = 0.24
            material['alphaMode'] = 'BLEND'
        elif 'eyehighlight' in lower:
            pbr['baseColorFactor'] = [1.0, 1.0, 1.0, 0.58]
            pbr['roughnessFactor'] = 0.18
            material['alphaMode'] = 'BLEND'
        elif 'face_00_skin' in lower or 'faceskin' in lower:
            pbr['baseColorFactor'] = [0.92, 0.92, 0.92, 1.0]
            pbr['roughnessFactor'] = 0.50
            material['alphaMode'] = 'OPAQUE'
        elif 'eyewhite' in lower:
            pbr['baseColorFactor'] = [0.92, 0.94, 0.97, 1.0]
            pbr['roughnessFactor'] = 0.30
            material['alphaMode'] = 'OPAQUE'
        elif 'facebrow' in lower or 'faceeyeline' in lower:
            pbr['baseColorFactor'] = [1.0, 1.0, 1.0, 0.95]
            pbr['roughnessFactor'] = 0.46
            material['alphaMode'] = 'BLEND'
        elif 'hairback' in lower or 'silver_updo' in lower:
            pbr['baseColorFactor'] = [0.70, 0.72, 0.82, 1.0]
            pbr['metallicFactor'] = 0.02
            pbr['roughnessFactor'] = 0.36
        elif 'uniform' in lower:
            pbr['baseColorFactor'] = [0.72, 0.75, 0.84, 1.0]
            pbr['metallicFactor'] = 0.03
            pbr['roughnessFactor'] = 0.32
        elif 'aina_core' in lower:
            pbr['baseColorFactor'] = [0.025, 0.25, 0.85, 1.0]
            material['emissiveFactor'] = [0.02, 0.33, 1.0]

    document.setdefault('asset', {})['generator'] = 'AINA V9 identity refinement pipeline'
    document['asset']['copyright'] = 'AINA Digital Human'

    extension = document.get('extensions', {}).get('VRMC_vrm')
    if extension:
        extension.setdefault('meta', {})['name'] = 'AINA'
        extension['meta']['version'] = '1.0 V9 identity refinement'
        extension['meta']['references'] = ['AINA approved multi-view identity board']


def rebuild_binary(document: dict, binary: bytearray, image_replacements: dict[int, Image.Image]):
    replacement_views = {
        document['images'][index]['bufferView']: png_bytes(image)
        for index, image in image_replacements.items()
        if index < len(document.get('images', []))
    }
    payloads = []
    for view_index, view in enumerate(document['bufferViews']):
        start = view.get('byteOffset', 0)
        payloads.append(replacement_views.get(view_index, bytes(binary[start:start + view['byteLength']])))

    rebuilt = bytearray()
    for view, payload in zip(document['bufferViews'], payloads):
        while len(rebuilt) % 4:
            rebuilt.append(0)
        view['byteOffset'] = len(rebuilt)
        view['byteLength'] = len(payload)
        rebuilt.extend(payload)
    document['buffers'][0]['byteLength'] = len(rebuilt)
    return rebuilt


def build_one(source: Path, base_document: dict, base_binary: bytes, output: Path):
    document, binary = read_glb(source)
    replacements = recolor_reference_textures(base_document, base_binary, document, binary)
    geometry_report = deform_face(document, binary)
    updo_report = deform_updo(document, binary)
    patch_materials(document)
    rebuilt = rebuild_binary(document, binary, replacements)
    write_glb(output, document, rebuilt)
    return {
        'output': output.name,
        'bytes': output.stat().st_size,
        'geometry': geometry_report,
        'updo': updo_report,
        'materials': len(document.get('materials', [])),
    }


def main():
    if len(sys.argv) != 5:
        raise SystemExit('build_aina_v9.py V8_VRM V8_BLENDER_GLB BASE_VROID_VRM OUTPUT_DIR')
    v8_vrm = Path(sys.argv[1])
    v8_blender = Path(sys.argv[2])
    base_path = Path(sys.argv[3])
    output_dir = Path(sys.argv[4])
    output_dir.mkdir(parents=True, exist_ok=True)

    base_document, base_binary = read_glb(base_path)
    formal = build_one(v8_vrm, base_document, base_binary, output_dir / 'AINA_VRM_CANDIDATE_V9.vrm')
    blender = build_one(v8_blender, base_document, base_binary, output_dir / 'AINA_BLENDER_SOURCE_V9.glb')

    report = {
        'version': 'V9',
        'identity_basis': 'approved AINA multi-view board',
        'formal_vrm': formal,
        'blender_source': blender,
        'vrm1': True,
        'humanoid_bones_expected': 54,
        'face_morphs_expected': 57,
        'visual_changes': [
            'restored UV-correct skin and eye layers',
            'smaller almond eyes',
            'reduced upper cranium and forehead',
            'softened nose profile',
            'oval jaw and compact lips',
            'lower-energy silver hair and uniform materials',
            'reduced updo volume',
        ],
        'identity_lock': False,
        'visual_identity_lock': False,
    }
    (output_dir / 'AINA_V9_BUILD_REPORT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
