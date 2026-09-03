#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

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
        length, kind = struct.unpack_from('<II', data, offset)
        offset += 8
        chunk = data[offset:offset + length]
        offset += length
        if kind == 0x4E4F534A:
            document = json.loads(chunk.decode('utf-8').rstrip(' \0'))
        elif kind == 0x004E4942:
            binary = chunk
    if document is None:
        raise ValueError('GLB has no JSON chunk')
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


def read_accessor(document: dict, binary: bytes, index: int):
    acc = document['accessors'][index]
    view = document['bufferViews'][acc['bufferView']]
    dtype = np.dtype(DTYPES[acc['componentType']]).newbyteorder('<')
    components = COMPONENTS[acc['type']]
    start = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
    stride = view.get('byteStride', dtype.itemsize * components)
    if stride == dtype.itemsize * components:
        return np.frombuffer(
            binary,
            dtype=dtype,
            count=acc['count'] * components,
            offset=start,
        ).reshape(acc['count'], components).copy()
    return np.ndarray(
        (acc['count'], components),
        dtype=dtype,
        buffer=binary,
        offset=start,
        strides=(stride, dtype.itemsize),
    ).copy()


def write_accessor(document: dict, binary: bytearray, index: int, values):
    acc = document['accessors'][index]
    view = document['bufferViews'][acc['bufferView']]
    dtype = np.dtype(DTYPES[acc['componentType']]).newbyteorder('<')
    components = COMPONENTS[acc['type']]
    values = np.asarray(values, dtype=dtype).reshape(acc['count'], components)
    start = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
    stride = view.get('byteStride', dtype.itemsize * components)
    if stride == dtype.itemsize * components:
        binary[start:start + values.nbytes] = values.tobytes()
    else:
        for row_index, row in enumerate(values):
            binary[start + row_index * stride:start + row_index * stride + row.nbytes] = row.tobytes()
    if np.issubdtype(dtype, np.floating):
        acc['min'] = values.min(axis=0).astype(float).tolist()
        acc['max'] = values.max(axis=0).astype(float).tolist()


def primitive_sets(document: dict, binary: bytes):
    primitives = document['meshes'][0]['primitives']
    if len(primitives) < 7:
        raise RuntimeError('AINA V10 expects seven face primitives')
    return [
        np.unique(read_accessor(document, binary, primitive['indices']).reshape(-1).astype(np.int64))
        for primitive in primitives[:7]
    ]


def scale_each_side(x, y, ids, sx, sy, inward=0.0, vertical_shift=0.0):
    for sign in (-1.0, 1.0):
        group = ids[x[ids] * sign > 0.0]
        if len(group) == 0:
            continue
        cx = float(x[group].mean())
        cy = float(y[group].mean())
        x[group] = cx + (x[group] - cx) * sx - sign * inward
        y[group] = cy + (y[group] - cy) * sy + vertical_shift


def deform_face(document: dict, binary: bytearray):
    primitive = document['meshes'][0]['primitives'][0]
    position_index = primitive['attributes']['POSITION']
    vertices = read_accessor(document, binary, position_index).astype(np.float64)
    mouth, iris, highlight, skin, eye_white, brow, eyeline = primitive_sets(document, binary)
    all_face = np.unique(np.concatenate((mouth, iris, highlight, skin, eye_white, brow, eyeline)))
    x, y, z = vertices.T
    old_y = y.copy()

    eye_anchor = float(y[eye_white].mean())
    mouth_anchor = float(y[mouth].mean())
    chin_anchor = float(y[skin].min())

    above = all_face[old_y[all_face] >= eye_anchor]
    middle = all_face[(old_y[all_face] < eye_anchor) & (old_y[all_face] >= mouth_anchor)]
    lower = all_face[old_y[all_face] < mouth_anchor]
    y[above] = eye_anchor + (old_y[above] - eye_anchor) * 0.60
    y[middle] = eye_anchor + (old_y[middle] - eye_anchor) * 1.55
    mouth_target = eye_anchor + (mouth_anchor - eye_anchor) * 1.55
    y[lower] = mouth_target + (old_y[lower] - mouth_anchor) * 1.30

    x[all_face] *= 0.96
    upper_amount = np.clip((old_y - eye_anchor) / max(1e-6, old_y[skin].max() - eye_anchor), 0.0, 1.0)
    x[all_face] *= 1.0 - 0.18 * upper_amount[all_face] ** 0.85

    cheek = np.exp(-0.5 * ((old_y - 1.405) / 0.027) ** 2)
    x[skin] *= 1.0 + 0.020 * cheek[skin]
    jaw_amount = np.clip((mouth_anchor - old_y) / max(1e-6, mouth_anchor - chin_anchor), 0.0, 1.0)
    x[skin] *= 1.0 - 0.14 * jaw_amount[skin] ** 1.10

    rear = skin[z[skin] > -0.010]
    z[rear] *= 0.82
    upper_skin = skin[old_y[skin] > eye_anchor]
    z[upper_skin] *= 0.90

    scale_each_side(x, y, eye_white, sx=0.72, sy=0.46, inward=0.0062, vertical_shift=-0.0004)
    scale_each_side(x, y, eyeline, sx=0.72, sy=0.46, inward=0.0062, vertical_shift=-0.0001)
    scale_each_side(x, y, iris, sx=0.78, sy=0.50, inward=0.0062, vertical_shift=-0.0004)
    scale_each_side(x, y, highlight, sx=0.78, sy=0.50, inward=0.0062, vertical_shift=-0.0004)
    scale_each_side(x, y, brow, sx=0.83, sy=0.70, inward=0.0038, vertical_shift=0.0004)

    mouth_cx = float(x[mouth].mean())
    mouth_cy = float(y[mouth].mean())
    x[mouth] = mouth_cx + (x[mouth] - mouth_cx) * 0.96
    y[mouth] = mouth_cy + (y[mouth] - mouth_cy) * 0.45

    bridge = np.exp(-0.5 * ((x / 0.012) ** 2 + ((y - (eye_anchor - 0.025)) / 0.032) ** 2))
    bridge_ids = skin[bridge[skin] > 0.025]
    z[bridge_ids] -= 0.0024 * bridge[bridge_ids]
    tip = np.exp(-0.5 * ((x / 0.017) ** 2 + ((y - (eye_anchor - 0.053)) / 0.017) ** 2))
    tip_ids = skin[tip[skin] > 0.03]
    z[tip_ids] -= 0.0020 * tip[tip_ids]
    central = skin[(np.abs(x[skin]) < 0.022) & (y[skin] > eye_anchor - 0.072) & (y[skin] < eye_anchor - 0.020)]
    z[central] = np.maximum(z[central], -0.0795)

    chin_ids = skin[y[skin] < mouth_target - 0.025]
    z[chin_ids] += 0.0025 * np.clip((mouth_target - 0.025 - y[chin_ids]) / 0.020, 0.0, 1.0)

    vertices[:, 0] = x
    vertices[:, 1] = y
    vertices[:, 2] = z
    write_accessor(document, binary, position_index, vertices.astype(np.float32))
    return {
        'face_vertices': len(vertices),
        'morph_targets': len(primitive.get('targets', [])),
        'eye_anchor': eye_anchor,
        'mouth_anchor_old': mouth_anchor,
        'mouth_anchor_new': mouth_target,
        'new_min': vertices.min(axis=0).astype(float).tolist(),
        'new_max': vertices.max(axis=0).astype(float).tolist(),
    }


def compact_updo(document: dict, binary: bytearray):
    result = {'found': False, 'vertices': 0}
    for mesh in document.get('meshes', []):
        if mesh.get('name') != 'AINA_Updo':
            continue
        result['found'] = True
        for primitive in mesh.get('primitives', []):
            position_index = primitive.get('attributes', {}).get('POSITION')
            if position_index is None:
                continue
            vertices = read_accessor(document, binary, position_index).astype(np.float64)
            center = vertices.mean(axis=0)
            vertices[:, 0] = center[0] + (vertices[:, 0] - center[0]) * 0.78
            vertices[:, 1] = center[1] + (vertices[:, 1] - center[1]) * 0.78 - 0.010
            vertices[:, 2] = center[2] + (vertices[:, 2] - center[2]) * 0.74 + 0.004
            write_accessor(document, binary, position_index, vertices.astype(np.float32))
            result['vertices'] += len(vertices)
    return result


def patch_materials(document: dict):
    report = {}
    for material in document.get('materials', []):
        name = material.get('name', '')
        lower = name.lower()
        pbr = material.setdefault('pbrMetallicRoughness', {})
        pbr.setdefault('metallicFactor', 0.0)
        pbr.setdefault('roughnessFactor', 0.48)

        if 'aina_core' not in lower:
            material.pop('emissiveTexture', None)
            material['emissiveFactor'] = [0.0, 0.0, 0.0]

        if 'facemouth' in lower:
            pbr['baseColorFactor'] = [0.92, 0.78, 0.80, 1.0]
            pbr['roughnessFactor'] = 0.40
            material['alphaMode'] = 'OPAQUE'
        elif 'eyeiris' in lower:
            pbr['baseColorFactor'] = [0.62, 0.70, 0.86, 1.0]
            pbr['roughnessFactor'] = 0.20
            material['alphaMode'] = 'BLEND'
        elif 'eyehighlight' in lower:
            pbr['baseColorFactor'] = [0.72, 0.78, 0.88, 0.30]
            pbr['roughnessFactor'] = 0.12
            material['alphaMode'] = 'BLEND'
        elif 'face_00_skin' in lower or 'faceskin' in lower:
            pbr['baseColorFactor'] = [0.74, 0.69, 0.68, 1.0]
            pbr['roughnessFactor'] = 0.54
            material['alphaMode'] = 'OPAQUE'
        elif 'eyewhite' in lower:
            pbr['baseColorFactor'] = [0.67, 0.71, 0.79, 1.0]
            pbr['roughnessFactor'] = 0.33
            material['alphaMode'] = 'OPAQUE'
        elif 'facebrow' in lower:
            pbr['baseColorFactor'] = [0.56, 0.48, 0.50, 1.0]
            pbr['roughnessFactor'] = 0.52
            material['alphaMode'] = 'BLEND'
        elif 'faceeyeline' in lower:
            pbr['baseColorFactor'] = [0.0, 0.0, 0.0, 0.0]
            pbr['roughnessFactor'] = 0.52
            material['alphaMode'] = 'BLEND'
        elif 'body_00_skin' in lower:
            pbr['baseColorFactor'] = [0.72, 0.69, 0.72, 1.0]
            pbr['roughnessFactor'] = 0.52
            material['alphaMode'] = 'MASK'
        elif 'hairback' in lower:
            pbr['baseColorFactor'] = [0.43, 0.47, 0.60, 1.0]
            pbr['metallicFactor'] = 0.025
            pbr['roughnessFactor'] = 0.38
            material['alphaMode'] = 'MASK'
        elif 'silver_updo' in lower:
            pbr['baseColorFactor'] = [0.42, 0.46, 0.60, 1.0]
            pbr['metallicFactor'] = 0.04
            pbr['roughnessFactor'] = 0.36
        elif 'hairpins' in lower:
            pbr['baseColorFactor'] = [0.22, 0.28, 0.44, 1.0]
            pbr['metallicFactor'] = 0.72
            pbr['roughnessFactor'] = 0.22
        elif 'aina_core' in lower:
            pbr['baseColorFactor'] = [0.018, 0.20, 0.80, 1.0]
            pbr['metallicFactor'] = 0.05
            pbr['roughnessFactor'] = 0.18
            material['emissiveFactor'] = [0.02, 0.28, 1.0]
        elif 'uniform' in lower:
            pbr['baseColorFactor'] = [0.64, 0.67, 0.77, 1.0]
            pbr['metallicFactor'] = 0.04
            pbr['roughnessFactor'] = 0.34

        report[name] = {
            'baseColorFactor': pbr.get('baseColorFactor'),
            'emissiveFactor': material.get('emissiveFactor'),
            'hasEmissiveTexture': 'emissiveTexture' in material,
            'alphaMode': material.get('alphaMode'),
        }

    document.setdefault('asset', {})['generator'] = 'AINA V10 proportion and material correction pipeline'
    extension = document.get('extensions', {}).get('VRMC_vrm')
    if extension:
        meta = extension.setdefault('meta', {})
        meta['name'] = 'AINA'
        meta['version'] = '1.0 V10 visual refinement'
        meta['references'] = ['AINA approved high-resolution multi-view identity board']
    return report


def build(source: Path, output: Path):
    document, binary = read_glb(source)
    geometry = deform_face(document, binary)
    updo = compact_updo(document, binary)
    materials = patch_materials(document)
    write_glb(output, document, binary)
    return {
        'output': output.name,
        'bytes': output.stat().st_size,
        'geometry': geometry,
        'updo': updo,
        'materials': materials,
    }


def main():
    if len(sys.argv) != 4:
        raise SystemExit('build_aina_v10.py V9_FORMAL_VRM V9_BLENDER_GLB OUTPUT_DIR')
    formal_source = Path(sys.argv[1])
    blender_source = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)

    formal = build(formal_source, output_dir / 'AINA_VRM_CANDIDATE_V10.vrm')
    blender = build(blender_source, output_dir / 'AINA_BLENDER_SOURCE_V10.glb')
    report = {
        'version': 'V10',
        'formal': formal,
        'blender': blender,
        'structural_requirements': {
            'vrm1': True,
            'humanoid_bones': 54,
            'face_morphs': 57,
            'expression_presets': 14,
        },
        'visual_changes': [
            'all non-core emissive channels removed',
            'forehead compressed and lower face lengthened',
            'eye opening reduced and eyes moved inward',
            'lower jaw tapered',
            'nose bridge and tip rebalanced',
            'existing black eyeline layer hidden',
            'hair and uniform response darkened',
        ],
        'identity_lock': False,
        'visual_identity_lock': False,
    }
    (output_dir / 'AINA_V10_BUILD_REPORT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
