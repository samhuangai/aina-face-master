#!/usr/bin/env python3
"""Re-export the existing AINA_MASTER.blend while preserving all 52 morph targets.

This is a focused production repair: it does not rebuild or alter the character.
It disables Blender 4.5's armature-object-removal optimization inside the VRM
Addon because that preprocessing path is the only remaining delta from the
known-good 52-target glTF probe.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import traceback
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aina_vrm_addon_runtime import ensure_vrm_addon

SHAPE_KEYS = [
    'browDownLeft','browDownRight','browInnerUp','browOuterUpLeft','browOuterUpRight',
    'cheekPuff','cheekSquintLeft','cheekSquintRight',
    'eyeBlinkLeft','eyeBlinkRight','eyeLookDownLeft','eyeLookDownRight','eyeLookInLeft','eyeLookInRight','eyeLookOutLeft','eyeLookOutRight','eyeLookUpLeft','eyeLookUpRight','eyeSquintLeft','eyeSquintRight','eyeWideLeft','eyeWideRight',
    'jawForward','jawLeft','jawOpen','jawRight',
    'mouthClose','mouthDimpleLeft','mouthDimpleRight','mouthFrownLeft','mouthFrownRight','mouthFunnel','mouthLeft','mouthLowerDownLeft','mouthLowerDownRight','mouthPressLeft','mouthPressRight','mouthPucker','mouthRight','mouthRollLower','mouthRollUpper','mouthShrugLower','mouthShrugUpper','mouthSmileLeft','mouthSmileRight','mouthStretchLeft','mouthStretchRight','mouthUpperUpLeft','mouthUpperUpRight',
    'noseSneerLeft','noseSneerRight','tongueOut',
]
EXPECTED_PRESET_BINDS = {
    'happy':4,'angry':4,'sad':3,'relaxed':4,'surprised':4,'neutral':0,
    'aa':2,'ih':2,'ou':2,'ee':4,'oh':2,'blink':2,
    'blinkLeft':1,'blinkRight':1,'lookUp':2,'lookDown':2,'lookLeft':2,'lookRight':2,
}


def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    return ap.parse_args(argv)


def parse_glb_json(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 20:
        raise RuntimeError('VRM/GLB is too small')
    magic, version, total = struct.unpack_from('<4sII', data, 0)
    if magic != b'glTF' or version != 2 or total != len(data):
        raise RuntimeError(f'Invalid GLB header: magic={magic!r} version={version} total={total} actual={len(data)}')
    off = 12
    while off + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from('<II', data, off)
        off += 8
        chunk = data[off:off + chunk_len]
        off += chunk_len
        if chunk_type == 0x4E4F534A:
            return json.loads(chunk.rstrip(b' \x00').decode('utf-8'))
    raise RuntimeError('GLB JSON chunk not found')


def verify_binary(path: Path) -> dict:
    doc = parse_glb_json(path)
    meshes = doc.get('meshes') or []
    face = None
    for i, mesh in enumerate(meshes):
        if not isinstance(mesh, dict):
            continue
        extras = mesh.get('extras') or {}
        names = extras.get('targetNames') if isinstance(extras, dict) else None
        weights = mesh.get('weights')
        if isinstance(names, list) and len(names) == 52 and set(SHAPE_KEYS).issubset(set(map(str, names))):
            face = (i, mesh, list(map(str, names)), weights)
            break
    if face is None:
        candidates = [{
            'index': i,
            'name': m.get('name') if isinstance(m, dict) else None,
            'weights': len(m.get('weights', [])) if isinstance(m, dict) and isinstance(m.get('weights'), list) else 0,
            'target_names': len((m.get('extras') or {}).get('targetNames', [])) if isinstance(m, dict) and isinstance(m.get('extras'), dict) else 0,
        } for i, m in enumerate(meshes)]
        raise RuntimeError(f'No exported mesh carries all 52 AINA morph targets: {candidates}')

    mesh_i, mesh, names, weights = face
    primitives = mesh.get('primitives') or []
    primitive_target_counts = [
        len(p.get('targets') or []) if isinstance(p, dict) else 0 for p in primitives
    ]
    missing = [name for name in SHAPE_KEYS if name not in names]
    if missing or not isinstance(weights, list) or len(weights) != 52:
        raise RuntimeError(f'AINA morph metadata incomplete: missing={missing} weights={0 if not isinstance(weights, list) else len(weights)}')
    if not primitives or any(n != 52 for n in primitive_target_counts):
        raise RuntimeError(f'AINA primitive morph target count mismatch: {primitive_target_counts}')

    vrm = ((doc.get('extensions') or {}).get('VRMC_vrm') or {})
    preset = ((vrm.get('expressions') or {}).get('preset') or {})
    actual_preset_counts = {
        name: len((preset.get(name) or {}).get('morphTargetBinds') or [])
        for name in EXPECTED_PRESET_BINDS
    }
    missing_presets = [name for name in EXPECTED_PRESET_BINDS if name not in preset]
    if missing_presets:
        raise RuntimeError(f'Missing VRM preset entries in binary: {missing_presets}')
    if actual_preset_counts != EXPECTED_PRESET_BINDS:
        raise RuntimeError(f'VRM preset bind counts differ: expected={EXPECTED_PRESET_BINDS} actual={actual_preset_counts}')

    return {
        'binary_pass': True,
        'vrm_bytes': path.stat().st_size,
        'face_mesh_index': mesh_i,
        'face_mesh_name': mesh.get('name'),
        'shape_controls': len(names),
        'shape_control_names': names,
        'primitive_target_counts': primitive_target_counts,
        'preset_count': len(EXPECTED_PRESET_BINDS),
        'preset_bind_counts': actual_preset_counts,
        'total_preset_morph_binds': sum(actual_preset_counts.values()),
        'armature_object_remove_forced_off': True,
    }


def main():
    a = parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'QA').mkdir(exist_ok=True)

    ensure_vrm_addon(Path.cwd())

    from io_scene_vrm.exporter.vrm1_exporter import Vrm1Exporter

    Vrm1Exporter.gltf_export_armature_object_remove = staticmethod(
        lambda context, mesh_object_names: False
    )

    head = bpy.data.objects.get('AINA_Face_v15_5')
    if head is None or head.type != 'MESH' or not head.data.shape_keys:
        raise RuntimeError('AINA face with Shape Keys is missing from AINA_MASTER.blend')
    keys = [k.name for k in head.data.shape_keys.key_blocks if k.name != 'Basis']
    missing = [k for k in SHAPE_KEYS if k not in keys]
    if len(keys) != 52 or missing:
        raise RuntimeError(f'MASTER does not contain exact 52 controls: count={len(keys)} missing={missing}')

    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.update()

    vrm_path = out / 'AINA.vrm'
    result = bpy.ops.export_scene.vrm(filepath=str(vrm_path))
    if result != {'FINISHED'}:
        raise RuntimeError(f'VRM export failed: {result}')
    if not vrm_path.exists() or vrm_path.stat().st_size < 100_000:
        raise RuntimeError('AINA.vrm missing or implausibly small after export')

    qa = verify_binary(vrm_path)
    qa['master_shape_controls'] = len(keys)
    qa_path = out / 'QA' / 'AINA_VRM_BINARY_MORPH_QA.json'
    qa_path.write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print('[AINA_MORPH_REEXPORT]', json.dumps(qa, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
