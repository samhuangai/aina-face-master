#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import bpy


def glb_json(path: Path) -> dict:
    data=path.read_bytes()
    if data[:4] != b'glTF': raise RuntimeError(f'Not GLB: {path}')
    version,total=struct.unpack_from('<II',data,4)
    off=12
    while off+8 <= len(data):
        length,typ=struct.unpack_from('<II',data,off); off+=8
        chunk=data[off:off+length]; off+=length
        if typ == 0x4E4F534A:
            return json.loads(chunk.rstrip(b'\x00 \t\r\n').decode('utf-8'))
    raise RuntimeError('No JSON chunk')


def summarize(path: Path) -> dict:
    g=glb_json(path); out=[]
    for i,m in enumerate(g.get('meshes',[])):
        targets=sum(len(p.get('targets',[])) for p in m.get('primitives',[]))
        out.append({'i':i,'name':m.get('name'),'primitive_count':len(m.get('primitives',[])),'target_slots_sum':targets,'weights':len(m.get('weights',[])),'target_names':len((m.get('extras') or {}).get('targetNames',[]))})
    return {'meshes':out,'morph_meshes':[x for x in out if x['target_slots_sum'] or x['weights'] or x['target_names']]}


def main():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    out=Path(argv[0] if argv else 'morph_probe');out.mkdir(parents=True,exist_ok=True)
    head=bpy.data.objects.get('AINA_Face_v15_5')
    if not head or head.type!='MESH': raise RuntimeError('AINA_Face_v15_5 not found')
    keys=list(head.data.shape_keys.key_blocks) if head.data.shape_keys else []
    info={'object':head.name,'parent':head.parent.name if head.parent else None,'parent_type':head.parent_type,'parent_bone':head.parent_bone,'modifier_types':[m.type for m in head.modifiers],'shape_key_count_including_basis':len(keys),'shape_key_names':[k.name for k in keys]}
    print('HEAD_INFO='+json.dumps(info),flush=True)
    if len(keys)!=53: raise RuntimeError(f'Expected Basis + 52 shape keys, got {len(keys)}')

    cp=head.copy();cp.data=head.data.copy();cp.name='AINA_Face_CopyProbe';bpy.context.collection.objects.link(cp)
    cp_keys=list(cp.data.shape_keys.key_blocks) if cp.data.shape_keys else []
    print('COPY_INFO='+json.dumps({'keys':len(cp_keys),'names':[k.name for k in cp_keys[:5]]}),flush=True)

    # Native glTF exporter probe. If this contains morph targets, Blender shape key
    # data itself is valid and loss occurs in VRM preprocessing/export selection.
    bpy.ops.object.select_all(action='DESELECT');head.hide_set(False);head.select_set(True);bpy.context.view_layer.objects.active=head
    raw=out/'raw_head.glb'
    r=bpy.ops.export_scene.gltf(filepath=str(raw),check_existing=False,export_format='GLB',use_selection=True,export_animations=False,export_morph=True,export_apply=False,export_skins=False,export_extras=True)
    print('RAW_GLTF_RESULT='+repr(r),flush=True)
    raw_summary=summarize(raw);print('RAW_GLTF_SUMMARY='+json.dumps(raw_summary),flush=True)

    bpy.ops.object.select_all(action='DESELECT');cp.hide_set(False);cp.select_set(True);bpy.context.view_layer.objects.active=cp
    copied=out/'copied_head.glb'
    r=bpy.ops.export_scene.gltf(filepath=str(copied),check_existing=False,export_format='GLB',use_selection=True,export_animations=False,export_morph=True,export_apply=False,export_skins=False,export_extras=True)
    print('COPY_GLTF_RESULT='+repr(r),flush=True)
    copy_summary=summarize(copied);print('COPY_GLTF_SUMMARY='+json.dumps(copy_summary),flush=True)

    report={'head':info,'copy_shape_keys':len(cp_keys),'raw':raw_summary,'copy':copy_summary}
    (out/'MORPH_PROBE.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
