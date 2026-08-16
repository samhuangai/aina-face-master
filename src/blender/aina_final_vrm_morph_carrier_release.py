#!/usr/bin/env python3
"""Final AINA assembly entry point using a morph-preserving native GLB carrier."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import bpy
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
for p in (HERE,ROOT/'src'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import aina_final_vrm_release_addonfix as entry
from merge_aina_vrm_carrier import merge
core=entry.release.core

if os.environ.get('AINA_FAST_NO_PREVIEW')=='1':
    def no_preview(out):
        p=Path(out)/'Preview';p.mkdir(parents=True,exist_ok=True);return []
    core.setup_render=no_preview

def cli_out():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    if '--out' not in argv:raise RuntimeError('--out is required')
    return Path(argv[argv.index('--out')+1])

if __name__=='__main__':
    core.main()
    out=cli_out();semantic=out/'AINA_SEMANTIC_NO_MORPH.vrm';final=out/'AINA.vrm';native=out/'AINA_NATIVE_52_MORPH.glb'
    if semantic.exists():semantic.unlink()
    final.replace(semantic)
    result=bpy.ops.export_scene.gltf(filepath=str(native),check_existing=False,export_format='GLB',use_selection=False,export_extras=True,export_morph=True,export_morph_normal=False,export_morph_tangent=False,export_apply=False,export_animations=False)
    if result!={'FINISHED'}:raise RuntimeError(f'Native 52-morph carrier export failed: {result}')
    merge(semantic,native,final,out/'QA'/'AINA_VRM_MORPH_CARRIER_MERGE.json')
    qa_path=out/'QA'/'AINA_FINAL_ASSEMBLY_QA.json';qa=json.loads(qa_path.read_text());qa['files']['vrm']=str(final);qa['files']['vrm_bytes']=final.stat().st_size;qa['morph_carrier_merge']=True;qa['semantic_vrm_bytes']=semantic.stat().st_size;qa['native_morph_carrier_bytes']=native.stat().st_size;qa_path.write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print('[AINA_FINAL] Final merged VRM bytes:',final.stat().st_size,flush=True)
