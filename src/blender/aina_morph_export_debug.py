#!/usr/bin/env python3
from __future__ import annotations
import json,struct,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import aina_final_vrm_release_addonfix as entry
core=entry.release.core

def no_render(out):
    p=Path(out)/'Preview';p.mkdir(parents=True,exist_ok=True)
    return []
core.setup_render=no_render
orig=core.create_shape_keys

def wrapped(head,base,tongue):
    stats=orig(head,base,tongue)
    kb=head.data.shape_keys.key_blocks;basis=np.array([p.co[:] for p in kb['Basis'].data],float)
    actual={}
    for k in kb:
        if k.name=='Basis':continue
        co=np.array([p.co[:] for p in k.data],float);d=np.linalg.norm(co-basis,axis=1);actual[k.name]={'max_m':float(d.max()),'moved':int((d>1e-6).sum())}
    print('AINA_PREEXPORT_SHAPE_COUNT',len(actual));print('AINA_PREEXPORT_SHAPES',json.dumps(actual,sort_keys=True))
    return stats
core.create_shape_keys=wrapped

def parse_glb(path):
    b=Path(path).read_bytes();magic,ver,total=struct.unpack_from('<4sII',b,0);assert magic==b'glTF' and ver==2
    off=12;j=None
    while off<total:
        n,t=struct.unpack_from('<II',b,off);off+=8;chunk=b[off:off+n];off+=n
        if t==0x4E4F534A:j=json.loads(chunk.rstrip(b' \t\r\n\x00').decode())
    assert j is not None;return j

if __name__=='__main__':
    core.main()
    # read --out from argv
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    out=Path(argv[argv.index('--out')+1]) if '--out' in argv else Path('output_debug')
    j=parse_glb(out/'AINA.vrm');meshes=j.get('meshes',[]);nodes=j.get('nodes',[])
    report={'mesh_count':len(meshes),'nodes':[]}
    for i,n in enumerate(nodes):
        name=n.get('name','');mi=n.get('mesh')
        if name=='AINA_Face_v15_5' or (isinstance(mi,int) and 0<=mi<len(meshes) and 'AINA_Face' in meshes[mi].get('name','')):
            m=meshes[mi];report['nodes'].append({'node_index':i,'node_name':name,'mesh_index':mi,'mesh_name':m.get('name'),'target_names':m.get('extras',{}).get('targetNames',[]),'primitive_target_counts':[len(p.get('targets',[])) for p in m.get('primitives',[])]})
    ext=j.get('extensions',{}).get('VRMC_vrm',{});expr=ext.get('expressions',{}).get('preset',{});report['preset_bind_counts']={k:len(v.get('morphTargetBinds',[])) for k,v in expr.items()}
    (out/'QA'/'AINA_RAW_VRM_MORPH_DEBUG.json').write_text(json.dumps(report,indent=2));print('AINA_RAW_VRM_MORPH_DEBUG',json.dumps(report))
