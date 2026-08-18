#!/usr/bin/env python3
"""Deterministically restore AINA's verified 52 morph targets into the final VRM.

Inputs are two already-proven artifacts:
1) AINA.vrm from final production (rig, Humanoid, LookAt, SpringBone, materials)
2) raw_head.glb from the 52-target Blender glTF probe.

The face primitives share exact POSITION/NORMAL vertex identities. The VRM
exporter's preprocessing only collapses a small number of duplicate vertices.
Those duplicates have identical deltas across all 52 targets, so the morph data
can be mapped losslessly to the final VRM vertex streams.
"""
from __future__ import annotations
import argparse
import collections
import json
import struct
from pathlib import Path

import numpy as np

COMP_DTYPE={5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
TYPE_N={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT2':4,'MAT3':9,'MAT4':16}

SHAPE_KEYS=[
    'browDownLeft','browDownRight','browInnerUp','browOuterUpLeft','browOuterUpRight',
    'cheekPuff','cheekSquintLeft','cheekSquintRight',
    'eyeBlinkLeft','eyeBlinkRight','eyeLookDownLeft','eyeLookDownRight','eyeLookInLeft','eyeLookInRight','eyeLookOutLeft','eyeLookOutRight','eyeLookUpLeft','eyeLookUpRight','eyeSquintLeft','eyeSquintRight','eyeWideLeft','eyeWideRight',
    'jawForward','jawLeft','jawOpen','jawRight',
    'mouthClose','mouthDimpleLeft','mouthDimpleRight','mouthFrownLeft','mouthFrownRight','mouthFunnel','mouthLeft','mouthLowerDownLeft','mouthLowerDownRight','mouthPressLeft','mouthPressRight','mouthPucker','mouthRight','mouthRollLower','mouthRollUpper','mouthShrugLower','mouthShrugUpper','mouthSmileLeft','mouthSmileRight','mouthStretchLeft','mouthStretchRight','mouthUpperUpLeft','mouthUpperUpRight',
    'noseSneerLeft','noseSneerRight','tongueOut',
]
PRESET_BINDS={
    'happy':[('mouthSmileLeft',.82),('mouthSmileRight',.82),('cheekSquintLeft',.30),('cheekSquintRight',.30)],
    'angry':[('browDownLeft',.82),('browDownRight',.82),('mouthFrownLeft',.45),('mouthFrownRight',.45)],
    'sad':[('browInnerUp',.72),('mouthFrownLeft',.72),('mouthFrownRight',.72)],
    'relaxed':[('mouthSmileLeft',.22),('mouthSmileRight',.22),('eyeSquintLeft',.10),('eyeSquintRight',.10)],
    'surprised':[('browInnerUp',.55),('eyeWideLeft',.86),('eyeWideRight',.86),('jawOpen',.58)],
    'neutral':[],
    'aa':[('jawOpen',.72),('mouthFunnel',.20)],
    'ih':[('mouthStretchLeft',.62),('mouthStretchRight',.62)],
    'ou':[('mouthPucker',.78),('mouthFunnel',.55)],
    'ee':[('mouthSmileLeft',.38),('mouthSmileRight',.38),('mouthStretchLeft',.52),('mouthStretchRight',.52)],
    'oh':[('jawOpen',.48),('mouthFunnel',.78)],
    'blink':[('eyeBlinkLeft',1.0),('eyeBlinkRight',1.0)],
    'blinkLeft':[('eyeBlinkLeft',1.0)],
    'blinkRight':[('eyeBlinkRight',1.0)],
    'lookUp':[('eyeLookUpLeft',1.0),('eyeLookUpRight',1.0)],
    'lookDown':[('eyeLookDownLeft',1.0),('eyeLookDownRight',1.0)],
    'lookLeft':[('eyeLookOutLeft',.72),('eyeLookInRight',.72)],
    'lookRight':[('eyeLookInLeft',.72),('eyeLookOutRight',.72)],
}


def args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-vrm',type=Path,required=True)
    ap.add_argument('--morph-glb',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    return ap.parse_args()


def parse_glb(path:Path):
    data=path.read_bytes()
    if len(data)<20: raise RuntimeError(f'GLB too small: {path}')
    magic,ver,total=struct.unpack_from('<4sII',data,0)
    if magic!=b'glTF' or ver!=2 or total!=len(data):
        raise RuntimeError(f'Invalid GLB header for {path}')
    off=12; doc=None; binb=None
    while off+8<=len(data):
        n,t=struct.unpack_from('<II',data,off); off+=8
        chunk=data[off:off+n]; off+=n
        if t==0x4E4F534A: doc=json.loads(chunk.rstrip(b' \x00').decode('utf-8'))
        elif t==0x004E4942: binb=chunk
    if doc is None or binb is None: raise RuntimeError(f'Missing JSON/BIN chunk: {path}')
    return doc,binb


def accessor(doc,binb,idx):
    a=doc['accessors'][idx]
    dtype=np.dtype(COMP_DTYPE[a['componentType']]); ncomp=TYPE_N[a['type']]; count=a['count']
    arr=np.zeros((count,ncomp),dtype=dtype)
    if 'bufferView' in a:
        bv=doc['bufferViews'][a['bufferView']]
        off=bv.get('byteOffset',0)+a.get('byteOffset',0)
        stride=bv.get('byteStride',dtype.itemsize*ncomp)
        if stride==dtype.itemsize*ncomp:
            arr[:]=np.frombuffer(binb,dtype=dtype,count=count*ncomp,offset=off).reshape(count,ncomp)
        else:
            arr[:]=np.ndarray((count,ncomp),dtype=dtype,buffer=binb,offset=off,strides=(stride,dtype.itemsize))
    sp=a.get('sparse')
    if sp:
        sc=sp['count']; si=sp['indices']; sv=sp['values']
        ibv=doc['bufferViews'][si['bufferView']]
        idtype=np.dtype(COMP_DTYPE[si['componentType']])
        ioff=ibv.get('byteOffset',0)+si.get('byteOffset',0)
        inds=np.frombuffer(binb,dtype=idtype,count=sc,offset=ioff)
        vbv=doc['bufferViews'][sv['bufferView']]
        voff=vbv.get('byteOffset',0)+sv.get('byteOffset',0)
        vals=np.frombuffer(binb,dtype=dtype,count=sc*ncomp,offset=voff).reshape(sc,ncomp)
        arr[inds]=vals
    return arr


def append_aligned(buf:bytearray,data:bytes):
    while len(buf)%4: buf.append(0)
    off=len(buf); buf.extend(data); return off,len(data)


def sparse_vec3(doc,buf,arr):
    arr=np.asarray(arr,np.float32)
    nz=np.flatnonzero(np.any(arr!=0.0,axis=1))
    acc={'componentType':5126,'count':int(len(arr)),'type':'VEC3','min':[float(x) for x in arr.min(axis=0)],'max':[float(x) for x in arr.max(axis=0)]}
    if len(nz):
        inds=nz.astype(np.uint32); vals=np.ascontiguousarray(arr[nz],dtype=np.float32)
        ioff,ilen=append_aligned(buf,inds.tobytes())
        ibv=len(doc.setdefault('bufferViews',[]))
        doc['bufferViews'].append({'buffer':0,'byteOffset':ioff,'byteLength':ilen})
        voff,vlen=append_aligned(buf,vals.tobytes())
        vbv=len(doc['bufferViews'])
        doc['bufferViews'].append({'buffer':0,'byteOffset':voff,'byteLength':vlen})
        acc['sparse']={'count':int(len(nz)),'indices':{'bufferView':ibv,'componentType':5125},'values':{'bufferView':vbv}}
    idx=len(doc.setdefault('accessors',[])); doc['accessors'].append(acc)
    return idx,int(len(nz))


def pack_glb(doc,binb,path):
    doc['buffers'][0]['byteLength']=len(binb)
    j=json.dumps(doc,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    while len(j)%4: j+=b' '
    b=bytearray(binb)
    while len(b)%4: b.append(0)
    total=12+8+len(j)+8+len(b)
    out=bytearray(struct.pack('<4sII',b'glTF',2,total))
    out+=struct.pack('<II',len(j),0x4E4F534A)+j
    out+=struct.pack('<II',len(b),0x004E4942)+b
    path.write_bytes(out)


def main():
    a=args(); a.out.mkdir(parents=True,exist_ok=True); (a.out/'QA').mkdir(exist_ok=True)
    doc,binb=parse_glb(a.base_vrm)
    rdoc,rbin=parse_glb(a.morph_glb)
    buf=bytearray(binb)

    face_matches=[(i,m) for i,m in enumerate(doc.get('meshes',[])) if m.get('name')=='AINA_Face_v15_5_Mesh']
    if len(face_matches)!=1: raise RuntimeError(f'Expected one final face mesh, got {len(face_matches)}')
    face_i,face=face_matches[0]
    raw=rdoc['meshes'][0]
    names=list((raw.get('extras') or {}).get('targetNames') or [])
    if names!=SHAPE_KEYS: raise RuntimeError('Probe target names are not the canonical 52 AINA controls')
    if len(raw.get('primitives',[]))!=3 or len(face.get('primitives',[]))!=3:
        raise RuntimeError('Unexpected face primitive count')

    primitive_stats=[]
    for pi,(rp,fp) in enumerate(zip(raw['primitives'],face['primitives'])):
        rpos=accessor(rdoc,rbin,rp['attributes']['POSITION']).astype(np.float32)
        rno=accessor(rdoc,rbin,rp['attributes']['NORMAL']).astype(np.float32)
        fpos=accessor(doc,bytes(buf),fp['attributes']['POSITION']).astype(np.float32)
        fno=accessor(doc,bytes(buf),fp['attributes']['NORMAL']).astype(np.float32)

        lookup=collections.defaultdict(list)
        for i in range(len(rpos)): lookup[(rpos[i].tobytes(),rno[i].tobytes())].append(i)
        mapping=[]; missing=[]
        for j in range(len(fpos)):
            candidates=lookup.get((fpos[j].tobytes(),fno[j].tobytes()),[])
            if not candidates: missing.append(j); mapping.append(-1)
            else: mapping.append(candidates[0])
        if missing: raise RuntimeError(f'Primitive {pi}: {len(missing)} final vertices cannot map to probe')
        mapping=np.asarray(mapping,np.int64)

        duplicate_groups=[c for c in lookup.values() if len(c)>1]
        targets=[]; pos_sparse=[]; normal_sparse=[]
        for target in rp['targets']:
            pfull=accessor(rdoc,rbin,target['POSITION']).astype(np.float32)
            nfull=accessor(rdoc,rbin,target['NORMAL']).astype(np.float32)
            for candidates in duplicate_groups:
                if np.any(pfull[candidates]!=pfull[candidates[0]]):
                    raise RuntimeError(f'Primitive {pi}: duplicate vertex has divergent POSITION morph delta')
                if np.any(nfull[candidates]!=nfull[candidates[0]]):
                    raise RuntimeError(f'Primitive {pi}: duplicate vertex has divergent NORMAL morph delta')
            out={}
            pidx,pnz=sparse_vec3(doc,buf,pfull[mapping])
            nidx,nnz=sparse_vec3(doc,buf,nfull[mapping])
            out['POSITION']=pidx; out['NORMAL']=nidx
            pos_sparse.append(pnz); normal_sparse.append(nnz); targets.append(out)
        fp['targets']=targets
        primitive_stats.append({
            'primitive':pi,'raw_vertices':len(rpos),'final_vertices':len(fpos),
            'removed_duplicate_vertices':len(rpos)-len(fpos),
            'position_sparse_rows':pos_sparse,'normal_sparse_rows':normal_sparse,
        })

    face['weights']=[0.0]*52
    extras=face.get('extras') if isinstance(face.get('extras'),dict) else {}
    extras['targetNames']=SHAPE_KEYS
    face['extras']=extras

    face_nodes=[i for i,n in enumerate(doc.get('nodes',[])) if n.get('mesh')==face_i]
    if len(face_nodes)!=1: raise RuntimeError(f'Expected one face node, got {face_nodes}')
    face_node=face_nodes[0]
    preset=doc['extensions']['VRMC_vrm'].setdefault('expressions',{}).setdefault('preset',{})
    name_to_index={name:i for i,name in enumerate(SHAPE_KEYS)}
    for pname,items in PRESET_BINDS.items():
        expr=preset.setdefault(pname,{})
        expr['morphTargetBinds']=[
            {'node':face_node,'index':name_to_index[key],'weight':float(weight)}
            for key,weight in items
        ]

    out_vrm=a.out/'AINA.vrm'
    pack_glb(doc,bytes(buf),out_vrm)

    check,cbin=parse_glb(out_vrm)
    cm=check['meshes'][face_i]
    target_counts=[len(p.get('targets') or []) for p in cm['primitives']]
    target_names=list((cm.get('extras') or {}).get('targetNames') or [])
    weights=cm.get('weights') or []
    cpreset=check['extensions']['VRMC_vrm']['expressions']['preset']
    bind_counts={k:len((cpreset.get(k) or {}).get('morphTargetBinds') or []) for k in PRESET_BINDS}
    expected_counts={k:len(v) for k,v in PRESET_BINDS.items()}
    qa={
        'product':'AINA Final VRM Deterministic Morph Restore',
        'binary_pass':(
            len(target_names)==52 and target_names==SHAPE_KEYS and len(weights)==52 and
            target_counts==[52,52,52] and bind_counts==expected_counts
        ),
        'vrm_bytes':out_vrm.stat().st_size,
        'shape_controls':len(target_names),
        'primitive_target_counts':target_counts,
        'preset_count':len(bind_counts),
        'preset_bind_counts':bind_counts,
        'expected_preset_bind_counts':expected_counts,
        'total_preset_morph_binds':sum(bind_counts.values()),
        'face_mesh_index':face_i,'face_node_index':face_node,
        'primitive_mapping':primitive_stats,
    }
    (a.out/'QA'/'AINA_VRM_BINARY_MORPH_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps(qa,indent=2))
    if not qa['binary_pass']: raise RuntimeError('Patched VRM binary gate failed')


if __name__=='__main__':
    main()
