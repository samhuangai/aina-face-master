#!/usr/bin/env python3
"""Restore the exact proven 52-morph face primitives into the final AINA VRM.

The final VRM exporter collapsed a few static duplicate face vertices only
because morph targets were absent. The known-good raw_head.glb keeps those
vertices split so their morph normal deltas remain lossless. This repair first
proves that collapsing the raw primitives reproduces the final VRM triangle
indices exactly, then restores the original unfused POSITION/NORMAL/indices and
all 52 morph targets while leaving the final VRM nodes, rig, materials,
Humanoid, LookAt and SpringBone data untouched.
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


def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-vrm',type=Path,required=True)
    ap.add_argument('--morph-glb',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    return ap.parse_args()


def parse_glb(path):
    data=Path(path).read_bytes()
    if len(data)<20: raise RuntimeError(f'GLB too small: {path}')
    magic,ver,total=struct.unpack_from('<4sII',data,0)
    if magic!=b'glTF' or ver!=2 or total!=len(data):
        raise RuntimeError(f'Invalid GLB header: {path}')
    off=12; doc=binb=None
    while off+8<=len(data):
        n,t=struct.unpack_from('<II',data,off); off+=8
        chunk=data[off:off+n]; off+=n
        if t==0x4E4F534A: doc=json.loads(chunk.rstrip(b' \x00').decode('utf-8'))
        elif t==0x004E4942: binb=chunk
    if doc is None or binb is None: raise RuntimeError(f'Missing JSON/BIN: {path}')
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
        ibv=doc['bufferViews'][si['bufferView']]; idt=np.dtype(COMP_DTYPE[si['componentType']])
        inds=np.frombuffer(binb,dtype=idt,count=sc,offset=ibv.get('byteOffset',0)+si.get('byteOffset',0))
        vbv=doc['bufferViews'][sv['bufferView']]
        vals=np.frombuffer(binb,dtype=dtype,count=sc*ncomp,offset=vbv.get('byteOffset',0)+sv.get('byteOffset',0)).reshape(sc,ncomp)
        arr[inds]=vals
    return arr


def append_aligned(buf,data):
    while len(buf)%4: buf.append(0)
    off=len(buf); buf.extend(data); return off,len(data)


def dense_accessor(doc,buf,arr,component_type,type_name,target=None,include_bounds=False):
    dtype=np.dtype(COMP_DTYPE[component_type])
    arr=np.ascontiguousarray(arr,dtype=dtype)
    if arr.ndim==1: arr=arr.reshape(-1,1)
    off,n=append_aligned(buf,arr.tobytes())
    bv={'buffer':0,'byteOffset':off,'byteLength':n}
    if target is not None: bv['target']=target
    bvi=len(doc.setdefault('bufferViews',[])); doc['bufferViews'].append(bv)
    a={'bufferView':bvi,'componentType':component_type,'count':int(len(arr)),'type':type_name}
    if include_bounds:
        a['min']=[float(x) if np.issubdtype(dtype,np.floating) else int(x) for x in arr.min(axis=0)]
        a['max']=[float(x) if np.issubdtype(dtype,np.floating) else int(x) for x in arr.max(axis=0)]
    ai=len(doc.setdefault('accessors',[])); doc['accessors'].append(a); return ai


def sparse_vec3(doc,buf,arr):
    arr=np.asarray(arr,np.float32)
    nz=np.flatnonzero(np.any(arr!=0.0,axis=1))
    acc={'componentType':5126,'count':int(len(arr)),'type':'VEC3',
         'min':[float(x) for x in arr.min(axis=0)],'max':[float(x) for x in arr.max(axis=0)]}
    if len(nz):
        inds=nz.astype(np.uint32); vals=np.ascontiguousarray(arr[nz],dtype=np.float32)
        ioff,ilen=append_aligned(buf,inds.tobytes())
        ibv=len(doc.setdefault('bufferViews',[])); doc['bufferViews'].append({'buffer':0,'byteOffset':ioff,'byteLength':ilen})
        voff,vlen=append_aligned(buf,vals.tobytes())
        vbv=len(doc['bufferViews']); doc['bufferViews'].append({'buffer':0,'byteOffset':voff,'byteLength':vlen})
        acc['sparse']={'count':int(len(nz)),'indices':{'bufferView':ibv,'componentType':5125},'values':{'bufferView':vbv}}
    ai=len(doc.setdefault('accessors',[])); doc['accessors'].append(acc); return ai,int(len(nz))


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
    Path(path).write_bytes(out)


def static_key(pos,no,i):
    return (pos[i].tobytes(),no[i].tobytes())


def main():
    a=parse_args(); a.out.mkdir(parents=True,exist_ok=True); (a.out/'QA').mkdir(exist_ok=True)
    doc,binb=parse_glb(a.base_vrm); rdoc,rbin=parse_glb(a.morph_glb); buf=bytearray(binb)
    face_matches=[(i,m) for i,m in enumerate(doc.get('meshes',[])) if m.get('name')=='AINA_Face_v15_5_Mesh']
    if len(face_matches)!=1: raise RuntimeError(f'Expected one face mesh, got {len(face_matches)}')
    face_i,face=face_matches[0]; raw=rdoc['meshes'][0]
    names=list((raw.get('extras') or {}).get('targetNames') or [])
    if names!=SHAPE_KEYS: raise RuntimeError('Probe target names are not canonical AINA 52')
    if len(raw.get('primitives',[]))!=3 or len(face.get('primitives',[]))!=3: raise RuntimeError('Unexpected primitive count')

    primitive_stats=[]
    for pi,(rp,fp) in enumerate(zip(raw['primitives'],face['primitives'])):
        if set(rp.get('attributes',{}))!={'POSITION','NORMAL'} or set(fp.get('attributes',{}))!={'POSITION','NORMAL'}:
            raise RuntimeError(f'Primitive {pi}: unexpected attributes raw={rp.get("attributes")} final={fp.get("attributes")}')
        rpos=accessor(rdoc,rbin,rp['attributes']['POSITION']).astype(np.float32)
        rno=accessor(rdoc,rbin,rp['attributes']['NORMAL']).astype(np.float32)
        fpos=accessor(doc,binb,fp['attributes']['POSITION']).astype(np.float32)
        fno=accessor(doc,binb,fp['attributes']['NORMAL']).astype(np.float32)
        ridx=accessor(rdoc,rbin,rp['indices']).reshape(-1).astype(np.int64)
        fidx=accessor(doc,binb,fp['indices']).reshape(-1).astype(np.int64)

        rlookup=collections.defaultdict(list); flookup=collections.defaultdict(list)
        for i in range(len(rpos)): rlookup[static_key(rpos,rno,i)].append(i)
        for i in range(len(fpos)): flookup[static_key(fpos,fno,i)].append(i)
        if set(rlookup)!=set(flookup): raise RuntimeError(f'Primitive {pi}: static vertex sets differ')
        raw_to_final=np.full(len(rpos),-1,dtype=np.int64)
        for k,rl in rlookup.items():
            fl=flookup[k]
            if len(fl)==1:
                for ri in rl: raw_to_final[ri]=fl[0]
            elif len(fl)==len(rl):
                for ri,fi in zip(rl,fl): raw_to_final[ri]=fi
            else:
                raise RuntimeError(f'Primitive {pi}: unsupported duplicate multiplicity raw={len(rl)} final={len(fl)}')
        if np.any(raw_to_final<0) or len(ridx)!=len(fidx) or not np.array_equal(raw_to_final[ridx],fidx):
            raise RuntimeError(f'Primitive {pi}: final topology is not the exact static-collapse of the 52-morph probe')

        pos_acc=dense_accessor(doc,buf,rpos,5126,'VEC3',target=34962,include_bounds=True)
        no_acc=dense_accessor(doc,buf,rno,5126,'VEC3',target=34962,include_bounds=False)
        raw_idx_meta=rdoc['accessors'][rp['indices']]
        idx_component=raw_idx_meta['componentType']
        raw_index_array=accessor(rdoc,rbin,rp['indices']).reshape(-1)
        idx_acc=dense_accessor(doc,buf,raw_index_array,idx_component,'SCALAR',target=34963,include_bounds=True)
        fp['attributes']={'POSITION':pos_acc,'NORMAL':no_acc}
        fp['indices']=idx_acc

        targets=[]; ps=[]; ns=[]
        for t in rp['targets']:
            pfull=accessor(rdoc,rbin,t['POSITION']).astype(np.float32)
            nfull=accessor(rdoc,rbin,t['NORMAL']).astype(np.float32)
            pa,pnz=sparse_vec3(doc,buf,pfull); na,nnz=sparse_vec3(doc,buf,nfull)
            targets.append({'POSITION':pa,'NORMAL':na}); ps.append(pnz); ns.append(nnz)
        fp['targets']=targets
        primitive_stats.append({
            'primitive':pi,'raw_vertices':int(len(rpos)),'final_collapsed_vertices':int(len(fpos)),
            'restored_vertices':int(len(rpos)),'restored_duplicate_vertices':int(len(rpos)-len(fpos)),
            'index_count':int(len(ridx)),'collapse_topology_exact':True,
            'position_sparse_rows':ps,'normal_sparse_rows':ns,
        })

    face['weights']=[0.0]*52
    extras=face.get('extras') if isinstance(face.get('extras'),dict) else {}
    extras['targetNames']=SHAPE_KEYS; face['extras']=extras
    face_nodes=[i for i,n in enumerate(doc.get('nodes',[])) if n.get('mesh')==face_i]
    if len(face_nodes)!=1: raise RuntimeError(f'Expected one face node, got {face_nodes}')
    face_node=face_nodes[0]
    preset=doc['extensions']['VRMC_vrm'].setdefault('expressions',{}).setdefault('preset',{})
    name_to_index={n:i for i,n in enumerate(SHAPE_KEYS)}
    for pname,items in PRESET_BINDS.items():
        expr=preset.setdefault(pname,{})
        expr['morphTargetBinds']=[{'node':face_node,'index':name_to_index[k],'weight':float(w)} for k,w in items]

    out_vrm=a.out/'AINA.vrm'; pack_glb(doc,bytes(buf),out_vrm)
    check,cbin=parse_glb(out_vrm); cm=check['meshes'][face_i]
    target_counts=[len(p.get('targets') or []) for p in cm['primitives']]
    target_names=list((cm.get('extras') or {}).get('targetNames') or [])
    weights=cm.get('weights') or []
    cpreset=check['extensions']['VRMC_vrm']['expressions']['preset']
    bind_counts={k:len((cpreset.get(k) or {}).get('morphTargetBinds') or []) for k in PRESET_BINDS}
    expected={k:len(v) for k,v in PRESET_BINDS.items()}
    restored_counts=[check['accessors'][p['attributes']['POSITION']]['count'] for p in cm['primitives']]
    expected_restored=[rdoc['accessors'][p['attributes']['POSITION']]['count'] for p in raw['primitives']]
    qa={
        'product':'AINA Final VRM Exact Primitive + Morph Restore',
        'binary_pass':len(target_names)==52 and target_names==SHAPE_KEYS and len(weights)==52 and target_counts==[52,52,52] and bind_counts==expected and restored_counts==expected_restored,
        'vrm_bytes':out_vrm.stat().st_size,'shape_controls':len(target_names),'primitive_target_counts':target_counts,
        'restored_vertex_counts':restored_counts,'expected_restored_vertex_counts':expected_restored,
        'preset_count':len(bind_counts),'preset_bind_counts':bind_counts,'expected_preset_bind_counts':expected,
        'total_preset_morph_binds':sum(bind_counts.values()),'face_mesh_index':face_i,'face_node_index':face_node,
        'primitive_restore':primitive_stats,
    }
    (a.out/'QA'/'AINA_VRM_BINARY_MORPH_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps(qa,indent=2))
    if not qa['binary_pass']: raise RuntimeError('Patched VRM binary gate failed')


if __name__=='__main__':
    main()
