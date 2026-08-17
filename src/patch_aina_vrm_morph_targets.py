#!/usr/bin/env python3
"""Restore AINA's deterministic 52 facial morph targets into an exported VRM.

The Blender master contains all 52 Shape Keys, but the VRM Addon v4.5 export
path used by this project can emit the face mesh without glTF morph targets.
This production repair is deterministic: it rebuilds the exact same deltas from
the locked v15.5 OBJ, maps exported glTF face vertices back to their source
vertices by position, writes sparse POSITION morph accessors, and restores all
18 VRM preset morphTargetBinds. No face geometry or identity is changed.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

K=np.array([1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,13695,12973,13422,6543,6537],dtype=np.int64)

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
    'blink_left':[('eyeBlinkLeft',1.0)],
    'blink_right':[('eyeBlinkRight',1.0)],
    'look_up':[('eyeLookUpLeft',1.0),('eyeLookUpRight',1.0)],
    'look_down':[('eyeLookDownLeft',1.0),('eyeLookDownRight',1.0)],
    'look_left':[('eyeLookOutLeft',.72),('eyeLookInRight',.72)],
    'look_right':[('eyeLookInLeft',.72),('eyeLookOutRight',.72)],
}
VRM_PRESET_NAME={'blink_left':'blinkLeft','blink_right':'blinkRight','look_up':'lookUp','look_down':'lookDown','look_left':'lookLeft','look_right':'lookRight'}


def components(nv,faces):
    e=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    a=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(nv,nv));a=(a+a.T).tocsr()
    n,lab=connected_components(a,directed=False)
    return [np.flatnonzero(lab==i) for i in range(n)]


def weights(coords,c,r,inner=.25,outer=1.20):
    c=np.asarray(c,float);r=np.asarray(r,float);q=np.sqrt(np.sum(((coords-c)/r)**2,axis=1));w=np.zeros(len(coords));w[q<=inner]=1
    m=(q>inner)&(q<outer)
    if np.any(m):
        t=(q[m]-inner)/(outer-inner);w[m]=.5*(1+np.cos(np.pi*t))
    return w


def shift_region(coords,c,r,d,inner=.25,outer=1.20):
    coords += weights(coords,c,r,inner,outer)[:,None]*np.asarray(d,float)


def scale_region(coords,c,r,s,inner=.25,outer=1.20):
    w=weights(coords,c,r,inner,outer)[:,None];c=np.asarray(c,float);target=c+(coords-c)*np.asarray(s,float);coords += w*(target-coords)


def shape_deltas(base,tongue_ids):
    lm=base[K];browL=lm[22:27].mean(0);browR=lm[17:22].mean(0);eyeL=lm[42:48].mean(0);eyeR=lm[36:42].mean(0)
    mouth=lm[48:60].mean(0);chin=lm[8];jaw=(mouth+chin)/2;cornerL=lm[54];cornerR=lm[48]
    upperL=lm[[52,53,54]].mean(0);upperR=lm[[48,49,50]].mean(0);lowerL=lm[[54,55,56]].mean(0);lowerR=lm[[48,58,59]].mean(0)
    cheekL=(eyeL+lm[35]+cornerL)/3;cheekR=(eyeR+lm[31]+cornerR)/3;noseL=lm[35];noseR=lm[31]
    result={}
    for name in SHAPE_KEYS:
        c=base.copy();side=1 if 'Left' in name else (-1 if 'Right' in name else 0)
        eye=eyeL if side==1 else eyeR;brow=browL if side==1 else browR;corner=cornerL if side==1 else cornerR;cheek=cheekL if side==1 else cheekR
        if name.startswith('browDown'):shift_region(c,brow,(.035,.025,.022),(0,0,-.005))
        elif name=='browInnerUp':
            for cc in (lm[21],lm[22]):shift_region(c,cc,(.022,.022,.022),(0,0,.006))
        elif name.startswith('browOuterUp'):shift_region(c,lm[[25,26]].mean(0) if side==1 else lm[[17,18]].mean(0),(.025,.022,.022),(0,0,.005))
        elif name=='cheekPuff':shift_region(c,cheekL,(.040,.035,.040),(.001,-.0045,.0005));shift_region(c,cheekR,(.040,.035,.040),(-.001,-.0045,.0005))
        elif name.startswith('cheekSquint'):shift_region(c,cheek,(.035,.032,.030),(0,-.001,.0030))
        elif name.startswith('eyeBlink'):scale_region(c,eye,(.037,.026,.020),(1,1,.08))
        elif name.startswith('eyeSquint'):scale_region(c,eye,(.037,.026,.021),(1,1,.55));shift_region(c,cheek,(.032,.030,.026),(0,-.0007,.0017))
        elif name.startswith('eyeWide'):scale_region(c,eye,(.037,.026,.021),(1,1,1.30))
        elif name.startswith('eyeLook'):
            direction=np.zeros(3);direction[0]=(.0035 if ('OutLeft' in name or 'InRight' in name) else (-.0035 if ('InLeft' in name or 'OutRight' in name) else 0));direction[2]=(.003 if 'Up' in name else (-.003 if 'Down' in name else 0));shift_region(c,eye,(.030,.022,.019),direction)
        elif name=='jawForward':shift_region(c,jaw,(.050,.055,.045),(0,-.004,0))
        elif name=='jawLeft':shift_region(c,jaw,(.055,.060,.048),(.004,0,0))
        elif name=='jawOpen':shift_region(c,jaw,(.050,.060,.048),(0,0,-.010));shift_region(c,mouth,(.040,.035,.030),(0,0,-.005))
        elif name=='jawRight':shift_region(c,jaw,(.055,.060,.048),(-.004,0,0))
        elif name=='mouthClose':scale_region(c,mouth,(.040,.028,.025),(1,1,.25))
        elif name.startswith('mouthDimple'):shift_region(c,corner,(.026,.025,.022),(.0018*side,.0012,-.0010))
        elif name.startswith('mouthFrown'):shift_region(c,corner,(.027,.025,.022),(.0008*side,.0005,-.0040))
        elif name=='mouthFunnel':scale_region(c,mouth,(.040,.030,.030),(.72,1.05,1.05));shift_region(c,mouth,(.038,.030,.028),(0,-.003,0))
        elif name=='mouthLeft':shift_region(c,mouth,(.045,.030,.028),(.005,0,0))
        elif name.startswith('mouthLowerDown'):shift_region(c,lowerL if side==1 else lowerR,(.030,.025,.022),(0,0,-.0035))
        elif name.startswith('mouthPress'):scale_region(c,corner,(.026,.024,.020),(.96,1,.68))
        elif name=='mouthPucker':scale_region(c,mouth,(.040,.030,.030),(.76,1.0,1.15));shift_region(c,mouth,(.036,.030,.026),(0,-.0035,0))
        elif name=='mouthRight':shift_region(c,mouth,(.045,.030,.028),(-.005,0,0))
        elif name=='mouthRollLower':shift_region(c,lm[[55,56,57,58,59]].mean(0),(.038,.024,.020),(0,.002,.001))
        elif name=='mouthRollUpper':shift_region(c,lm[[49,50,51,52,53]].mean(0),(.038,.024,.020),(0,.002,-.001))
        elif name=='mouthShrugLower':shift_region(c,lm[[55,56,57,58,59]].mean(0),(.038,.024,.021),(0,0,.003))
        elif name=='mouthShrugUpper':shift_region(c,lm[[49,50,51,52,53]].mean(0),(.038,.024,.021),(0,0,.003))
        elif name.startswith('mouthSmile'):shift_region(c,corner,(.025,.025,.022),(.0018*side,-.0004,.0042))
        elif name.startswith('mouthStretch'):shift_region(c,corner,(.026,.025,.022),(.0042*side,0,0))
        elif name.startswith('mouthUpperUp'):shift_region(c,upperL if side==1 else upperR,(.028,.023,.020),(0,0,.0032))
        elif name.startswith('noseSneer'):
            nc=noseL if side==1 else noseR;shift_region(c,nc,(.022,.023,.024),(.0006*side,-.0012,.0030))
        elif name=='tongueOut':
            if len(tongue_ids):c[tongue_ids]+=np.array([0,-.008,-.003])
            else:shift_region(c,mouth,(.030,.030,.020),(0,-.006,-.002))
        else:raise RuntimeError(f'Unhandled shape key {name}')
        result[name]=c-base
    return result


def load_obj(path):
    verts=[];faces=[]
    for line in Path(path).read_text(errors='ignore').splitlines():
        if line.startswith('v '):
            q=line.split();verts.append((float(q[1]),float(q[2]),float(q[3])))
        elif line.startswith('f '):
            ids=[int(x.split('/')[0])-1 for x in line.split()[1:]]
            for i in range(1,len(ids)-1):faces.append((ids[0],ids[i],ids[i+1]))
    return np.asarray(verts,float),np.asarray(faces,np.int64)


def parse_glb(path):
    data=Path(path).read_bytes()
    if data[:4]!=b'glTF':raise RuntimeError('Input is not a GLB/VRM')
    magic,version,total=struct.unpack_from('<4sII',data,0)
    if version!=2 or total!=len(data):raise RuntimeError('Invalid GLB header')
    off=12;chunks=[]
    while off+8<=len(data):
        length,typ=struct.unpack_from('<II',data,off);off+=8;chunk=data[off:off+length];off+=length;chunks.append((typ,chunk))
    json_chunk=next(c for t,c in chunks if t==0x4E4F534A);bin_chunk=next(c for t,c in chunks if t==0x004E4942)
    return json.loads(json_chunk.rstrip(b'\x00 \t\r\n').decode('utf-8')),bytearray(bin_chunk)


def accessor_array(g,bin_data,index):
    a=g['accessors'][index];bv=g['bufferViews'][a['bufferView']];comps={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
    dtype={5126:np.float32,5125:np.uint32,5123:np.uint16,5121:np.uint8,5122:np.int16}[a['componentType']];item=np.dtype(dtype).itemsize*comps;offset=bv.get('byteOffset',0)+a.get('byteOffset',0);stride=bv.get('byteStride',item)
    if stride==item:return np.frombuffer(bin_data,dtype=dtype,count=a['count']*comps,offset=offset).reshape(a['count'],comps).copy()
    out=np.empty((a['count'],comps),dtype=dtype)
    for i in range(a['count']):out[i]=np.frombuffer(bin_data,dtype=dtype,count=comps,offset=offset+i*stride)
    return out


def append_view(g,bin_data,raw):
    while len(bin_data)%4:bin_data.append(0)
    offset=len(bin_data);bin_data.extend(raw)
    while len(bin_data)%4:bin_data.append(0)
    index=len(g.setdefault('bufferViews',[]));g['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':len(raw)})
    return index


def pack_glb(g,bin_data,path):
    g['buffers'][0]['byteLength']=len(bin_data)
    j=json.dumps(g,separators=(',',':'),ensure_ascii=False).encode('utf-8');j+=b' '*((-len(j))%4)
    b=bytes(bin_data);b+=b'\x00'*((-len(b))%4)
    total=12+8+len(j)+8+len(b);out=bytearray(struct.pack('<4sII',b'glTF',2,total));out+=struct.pack('<II',len(j),0x4E4F534A)+j;out+=struct.pack('<II',len(b),0x004E4942)+b
    Path(path).write_bytes(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--vrm',type=Path,required=True);ap.add_argument('--locked-obj',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--report',type=Path);a=ap.parse_args()
    g,bin_data=parse_glb(a.vrm);v,f=load_obj(a.locked_obj)
    base=np.empty_like(v);base[:,0]=v[:,0]*1.08;base[:,1]=v[:,2]*1.08;base[:,2]=-v[:,1]*1.08;base[:,2]+=1.72-float(base[:,2].max())
    cs=components(len(v),f);head=max(cs,key=len);eyes=[q for q in cs if 650<len(q)<900];oral=sorted([q for q in cs if not np.array_equal(q,head) and all(not np.array_equal(q,e) for e in eyes)],key=len,reverse=True);tongue=oral[-1] if oral else np.array([],dtype=np.int64)
    deltas=shape_deltas(base,tongue);canonical=np.column_stack([base[:,0],base[:,2],-base[:,1]]);tree=cKDTree(canonical)
    mesh_index=next((i for i,m in enumerate(g.get('meshes',[])) if m.get('name')=='AINA_Face_v15_5_Mesh'),None)
    if mesh_index is None:raise RuntimeError('AINA face mesh missing from VRM')
    node_index=next((i for i,n in enumerate(g.get('nodes',[])) if n.get('mesh')==mesh_index),None)
    if node_index is None:raise RuntimeError('AINA face node missing from VRM')
    mesh=g['meshes'][mesh_index];maps=[];mapping_max=0.0
    for primitive in mesh.get('primitives',[]):
        pos=accessor_array(g,bin_data,primitive['attributes']['POSITION']);dist,idx=tree.query(pos,k=1);mapping_max=max(mapping_max,float(dist.max()));maps.append(idx);primitive['targets']=[]
    if mapping_max>2e-6:raise RuntimeError(f'Face vertex mapping error too large: {mapping_max}')
    mesh['weights']=[0.0]*len(SHAPE_KEYS);extras=mesh.get('extras') or {};extras['targetNames']=list(SHAPE_KEYS);mesh['extras']=extras
    sparse_counts={}
    for name in SHAPE_KEYS:
        d=deltas[name];dg=np.column_stack([d[:,0],d[:,2],-d[:,1]]).astype(np.float32);sparse_counts[name]=[]
        for primitive,indices in zip(mesh['primitives'],maps):
            values=dg[indices];nz=np.flatnonzero(np.linalg.norm(values,axis=1)>1e-9).astype(np.uint32);sparse_counts[name].append(int(len(nz)))
            ibv=append_view(g,bin_data,nz.astype('<u4').tobytes());vbv=append_view(g,bin_data,np.ascontiguousarray(values[nz].astype('<f4')).tobytes())
            minimum=np.minimum(0,values[nz].min(axis=0) if len(nz) else np.zeros(3)).astype(float).tolist();maximum=np.maximum(0,values[nz].max(axis=0) if len(nz) else np.zeros(3)).astype(float).tolist()
            accessor_index=len(g.setdefault('accessors',[]));g['accessors'].append({'componentType':5126,'count':int(len(indices)),'type':'VEC3','min':minimum,'max':maximum,'sparse':{'count':int(len(nz)),'indices':{'bufferView':ibv,'componentType':5125},'values':{'bufferView':vbv}}});primitive['targets'].append({'POSITION':accessor_index})
    presets=g['extensions']['VRMC_vrm']['expressions']['preset'];name_to_index={name:i for i,name in enumerate(SHAPE_KEYS)}
    for preset_name,items in PRESET_BINDS.items():
        vrm_name=VRM_PRESET_NAME.get(preset_name,preset_name);expr=presets.setdefault(vrm_name,{'isBinary':False,'overrideBlink':'none','overrideLookAt':'none','overrideMouth':'none'});expr['morphTargetBinds']=[{'node':int(node_index),'index':int(name_to_index[key]),'weight':float(weight)} for key,weight in items]
    pack_glb(g,bin_data,a.out)
    bind_counts={k:len(v.get('morphTargetBinds',[])) for k,v in presets.items()};report={'pass':True,'source_vrm_bytes':a.vrm.stat().st_size,'patched_vrm_bytes':a.out.stat().st_size,'face_mesh_index':mesh_index,'face_node_index':node_index,'source_vertex_mapping_max_error_m':mapping_max,'shape_key_count':len(SHAPE_KEYS),'primitive_target_counts':[len(p.get('targets',[])) for p in mesh['primitives']],'preset_count':len(PRESET_BINDS),'preset_morph_bind_total':sum(bind_counts.values()),'preset_bind_counts':bind_counts,'sparse_nonzero_vertex_counts':sparse_counts}
    report_path=a.report or a.out.with_suffix('.morph_patch.json');report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
    if report['primitive_target_counts']!=[52]*len(mesh['primitives']) or report['preset_morph_bind_total']<=0:raise RuntimeError('Morph patch validation failed')

if __name__=='__main__':main()
