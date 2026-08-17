#!/usr/bin/env python3
"""glTF-2.0-compliant AINA morph-target repair.

Uses the deterministic geometry and shape algorithms from
patch_aina_vrm_morph_targets.py, but emits zero-only primitive targets as
implicit-zero accessors (no bufferView and no sparse block). glTF sparse.count
must be >= 1, so this keeps all 52 target slots valid on every material split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import patch_aina_vrm_morph_targets as p


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--vrm',type=Path,required=True);ap.add_argument('--locked-obj',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--report',type=Path);a=ap.parse_args()
    g,bin_data=p.parse_glb(a.vrm);v,f=p.load_obj(a.locked_obj)
    base=np.empty_like(v);base[:,0]=v[:,0]*1.08;base[:,1]=v[:,2]*1.08;base[:,2]=-v[:,1]*1.08;base[:,2]+=1.72-float(base[:,2].max())
    cs=p.components(len(v),f);head=max(cs,key=len);eyes=[q for q in cs if 650<len(q)<900];oral=sorted([q for q in cs if not np.array_equal(q,head) and all(not np.array_equal(q,e) for e in eyes)],key=len,reverse=True);tongue=oral[-1] if oral else np.array([],dtype=np.int64)
    deltas=p.shape_deltas(base,tongue);canonical=np.column_stack([base[:,0],base[:,2],-base[:,1]]);tree=cKDTree(canonical)
    mesh_index=next((i for i,m in enumerate(g.get('meshes',[])) if m.get('name')=='AINA_Face_v15_5_Mesh'),None)
    if mesh_index is None:raise RuntimeError('AINA face mesh missing from VRM')
    node_index=next((i for i,n in enumerate(g.get('nodes',[])) if n.get('mesh')==mesh_index),None)
    if node_index is None:raise RuntimeError('AINA face node missing from VRM')
    mesh=g['meshes'][mesh_index];maps=[];mapping_max=0.0
    for primitive in mesh.get('primitives',[]):
        pos=p.accessor_array(g,bin_data,primitive['attributes']['POSITION']);dist,idx=tree.query(pos,k=1);mapping_max=max(mapping_max,float(dist.max()));maps.append(idx);primitive['targets']=[]
    if mapping_max>2e-6:raise RuntimeError(f'Face vertex mapping error too large: {mapping_max}')
    mesh['weights']=[0.0]*len(p.SHAPE_KEYS);extras=mesh.get('extras') or {};extras['targetNames']=list(p.SHAPE_KEYS);mesh['extras']=extras
    sparse_counts={}
    for name in p.SHAPE_KEYS:
        d=deltas[name];dg=np.column_stack([d[:,0],d[:,2],-d[:,1]]).astype(np.float32);sparse_counts[name]=[]
        for primitive,indices in zip(mesh['primitives'],maps):
            values=dg[indices];nz=np.flatnonzero(np.linalg.norm(values,axis=1)>1e-9).astype(np.uint32);sparse_counts[name].append(int(len(nz)))
            accessor={'componentType':5126,'count':int(len(indices)),'type':'VEC3','min':[0.0,0.0,0.0],'max':[0.0,0.0,0.0]}
            if len(nz):
                ibv=p.append_view(g,bin_data,nz.astype('<u4').tobytes());vbv=p.append_view(g,bin_data,np.ascontiguousarray(values[nz].astype('<f4')).tobytes())
                accessor['min']=np.minimum(0,values[nz].min(axis=0)).astype(float).tolist();accessor['max']=np.maximum(0,values[nz].max(axis=0)).astype(float).tolist()
                accessor['sparse']={'count':int(len(nz)),'indices':{'bufferView':ibv,'componentType':5125},'values':{'bufferView':vbv}}
            accessor_index=len(g.setdefault('accessors',[]));g['accessors'].append(accessor);primitive['targets'].append({'POSITION':accessor_index})
    presets=g['extensions']['VRMC_vrm']['expressions']['preset'];name_to_index={name:i for i,name in enumerate(p.SHAPE_KEYS)}
    for preset_name,items in p.PRESET_BINDS.items():
        vrm_name=p.VRM_PRESET_NAME.get(preset_name,preset_name);expr=presets.setdefault(vrm_name,{'isBinary':False,'overrideBlink':'none','overrideLookAt':'none','overrideMouth':'none'});expr['morphTargetBinds']=[{'node':int(node_index),'index':int(name_to_index[key]),'weight':float(weight)} for key,weight in items]
    p.pack_glb(g,bin_data,a.out)
    bind_counts={k:len(v.get('morphTargetBinds',[])) for k,v in presets.items()};zero_targets=sum(1 for counts in sparse_counts.values() for count in counts if count==0)
    report={'pass':True,'glTF_sparse_zero_count_blocks':0,'implicit_zero_target_accessors':zero_targets,'source_vrm_bytes':a.vrm.stat().st_size,'patched_vrm_bytes':a.out.stat().st_size,'face_mesh_index':mesh_index,'face_node_index':node_index,'source_vertex_mapping_max_error_m':mapping_max,'shape_key_count':len(p.SHAPE_KEYS),'primitive_target_counts':[len(pr.get('targets',[])) for pr in mesh['primitives']],'preset_count':len(p.PRESET_BINDS),'preset_morph_bind_total':sum(bind_counts.values()),'preset_bind_counts':bind_counts,'sparse_nonzero_vertex_counts':sparse_counts}
    report_path=a.report or a.out.with_suffix('.morph_patch.json');report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
    if report['primitive_target_counts']!=[52]*len(mesh['primitives']) or report['preset_morph_bind_total']<=0:raise RuntimeError('Morph patch validation failed')

if __name__=='__main__':main()
