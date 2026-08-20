#!/usr/bin/env python3
"""AINA topology-preserving soft-tissue surface polish.

Runs a bounded masked Taubin pass on the actual FaceVerse head surface after the
front/3Q identity corrections.  It smooths forehead, under-eye, apple-cheek and
cheek-to-jaw transitions while anchoring all 68 semantic identity vertices and
protecting eyelids, nose, lips, jaw silhouette, ears and low neck.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

K = np.array([
    1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,
    10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,
    10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,
    4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,
    6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,
    6817,6550,13695,12973,13422,6543,6537,
], dtype=np.int64)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def map_to_blender(v: np.ndarray, height=1.72):
    s = 1.08
    out = np.empty_like(v, dtype=np.float64)
    out[:,0] = v[:,0] * s; out[:,1] = v[:,2] * s; out[:,2] = -v[:,1] * s
    offset = height - float(out[:,2].max()); out[:,2] += offset
    return out, offset


def map_from_blender(v: np.ndarray, offset: float):
    s = 1.08
    out = np.empty_like(v, dtype=np.float64)
    out[:,0] = v[:,0] / s; out[:,2] = v[:,1] / s; out[:,1] = -(v[:,2] - offset) / s
    return out


def components(n: int, faces: np.ndarray):
    parent = np.arange(n, dtype=np.int64)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = int(parent[x])
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra != rb: parent[rb] = ra
    for a,b,c in faces:
        union(int(a),int(b)); union(int(b),int(c)); union(int(c),int(a))
    roots=np.array([find(i) for i in range(n)],dtype=np.int64)
    groups={}
    for i,r in enumerate(roots): groups.setdefault(int(r),[]).append(i)
    return roots,{r:np.asarray(ids,dtype=np.int64) for r,ids in groups.items()}


def adjacency(n: int, faces: np.ndarray, ids: np.ndarray):
    mask=np.zeros(n,bool); mask[ids]=True
    nbr=[set() for _ in range(n)]
    for a,b,c in faces:
        a,b,c=int(a),int(b),int(c)
        if mask[a] and mask[b]: nbr[a].add(b); nbr[b].add(a)
        if mask[b] and mask[c]: nbr[b].add(c); nbr[c].add(b)
        if mask[c] and mask[a]: nbr[c].add(a); nbr[a].add(c)
    return [np.fromiter(x,dtype=np.int64) if x else np.empty(0,dtype=np.int64) for x in nbr]


def gaussian(points: np.ndarray, center, radii):
    c=np.asarray(center,float); r=np.asarray(radii,float)
    q=np.sum(((points-c)/np.maximum(r,1e-8))**2,axis=1)
    return np.exp(-0.5*q*q)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        f.write('# AINA bounded soft-tissue Taubin surface polish\n')
        for x,y,z in vertices: f.write(f'v {x:.9f} {y:.9f} {z:.9f}\n')
        for a,b,c in faces: f.write(f'f {int(a)+1} {int(b)+1} {int(c)+1}\n')


def main():
    args=parse_args()
    mesh=trimesh.load(args.mesh,process=False,maintain_order=True)
    if not isinstance(mesh,trimesh.Trimesh): raise RuntimeError('Expected one triangulated AINA OBJ')
    raw=np.asarray(mesh.vertices,dtype=np.float64); faces=np.asarray(mesh.faces,dtype=np.int64)
    if int(K.max())>=len(raw): raise RuntimeError('AINA semantic vertex order is missing')
    points,offset=map_to_blender(raw); original=points.copy()
    roots,groups=components(len(points),faces); head=max(groups.values(),key=len)
    head_mask=np.zeros(len(points),bool); head_mask[head]=True
    lm=points[K].copy(); p=points

    # Region weights: only the visible soft-tissue shell.
    front=np.clip((0.070-p[:,1])/0.090,0,1)
    vertical=np.clip((p[:,2]-(lm[8,2]-0.004))/0.060,0,1)
    central=np.clip((0.104-np.abs(p[:,0]))/0.038,0,1)
    base=0.18*front*vertical*central*head_mask.astype(float)

    brow=lm[17:27].mean(0)
    forehead=brow+np.array([0.0,0.010,0.046])
    base=np.maximum(base,0.68*gaussian(p,forehead,(0.080,0.050,0.066))*front*head_mask)

    cheek_r=(lm[40]+lm[31]+lm[48])/3.0
    cheek_l=(lm[46]+lm[35]+lm[54])/3.0
    for c in (cheek_r,cheek_l):
        base=np.maximum(base,0.78*gaussian(p,c,(0.052,0.044,0.048))*front*head_mask)
        base=np.maximum(base,0.52*gaussian(p,c+np.array([0.0,0.006,-0.035]),(0.052,0.048,0.050))*front*head_mask)

    # Protect the actual identity features and silhouette.
    anchor=np.zeros(len(points),dtype=np.float64)
    groups_spec=[
        (np.arange(0,17),0.018,0.96),
        (np.arange(17,27),0.014,0.86),
        (np.arange(27,36),0.015,0.98),
        (np.arange(36,48),0.016,1.00),
        (np.arange(48,68),0.017,1.00),
    ]
    for ids,radius,strength in groups_spec:
        for c in lm[ids]: anchor=np.maximum(anchor,strength*gaussian(p,c,(radius,radius,radius)))
    # Ear-side, rear skull and low neck remain fixed.
    for idx in (0,16): anchor=np.maximum(anchor,gaussian(p,lm[idx],(0.044,0.052,0.065)))
    anchor=np.maximum(anchor,np.clip((p[:,1]-0.040)/0.060,0,1))
    anchor=np.maximum(anchor,np.clip((lm[8,2]-0.010-p[:,2])/0.050,0,1))
    mask=np.clip(base*(1.0-anchor),0,0.82)
    mask[~head_mask]=0
    mask[K]=0

    nbr=adjacency(len(points),faces,head)
    current=points.copy()
    lamb,mu=0.31,-0.325
    for _ in range(3):
        lap=np.zeros_like(current)
        for i in head:
            ns=nbr[int(i)]
            if len(ns): lap[i]=current[ns].mean(axis=0)-current[i]
        current += (lamb*mask)[:,None]*lap
        current[K]=original[K]
        lap=np.zeros_like(current)
        for i in head:
            ns=nbr[int(i)]
            if len(ns): lap[i]=current[ns].mean(axis=0)-current[i]
        current += (mu*mask)[:,None]*lap
        current[K]=original[K]

    delta=current-original
    length=np.linalg.norm(delta,axis=1)
    delta*=np.minimum(1.0,0.00095/np.maximum(length,1e-12))[:,None]
    delta[~head_mask]=0; delta[K]=0
    refined=original+delta

    tri0=original[faces]; tri1=refined[faces]
    area0=.5*np.linalg.norm(np.cross(tri0[:,1]-tri0[:,0],tri0[:,2]-tri0[:,0]),axis=1)
    area1=.5*np.linalg.norm(np.cross(tri1[:,1]-tri1[:,0],tri1[:,2]-tri1[:,0]),axis=1)
    ratio=area1/np.maximum(area0,1e-12)
    raw1=map_from_blender(refined,offset)
    write_obj(args.out,raw1,faces)
    reload=trimesh.load(args.out,process=False,maintain_order=True)
    if not isinstance(reload,trimesh.Trimesh): raise RuntimeError('Polished OBJ failed to reload')

    # Laplacian-energy measurement on the soft tissue region.
    def energy(v):
        vals=[]
        active=np.flatnonzero(mask>0.15)
        for i in active:
            ns=nbr[int(i)]
            if len(ns): vals.append(float(np.dot(v[ns].mean(axis=0)-v[i],v[ns].mean(axis=0)-v[i])))
        return float(np.mean(vals)) if vals else 0.0
    e0,e1=energy(original),energy(refined)
    report={
        'product':'AINA topology-preserving real soft-tissue surface polish',
        'source':str(args.mesh),'output':str(args.out),'topology_changed':False,
        'semantic_vertex_order_preserved':len(reload.vertices)==len(raw),
        'active_soft_tissue_vertices':int(np.sum(mask>0.15)),
        'max_vertex_displacement_m':float(np.linalg.norm(delta,axis=1).max()),
        'rms_vertex_displacement_m':float(np.sqrt(np.mean(np.sum(delta*delta,axis=1)))),
        'semantic_landmark_max_displacement_m':float(np.linalg.norm(delta[K],axis=1).max()),
        'soft_tissue_laplacian_energy_before':e0,
        'soft_tissue_laplacian_energy_after':e1,
        'triangle_area_ratio_p01':float(np.percentile(ratio,1)),
        'triangle_area_ratio_p99':float(np.percentile(ratio,99)),
        'checks':{
            'vertex_count_preserved':len(reload.vertices)==len(raw),
            'triangle_count_preserved':len(reload.faces)==len(faces),
            'finite_vertices':bool(np.isfinite(raw1).all()),
            'semantic_landmarks_fixed':float(np.linalg.norm(delta[K],axis=1).max())<1e-10,
            'bounded_displacement':float(np.linalg.norm(delta,axis=1).max())<=0.00096,
            'surface_energy_not_worse':e1<=e0*1.001+1e-12,
            'triangle_quality_safe':float(np.percentile(ratio,1))>0.55 and float(np.percentile(ratio,99))<1.55,
        },
        'visual_lock':False,
        'visual_gate':'rerender exact polished OBJ in beauty and clay front/20-degree-3Q views',
    }
    report['pass']=bool(all(report['checks'].values()))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not report['pass']: raise SystemExit('AINA soft-tissue surface QA failed')

if __name__=='__main__': main()
