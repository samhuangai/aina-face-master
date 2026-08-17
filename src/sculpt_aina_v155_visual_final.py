#!/usr/bin/env python3
"""AINA v15.5 final visual identity sculpt.

This does not create another face version. It takes the existing v15.5 locked
FaceVerse topology and applies the art-directed macro changes found during real
front / shallow-3Q / profile review: compact cranium, reduced head depth,
smooth feminine lower-face taper, softer orbital planes, almond eye opening,
short delicate nose, compact lips and shorter lower face.

The output intentionally sets identity_lock=false until a real Blender portrait
review passes. 68-point RMSE alone is not allowed to lock identity anymore.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components

K=np.array([1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,13695,12973,13422,6543,6537],dtype=np.int64)


def components(nv,faces):
    e=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    a=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(nv,nv));a=(a+a.T).tocsr()
    n,lab=connected_components(a,directed=False)
    return [np.flatnonzero(lab==i) for i in range(n)]


def ell(p,c,r,inner=.25,outer=1.15):
    c=np.asarray(c,float);r=np.asarray(r,float);q=np.sqrt(np.sum(((p-c)/r)**2,axis=1));w=np.zeros(len(p));w[q<=inner]=1
    m=(q>inner)&(q<outer)
    if np.any(m):
        t=(q[m]-inner)/(outer-inner);w[m]=.5*(1+np.cos(np.pi*t))
    return w


def affine(p,c,r,s=(1,1,1),shift=(0,0,0),inner=.25,outer=1.15):
    w=ell(p,c,r,inner,outer)[:,None];c=np.asarray(c,float);target=c+(p-c)*np.asarray(s,float)+np.asarray(shift,float);p+=w*(target-p)


def head_adjacency(head,faces,nv):
    hm=np.zeros(nv,bool);hm[head]=True;hf=faces[hm[faces].all(1)];g={int(q):i for i,q in enumerate(head)}
    lf=np.asarray([[g[int(a)],g[int(b)],g[int(c)]] for a,b,c in hf],dtype=np.int64)
    rows=np.concatenate([lf[:,0],lf[:,1],lf[:,2],lf[:,1],lf[:,2],lf[:,0]]);cols=np.concatenate([lf[:,1],lf[:,2],lf[:,0],lf[:,0],lf[:,1],lf[:,2]])
    A=sparse.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(head),len(head))).tocsr();deg=np.asarray(A.sum(1)).ravel()
    return A,deg,hm,g


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    m=trimesh.load(args.base,process=False,maintain_order=True);v0=np.asarray(m.vertices,float);f=np.asarray(m.faces,np.int64);v=v0.copy();cs=components(len(v),f);head=max(cs,key=len);A,deg,hm,g=head_adjacency(head,f,len(v));kl=np.asarray([g[int(q)] for q in K],np.int64);p=v[head].copy();start=p.copy();lm=p[kl].copy()

    # Compact round cranium above the eyes without flattening the crown.
    eye_y=float(lm[36:48,1].mean());mouth_y=float(lm[48:60,1].mean());chin_y=float(lm[8,1]);y=p[:,1].copy();anchor=eye_y-.012;u=np.clip((anchor-y)/.09,0,1)
    p[:,1]+=u*((anchor+(y-anchor)*.82)-y);p[:,0]*=(1-.08*u)

    # Smooth lower-face taper across the whole skull cross section. Ear zones are
    # excluded so the taper cannot create the old jaw/ear seam artifacts.
    t=np.clip((p[:,1]-eye_y)/max(chin_y-eye_y,1e-6),0,1);scale_x=1-.17*(t**1.35)
    earzone=(np.abs(p[:,0])>.058)&(p[:,2]>.035)&(p[:,1]<.010)&(p[:,1]>-.060);p[:,0]*=np.where(earzone,1.0,scale_x)
    temple=np.exp(-.5*((p[:,1]+.010)/.035)**2)*np.clip((np.abs(p[:,0])-.050)/.030,0,1);p[:,0]*=(1-.035*temple)
    for sg in (-1,1):affine(p,[sg*.074,-.025,.064],[.027,.041,.039],s=(.70,.74,.72),shift=(-sg*.004,0,.003),inner=.20,outer=1.10)

    # The previous mesh was far too deep in profile. Compress the complete skull
    # depth and then rebuild only the small AINA facial projections.
    p[:,2]*=.72

    # Relax harsh adult orbital/midface planes while protecting feature edges.
    lm=p[kl].copy();rel=ell(p,[0,.004,0],[.064,.082,.040],.08,1.0);protect=np.zeros(len(p))
    for ids,r in ((np.arange(36,48),[.025,.018,.015]),(np.arange(30,36),[.020,.017,.015]),(np.arange(48,68),[.031,.017,.015])):
        protect=np.maximum(protect,ell(p,lm[ids].mean(0),r,.30,1.0))
    rel*=1-.70*protect
    for _ in range(6):
        av=(A@p[:,2])/np.maximum(deg,1);p[:,2]+=.16*rel*(av-p[:,2])

    # Soft almond eyes: moderate aperture, not the rejected oversized round-eye
    # experiment. Reduce the supraorbital ridge and keep the lid core slightly forward.
    lm=p[kl].copy()
    for ids in (np.arange(36,42),np.arange(42,48)):
        ec=lm[ids].mean(0);affine(p,ec,[.031,.019,.018],s=(1.05,1.30,1.0),inner=.25,outer=1.06)
        wo=ell(p,ec,[.035,.029,.022],.30,1.22);wi=ell(p,ec,[.020,.014,.014],.45,1.06);p[:,2]+=.0018*wo*(1-.80*wi);p[:,2]-=.0008*wi
    lm=p[kl].copy()
    for ids in (np.arange(17,22),np.arange(22,27)):
        bc=lm[ids].mean(0);affine(p,bc,[.034,.024,.020],s=(1,.95,.86),shift=(0,-.0010,.0018),inner=.25,outer=1.16)

    # Small, short nose with enough projection to survive the global depth compression.
    lm=p[kl].copy();bridge=lm[27:30].mean(0);lower=lm[30:36].mean(0)
    affine(p,bridge,[.018,.034,.020],s=(.80,.92,.90),shift=(0,-.0007,-.0015),inner=.28,outer=1.16)
    affine(p,lower,[.024,.024,.021],s=(.72,.78,.86),shift=(0,-.0022,-.0028),inner=.26,outer=1.16)

    # Compact lips and shorter lower face.
    lm=p[kl].copy();mc=lm[48:60].mean(0);affine(p,mc,[.037,.022,.020],s=(.84,.70,.88),shift=(0,-.0010,-.0008),inner=.25,outer=1.14)
    tt=np.clip((p[:,1]-mouth_y)/max(chin_y-mouth_y,1e-6),0,1);p[:,1]-=.005*tt
    lm=p[kl].copy();cL=(lm[42:48].mean(0)+lm[35]+lm[54])/3;cR=(lm[36:42].mean(0)+lm[31]+lm[48])/3
    for c in (cL,cR):p[:,2]-=.0015*ell(p,c+[0,.003,0],[.031,.029,.022],.22,1.14)
    lm=p[kl].copy();chin=lm[8];affine(p,chin,[.032,.027,.024],s=(.78,.84,.90),shift=(0,-.0008,-.0005),inner=.22,outer=1.12)
    v[head]=p

    # Refit the actual eye components behind the new lids.
    eyes=sorted([q for q in cs if 650<len(q)<900],key=lambda q:v0[q].mean(0)[0])
    if len(eyes)!=2:raise RuntimeError(f'Expected two eye components, got {len(eyes)}')
    for q,ids in zip(eyes,(np.arange(36,42),np.arange(42,48))):
        raw=v0[q].copy();raw[:,2]*=.72;c=raw.mean(0);rim=v[K][ids].mean(0);target=np.asarray([rim[0],rim[1],rim[2]+.0048]);v[q]=target+(raw-c)*.95

    # Oral disconnected components follow the compact mouth and the same depth scale.
    old_lm=v0[K];new_lm=v[K];mouth_xy=new_lm[48:60,:2].mean(0)-old_lm[48:60,:2].mean(0);mouth_z=float(new_lm[48:60,2].mean()-old_lm[48:60,2].mean()*.72)
    for q in cs:
        if np.array_equal(q,head) or any(np.array_equal(q,e) for e in eyes):continue
        vv=v0[q].copy();vv[:,2]*=.72;vv[:,:2]+=mouth_xy;vv[:,2]+=mouth_z;v[q]=vv

    out=trimesh.Trimesh(vertices=v,faces=f,process=False)
    for ext in ('obj','glb','ply'):out.export(args.out/f'AINA_FACEVERSE_FULL_v15.5_VISUAL_FINAL.{ext}')
    keep=hm.copy();[keep.__setitem__(q,True) for q in eyes];fi=np.flatnonzero(keep[f].all(1));clay=out.submesh([fi],append=True,repair=False)
    for ext in ('obj','glb','ply'):clay.export(args.out/f'AINA_FACEVERSE_IDENTITY_CLAY_v15.5_VISUAL_FINAL.{ext}')

    tri0=v0[f];tri1=v[f];a0=.5*np.linalg.norm(np.cross(tri0[:,1]-tri0[:,0],tri0[:,2]-tri0[:,0]),axis=1);a1=.5*np.linalg.norm(np.cross(tri1[:,1]-tri1[:,0],tri1[:,2]-tri1[:,0]),axis=1);ratio=a1/np.maximum(a0,1e-12);delta=np.linalg.norm(p-start,axis=1)
    report={
      'version':'AINA Face Master v15.5 Visual Final Candidate','topology_changed':False,'identity_lock':False,'visual_review_required':True,
      'source':str(args.base),'full_vertices':int(len(v)),'full_triangles':int(len(f)),'two_eye_components':len(eyes)==2,
      'triangle_area_ratio_p01':float(np.percentile(ratio,1)),'triangle_area_ratio_p99':float(np.percentile(ratio,99)),
      'max_head_delta_m':float(delta.max()),'rms_head_delta_m':float(np.sqrt(np.mean(delta*delta))),
      'visual_changes':['compact cranium','0.72 depth compression','smooth lower-face taper','soft almond orbital treatment','short delicate nose','compact lips','shorter lower face'],
      'note':'68-point numerical lock is intentionally revoked. identity_lock may only become true after real Blender front/3Q/profile visual review.'
    }
    (args.out/'AINA_FACEVERSE_v15.5_VISUAL_FINAL_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
