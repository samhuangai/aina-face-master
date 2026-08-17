#!/usr/bin/env python3
"""Art-direct AINA identity on the clean continuous MPFB2 female topology.

Unlike the rejected FaceVerse chain, this preserves one continuous native human
body/head topology and the MPFB rig vertex order.  The deformation is smooth,
semantic, bounded and intentionally conservative.  It targets the approved
AINA silhouette: large soft almond eyes, small short nose, compact lips, apple
cheeks, narrow feminine jaw/chin, small close ears and a youthful forehead.

This file never sets identity_lock=true. A real Blender visual review must pass
before the lock can be promoted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def ell(p,c,r,inner=.28,outer=1.20):
    c=np.asarray(c,float);r=np.asarray(r,float);q=np.sqrt(np.sum(((p-c)/r)**2,axis=1));w=np.zeros(len(p));w[q<=inner]=1
    m=(q>inner)&(q<outer)
    if np.any(m):
        t=(q[m]-inner)/(outer-inner);w[m]=.5*(1+np.cos(np.pi*t))
    return w


def affine(p,c,r,s=(1,1,1),shift=(0,0,0),inner=.28,outer=1.20,mask=None):
    w=ell(p,c,r,inner,outer)
    if mask is not None:w*=mask
    c=np.asarray(c,float);target=c+(p-c)*np.asarray(s,float)+np.asarray(shift,float);p+=w[:,None]*(target-p)


def adjacency(n,faces):
    e=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);A=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(n,n));A=(A+A.T).tocsr();deg=np.asarray(A.sum(1)).ravel();return A,deg


def write_obj(path,v,f):
    with path.open('w',encoding='utf-8') as h:
        for x,y,z in v:h.write(f'v {x:.9f} {y:.9f} {z:.9f}\n')
        for a,b,c in f:h.write(f'f {int(a)+1} {int(b)+1} {int(c)+1}\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--topology',type=Path,required=True);ap.add_argument('--qa',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    dat=np.load(a.topology);v0=np.asarray(dat['vertices_local'],float);faces=np.asarray(dat['faces'],np.int32);v=v0.copy();qa=json.loads(a.qa.read_text())
    info=qa['interesting_head_face_groups'];head=np.asarray(info['head']['strong_indices'],np.int64);lips=np.asarray(info['lips']['strong_indices'],np.int64);ears=np.asarray(info['ears']['strong_indices'],np.int64)
    if len(v)!=19158 or len(head)<4500 or len(lips)<300:raise RuntimeError('Unexpected MPFB topology/group layout')
    p=v[head].copy();p0=p.copy();lo=p.min(0);hi=p.max(0)
    # Coordinate system from probe: X left/right, -Y face-forward, Z up.
    eye_z=1.565;eye_x=.0355;mouth_z=float(v[lips,2].mean());face_front=np.clip((-p[:,1]-.090)/.065,0,1)

    # 1) Youthful/feminine silhouette. Keep upper cranium smooth while tapering
    # mid/lower face strongly toward the approved AINA V-line jaw.
    t=np.clip((1.545-p[:,2])/.125,0,1);frontmask=np.clip((-p[:,1]-.030)/.080,0,1)
    p[:,0]*=(1-.115*(t**1.25)*frontmask)
    chinband=np.clip((1.485-p[:,2])/.075,0,1);p[:,0]*=(1-.075*chinband*frontmask)
    # Shorten lower face without moving the neck seam.
    lower=np.clip((mouth_z-p[:,2])/.070,0,1)*frontmask;p[:,2]+=.0075*lower

    # 2) Temples/forehead: subtly narrower temple sides, slightly fuller/smoother
    # frontal forehead rather than the old hard adult brow plane.
    temple=np.exp(-.5*((p[:,2]-1.585)/.052)**2)*np.clip((np.abs(p[:,0])-.055)/.045,0,1)*frontmask;p[:,0]*=(1-.035*temple)
    forehead=ell(p,[0,-.115,1.615],[.078,.050,.055],.22,1.10);p[:,1]-=.0022*forehead

    # 3) Eyes: open the existing sockets into large but not round AINA almond
    # shapes. Width increase > height increase, with small raised outer tails.
    for sg in (-1,1):
        ec=np.array([sg*eye_x,-.145,eye_z]);local_front=np.clip((-p[:,1]-.105)/.045,0,1)
        affine(p,ec,[.035,.030,.024],s=(1.10,1.0,1.18),inner=.22,outer=1.12,mask=local_front)
        outer=np.array([sg*.053,-.145,1.566]);p[:,2]+=.0018*ell(p,outer,[.015,.024,.014],.18,1.10)*local_front
        # Soften upper orbital ridge and slightly support apple-cheek transition.
        brow=np.array([sg*.034,-.124,1.590]);p[:,1]+=.0018*ell(p,brow,[.035,.026,.022],.24,1.15)
        cheek=np.array([sg*.039,-.137,1.525]);p[:,1]-=.0022*ell(p,cheek,[.038,.030,.031],.20,1.16)

    # 4) Nose: base nose is too broad/long. Narrow the bridge and alae, shorten
    # the lower nose, retract the tip slightly while preserving a readable profile.
    affine(p,[0,-.145,1.545],[.022,.033,.041],s=(.82,.96,.92),shift=(0,.0010,.0010),inner=.25,outer=1.18,mask=face_front)
    affine(p,[0,-.158,1.515],[.029,.028,.026],s=(.76,.84,.78),shift=(0,.0032,.0030),inner=.25,outer=1.18,mask=face_front)

    # 5) Lips: use the exact native lip vertex group. Compact width/height and
    # gently retain volume instead of the tight/puckered FaceVerse result.
    mc=v[lips].mean(0);q=v[lips].copy();q=mc+(q-mc)*np.array([.86,1.03,.88]);q[:,2]+=.0010;v[lips]=q
    # Smooth perioral transition on head shell.
    affine(p,mc,[.040,.035,.026],s=(.93,.98,.94),shift=(0,.0008,.0006),inner=.24,outer=1.15,mask=face_front)

    # 6) Chin/jaw: compact rounded chin, retract adult-heavy lower-face profile.
    affine(p,[0,-.125,1.438],[.045,.050,.044],s=(.82,.86,.88),shift=(0,.0040,.0040),inner=.25,outer=1.18,mask=frontmask)
    jawmask=np.clip((1.505-p[:,2])/.090,0,1)*frontmask;p[:,1]+=.0035*jawmask

    # 7) Small close ears, preserving their native topology/detail.
    for sg in (-1,1):
        ids=ears[np.sign(v[ears,0])==sg];c=v[ids].mean(0);v[ids]=c+(v[ids]-c)*np.array([.78,.84,.84]);v[ids,0]-=sg*.0030

    # Write head back and lightly relax only the face-depth coordinate. Feature
    # cores are protected so topology stays clean instead of becoming melted.
    v[head]=p;A,deg=adjacency(len(v),faces);z=v[:,1].copy();region=np.zeros(len(v));region[head]=np.clip((-v[head,1]-.075)/.075,0,1)*np.exp(-.5*((v[head,2]-1.525)/.100)**2);protect=np.zeros(len(v));protect[lips]=1
    for sg in (-1,1):protect=np.maximum(protect,ell(v,[sg*eye_x,-.145,eye_z],[.030,.025,.021],.25,1.0))
    protect=np.maximum(protect,ell(v,[0,-.155,1.525],[.023,.027,.034],.25,1.0));region*=1-.82*protect
    for _ in range(3):
        av=(A@z)/np.maximum(deg,1);z+=.10*region*(av-z)
    v[:,1]=z

    # Re-apply explicit lip compacting after shell relaxation.
    mc=v0[lips].mean(0);q=v0[lips].copy();q=mc+(q-mc)*np.array([.86,1.03,.88]);q[:,2]+=.0010;v[lips]=q

    write_obj(a.out/'AINA_MPFB_FULL_v15.5_IDENTITY_CANDIDATE.obj',v,faces)
    np.savez_compressed(a.out/'AINA_MPFB_FULL_v15.5_IDENTITY_CANDIDATE.npz',vertices=v,faces=faces,head_indices=head,lips_indices=lips,ears_indices=ears)
    d=np.linalg.norm(v[head]-v0[head],axis=1)
    rep={'version':'AINA v15.5 MPFB Continuous Female Identity Candidate','identity_lock':False,'visual_review_required':True,'topology_changed':False,'continuous_body':True,'vertices':int(len(v)),'triangles':int(len(faces)),'head_vertices':int(len(head)),'lip_vertices':int(len(lips)),'ear_vertices':int(len(ears)),'max_head_delta_m':float(d.max()),'rms_head_delta_m':float(np.sqrt(np.mean(d*d))),'semantic_targets':['youthful V-line silhouette','soft almond eyes','small short nose','compact soft lips','apple cheeks','rounded compact chin','small close ears'],'note':'Same native MPFB body/head vertex order and rig topology. Lock is forbidden until real Blender front/3Q/profile visually pass.'}
    (a.out/'AINA_MPFB_v15.5_IDENTITY_REPORT.json').write_text(json.dumps(rep,indent=2),encoding='utf-8');print(json.dumps(rep,indent=2))

if __name__=='__main__':main()
