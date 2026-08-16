#!/usr/bin/env python3
"""Finalize the existing AINA v15.5 mesh into the single identity-locked production face.

This is not a new face version. It consumes the clean v15.5 mesh, applies the
last art-directed semantic/depth polish plus a smooth bounded front-landmark
convergence pass, preserves topology, and emits v15.5 IDENTITY_LOCKED assets.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components

K=np.array([1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,13695,12973,13422,6543,6537],dtype=np.int64)
W=np.ones(68,float);W[:17]=4.8;W[17:27]=1.5;W[27:36]=4.2;W[36:48]=6.2;W[48:60]=5.2;W[60:]=2.2

def components(nv,f):
    e=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    a=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(nv,nv));a=(a+a.T).tocsr()
    n,lab=connected_components(a,directed=False)
    return [np.flatnonzero(lab==i) for i in range(n)]

def ell(p,c,r,inner=.35,outer=1.35):
    c=np.asarray(c,float);r=np.asarray(r,float)
    q=np.sqrt(np.sum(((p-c)/r)**2,axis=1));w=np.zeros(len(p));w[q<=inner]=1
    m=(q>inner)&(q<outer)
    if np.any(m):
        t=(q[m]-inner)/(outer-inner);w[m]=.5*(1+np.cos(np.pi*t))
    return w

def affine(p,c,r,s=(1,1,1),shift=(0,0,0),inner=.35,outer=1.35):
    ww=ell(p,c,r,inner,outer)[:,None];c=np.asarray(c,float)
    target=c+(p-c)*np.asarray(s,float)+np.asarray(shift,float)
    p += ww*(target-p)

def bump(p,c,r,d,inner=.25,outer=1.35):
    p += ell(p,c,r,inner,outer)[:,None]*np.asarray(d,float)

def local_rbf_xy(p,ctrl,disp,sigma=.0065,zsigma=.014,strength=.78):
    ctrl=np.asarray(ctrl,float);disp=np.asarray(disp,float);s2=sigma*sigma;zs2=zsigma*zsigma
    for st in range(0,len(p),4096):
        pp=p[st:st+4096]
        dx=pp[:,None,0]-ctrl[None,:,0];dy=pp[:,None,1]-ctrl[None,:,1];dz=pp[:,None,2]-ctrl[None,:,2]
        ww=np.exp(-(dx*dx+dy*dy)/(2*s2)-dz*dz/(2*zs2));sw=ww.sum(1)
        val=(ww@disp)/(sw[:,None]+1e-12);env=1-np.exp(-1.2*sw)
        p[st:st+len(pp),:2]+=strength*val*env[:,None]

def load_target(path:Path):
    d=json.loads(path.read_text());pts=np.asarray(d['landmarks_xy'],float)
    if pts.shape!=(68,2):raise RuntimeError(f'Expected 68x2 target landmarks, got {pts.shape}')
    return pts

def similarity_desired(cur,target):
    ww=W[:,None];mx=(cur*ww).sum(0)/W.sum();my=(target*ww).sum(0)/W.sum();X=cur-mx;Y=target-my
    H=(X*ww).T@Y;U,S,Vt=np.linalg.svd(H);R=U@Vt
    if np.linalg.det(R)<0:Vt[-1]*=-1;R=U@Vt
    scale=float(S.sum()/max(np.sum(ww*(X*X)),1e-12))
    desired=((target-my)@R.T)/max(scale,1e-12)+mx
    pred=scale*(X@R)+my;err=np.linalg.norm(pred-target,axis=1)
    return desired,float(np.sqrt(np.mean(err*err))),float(err.max())

def target_rbf_pass(p,kl,target,iters=6):
    history=[]
    for i in range(iters):
        lm=p[kl];desired,rmse,maxe=similarity_desired(lm[:,:2],target);disp=desired-lm[:,:2]
        dn=np.linalg.norm(disp,axis=1);cap=.0045;bad=dn>cap
        if np.any(bad):disp[bad]*=(cap/dn[bad])[:,None]
        sigma=.011 if i<2 else .0095;strength=.58 if i<2 else .45;s2=sigma*sigma;zs2=.026*.026
        for st in range(0,len(p),2048):
            pp=p[st:st+2048];dx=pp[:,None,0]-lm[None,:,0];dy=pp[:,None,1]-lm[None,:,1];dz=pp[:,None,2]-lm[None,:,2]
            ww=np.exp(-(dx*dx+dy*dy)/(2*s2)-dz*dz/(2*zs2));sw=ww.sum(1);val=(ww@disp)/(sw[:,None]+1e-12);env=1-np.exp(-.9*sw)
            face=(1/(1+np.exp((pp[:,2]-.070)/.008)))*(1/(1+np.exp((-pp[:,1]-.055)/.007)))
            p[st:st+len(pp),:2]+=strength*val*env[:,None]*face[:,None]
        history.append({'iteration':i+1,'rmse_px':rmse,'max_px':maxe,'max_requested_move_mm':float(dn.max()*1000)})
    return history

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--target-landmarks',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    target=load_target(a.target_landmarks)
    m=trimesh.load(a.base,process=False,maintain_order=True);v0=np.asarray(m.vertices,float);f=np.asarray(m.faces,np.int64);v=v0.copy();cs=components(len(v),f)
    head=max(cs,key=len);hm=np.zeros(len(v),bool);hm[head]=True;g={int(q):i for i,q in enumerate(head)};kl=np.array([g[int(q)] for q in K]);p=v[head].copy();p_start=p.copy();lm=p[kl].copy()

    y=p[:,1];front=1/(1+np.exp((p[:,2]-.055)/.010));side=1/(1+np.exp((np.abs(p[:,0])-.078)/.005));low=np.clip((y-.004)/.070,0,1)*front*side
    p[:,0]*=(1-.080*low)
    lm=p[kl].copy();chin=lm[8];affine(p,chin,[.037,.031,.040],s=(.82,1.01,.96),shift=(0,.0008,.0011),inner=.30,outer=1.35)

    lm=p[kl].copy()
    for ids,sg in ((np.arange(36,42),+1),(np.arange(42,48),-1)):
        c=lm[ids].mean(0);affine(p,c,[.033,.022,.025],s=(.955,1.17,.985),shift=(sg*.0013,0,-.00025),inner=.38,outer=1.28)
    lm=p[kl].copy();ctrl=[];disp=[]
    for outerid,innerid in ((36,39),(45,42)):
        ctrl.extend([lm[innerid],lm[outerid]]);disp.extend([[0,-.0025],[0,.0010]])
    local_rbf_xy(p,np.asarray(ctrl),np.asarray(disp))

    lm=p[kl].copy();bridge=lm[27:30].mean(0);affine(p,bridge,[.018,.034,.028],s=(.84,.99,.90),shift=(0,.0004,.0022),inner=.36,outer=1.30)
    lm=p[kl].copy();lower=lm[30:36].mean(0);affine(p,lower,[.025,.025,.030],s=(.78,.96,.80),shift=(0,.0008,.0030),inner=.35,outer=1.32)
    lm=p[kl].copy();tip=lm[30];affine(p,tip,[.014,.016,.020],s=(.88,.93,.78),shift=(0,.0007,.0020),inner=.32,outer=1.22)

    lm=p[kl].copy();cL=(lm[42:48].mean(0)+lm[35]+lm[54])/3;cR=(lm[36:42].mean(0)+lm[31]+lm[48])/3
    for c in (cL,cR):bump(p,c+[0,.007,0],[.034,.036,.038],(0,0,-.0026),inner=.25,outer=1.35)
    lm=p[kl].copy();mouth=lm[48:60].mean(0);affine(p,mouth,[.038,.026,.028],s=(1.035,1.02,.94),shift=(0,.0008,.0007),inner=.34,outer=1.32);bump(p,mouth,[.033,.017,.020],(0,0,-.0018),inner=.28,outer=1.22)
    lm=p[kl].copy();jawc=(lm[8]+mouth)/2;bump(p,jawc+[0,.010,0],[.050,.045,.050],(0,0,.0018),inner=.22,outer=1.28)

    history=target_rbf_pass(p,kl,target,iters=6)

    for sg in (-1,1):
        affine(p,[sg*.079,-.028,.070],[.021,.043,.038],s=(.78,.84,.85),shift=(-sg*.0022,0,.0010),inner=.25,outer=1.15)
    lm=p[kl].copy();cL=(lm[42:48].mean(0)+lm[35]+lm[54])/3;cR=(lm[36:42].mean(0)+lm[31]+lm[48])/3
    for c in (cL,cR):bump(p,c+[0,.006,0],[.034,.034,.038],(0,0,-.0009),inner=.30,outer=1.25)
    lm=p[kl].copy();chin=lm[8];affine(p,chin,[.033,.030,.038],s=(1.00,1.03,.97),shift=(0,.0020,-.0010),inner=.30,outer=1.25)
    v[head]=p

    lm=v[K];eyes=sorted([q for q in cs if 650<len(q)<900],key=lambda q:v0[q].mean(0)[0])
    if len(eyes)!=2:raise RuntimeError(f'Expected 2 eye components, got {len(eyes)}')
    for ids,el in zip(eyes,(lm[36:42],lm[42:48])):
        c=v[ids].mean(0);rim=el.mean(0);v[ids]+=np.array([rim[0],rim[1],rim[2]+.0063])-c
    old=v0[K];new=v[K];mouth_shift=new[48:60].mean(0)-old[48:60].mean(0)
    for ids in cs:
        if np.array_equal(ids,head) or any(np.array_equal(ids,e) for e in eyes):continue
        v[ids]+=mouth_shift

    out=trimesh.Trimesh(vertices=v,faces=f,process=False)
    for ext in ('obj','glb','ply'):out.export(a.out/f'AINA_FACEVERSE_FULL_v15.5_IDENTITY_LOCKED.{ext}')
    keep=hm.copy();[keep.__setitem__(e,True) for e in eyes];fid=np.flatnonzero(keep[f].all(1));clay=out.submesh([fid],append=True,repair=False)
    for ext in ('obj','glb','ply'):clay.export(a.out/f'AINA_FACEVERSE_IDENTITY_CLAY_v15.5_LOCKED.{ext}')

    _,rmse,maxe=similarity_desired(v[K,:2],target)
    tri0=v0[f];tri1=v[f];a0=.5*np.linalg.norm(np.cross(tri0[:,1]-tri0[:,0],tri0[:,2]-tri0[:,0]),axis=1);a1=.5*np.linalg.norm(np.cross(tri1[:,1]-tri1[:,0],tri1[:,2]-tri1[:,0]),axis=1);ratio=a1/np.maximum(a0,1e-12)
    p01=float(np.percentile(ratio,1));p99=float(np.percentile(ratio,99));delta=np.linalg.norm(p-p_start,axis=1)
    checks={
        'topology_preserved': bool(len(v)==len(v0) and len(f)==len(np.asarray(m.faces))),
        'finite_geometry': bool(np.isfinite(v).all()),
        'two_eye_components': bool(len(eyes)==2),
        'front_landmark_rmse_le_1_10px': bool(rmse<=1.10),
        'front_landmark_max_le_3_0px': bool(maxe<=3.0),
        'triangle_area_p01_ge_0_50': bool(p01>=.50),
        'triangle_area_p99_le_1_55': bool(p99<=1.55),
    }
    lock=bool(all(checks.values()))
    report={
        'version':'AINA Face Master v15.5 Identity Lock','base':str(a.base),'topology_changed':False,
        'identity_lock':lock,'candidate':False,'checks':checks,'front_landmark_rmse_px':rmse,'front_landmark_max_px':maxe,
        'triangle_area_ratio_p01':p01,'triangle_area_ratio_p99':p99,'max_head_delta_m':float(delta.max()),'rms_head_delta_m':float(np.sqrt(np.mean(delta*delta))),
        'qa_gate':'locked naked-clay front landmark convergence + visually calibrated shallow 3Q (-20) + correctly oriented left-facing profile (+90)',
        'note':'No new AINA effect/reference image was generated. This is the final lock pass on the existing v15.5 real mesh.',
        'convergence_history':history,
    }
    (a.out/'AINA_FACEVERSE_v15.5_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not lock:raise SystemExit('AINA v15.5 identity lock gate failed')
if __name__=='__main__':main()
