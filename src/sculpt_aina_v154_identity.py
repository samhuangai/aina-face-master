#!/usr/bin/env python3
"""AINA v15.4 — calibrated multi-view real-mesh convergence.

Actual topology-preserving 3D geometry edit only. No effect/reference image is
generated. v15.4 corrects two QA-camera mistakes discovered after v15.3:
- the approved 3/4 artwork is a shallow turn (diagnostic sweep around 15–25 deg),
  not a literal 45-degree camera;
- the approved strict profile faces left, so it must be compared to the +90 model
  render rather than the old -90 render.

Geometry work therefore focuses only on mismatches that survive calibrated views:
V-line lower face, wider/flatter almond lids, softer accumulated orbital grooves,
delicate lips, and a real side-profile depth rebuild that gives the lower nose a
proper projection while keeping the upper bridge subtle.

Identity lock remains false until naked-clay calibrated front/3Q/profile pass.
"""
from __future__ import annotations
import argparse,json
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

def ell(p,c,r,inner=.45,outer=1.45):
    c=np.asarray(c,float);r=np.asarray(r,float)
    q=np.sqrt(np.sum(((p-c)/r)**2,axis=1));w=np.zeros(len(p),float);w[q<=inner]=1.
    m=(q>inner)&(q<outer);t=(q[m]-inner)/max(outer-inner,1e-9);w[m]=.5*(1+np.cos(np.pi*t));return w

def rbfxy(p,ctrl,disp,sigma=.006,zsigma=.010,strength=1.):
    ctrl=np.asarray(ctrl,float);disp=np.asarray(disp,float);s2=sigma*sigma;zs2=zsigma*zsigma
    for st in range(0,len(p),4096):
        pp=p[st:st+4096];dx=pp[:,None,0]-ctrl[None,:,0];dy=pp[:,None,1]-ctrl[None,:,1];dz=pp[:,None,2]-ctrl[None,:,2]
        w=np.exp(-(dx*dx+dy*dy)/(2*s2)-dz*dz/(2*zs2));sw=w.sum(1)
        val=(w@disp)/(sw[:,None]+1e-12);env=1-np.exp(-1.15*sw)
        p[st:st+len(pp),:2]+=val*env[:,None]*strength

def head_adjacency(head,faces,nv):
    hm=np.zeros(nv,bool);hm[head]=1;hf=faces[hm[faces].all(1)];g=-np.ones(nv,int);g[head]=np.arange(len(head));le=g[hf]
    rows=np.concatenate([le[:,0],le[:,1],le[:,2],le[:,1],le[:,2],le[:,0]])
    cols=np.concatenate([le[:,1],le[:,2],le[:,0],le[:,0],le[:,1],le[:,2]])
    a=sparse.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(head),len(head))).tocsr()
    return a,np.asarray(a.sum(1)).ravel()

def sculpt(p,kl,adj,deg):
    p0=p.copy();lm=p[kl].copy()
    # Adult feminine V-line, preserving upper cranium and face center.
    y=p[:,1];x=np.abs(p[:,0]);z=p[:,2]
    front=1/(1+np.exp((z-.030)/.006));center=1/(1+np.exp((x-.073)/.006))
    wmid=np.clip((y+.005)/.055,0,1)*front*center;low=np.clip((y-.018)/.045,0,1)*front*center
    p[:,0]*=(1-.075*wmid)*(1-.060*low)

    # Almond lids pass 1.
    lm=p[kl].copy()
    for ids,sign in ((np.arange(36,42),-1.),(np.arange(42,48),1.)):
        c=lm[ids].mean(0);tar=lm[ids,:2].copy();tar[:,0]=c[0]+1.115*(lm[ids,0]-c[0]);tar[:,1]=c[1]+.90*(lm[ids,1]-c[1])
        tail=(tar[:,0]-c[0])*sign;tar[:,1]-=.0008*np.clip(tail/.012,0,1)
        rbfxy(p,lm[ids],tar-lm[ids,:2],.0054,.009,.76)

    # Shorter lower-nose XY, subtle upper bridge.
    lm=p[kl].copy();lower=np.arange(30,36);tar=lm[lower,:2].copy();tar[:,1]-=.00125
    rbfxy(p,lm[lower],tar-lm[lower,:2],.0055,.010,.55)
    bridge=lm[27:31].mean(0);tip=lm[30]
    p[:,2]+=.0010*ell(p,bridge,[.017,.032,.020],.5,1.35)
    p[:,2]+=.0007*ell(p,tip,[.015,.017,.018],.5,1.30)

    # Delicate neutral lips.
    lm=p[kl].copy();ids=np.arange(48,60);mc=lm[ids].mean(0);tar=lm[ids,:2].copy()
    tar[:,0]=mc[0]+.92*(lm[ids,0]-mc[0]);tar[:,1]=(mc[1]-.0008)+.80*(lm[ids,1]-mc[1])
    rbfxy(p,lm[ids],tar-lm[ids,:2],.0055,.009,.70)

    # Cheek / muzzle balance after calibrated shallow 3Q inspection.
    lm=p[kl].copy();mc=lm[48:60].mean(0);p[:,2]+=.0019*ell(p,mc,[.033,.026,.028],.45,1.50)
    for sign in (-1.,1.):
        p[:,2]+=.0015*ell(p,[sign*.036,.020,.010],[.035,.035,.035],.45,1.35)
        p[:,2]-=.0005*ell(p,[sign*.034,.002,.004],[.028,.025,.030],.50,1.30)
    chin=lm[8];wc=ell(p,chin,[.032,.030,.038],.42,1.35);p[:,0]+=wc*(-p[:,0])*.045;p[:,2]+=.0009*wc

    # Soften accumulated orbital/under-eye depth grooves without erasing lid rims.
    lm=p[kl].copy();wr=np.zeros(len(p))
    for ids in (np.arange(36,42),np.arange(42,48)):
        c=lm[ids].mean(0);worb=ell(p,c,[.033,.026,.027],.50,1.35);wlid=ell(p,c,[.020,.013,.018],.60,1.20);wr=np.maximum(wr,worb*(1-.72*wlid))
    q=p[:,2].copy()
    for _ in range(3):
        avg=(adj@q)/np.maximum(deg,1);q=q+.22*wr*(avg-q)
    p[:,2]=q

    # Almond lids pass 2: deliberately larger horizontally, flatter vertically.
    lm=p[kl].copy()
    for ids in (np.arange(36,42),np.arange(42,48)):
        c=lm[ids].mean(0);tar=lm[ids,:2].copy();tar[:,0]=c[0]+1.10*(lm[ids,0]-c[0]);tar[:,1]=c[1]+.90*(lm[ids,1]-c[1])
        rbfxy(p,lm[ids],tar-lm[ids,:2],.0052,.009,.82)

    # True profile-depth rebuild. FaceVerse v15.3 had lower nose nearly behind the forehead plane.
    # Build projection only at the lower nose; keep bridge soft in front view.
    lm=p[kl].copy();nc=np.array([lm[30,0],-.002,.002]);bc=np.array([lm[30,0],.005,.006])
    p[:,2]-=.0075*ell(p,nc,[.018,.019,.026],.28,1.38)
    p[:,2]-=.0030*ell(p,bc,[.022,.014,.025],.30,1.30)
    # Lips sit slightly forward of chin, but remain far behind the nose tip.
    mc=lm[48:60].mean(0);p[:,2]+=.0004*ell(p,mc,[.030,.021,.026],.30,1.35)
    p[:,2]+=.0003*ell(p,[mc[0],mc[1]-.012,mc[2]],[.020,.018,.024],.35,1.30)
    return p,p-p0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-full',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_v154'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    mesh=trimesh.load(args.base_full,process=False,maintain_order=True);v0=np.asarray(mesh.vertices,float);f=np.asarray(mesh.faces,np.int64);v=v0.copy();cs=components(len(v),f);head=max(cs,key=len);hm=np.zeros(len(v),bool);hm[head]=1
    g={int(q):i for i,q in enumerate(head)}
    if not np.all(hm[K]):raise RuntimeError('semantic keypoints escaped main head shell')
    kl=np.array([g[int(q)] for q in K]);adj,deg=head_adjacency(head,f,len(v));p,delta=sculpt(v[head].copy(),kl,adj,deg);v[head]=p
    lm=v[K].copy();eyes=sorted([c for c in cs if 650<len(c)<900],key=lambda q:v0[q].mean(0)[0])
    if len(eyes)!=2:raise RuntimeError(f'expected 2 eyeballs, got {len(eyes)}')
    before=[];after=[]
    for ids,el in zip(eyes,(lm[36:42],lm[42:48])):
        c=v[ids].mean(0);before.append(c.tolist());rim=el.mean(0);tc=np.array([rim[0],rim[1],rim[2]+.0072]);v[ids]+=tc-c;after.append(v[ids].mean(0).tolist())
    # Oral disconnected components follow mouth translation only.
    old=v0[K];new=v[K];shift=new[48:60].mean(0)-old[48:60].mean(0)
    for ids in cs:
        if np.array_equal(ids,head) or any(np.array_equal(ids,e) for e in eyes):continue
        v[ids]+=shift
    outm=trimesh.Trimesh(vertices=v,faces=f,process=False)
    for ext in ('obj','glb','ply'):outm.export(args.out/f'AINA_FACEVERSE_FULL_v15.4_IDENTITY.{ext}')
    keep=hm.copy()
    for e in eyes:keep[e]=1
    fid=np.flatnonzero(keep[f].all(1));clay=outm.submesh([fid],append=True,repair=False)
    for ext in ('obj','glb','ply'):clay.export(args.out/f'AINA_FACEVERSE_IDENTITY_CLAY_v15.4.{ext}')
    lm=v[K];eyeplane=float(np.mean([lm[36:42,2].mean(),lm[42:48,2].mean()]));nose_tip=float(lm[30,2]);mouth=float(lm[48:60,2].mean());chin=float(lm[8,2])
    rep={'version':'AINA Face Master v15.4 Calibrated Multi-View Real-Mesh Convergence','base':str(args.base_full),'topology_changed':False,'full_vertices':int(len(v)),'full_triangles':int(len(f)),'head_vertices':int(len(head)),'max_head_delta_from_v153_m':float(np.linalg.norm(delta,axis=1).max()),'rms_head_delta_from_v153_m':float(np.sqrt(np.mean(np.sum(delta*delta,axis=1)))),'eye_center_distance_m':float(abs(lm[42:48,0].mean()-lm[36:42,0].mean())),'right_eye_span_xy_m':np.ptp(lm[36:42,:2],axis=0).tolist(),'left_eye_span_xy_m':np.ptp(lm[42:48,:2],axis=0).tolist(),'mouth_span_m':float(np.ptp(lm[48:60,0])),'profile_depth_semantic_m':{'eye_plane_z':eyeplane,'nose_tip_z':nose_tip,'mouth_mean_z':mouth,'chin_z':chin,'nose_tip_forward_of_eye_plane':eyeplane-nose_tip,'mouth_forward_of_chin':chin-mouth},'eyeball_centers_before':before,'eyeball_centers_after':after,'qa_camera_correction':{'approved_3q':'compare against shallow yaw sweep -15/-20/-25 degrees; primary -20, not old -45','approved_side':'left-facing reference compares to model yaw +90, not old -90'},'identity_lock':False,'candidate':True,'qa_gate':'actual naked-clay front + calibrated shallow 3Q + correctly oriented strict profile vs approved AINA references','note':'No new AINA effect/reference image generated. Only real OBJ/GLB/PLY geometry is edited.','next':'Use calibrated QA; if likeness still fails, continue depth/feature sculpt without distorting mesh toward mis-calibrated cameras.'}
    (args.out/'AINA_FACEVERSE_v15.4_REPORT.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
