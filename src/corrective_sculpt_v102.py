#!/usr/bin/env python3
"""AINA Face Master v10.2 — multi-view identity corrective sculpt.

This pass deliberately steps outside the 170D GNM identity PCA space. Starting
from the v10.1 neutral GNM identity, it converts landmark residuals from the
approved AINA front/3Q/profile references into smooth local free-form skin
corrections. GNM topology is preserved; only vertex positions are changed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import face_alignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
import trimesh

from gnm.shape import gnm_numpy
from gnm.shape import gnm_landmarks

VIEW_ORDER = ("front", "three_quarter", "side")


def image_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def detect_68(fa, image):
    h,w=image.shape[:2]
    scale=max(1.0,720.0/max(h,w))
    work=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC) if scale>1 else image
    preds=fa.get_landmarks_from_image(work)
    if not preds: raise RuntimeError('68-point detector found no face')
    ctr=np.array([work.shape[1]/2,work.shape[0]/2],dtype=np.float32)
    best=min(preds,key=lambda p:float(np.linalg.norm(np.asarray(p)[:,:2].mean(0)-ctr)))
    pts=np.asarray(best,dtype=np.float64)[:,:2]/scale
    if pts.shape!=(68,2): raise RuntimeError(str(pts.shape))
    return pts


def normalize_target(points_px, shape):
    h,w=shape[:2]; s=.5*max(w,h); ctr=np.array([w*.5,h*.5])
    return (points_px-ctr)/s


def landmarks_from_vertices(vertices, idx, weights):
    return (vertices[idx]*weights[...,None]).sum(axis=-2)


def project(points, cam):
    r=np.asarray(cam['rotation_rows'],dtype=np.float64)
    scale=float(cam['scale']); trans=np.asarray(cam['translation'],dtype=np.float64)
    return scale*(points@r.T)[:,:2]+trans


def visible_ids(name):
    if name=='front': return np.arange(68,dtype=np.int64)
    if name=='three_quarter':
        return np.concatenate([np.arange(0,17),np.arange(17,27),np.arange(27,36),np.arange(36,48),np.arange(48,68)]).astype(np.int64)
    # strict profile detector has unstable hidden-side eye/brow points
    return np.concatenate([np.arange(0,17),np.arange(27,36),np.arange(36,42),np.arange(48,60)]).astype(np.int64)


def feature_gain(ids):
    g=np.ones(len(ids),dtype=np.float64)
    for j,i in enumerate(ids):
        if 36<=i<48: g[j]=1.35       # eye aperture / corners
        elif 27<=i<36: g[j]=1.18     # nose
        elif 48<=i<68: g[j]=1.22     # mouth / lips
        elif 0<=i<17: g[j]=1.15      # jaw/chin silhouette
        else: g[j]=0.78              # brows less identity-critical here
    return g


def corrective_pass(vertices, skin_ids, lm_idx, lm_w, target, cam, name, gain, sigma, max_step):
    lm=landmarks_from_vertices(vertices,lm_idx,lm_w)
    pred=project(lm,cam)
    ids=visible_ids(name)
    r=np.asarray(cam['rotation_rows'],dtype=np.float64)
    scale=float(cam['scale'])
    screen_delta=(target[name][ids]-pred[ids])/scale
    fg=feature_gain(ids)[:,None]
    delta_world=(screen_delta[:,0,None]*r[0][None,:]+screen_delta[:,1,None]*r[1][None,:])*fg

    q_lm=(lm@r.T)[ids,:2]
    q_skin=(vertices[skin_ids]@r.T)[:,:2]
    # Deduplicate very close controls for RBF numerical stability.
    key=np.round(q_lm,6)
    _,uniq=np.unique(key,axis=0,return_index=True)
    q_ctl=q_lm[uniq]; d_ctl=delta_world[uniq]

    # Zero anchors selected from skin positions far from any facial landmark
    # keep scalp, ears and neck stable while allowing unrestricted facial sculpt.
    tree=cKDTree(q_ctl)
    dmin,_=tree.query(q_skin,k=1)
    far=np.where(dmin>sigma*1.55)[0]
    if len(far):
        # deterministic spatially diverse sampling
        stride=max(1,len(far)//48)
        anchor_idx=far[::stride][:48]
        q_fit=np.vstack([q_ctl,q_skin[anchor_idx]])
        d_fit=np.vstack([d_ctl,np.zeros((len(anchor_idx),3))])
    else:
        q_fit,d_fit=q_ctl,d_ctl

    try:
        rbf=RBFInterpolator(q_fit,d_fit,kernel='thin_plate_spline',smoothing=2e-7)
        disp=rbf(q_skin)
    except Exception:
        rbf=RBFInterpolator(q_fit,d_fit,kernel='linear',smoothing=1e-6)
        disp=rbf(q_skin)

    # Compact facial influence; landmarks are exact/near-exact, cranium fades out.
    dnear,_=cKDTree(q_ctl).query(q_skin,k=1)
    influence=np.exp(-0.5*(dnear/max(sigma,1e-6))**4)
    disp*=influence[:,None]*gain
    norm=np.linalg.norm(disp,axis=1)
    scl=np.minimum(1.0,max_step/np.maximum(norm,1e-12))
    disp*=scl[:,None]
    out=vertices.copy(); out[skin_ids]+=disp

    lm2=landmarks_from_vertices(out,lm_idx,lm_w)
    pred2=project(lm2,cam)
    err0=np.linalg.norm(pred-target[name],axis=1)
    err1=np.linalg.norm(pred2-target[name],axis=1)
    return out, {
        'view':name,'gain':gain,'sigma':sigma,'max_step_m':max_step,
        'rmse_before':float(np.sqrt(np.mean(err0[ids]**2))),
        'rmse_after':float(np.sqrt(np.mean(err1[ids]**2))),
        'median_before':float(np.median(err0[ids])),
        'median_after':float(np.median(err1[ids])),
        'max_vertex_step_m':float(norm.max() if len(norm) else 0),
    }


def relax_non_landmark_noise(vertices, skin_faces_global, skin_ids, protected_global, amount=.06, iterations=1):
    # Very light topology smoothing only on non-landmark skin vertices. It removes
    # RBF ripple without washing out eyes/nose/lips/jaw controls.
    local={int(g):i for i,g in enumerate(skin_ids)}
    faces=np.vectorize(local.get)(skin_faces_global)
    sv=vertices[skin_ids].copy()
    adj=[set() for _ in range(len(sv))]
    for a,b,c in faces:
        adj[a].update((b,c)); adj[b].update((a,c)); adj[c].update((a,b))
    protected=set(local[g] for g in protected_global if int(g) in local)
    for _ in range(iterations):
        old=sv.copy(); new=sv.copy()
        for i,nbr in enumerate(adj):
            if i in protected or not nbr: continue
            mean=old[list(nbr)].mean(axis=0)
            new[i]=old[i]*(1-amount)+mean*amount
        sv=new
    out=vertices.copy(); out[skin_ids]=sv
    return out


def save_overlay(image,target_px,pred_norm,out_path,title):
    h,w=image.shape[:2]; s=.5*max(w,h); ctr=np.array([w*.5,h*.5]); pred_px=pred_norm*s+ctr
    fig,ax=plt.subplots(figsize=(6,6),dpi=160); ax.imshow(image)
    ax.scatter(target_px[:,0],target_px[:,1],s=10,label='reference 68')
    ax.scatter(pred_px[:,0],pred_px[:,1],s=9,marker='x',label='v10.2 projection')
    ax.set_title(title); ax.axis('off'); ax.legend(loc='lower right',fontsize=7); fig.tight_layout(.2)
    fig.savefig(out_path,bbox_inches='tight'); plt.close(fig)


def render(vertices,faces,base_r,yaw_deg,path,title):
    right,up,forward=base_r[0],base_r[1],base_r[2]; a=math.radians(yaw_deg)
    right2=math.cos(a)*right+math.sin(a)*forward; forward2=-math.sin(a)*right+math.cos(a)*forward
    basis=np.stack([right2,up,forward2]); p=vertices@basis.T; xy=p[:,:2]; tri3=p[faces]
    n=np.cross(tri3[:,1]-tri3[:,0],tri3[:,2]-tri3[:,0]); n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
    visible=n[:,2]<-0.01; faces2=faces[visible]; tri3=tri3[visible]; n=n[visible]
    if len(faces2)>40000:
        step=int(math.ceil(len(faces2)/40000)); faces2=faces2[::step];tri3=tri3[::step];n=n[::step]
    order=np.argsort(tri3[:,:,2].mean(1))[::-1]; faces2=faces2[order];n=n[order]; tri2=xy[faces2]
    diffuse=np.clip(-n[:,2],0,1); side=np.clip(-.3*n[:,0]-.25*n[:,1]-.65*n[:,2],0,1)
    inten=np.clip(.67+.20*diffuse+.11*side,.52,.98); col=np.stack([inten*.96,inten*.97,inten],axis=1)
    lo=np.percentile(xy,1.5,axis=0);hi=np.percentile(xy,98.5,axis=0);ctr=(lo+hi)/2;extent=max((hi-lo).max(),1e-6)*.57
    fig,ax=plt.subplots(figsize=(5,5),dpi=190); ax.add_collection(PolyCollection(tri2,facecolors=col,edgecolors='none',closed=True))
    ax.set_xlim(ctr[0]-extent,ctr[0]+extent); ax.set_ylim(ctr[1]+extent,ctr[1]-extent);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10)
    fig.tight_layout(.15);fig.savefig(path,bbox_inches='tight',pad_inches=.02);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--front',type=Path,required=True);ap.add_argument('--three-quarter',type=Path,required=True);ap.add_argument('--side',type=Path,required=True)
    ap.add_argument('--identity',type=Path,required=True);ap.add_argument('--cameras',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_v102'));args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);qa=args.out/'QA';qa.mkdir(exist_ok=True)
    refs={'front':image_rgb(args.front),'three_quarter':image_rgb(args.three_quarter),'side':image_rgb(args.side)}
    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cpu',face_detector='sfd')
    target_px={k:detect_68(fa,refs[k]) for k in VIEW_ORDER};target={k:normalize_target(target_px[k],refs[k].shape) for k in VIEW_ORDER}
    cams=json.loads(args.cameras.read_text())

    gnm=gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,variant=gnm_numpy.GNMVariant.HEAD)
    ident=np.load(args.identity).astype(np.float64); ident=ident.reshape(1,-1)
    vertices=np.asarray(gnm(identity=ident))[0].astype(np.float64); triangles=np.asarray(gnm.triangles,dtype=np.int64)
    skin_tri_idx=np.asarray(gnm.triangle_indices_for_group('skin'),dtype=np.int64); skin_faces_global=triangles[skin_tri_idx]; skin_ids=np.unique(skin_faces_global.reshape(-1))
    lm_cfg=gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68);lm_idx=np.asarray(lm_cfg.indices,dtype=np.int64);lm_w=np.asarray(lm_cfg.weights,dtype=np.float64)

    history=[]
    schedule=[
        ('front',.94,.043,.0060),('three_quarter',.44,.044,.0040),('side',.18,.048,.0030),
        ('front',.82,.039,.0048),('three_quarter',.32,.041,.0034),('side',.11,.045,.0024),
        ('front',.72,.036,.0038),('three_quarter',.22,.039,.0027),('front',.58,.033,.0028),
    ]
    for name,gain,sigma,maxstep in schedule:
        vertices,rec=corrective_pass(vertices,skin_ids,lm_idx,lm_w,target,cams[name],name,gain,sigma,maxstep); history.append(rec);print(json.dumps(rec))

    # One ultra-light cleanup; landmark control vertices are protected.
    protected=np.unique(lm_idx.reshape(-1)); vertices=relax_non_landmark_noise(vertices,skin_faces_global,skin_ids,protected,.025,1)

    # Export actual deformed skin while preserving original GNM skin topology.
    skin_global_to_local={int(g):i for i,g in enumerate(skin_ids)}
    sf=np.vectorize(skin_global_to_local.get)(skin_faces_global)
    sv=vertices[skin_ids]
    skin=trimesh.Trimesh(vertices=sv,faces=sf,process=False)
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.2.obj');skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.2.ply');skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.2.glb')
    # Full topology is also exported for engineering inspection; internal components remain v10.1 until identity lock.
    full=trimesh.Trimesh(vertices=vertices,faces=triangles,process=False);full.export(args.out/'AINA_FACE_MASTER_GNM_v10.2_FULL_TOPOLOGY.obj');full.export(args.out/'AINA_FACE_MASTER_GNM_v10.2_FULL_TOPOLOGY.glb')

    final_lm=landmarks_from_vertices(vertices,lm_idx,lm_w);metrics={}
    for name in VIEW_ORDER:
        pred=project(final_lm,cams[name]);err=np.linalg.norm(pred-target[name],axis=1);ids=visible_ids(name)
        metrics[name]={'rmse':float(np.sqrt(np.mean(err[ids]**2))),'median':float(np.median(err[ids])),'p90':float(np.percentile(err[ids],90))}
        save_overlay(refs[name],target_px[name],pred,qa/f'AINA_{name}_overlay_v10.2.png',f'AINA v10.2 {name}: reference vs corrective mesh')

    front_r=np.asarray(cams['front']['rotation_rows'],dtype=np.float64);paths=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p=qa/f'AINA_CLAY_{label}_v10.2.png';render(sv,sf,front_r,yaw,p,f'AINA v10.2 Clay {label.replace("_"," ")}');paths.append(p)
    ims=[Image.open(p).convert('RGB') for p in paths];h=max(i.height for i in ims);w=max(i.width for i in ims);sheet=Image.new('RGB',(w*5,h),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*w+(w-im.width)//2,(h-im.height)//2))
    sheet.save(qa/'AINA_CLAY_5VIEW_v10.2.png')
    report={'version':'AINA Face Master v10.2','base':'Google GNM v3 HEAD + v10.1 identity','method':'multi-view free-form RBF identity corrective; topology preserved','skin_vertices':int(len(sv)),'skin_triangles':int(len(sf)),'metrics':metrics,'history':history,'identity_lock':False,'note':'Actual deformed mesh. Identity lock remains false until clay five-view is visually accepted.'}
    (args.out/'AINA_v10.2_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
