#!/usr/bin/env python3
"""AINA v12.6 — multi-view depth sculpt on v12.5.

v12.5 is intentionally frozen in frontal X/Y because its 468 skin points match
the approved front art closely. v12.6 uses approved 3/4 and strict-profile
MediaPipe correspondences only to infer Z/depth corrections. Each view receives
a robust scaled-orthographic camera; pointwise image residuals are converted to
front-axis depth deltas, blended by semantic region, then propagated as a smooth
scalar field over the largest head/skin shell. No frontal X/Y is changed.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import fit_aina_faceverse_v121_dense as sem


def detect_native(path: Path):
    bgr=cv2.imread(str(path))
    if bgr is None: raise RuntimeError(f'missing image {path}')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=.18) as fm:
        res=fm.process(rgb)
    if not res.multi_face_landmarks: raise RuntimeError(f'MediaPipe failed on {path.name}')
    lm=res.multi_face_landmarks[0].landmark
    if len(lm)!=478: raise RuntimeError(f'{path.name}: expected 478, got {len(lm)}')
    return rgb,np.asarray([[x.x-.5,x.y-.5] for x in lm],np.float64)


def comps(nv,faces):
    e=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);a=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(nv,nv));a=(a+a.T).tocsr();n,lab=connected_components(a,directed=False);return [np.flatnonzero(lab==i) for i in range(n)]


def fit_cam(X,Y,w,robust=True):
    keep=w>0
    for _ in range(3 if robust else 1):
        ww=w[keep][:,None];XX=X[keep];YY=Y[keep];sw=max(float(ww.sum()),1e-12);mx=(XX*ww).sum(0)/sw;my=(YY*ww).sum(0)/sw;Xc=XX-mx;Yc=YY-my
        beta=np.linalg.lstsq(Xc*np.sqrt(ww),Yc*np.sqrt(ww),rcond=None)[0].T
        U,S,Vt=np.linalg.svd(beta,full_matrices=True);R2=U@np.eye(2,3)@Vt;scale=max(float(np.mean(S)),1e-9);t=my-scale*(R2@mx);pred=scale*(X@R2.T)+t
        if not robust:break
        err=np.linalg.norm(pred-Y,axis=1);valid=np.flatnonzero(w>0);thr=float(np.percentile(err[valid],72));keep=(w>0)&(err<=thr)
    return R2,scale,t,pred,keep


def depth_from_view(X,Y,w):
    R2,s,t,pred,keep=fit_cam(X,Y,w,robust=True);res=Y-pred;a=s*R2[:,2];den=max(float(a@a),1e-10);dz=(res@a)/den;err=np.linalg.norm(res,axis=1)
    # Detector points rejected from the robust camera are not trusted for depth.
    dz[~keep]=0.;return dz,{'scale':s,'rotation_rows':R2.tolist(),'translation':t.tolist(),'camera_inliers':int(keep.sum()),'rmse_all':float(np.sqrt(np.mean(err[w>0]**2))),'rmse_inliers':float(np.sqrt(np.mean(err[keep]**2)))}


def clamp1(x,cap):return np.clip(x,-cap,cap)

def area(v,f):
    t=v[f];return .5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)

def render(v,f,yaw,path,title):
    a=math.radians(yaw);c=math.cos(a);s=math.sin(a);p=v.copy();x=c*p[:,0]+s*p[:,2];z=-s*p[:,0]+c*p[:,2];p[:,0]=x;p[:,2]=z;tri=p[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9);order=np.argsort(-tri[:,:,2].mean(1));tri2=p[f[order],:2];nn=n[order];it=np.clip(.66+.22*np.abs(nn[:,2])+.08*np.clip(-.25*nn[:,0]-.18*nn[:,1]+.72*nn[:,2],0,1),.52,.98);col=np.stack([it*.96,it*.975,it],1);xy=p[:,:2];lo=np.percentile(xy,1.5,0);hi=np.percentile(xy,98.5,0);ctr=.5*(lo+hi);ext=max(float((hi-lo).max()),1e-6)*.57;fig,ax=plt.subplots(figsize=(5,5),dpi=190);ax.add_collection(PolyCollection(tri2,facecolors=col,edgecolors='none'));ax.set_xlim(ctr[0]-ext,ctr[0]+ext);ax.set_ylim(ctr[1]+ext,ctr[1]-ext);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10);fig.tight_layout(pad=.12);fig.savefig(path,bbox_inches='tight',pad_inches=.02);plt.close(fig)

def compare(a,b,o):
    x=Image.open(a).convert('RGB');y=Image.open(b).convert('RGB');H=max(x.height,y.height);xw=int(x.width*H/x.height);yw=int(y.width*H/y.height);s=Image.new('RGB',(xw+yw,H),'white');s.paste(x.resize((xw,H)),(0,0));s.paste(y.resize((yw,H)),(xw,0));s.save(o)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-full',type=Path,required=True);ap.add_argument('--three-quarter',type=Path,required=True);ap.add_argument('--side',type=Path,required=True);ap.add_argument('--faceverse-data',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_faceverse_v126'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);qa=args.out/'QA';qa.mkdir(exist_ok=True)
    mesh=trimesh.load(args.base_full,process=False,maintain_order=True);v=np.asarray(mesh.vertices,np.float64);f=np.asarray(mesh.faces,np.int64);v0=v.copy();rgbq,tq=detect_native(args.three_quarter);rgbs,ts=detect_native(args.side);fvd=np.load(args.faceverse_data,allow_pickle=True).item();mpinds=np.asarray(fvd['keypoints_mediapipe']).reshape(-1).astype(np.int64)
    weights=sem.make_weights().astype(np.float64);active=np.flatnonzero(weights>0);ids=mpinds[active];X=v[ids];w=weights[active]
    # Side camera relies mostly on silhouette/nose/lips; 3Q remains broadly useful.
    side_sem=np.ones(478,np.float64)*.28;side_sem[np.asarray(sem.FACE_OVAL)]=1.25;side_sem[np.asarray(sem.NOSE)]=1.8;side_sem[np.asarray(sem.LIPS)]=1.2;side_sem[np.asarray(sem.LEFT_EYE+sem.RIGHT_EYE)]=.35;side_sem[np.asarray(sem.IRIS)]=0.
    q_sem=np.ones(478,np.float64);q_sem[np.asarray(sem.NOSE)]=1.5;q_sem[np.asarray(sem.LIPS)]=1.3;q_sem[np.asarray(sem.FACE_OVAL)]=1.25;q_sem[np.asarray(sem.IRIS)]=0.
    dzq,camq=depth_from_view(X,tq[active],w*q_sem[active]);dzs,cams=depth_from_view(X,ts[active],w*side_sem[active])
    # Region-dependent blending: profile dominates nose/lips/oval, 3Q dominates eyes/cheeks.
    alpha=np.full(len(active),.30,np.float64);active_global=active.tolist();index={g:i for i,g in enumerate(active_global)}
    for g in sem.NOSE+sem.LIPS+sem.FACE_OVAL:
        if g in index:alpha[index[g]]=.62
    for g in sem.LEFT_EYE+sem.RIGHT_EYE+sem.LEFT_BROW+sem.RIGHT_BROW:
        if g in index:alpha[index[g]]=.20
    dz=alpha*dzs+(1-alpha)*dzq;dz=clamp1(dz,.0060)

    cs=comps(len(v),f);head=max(cs,key=len);mask=np.zeros(len(v),bool);mask[head]=True
    if not np.all(mask[ids]):raise RuntimeError('active MP controls escaped largest head shell')
    hf=f[mask[f].all(axis=1)];a0=area(v,hf);passes=[]
    for pi,gain in enumerate((.82,.46)):
        ctrl=v[ids,:2];rbf=RBFInterpolator(ctrl,dz[:,None],kernel='thin_plate_spline',smoothing=1.5e-6,degree=1);hp=v[head];field=rbf(hp[:,:2])[:,0];tree=cKDTree(ctrl);dxy,_=tree.query(hp[:,:2]);xyg=np.exp(-.5*(dxy/.020)**4);fz=float(np.median(v[ids,2]));zg=np.exp(-.5*((hp[:,2]-fz)/.048)**4);delta=gain*field*xyg*zg;delta=np.clip(delta,-(.0050 if pi==0 else .0025),(.0050 if pi==0 else .0025));v[head,2]+=delta
        # Recompute residual depth requests after first pass so the polish corrects remaining depth.
        X2=v[ids];dzq2,camq2=depth_from_view(X2,tq[active],w*q_sem[active]);dzs2,cams2=depth_from_view(X2,ts[active],w*side_sem[active]);dz=clamp1(alpha*dzs2+(1-alpha)*dzq2,.0032);camq, cams = camq2, cams2
        passes.append({'pass':pi,'gain':gain,'max_depth_vertex_shift_m':float(np.max(np.abs(delta))),'remaining_control_depth_rms_m':float(np.sqrt(np.mean(dz**2))),'three_quarter_camera_rmse':camq['rmse_inliers'],'side_camera_rmse':cams['rmse_inliers']});print(json.dumps(passes[-1]))
    a1=area(v,hf);rat=a1/np.maximum(a0,1e-12);q01=float(np.percentile(rat,1));q99=float(np.percentile(rat,99));mx=float(np.max(np.linalg.norm(v-v0,axis=1)))
    if q01<.35 or q99>2.8:raise RuntimeError(f'depth sculpt quality fail p01={q01:.4f} p99={q99:.4f}')
    outm=trimesh.Trimesh(vertices=v,faces=f,process=False);outm.export(args.out/'AINA_FACEVERSE_FULL_v12.6_MULTIVIEW_DEPTH.obj');outm.export(args.out/'AINA_FACEVERSE_FULL_v12.6_MULTIVIEW_DEPTH.glb');outm.export(args.out/'AINA_FACEVERSE_FULL_v12.6_MULTIVIEW_DEPTH.ply')
    views=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p=qa/f'AINA_FACEVERSE_CLAY_{label}_v12.6.png';render(v,f,yaw,p,f'AINA FaceVerse v12.6 {label}');views.append(p)
    ims=[Image.open(x).convert('RGB') for x in views];H=max(x.height for x in ims);W=max(x.width for x in ims);sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_FACEVERSE_CLAY_5VIEW_v12.6.png');compare(args.three_quarter,qa/'AINA_FACEVERSE_CLAY_left_45_v12.6.png',qa/'AINA_REFERENCE3Q_VS_CLAY45_v12.6.png');compare(args.side,qa/'AINA_FACEVERSE_CLAY_left_profile_v12.6.png',qa/'AINA_REFERENCE_SIDE_VS_CLAY_PROFILE_v12.6.png')
    rep={'version':'AINA FaceVerse v12.6 Multi-View Depth Sculpt','base':'v12.5 front-dense residual fit','front_xy_changed':False,'topology_changed':False,'max_total_depth_shift_m':mx,'triangle_area_ratio_p01':q01,'triangle_area_ratio_p99':q99,'three_quarter_camera':camq,'side_camera':cams,'passes':passes,'identity_lock':False,'acceptance_note':'Front shape is frozen; this version passes only if 3Q/profile visual identity also matches the approved AINA views.'};(args.out/'AINA_FACEVERSE_v12.6_REPORT.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
