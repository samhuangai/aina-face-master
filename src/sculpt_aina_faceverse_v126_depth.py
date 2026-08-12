#!/usr/bin/env python3
"""AINA v12.6 — hybrid multi-view depth sculpt on v12.5.

The front X/Y from v12.5 is frozen. The approved 3/4 view uses FaceVerse's
native MediaPipe 478 mapping for dense depth. Strict profile art is too extreme
for MediaPipe, so profile depth uses face_alignment's robust 68 points mapped
directly to FaceVerse's native 68 keypoints. The two depth fields are blended
semantically and propagated only as Z displacement over the largest head shell.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import cv2
import face_alignment
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


def detect_mp_native(path: Path):
    bgr=cv2.imread(str(path))
    if bgr is None: raise RuntimeError(f'missing image {path}')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=.18) as fm:
        res=fm.process(rgb)
    if not res.multi_face_landmarks: raise RuntimeError(f'MediaPipe failed on {path.name}')
    lm=res.multi_face_landmarks[0].landmark
    if len(lm)!=478: raise RuntimeError(f'{path.name}: expected 478, got {len(lm)}')
    return rgb,np.asarray([[x.x-.5,x.y-.5] for x in lm],np.float64)


def detect_68_native(path: Path):
    bgr=cv2.imread(str(path))
    if bgr is None: raise RuntimeError(f'missing image {path}')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
    scale=max(1.0,720.0/max(h,w))
    work=cv2.resize(rgb,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC) if scale>1 else rgb
    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cpu',face_detector='sfd')
    preds=fa.get_landmarks_from_image(work)
    if not preds: raise RuntimeError(f'68-point SFD detector failed on {path.name}')
    center=np.array([work.shape[1]*.5,work.shape[0]*.5],np.float32)
    best=min(preds,key=lambda p:float(np.linalg.norm(np.asarray(p)[:,:2].mean(0)-center)))
    pts=np.asarray(best,np.float64)[:,:2]/scale
    if pts.shape!=(68,2): raise RuntimeError(f'profile 68 shape {pts.shape}')
    # FaceVerse/MediaPipe native image convention: x,y in roughly [-.5,+.5], +Y down.
    pts[:,0]=(pts[:,0]-w*.5)/max(w,h);pts[:,1]=(pts[:,1]-h*.5)/max(w,h)
    return rgb,pts


def side68_weights():
    w=np.full(68,.06,np.float64)
    w[0:17]=.70
    w[5:12]=1.55
    w[27:36]=2.60
    w[48:60]=1.35
    w[17:27]=.15
    w[36:48]=.18
    w[60:68]=.65
    return w


def comps(nv,faces):
    e=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    a=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(nv,nv));a=(a+a.T).tocsr()
    n,lab=connected_components(a,directed=False)
    return [np.flatnonzero(lab==i) for i in range(n)]


def fit_cam(X,Y,w,robust=True):
    keep=w>0
    for _ in range(4 if robust else 1):
        ww=w[keep][:,None];XX=X[keep];YY=Y[keep];sw=max(float(ww.sum()),1e-12)
        mx=(XX*ww).sum(0)/sw;my=(YY*ww).sum(0)/sw;Xc=XX-mx;Yc=YY-my
        beta=np.linalg.lstsq(Xc*np.sqrt(ww),Yc*np.sqrt(ww),rcond=None)[0].T
        U,S,Vt=np.linalg.svd(beta,full_matrices=True);R2=U@np.eye(2,3)@Vt
        scale=max(float(np.mean(S)),1e-9);t=my-scale*(R2@mx);pred=scale*(X@R2.T)+t
        if not robust: break
        err=np.linalg.norm(pred-Y,axis=1);valid=np.flatnonzero(w>0)
        thr=float(np.percentile(err[valid],74));keep=(w>0)&(err<=thr)
    return R2,scale,t,pred,keep


def depth_from_view(X,Y,w):
    R2,s,t,pred,keep=fit_cam(X,Y,w,robust=True);res=Y-pred
    a=s*R2[:,2];den=max(float(a@a),1e-10);dz=(res@a)/den;err=np.linalg.norm(res,axis=1)
    dz[~keep]=0.
    return dz,{'scale':s,'rotation_rows':R2.tolist(),'translation':t.tolist(),'camera_inliers':int(keep.sum()),'rmse_all':float(np.sqrt(np.mean(err[w>0]**2))),'rmse_inliers':float(np.sqrt(np.mean(err[keep]**2)))}


def profile_field_on_mp(v,ids468,kp68,t68,w68):
    X68=v[kp68]
    dz68,cam=depth_from_view(X68,t68,w68)
    # Convert sparse profile constraints into a smooth front-plane depth field
    # evaluated at the dense 468 skin controls.
    rbf=RBFInterpolator(X68[:,:2],dz68[:,None],kernel='thin_plate_spline',smoothing=2.5e-6,degree=1)
    return rbf(v[ids468,:2])[:,0],cam,float(np.sqrt(np.mean(dz68[w68>0]**2)))


def clamp1(x,cap): return np.clip(x,-cap,cap)

def area(v,f):
    t=v[f];return .5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)

def render(v,f,yaw,path,title):
    a=math.radians(yaw);c=math.cos(a);s=math.sin(a);p=v.copy();x=c*p[:,0]+s*p[:,2];z=-s*p[:,0]+c*p[:,2];p[:,0]=x;p[:,2]=z
    tri=p[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
    order=np.argsort(-tri[:,:,2].mean(1));tri2=p[f[order],:2];nn=n[order]
    it=np.clip(.66+.22*np.abs(nn[:,2])+.08*np.clip(-.25*nn[:,0]-.18*nn[:,1]+.72*nn[:,2],0,1),.52,.98);col=np.stack([it*.96,it*.975,it],1)
    xy=p[:,:2];lo=np.percentile(xy,1.5,0);hi=np.percentile(xy,98.5,0);ctr=.5*(lo+hi);ext=max(float((hi-lo).max()),1e-6)*.57
    fig,ax=plt.subplots(figsize=(5,5),dpi=190);ax.add_collection(PolyCollection(tri2,facecolors=col,edgecolors='none'));ax.set_xlim(ctr[0]-ext,ctr[0]+ext);ax.set_ylim(ctr[1]+ext,ctr[1]-ext);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10);fig.tight_layout(pad=.12);fig.savefig(path,bbox_inches='tight',pad_inches=.02);plt.close(fig)
def compare(a,b,o):
    x=Image.open(a).convert('RGB');y=Image.open(b).convert('RGB');H=max(x.height,y.height);xw=int(x.width*H/x.height);yw=int(y.width*H/y.height);s=Image.new('RGB',(xw+yw,H),'white');s.paste(x.resize((xw,H)),(0,0));s.paste(y.resize((yw,H)),(xw,0));s.save(o)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-full',type=Path,required=True);ap.add_argument('--three-quarter',type=Path,required=True);ap.add_argument('--side',type=Path,required=True);ap.add_argument('--faceverse-data',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_faceverse_v126'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);qa=args.out/'QA';qa.mkdir(exist_ok=True)
    mesh=trimesh.load(args.base_full,process=False,maintain_order=True);v=np.asarray(mesh.vertices,np.float64);f=np.asarray(mesh.faces,np.int64);v0=v.copy()
    rgbq,tq=detect_mp_native(args.three_quarter);rgbs,ts68=detect_68_native(args.side)
    fvd=np.load(args.faceverse_data,allow_pickle=True).item();mpinds=np.asarray(fvd['keypoints_mediapipe']).reshape(-1).astype(np.int64);kp68=np.asarray(fvd['keypoints']).reshape(-1).astype(np.int64)
    if len(kp68)!=68: raise RuntimeError(f'FaceVerse keypoints has {len(kp68)}, expected 68')
    weights=sem.make_weights().astype(np.float64);active=np.flatnonzero(weights>0);ids=mpinds[active];w=weights[active]
    q_sem=np.ones(478,np.float64);q_sem[np.asarray(sem.NOSE)]=1.55;q_sem[np.asarray(sem.LIPS)]=1.35;q_sem[np.asarray(sem.FACE_OVAL)]=1.35;q_sem[np.asarray(sem.IRIS)]=0.
    w68=side68_weights()

    X=v[ids];dzq,camq=depth_from_view(X,tq[active],w*q_sem[active]);dzs,cams,profile_dz_rms=profile_field_on_mp(v,ids,kp68,ts68,w68)
    alpha=np.full(len(active),.24,np.float64);index={g:i for i,g in enumerate(active.tolist())}
    for g in sem.NOSE+sem.LIPS+sem.FACE_OVAL:
        if g in index: alpha[index[g]]=.60
    for g in sem.LEFT_EYE+sem.RIGHT_EYE+sem.LEFT_BROW+sem.RIGHT_BROW:
        if g in index: alpha[index[g]]=.12
    dz=clamp1(alpha*dzs+(1-alpha)*dzq,.0060)

    cs=comps(len(v),f);head=max(cs,key=len);mask=np.zeros(len(v),bool);mask[head]=True
    if not np.all(mask[ids]) or not np.all(mask[kp68]): raise RuntimeError('FaceVerse controls escaped largest head shell')
    hf=f[mask[f].all(axis=1)];a0=area(v,hf);passes=[]
    for pi,gain in enumerate((.78,.42)):
        ctrl=v[ids,:2];rbf=RBFInterpolator(ctrl,dz[:,None],kernel='thin_plate_spline',smoothing=1.6e-6,degree=1);hp=v[head];field=rbf(hp[:,:2])[:,0]
        tree=cKDTree(ctrl);dxy,_=tree.query(hp[:,:2]);xyg=np.exp(-.5*(dxy/.020)**4);fz=float(np.median(v[ids,2]));zg=np.exp(-.5*((hp[:,2]-fz)/.048)**4)
        delta=gain*field*xyg*zg;delta=np.clip(delta,-(.0048 if pi==0 else .0024),(.0048 if pi==0 else .0024));v[head,2]+=delta
        X2=v[ids];dzq2,camq2=depth_from_view(X2,tq[active],w*q_sem[active]);dzs2,cams2,profile_dz_rms=profile_field_on_mp(v,ids,kp68,ts68,w68);dz=clamp1(alpha*dzs2+(1-alpha)*dzq2,.0030);camq,cams=camq2,cams2
        passes.append({'pass':pi,'gain':gain,'max_depth_vertex_shift_m':float(np.max(np.abs(delta))),'remaining_control_depth_rms_m':float(np.sqrt(np.mean(dz**2))),'profile_68_depth_rms_m':profile_dz_rms,'three_quarter_camera_rmse':camq['rmse_inliers'],'side68_camera_rmse':cams['rmse_inliers']});print(json.dumps(passes[-1]))

    a1=area(v,hf);rat=a1/np.maximum(a0,1e-12);q01=float(np.percentile(rat,1));q99=float(np.percentile(rat,99));mx=float(np.max(np.linalg.norm(v-v0,axis=1)))
    if q01<.35 or q99>2.8: raise RuntimeError(f'depth sculpt quality fail p01={q01:.4f} p99={q99:.4f}')
    outm=trimesh.Trimesh(vertices=v,faces=f,process=False);outm.export(args.out/'AINA_FACEVERSE_FULL_v12.6_MULTIVIEW_DEPTH.obj');outm.export(args.out/'AINA_FACEVERSE_FULL_v12.6_MULTIVIEW_DEPTH.glb');outm.export(args.out/'AINA_FACEVERSE_FULL_v12.6_MULTIVIEW_DEPTH.ply')
    views=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p=qa/f'AINA_FACEVERSE_CLAY_{label}_v12.6.png';render(v,f,yaw,p,f'AINA FaceVerse v12.6 {label}');views.append(p)
    ims=[Image.open(x).convert('RGB') for x in views];H=max(x.height for x in ims);W=max(x.width for x in ims);sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_FACEVERSE_CLAY_5VIEW_v12.6.png');compare(args.three_quarter,qa/'AINA_FACEVERSE_CLAY_left_45_v12.6.png',qa/'AINA_REFERENCE3Q_VS_CLAY45_v12.6.png');compare(args.side,qa/'AINA_FACEVERSE_CLAY_left_profile_v12.6.png',qa/'AINA_REFERENCE_SIDE_VS_CLAY_PROFILE_v12.6.png')
    rep={'version':'AINA FaceVerse v12.6 Hybrid Multi-View Depth Sculpt','base':'v12.5 front-dense residual fit','front_xy_changed':False,'topology_changed':False,'profile_detector':'face_alignment SFD 68 -> FaceVerse native 68','max_total_depth_shift_m':mx,'triangle_area_ratio_p01':q01,'triangle_area_ratio_p99':q99,'three_quarter_camera':camq,'side68_camera':cams,'passes':passes,'identity_lock':False,'acceptance_note':'Front shape is frozen; this version passes only if 3Q/profile visual identity also matches the approved AINA views.'};(args.out/'AINA_FACEVERSE_v12.6_REPORT.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__': main()
