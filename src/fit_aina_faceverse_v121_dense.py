#!/usr/bin/env python3
"""AINA v12.1 — dense FaceVerse identity/expression fit to approved effect art.

The GNM route hit an anatomical identity ceiling. FaceVerse V4 supplies a much
richer coherent face model and, crucially, a native mapping to all 478
MediaPipe landmarks. This fitter optimizes identity plus a strongly-regularized
expression vector directly against the approved AINA front art. It therefore
uses eyelid/nose/lip/oval geometry rather than only 68 sparse points.

Acceptance remains visual Clay QA; numerical landmark error is diagnostic only.
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import torch
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image

ROOT=Path.cwd().resolve()
FVROOT=(ROOT/'vendor/faceverse-onnx').resolve()
sys.path.insert(0,str(FVROOT))
from faceversev4 import FaceVerseModel_torch

# MediaPipe semantic regions. These are not a sparse replacement for 478 pts;
# they merely tell the optimizer which parts are most identity-critical.
FACE_OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
LEFT_EYE=[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE=[362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
LEFT_BROW=[70,63,105,66,107,55,65,52,53,46]
RIGHT_BROW=[336,296,334,293,300,285,295,282,283,276]
NOSE=[1,2,4,5,6,19,94,97,98,129,168,195,197,326,327,358]
LIPS=[61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,80,191,78]
IRIS=list(range(468,478))


def load_rgb(path: Path):
    bgr=cv2.imread(str(path))
    if bgr is None: raise RuntimeError(f'missing image: {path}')
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)


def bbox_from_68(path: Path, image_shape, pad=.31):
    d=json.loads(path.read_text()); pts=np.asarray(d['landmarks_xy'],np.float64)
    lo=pts.min(0);hi=pts.max(0);c=.5*(lo+hi);size=max(float((hi-lo).max()),1.)*(1+2*pad)
    h,w=image_shape[:2]
    return np.array([max(0,c[0]-size*.5),max(0,c[1]-size*.53),min(w,c[0]+size*.5),min(h,c[1]+size*.47)],np.float32)


def heuristic_bbox(img, scale=.88):
    h,w=img.shape[:2];s=min(h,w)*scale;cx=w*.5;cy=h*.50
    return np.array([cx-s*.5,cy-s*.52,cx+s*.5,cy+s*.48],np.float32)


def preprocess(img,bbox):
    x1,y1,x2,y2=np.round(bbox).astype(int);x1=max(0,x1);y1=max(0,y1);x2=min(img.shape[1],x2);y2=min(img.shape[0],y2)
    crop=img[y1:y2,x1:x2]
    if crop.size==0: raise RuntimeError('empty FaceVerse crop')
    r=cv2.resize(crop,(256,256),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.
    return np.transpose(r,(2,0,1))[None]


def infer(session,img,bbox):
    return np.asarray(session.run([session.get_outputs()[0].name],{session.get_inputs()[0].name:preprocess(img,bbox)})[0][0],np.float32)


def detect_mp478(rgb):
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=.20) as fm:
        res=fm.process(rgb)
    if not res.multi_face_landmarks: raise RuntimeError('MediaPipe 478 failed on approved AINA front art')
    lm=res.multi_face_landmarks[0].landmark
    if len(lm)!=478: raise RuntimeError(f'expected 478 target landmarks, got {len(lm)}')
    # Model space is +Y up; MediaPipe image coordinates are +Y down.
    return np.asarray([[x.x-.5,.5-x.y] for x in lm],np.float32)


def selected_basis(fvd, inds, key, dims):
    base=np.asarray(fvd[key],np.float32)
    rows=(inds[:,None]*3+np.arange(3)[None,:]).reshape(-1)
    return base[rows,:dims].reshape(len(inds),3,dims)/100.0


def make_weights(n=478):
    w=np.ones(n,np.float32)
    for ids,val in [(FACE_OVAL,4.3),(LEFT_EYE,5.3),(RIGHT_EYE,5.3),(LEFT_BROW,2.0),(RIGHT_BROW,2.0),(NOSE,4.8),(LIPS,4.8)]:
        w[np.asarray(ids,np.int64)]=np.maximum(w[np.asarray(ids,np.int64)],val)
    # Iris positions are affected by eyeball rotation and should not drive skin identity.
    w[np.asarray(IRIS,np.int64)]=0.0
    return w


def init_camera(xy,target,w):
    active=w>0
    X=xy[active];Y=target[active];ww=w[active][:,None]
    mx=(X*ww).sum(0)/ww.sum();my=(Y*ww).sum(0)/ww.sum()
    Xc=X-mx;Yc=Y-my
    num=float((ww*Xc*Yc).sum());den=float((ww*Xc*Xc).sum())
    s=num/max(den,1e-9)
    if s<=0:
        s=float(np.sqrt((ww*Yc*Yc).sum()/max((ww*Xc*Xc).sum(),1e-9)))
    t=my-s*mx
    return s,t


def optimize_dense(mean,idb,expb,id0,exp0,target,weights,steps=520):
    device=torch.device('cpu')
    mean_t=torch.from_numpy(mean).to(device)
    idb_t=torch.from_numpy(idb).to(device)
    expb_t=torch.from_numpy(expb).to(device)
    target_t=torch.from_numpy(target).to(device)
    w_t=torch.from_numpy(weights).to(device)

    # Start with a restrained amount of inferred expression so the source network
    # may contribute eyelid aperture while mouth remains close to neutral.
    idv=torch.nn.Parameter(torch.from_numpy(id0[None]).float().to(device))
    exp_seed=(.42*exp0).astype(np.float32)
    expv=torch.nn.Parameter(torch.from_numpy(exp_seed[None]).float().to(device))

    with torch.no_grad():
        lm0=mean_t+torch.einsum('vci,bi->bvc',idb_t,idv)[0]+torch.einsum('vce,be->bvc',expb_t,expv)[0]
    s0,t0=init_camera(lm0[:,:2].numpy(),target,weights)
    log_s=torch.nn.Parameter(torch.tensor([math.log(max(abs(s0),1e-5))],dtype=torch.float32))
    trans=torch.nn.Parameter(torch.from_numpy(t0[None].astype(np.float32)))

    opt=torch.optim.Adam([
        {'params':[idv],'lr':.018},
        {'params':[expv],'lr':.010},
        {'params':[log_s,trans],'lr':.008},
    ])
    id_ref=torch.from_numpy(id0[None]).float(); exp_ref=torch.from_numpy(exp_seed[None]).float()
    hist=[]
    critical=torch.tensor(sorted(set(FACE_OVAL+LEFT_EYE+RIGHT_EYE+NOSE+LIPS)),dtype=torch.long)
    for step in range(steps):
        opt.zero_grad()
        lm=mean_t+torch.einsum('vci,bi->bvc',idb_t,idv)[0]+torch.einsum('vce,be->bvc',expb_t,expv)[0]
        pred=torch.exp(log_s)*lm[:,:2]+trans[0]
        err=pred-target_t
        data=((err.square().sum(1))*w_t).sum()/torch.clamp(w_t.sum(),min=1.)
        crit=torch.mean(err[critical].square())
        # Identity can move materially but stays near network estimate; expression
        # is much more tightly regularized to avoid baking a smile/grimace.
        rid=torch.mean((idv-id_ref).square())
        rexp=torch.mean((expv-exp_ref).square())
        # Penalize mouth-driving expression coefficient 49 and tongue tail.
        mouth_guard=expv[:,49].square().mean()+.25*expv[:,171:].square().mean()
        loss=data+1.5*crit+1.5e-4*rid+7.5e-4*rexp+1.0e-3*mouth_guard
        loss.backward(); torch.nn.utils.clip_grad_norm_([idv,expv],5.0);opt.step()
        if step in (0,24,49,99,199,319,419,steps-1):
            hist.append({'step':step,'loss':float(loss.detach()),'data':float(data.detach()),'critical':float(crit.detach()),'id_delta_rms':float(torch.sqrt(rid).detach()),'exp_delta_rms':float(torch.sqrt(rexp).detach()),'scale':float(torch.exp(log_s).detach()),'tx':float(trans[0,0].detach()),'ty':float(trans[0,1].detach())})
            print(json.dumps(hist[-1]))
    with torch.no_grad():
        lm=mean_t+torch.einsum('vci,bi->bvc',idb_t,idv)[0]+torch.einsum('vce,be->bvc',expb_t,expv)[0]
        pred=torch.exp(log_s)*lm[:,:2]+trans[0]
        e=torch.linalg.norm(pred-target_t,dim=1).numpy()
    return idv.detach().numpy()[0],expv.detach().numpy()[0],pred.numpy(),e,hist


def normalize_metric(v):
    v=np.asarray(v,np.float64).copy();c=np.median(v,axis=0);v-=c
    h=float(np.percentile(v[:,1],99)-np.percentile(v[:,1],1));s=.180/max(h,1e-9)
    return v*s,s,c


def render(v,f,yaw,path,title):
    a=math.radians(yaw);c=math.cos(a);s=math.sin(a)
    p=v.copy();x=c*p[:,0]+s*p[:,2];z=-s*p[:,0]+c*p[:,2];p[:,0]=x;p[:,2]=z
    tri=p[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
    # FaceVerse front surface has larger Z and must be painted last. Descending
    # depth in the collection plus the model's +Y-up display gives a real face,
    # not the misleading back/head surface from v12.0 QA.
    order=np.argsort(-tri[:,:,2].mean(1));tri2=p[f[order],:2];nn=n[order]
    dif=np.clip(np.abs(nn[:,2]),0,1);side=np.clip(-.25*nn[:,0]-.18*nn[:,1]+.72*nn[:,2],0,1);it=np.clip(.66+.22*dif+.09*side,.52,.98)
    col=np.stack([it*.96,it*.975,it],1)
    xy=p[:,:2];lo=np.percentile(xy,1.5,0);hi=np.percentile(xy,98.5,0);ctr=.5*(lo+hi);ext=max(float((hi-lo).max()),1e-6)*.57
    fig,ax=plt.subplots(figsize=(5,5),dpi=190);ax.add_collection(PolyCollection(tri2,facecolors=col,edgecolors='none'))
    ax.set_xlim(ctr[0]-ext,ctr[0]+ext);ax.set_ylim(ctr[1]-ext,ctr[1]+ext);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10)
    fig.tight_layout(pad=.12);fig.savefig(path,bbox_inches='tight',pad_inches=.02);plt.close(fig)


def landmark_overlay(rgb,target,pred,path):
    h,w=rgb.shape[:2];fig,ax=plt.subplots(figsize=(6,6),dpi=180);ax.imshow(rgb)
    tx=(target[:,0]+.5)*w;ty=(.5-target[:,1])*h;px=(pred[:,0]+.5)*w;py=(.5-pred[:,1])*h
    ax.scatter(tx,ty,s=4,label='AINA target');ax.scatter(px,py,s=4,marker='+',label='FaceVerse fit');ax.legend(loc='lower right',fontsize=7);ax.axis('off');fig.tight_layout(pad=0);fig.savefig(path,bbox_inches='tight',pad_inches=0);plt.close(fig)


def side_by_side(ref,act,out):
    a=Image.open(ref).convert('RGB');b=Image.open(act).convert('RGB');H=max(a.height,b.height);aw=int(a.width*H/a.height);bw=int(b.width*H/b.height)
    s=Image.new('RGB',(aw+bw,H),'white');s.paste(a.resize((aw,H)),(0,0));s.paste(b.resize((bw,H)),(aw,0));s.save(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--front',type=Path,required=True);ap.add_argument('--three-quarter',type=Path,required=True);ap.add_argument('--front-landmarks',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_faceverse_v121'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);qa=args.out/'QA';qa.mkdir(exist_ok=True)
    front=load_rgb(args.front);q=load_rgb(args.three_quarter);target=detect_mp478(front);weights=make_weights()

    sess=ort.InferenceSession(str(FVROOT/'data/faceverse_resnet50_float32.onnx'),providers=['CPUExecutionProvider'])
    cf=infer(sess,front,bbox_from_68(args.front_landmarks,front.shape));cq=infer(sess,q,heuristic_bbox(q))
    model=FaceVerseModel_torch(device=torch.device('cpu'),facevrsepath=str(FVROOT/'data/faceverse_v4_2.npy'),camera_distance=10,focal=1000,center=128)
    idd=int(model.id_dims);expd=int(model.exp_dims)
    id0=.72*cf[:idd]+.28*cq[:idd]
    exp0=cf[idd:idd+expd].copy()
    mp_inds=np.asarray(model.fvd['keypoints_mediapipe']).reshape(-1).astype(np.int64)
    if len(mp_inds)!=478: raise RuntimeError(f'FaceVerse keypoints_mediapipe has {len(mp_inds)}, expected 478')
    mean=np.asarray(model.fvd['meanshape'],np.float32)[mp_inds]/100.0
    idb=selected_basis(model.fvd,mp_inds,'idBase',idd);expb=selected_basis(model.fvd,mp_inds,'exBase',expd)

    id_fit,exp_fit,pred,e,hist=optimize_dense(mean,idb,expb,id0.astype(np.float32),exp0.astype(np.float32),target,weights)
    # Build final coherent FaceVerse mesh with fitted shape/expression; eye gaze is
    # zero so eyeballs face straight ahead. The optimized expression is baked only
    # because the target art's eyelid aperture is part of the requested identity.
    ztex=np.zeros(int(model.tex_dims),np.float32);zg=np.zeros(27,np.float32);za=np.zeros(3,np.float32);zt=np.zeros(3,np.float32);ze=np.zeros(4,np.float32)
    coeff=np.concatenate([id_fit,exp_fit,ztex,zg,za,zt,ze])[None]
    with torch.no_grad(): result=model.run(torch.from_numpy(coeff).float(),only_lms=False,use_color=False)
    verts=np.asarray(result['vertices'][0].cpu(),np.float64);faces=np.asarray(model.tri.cpu(),np.int64)
    metric,metric_scale,_=normalize_metric(verts);mesh=trimesh.Trimesh(vertices=metric,faces=faces,process=False)
    mesh.export(args.out/'AINA_FACEVERSE_FULL_v12.1_DENSE_FIT.obj');mesh.export(args.out/'AINA_FACEVERSE_FULL_v12.1_DENSE_FIT.glb');mesh.export(args.out/'AINA_FACEVERSE_FULL_v12.1_DENSE_FIT.ply')
    np.save(args.out/'AINA_FACEVERSE_IDENTITY_156_v12.1.npy',id_fit.astype(np.float32));np.save(args.out/'AINA_FACEVERSE_EXPRESSION_177_v12.1.npy',exp_fit.astype(np.float32));np.save(args.out/'AINA_TARGET_MP478_v12.1.npy',target.astype(np.float32))

    landmark_overlay(front,target,pred,qa/'AINA_MP478_OVERLAY_v12.1.png')
    views=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p=qa/f'AINA_FACEVERSE_CLAY_{label}_v12.1.png';render(metric,faces,yaw,p,f'AINA FaceVerse v12.1 {label}');views.append(p)
    ims=[Image.open(x).convert('RGB') for x in views];H=max(x.height for x in ims);W=max(x.width for x in ims);sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_FACEVERSE_CLAY_5VIEW_v12.1.png');side_by_side(args.front,qa/'AINA_FACEVERSE_CLAY_front_v12.1.png',qa/'AINA_REFERENCE_VS_FACEVERSE_FRONT_v12.1.png')

    active=weights>0;critical=np.asarray(sorted(set(FACE_OVAL+LEFT_EYE+RIGHT_EYE+NOSE+LIPS)),np.int64)
    report={'version':'AINA FaceVerse v12.1 Dense Identity Fit','source':'FaceVerse V4 + approved AINA MediaPipe 478','vertices':int(len(verts)),'faces':int(len(faces)),'identity_dims':idd,'expression_dims':expd,'active_landmarks':int(active.sum()),'weighted_rmse':float(np.sqrt(np.sum(weights[active]*e[active]**2)/np.sum(weights[active]))),'critical_rmse':float(np.sqrt(np.mean(e[critical]**2))),'median_error':float(np.median(e[active])),'p90_error':float(np.percentile(e[active],90)),'metric_scale':float(metric_scale),'optimization_history':hist,'identity_lock':False,'acceptance_note':'Only actual front/45/profile Clay likeness can pass this version. Dense RMSE is not identity acceptance.'}
    (args.out/'AINA_FACEVERSE_v12.1_REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__': main()
