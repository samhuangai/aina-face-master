#!/usr/bin/env python3
"""AINA v12.2 — strictly regularized FaceVerse dense fit.

v12.1 exposed a reference-storage alias: the trainable coefficient tensor and
its regularization reference shared NumPy memory, so the reported regularizer
stayed zero while identity coefficients escaped the valid FaceVerse region.
v12.2 fixes that bug, constrains every identity coefficient to id0 +/- 0.22,
and permits only the two documented eyelid expression channels (14,15).
"""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import torch
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image

import fit_aina_faceverse_v121_dense as base

ROOT=Path.cwd().resolve();FVROOT=(ROOT/'vendor/faceverse-onnx').resolve()
from faceversev4 import FaceVerseModel_torch


def optimize_strict(mean,idb,expb,id0,exp0,target,weights,steps=440):
    mean_t=torch.from_numpy(mean.copy()).float()
    idb_t=torch.from_numpy(idb.copy()).float()
    expb_t=torch.from_numpy(expb.copy()).float()
    target_t=torch.from_numpy(target.copy()).float()
    w_t=torch.from_numpy(weights.copy()).float()

    # IMPORTANT: clone() here breaks the v12.1 NumPy/Torch storage alias.
    id_ref=torch.tensor(id0.copy(),dtype=torch.float32).unsqueeze(0).clone().detach()
    idv=torch.nn.Parameter(id_ref.clone())
    eye_seed=np.clip(.35*exp0[14:16],-.25,.25).astype(np.float32)
    eye_ref=torch.tensor(eye_seed,dtype=torch.float32).unsqueeze(0).clone().detach()
    eyev=torch.nn.Parameter(eye_ref.clone())
    selector=torch.zeros((2,expb.shape[2]),dtype=torch.float32)
    selector[0,14]=1.;selector[1,15]=1.

    def expression(): return eyev@selector
    with torch.no_grad():
        lm0=mean_t+torch.einsum('vci,bi->bvc',idb_t,idv)[0]+torch.einsum('vce,be->bvc',expb_t,expression())[0]
    s0,t0=base.init_camera(lm0[:,:2].numpy(),target,weights)
    log_s=torch.nn.Parameter(torch.tensor([math.log(max(abs(s0),1e-5))],dtype=torch.float32))
    trans=torch.nn.Parameter(torch.tensor(t0[None].copy(),dtype=torch.float32))
    opt=torch.optim.Adam([
        {'params':[idv],'lr':.0060},
        {'params':[eyev],'lr':.0080},
        {'params':[log_s,trans],'lr':.0050},
    ])
    critical=torch.tensor(sorted(set(base.FACE_OVAL+base.LEFT_EYE+base.RIGHT_EYE+base.NOSE+base.LIPS)),dtype=torch.long)
    hist=[]
    for step in range(steps):
        opt.zero_grad();expv=expression()
        lm=mean_t+torch.einsum('vci,bi->bvc',idb_t,idv)[0]+torch.einsum('vce,be->bvc',expb_t,expv)[0]
        pred=torch.exp(log_s)*lm[:,:2]+trans[0];err=pred-target_t
        data=((err.square().sum(1))*w_t).sum()/torch.clamp(w_t.sum(),min=1.)
        crit=torch.mean(err[critical].square())
        rid=torch.mean((idv-id_ref).square());reye=torch.mean((eyev-eye_ref).square())
        # Strong coefficient-space prior. Data loss is ~1e-3..1e-2, so these
        # weights materially constrain shape while still allowing identity motion.
        loss=data+1.30*crit+.028*rid+.018*reye
        loss.backward();torch.nn.utils.clip_grad_norm_([idv,eyev],1.25);opt.step()
        with torch.no_grad():
            # Per-coefficient trust region around the network identity; also cap
            # absolute values to the range seen in ordinary FaceVerse inference.
            idv.copy_(torch.maximum(torch.minimum(idv,id_ref+.22),id_ref-.22))
            idv.clamp_(-.65,.65);eyev.clamp_(-.60,.60);log_s.clamp_(math.log(.03),math.log(1.5));trans.clamp_(-.5,.5)
        if step in (0,24,49,99,159,239,319,399,steps-1):
            with torch.no_grad():
                hist.append({'step':step,'loss':float(loss),'data':float(data),'critical':float(crit),'id_delta_rms':float(torch.sqrt(rid)),'eye_delta_rms':float(torch.sqrt(reye)),'id_rms':float(torch.sqrt(torch.mean(idv.square()))),'id_abs_max':float(torch.max(torch.abs(idv))),'eye14':float(eyev[0,0]),'eye15':float(eyev[0,1]),'scale':float(torch.exp(log_s))})
            print(json.dumps(hist[-1]))
    with torch.no_grad():
        expv=expression();lm=mean_t+torch.einsum('vci,bi->bvc',idb_t,idv)[0]+torch.einsum('vce,be->bvc',expb_t,expv)[0]
        pred=torch.exp(log_s)*lm[:,:2]+trans[0];e=torch.linalg.norm(pred-target_t,dim=1).numpy()
    exp_out=np.zeros(expb.shape[2],np.float32);exp_out[14:16]=eyev.detach().numpy()[0]
    return idv.detach().numpy()[0],exp_out,pred.numpy(),e,hist


def normalize_metric(v):
    v=np.asarray(v,np.float64).copy();c=np.median(v,axis=0);v-=c
    h=float(np.percentile(v[:,1],99)-np.percentile(v[:,1],1));s=.180/max(h,1e-9)
    return v*s,s


def render(v,f,yaw,path,title):
    a=math.radians(yaw);c=math.cos(a);s=math.sin(a);p=v.copy();x=c*p[:,0]+s*p[:,2];z=-s*p[:,0]+c*p[:,2];p[:,0]=x;p[:,2]=z
    tri=p[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
    order=np.argsort(-tri[:,:,2].mean(1));tri2=p[f[order],:2];nn=n[order];dif=np.clip(np.abs(nn[:,2]),0,1);side=np.clip(-.25*nn[:,0]-.18*nn[:,1]+.72*nn[:,2],0,1);it=np.clip(.66+.22*dif+.09*side,.52,.98);col=np.stack([it*.96,it*.975,it],1)
    xy=p[:,:2];lo=np.percentile(xy,1.5,0);hi=np.percentile(xy,98.5,0);ctr=.5*(lo+hi);ext=max(float((hi-lo).max()),1e-6)*.57
    fig,ax=plt.subplots(figsize=(5,5),dpi=190);ax.add_collection(PolyCollection(tri2,facecolors=col,edgecolors='none'));ax.set_xlim(ctr[0]-ext,ctr[0]+ext);ax.set_ylim(ctr[1]+ext,ctr[1]-ext);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10);fig.tight_layout(pad=.12);fig.savefig(path,bbox_inches='tight',pad_inches=.02);plt.close(fig)


def compare(a,b,o):
    x=Image.open(a).convert('RGB');y=Image.open(b).convert('RGB');H=max(x.height,y.height);xw=int(x.width*H/x.height);yw=int(y.width*H/y.height);s=Image.new('RGB',(xw+yw,H),'white');s.paste(x.resize((xw,H)),(0,0));s.paste(y.resize((yw,H)),(xw,0));s.save(o)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--front',type=Path,required=True);ap.add_argument('--three-quarter',type=Path,required=True);ap.add_argument('--front-landmarks',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_faceverse_v122'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);qa=args.out/'QA';qa.mkdir(exist_ok=True)
    front=base.load_rgb(args.front);q=base.load_rgb(args.three_quarter);target=base.detect_mp478(front);weights=base.make_weights()
    sess=ort.InferenceSession(str(FVROOT/'data/faceverse_resnet50_float32.onnx'),providers=['CPUExecutionProvider']);cf=base.infer(sess,front,base.bbox_from_68(args.front_landmarks,front.shape));cq=base.infer(sess,q,base.heuristic_bbox(q))
    model=FaceVerseModel_torch(device=torch.device('cpu'),facevrsepath=str(FVROOT/'data/faceverse_v4_2.npy'),camera_distance=10,focal=1000,center=128);idd=int(model.id_dims);expd=int(model.exp_dims);id0=.72*cf[:idd]+.28*cq[:idd];exp0=cf[idd:idd+expd].copy();mp_inds=np.asarray(model.fvd['keypoints_mediapipe']).reshape(-1).astype(np.int64)
    mean=np.asarray(model.fvd['meanshape'],np.float32)[mp_inds]/100.;idb=base.selected_basis(model.fvd,mp_inds,'idBase',idd);expb=base.selected_basis(model.fvd,mp_inds,'exBase',expd)
    id_fit,exp_fit,pred,e,hist=optimize_strict(mean,idb,expb,id0.astype(np.float32),exp0.astype(np.float32),target,weights)
    coeff=np.concatenate([id_fit,exp_fit,np.zeros(int(model.tex_dims),np.float32),np.zeros(27,np.float32),np.zeros(3,np.float32),np.zeros(3,np.float32),np.zeros(4,np.float32)])[None]
    with torch.no_grad():result=model.run(torch.from_numpy(coeff).float(),only_lms=False,use_color=False)
    verts=np.asarray(result['vertices'][0].cpu(),np.float64);faces=np.asarray(model.tri.cpu(),np.int64);metric,ms=normalize_metric(verts);mesh=trimesh.Trimesh(vertices=metric,faces=faces,process=False);mesh.export(args.out/'AINA_FACEVERSE_FULL_v12.2_STRICT.obj');mesh.export(args.out/'AINA_FACEVERSE_FULL_v12.2_STRICT.glb');mesh.export(args.out/'AINA_FACEVERSE_FULL_v12.2_STRICT.ply');np.save(args.out/'AINA_FACEVERSE_IDENTITY_156_v12.2.npy',id_fit.astype(np.float32));np.save(args.out/'AINA_FACEVERSE_EXPRESSION_177_v12.2.npy',exp_fit.astype(np.float32))
    base.landmark_overlay(front,target,pred,qa/'AINA_MP478_OVERLAY_v12.2.png');views=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p=qa/f'AINA_FACEVERSE_CLAY_{label}_v12.2.png';render(metric,faces,yaw,p,f'AINA FaceVerse v12.2 {label}');views.append(p)
    ims=[Image.open(x).convert('RGB') for x in views];H=max(x.height for x in ims);W=max(x.width for x in ims);sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_FACEVERSE_CLAY_5VIEW_v12.2.png');compare(args.front,qa/'AINA_FACEVERSE_CLAY_front_v12.2.png',qa/'AINA_REFERENCE_VS_FACEVERSE_FRONT_v12.2.png')
    act=weights>0;critical=np.asarray(sorted(set(base.FACE_OVAL+base.LEFT_EYE+base.RIGHT_EYE+base.NOSE+base.LIPS)),np.int64);delta=id_fit-id0
    rep={'version':'AINA FaceVerse v12.2 Strict Dense Fit','vertices':int(len(verts)),'faces':int(len(faces)),'weighted_rmse':float(np.sqrt(np.sum(weights[act]*e[act]**2)/np.sum(weights[act]))),'critical_rmse':float(np.sqrt(np.mean(e[critical]**2))),'id_initial_rms':float(np.sqrt(np.mean(id0**2))),'id_final_rms':float(np.sqrt(np.mean(id_fit**2))),'id_delta_rms':float(np.sqrt(np.mean(delta**2))),'id_delta_abs_max':float(np.max(np.abs(delta))),'eye_expression_14_15':exp_fit[14:16].tolist(),'metric_scale':float(ms),'history':hist,'identity_lock':False,'acceptance_note':'Must remain anatomically coherent and visually match AINA; no numerical score can override Clay QA.'};(args.out/'AINA_FACEVERSE_v12.2_REPORT.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
