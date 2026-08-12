#!/usr/bin/env python3
"""AINA v12.3 — similarity-normalized dense FaceVerse fit.

v12.2 stayed coherent but its trainable weak-perspective scale collapsed instead
of forcing geometry toward AINA. Here camera scale/translation are solved
analytically from the current 478-point shape every iteration. They are not
trainable escape variables. Identity therefore has to explain face proportions.
Only documented eyelid expression channels 14/15 are allowed.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import trimesh
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import fit_aina_faceverse_v121_dense as base
from faceversev4 import FaceVerseModel_torch

ROOT=Path.cwd().resolve();FVROOT=(ROOT/'vendor/faceverse-onnx').resolve()


def weighted_similarity(x,target,w):
    """Differentiable positive-scale 2D similarity without rotation."""
    ww=w[:,None];denw=torch.clamp(ww.sum(),min=1e-8)
    mx=(x*ww).sum(0)/denw;my=(target*ww).sum(0)/denw
    xc=x-mx;yc=target-my
    num=(ww*xc*yc).sum();den=torch.clamp((ww*xc*xc).sum(),min=1e-9)
    scale=torch.clamp(num/den,min=.02,max=2.0)
    pred=scale*xc+my
    return pred,scale,mx,my


def optimize(mean,idb,expb,id0,target,weights,steps=520):
    mean_t=torch.tensor(mean.copy(),dtype=torch.float32);idb_t=torch.tensor(idb.copy(),dtype=torch.float32);expb_t=torch.tensor(expb.copy(),dtype=torch.float32);target_t=torch.tensor(target.copy(),dtype=torch.float32);w_t=torch.tensor(weights.copy(),dtype=torch.float32)
    id_ref=torch.tensor(id0.copy(),dtype=torch.float32).unsqueeze(0).clone().detach();idv=torch.nn.Parameter(id_ref.clone())
    # FaceVerse channels 14/15 are eyelid closure. Negative starts open relative
    # to its overly narrow neutral eye while optimizer remains free to correct it.
    eye_ref=torch.tensor([[-.14,-.14]],dtype=torch.float32);eyev=torch.nn.Parameter(eye_ref.clone())
    selector=torch.zeros((2,expb.shape[2]),dtype=torch.float32);selector[0,14]=1.;selector[1,15]=1.
    def expv():return eyev@selector
    opt=torch.optim.Adam([{'params':[idv],'lr':.0065},{'params':[eyev],'lr':.0085}])
    critical=torch.tensor(sorted(set(base.FACE_OVAL+base.LEFT_EYE+base.RIGHT_EYE+base.NOSE+base.LIPS)),dtype=torch.long)
    hist=[]
    for step in range(steps):
        opt.zero_grad();lm=mean_t+torch.einsum('vci,bi->bvc',idb_t,idv)[0]+torch.einsum('vce,be->bvc',expb_t,expv())[0]
        pred,scale,_,_=weighted_similarity(lm[:,:2],target_t,w_t);err=pred-target_t
        data=(err.square().sum(1)*w_t).sum()/torch.clamp(w_t.sum(),min=1.);crit=err[critical].square().sum(1).mean();rid=(idv-id_ref).square().mean();reye=(eyev-eye_ref).square().mean()
        # Shape can move farther than v12.2, but every coefficient remains inside
        # a trust region and the global RMS prior prevents statistical-space abuse.
        loss=data+1.45*crit+.010*rid+.004*reye
        loss.backward();torch.nn.utils.clip_grad_norm_([idv,eyev],1.4);opt.step()
        with torch.no_grad():
            idv.copy_(torch.maximum(torch.minimum(idv,id_ref+.34),id_ref-.34));idv.clamp_(-.78,.78);eyev.clamp_(-.55,.18)
        if step in (0,24,49,99,159,239,319,419,steps-1):
            hist.append({'step':step,'loss':float(loss),'data':float(data),'critical':float(crit),'id_delta_rms':float(torch.sqrt(rid)),'id_rms':float(torch.sqrt(torch.mean(idv.square()))),'id_abs_max':float(torch.max(torch.abs(idv))),'eye14':float(eyev[0,0]),'eye15':float(eyev[0,1]),'analytic_scale':float(scale)});print(json.dumps(hist[-1]))
    with torch.no_grad():
        lm=mean_t+torch.einsum('vci,bi->bvc',idb_t,idv)[0]+torch.einsum('vce,be->bvc',expb_t,expv())[0];pred,scale,_,_=weighted_similarity(lm[:,:2],target_t,w_t);e=torch.linalg.norm(pred-target_t,dim=1).numpy()
    ex=np.zeros(expb.shape[2],np.float32);ex[14:16]=eyev.detach().numpy()[0]
    return idv.detach().numpy()[0],ex,pred.numpy(),e,float(scale),hist


def norm_metric(v):
    v=np.asarray(v,np.float64).copy();v-=np.median(v,axis=0);h=float(np.percentile(v[:,1],99)-np.percentile(v[:,1],1));s=.180/max(h,1e-9);return v*s,s

def render(v,f,yaw,path,title):
    a=math.radians(yaw);c=math.cos(a);s=math.sin(a);p=v.copy();x=c*p[:,0]+s*p[:,2];z=-s*p[:,0]+c*p[:,2];p[:,0]=x;p[:,2]=z;tri=p[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9);order=np.argsort(-tri[:,:,2].mean(1));tri2=p[f[order],:2];nn=n[order];it=np.clip(.66+.22*np.abs(nn[:,2])+.08*np.clip(-.25*nn[:,0]-.18*nn[:,1]+.72*nn[:,2],0,1),.52,.98);col=np.stack([it*.96,it*.975,it],1);xy=p[:,:2];lo=np.percentile(xy,1.5,0);hi=np.percentile(xy,98.5,0);ctr=.5*(lo+hi);ext=max(float((hi-lo).max()),1e-6)*.57;fig,ax=plt.subplots(figsize=(5,5),dpi=190);ax.add_collection(PolyCollection(tri2,facecolors=col,edgecolors='none'));ax.set_xlim(ctr[0]-ext,ctr[0]+ext);ax.set_ylim(ctr[1]+ext,ctr[1]-ext);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10);fig.tight_layout(pad=.12);fig.savefig(path,bbox_inches='tight',pad_inches=.02);plt.close(fig)
def compare(a,b,o):
    x=Image.open(a).convert('RGB');y=Image.open(b).convert('RGB');H=max(x.height,y.height);xw=int(x.width*H/x.height);yw=int(y.width*H/y.height);s=Image.new('RGB',(xw+yw,H),'white');s.paste(x.resize((xw,H)),(0,0));s.paste(y.resize((yw,H)),(xw,0));s.save(o)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--front',type=Path,required=True);ap.add_argument('--three-quarter',type=Path,required=True);ap.add_argument('--front-landmarks',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_faceverse_v123'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);qa=args.out/'QA';qa.mkdir(exist_ok=True)
    front=base.load_rgb(args.front);q=base.load_rgb(args.three_quarter);target=base.detect_mp478(front);weights=base.make_weights();weights[np.asarray(base.LEFT_EYE+base.RIGHT_EYE)]=7.2
    sess=ort.InferenceSession(str(FVROOT/'data/faceverse_resnet50_float32.onnx'),providers=['CPUExecutionProvider']);cf=base.infer(sess,front,base.bbox_from_68(args.front_landmarks,front.shape));cq=base.infer(sess,q,base.heuristic_bbox(q));model=FaceVerseModel_torch(device=torch.device('cpu'),facevrsepath=str(FVROOT/'data/faceverse_v4_2.npy'),camera_distance=10,focal=1000,center=128);idd=int(model.id_dims);expd=int(model.exp_dims);id0=.72*cf[:idd]+.28*cq[:idd];mpinds=np.asarray(model.fvd['keypoints_mediapipe']).reshape(-1).astype(np.int64);mean=np.asarray(model.fvd['meanshape'],np.float32)[mpinds]/100.;idb=base.selected_basis(model.fvd,mpinds,'idBase',idd);expb=base.selected_basis(model.fvd,mpinds,'exBase',expd)
    idf,exf,pred,e,asc,hist=optimize(mean,idb,expb,id0.astype(np.float32),target,weights)
    coeff=np.concatenate([idf,exf,np.zeros(int(model.tex_dims),np.float32),np.zeros(27,np.float32),np.zeros(3,np.float32),np.zeros(3,np.float32),np.zeros(4,np.float32)])[None]
    with torch.no_grad():res=model.run(torch.from_numpy(coeff).float(),only_lms=False,use_color=False)
    verts=np.asarray(res['vertices'][0].cpu(),np.float64);faces=np.asarray(model.tri.cpu(),np.int64);metric,ms=norm_metric(verts);mesh=trimesh.Trimesh(vertices=metric,faces=faces,process=False);mesh.export(args.out/'AINA_FACEVERSE_FULL_v12.3_SIMILARITY.obj');mesh.export(args.out/'AINA_FACEVERSE_FULL_v12.3_SIMILARITY.glb');mesh.export(args.out/'AINA_FACEVERSE_FULL_v12.3_SIMILARITY.ply');np.save(args.out/'AINA_FACEVERSE_IDENTITY_156_v12.3.npy',idf.astype(np.float32));np.save(args.out/'AINA_FACEVERSE_EXPRESSION_177_v12.3.npy',exf.astype(np.float32));base.landmark_overlay(front,target,pred,qa/'AINA_MP478_OVERLAY_v12.3.png')
    views=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p=qa/f'AINA_FACEVERSE_CLAY_{label}_v12.3.png';render(metric,faces,yaw,p,f'AINA FaceVerse v12.3 {label}');views.append(p)
    ims=[Image.open(x).convert('RGB') for x in views];H=max(x.height for x in ims);W=max(x.width for x in ims);sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_FACEVERSE_CLAY_5VIEW_v12.3.png');compare(args.front,qa/'AINA_FACEVERSE_CLAY_front_v12.3.png',qa/'AINA_REFERENCE_VS_FACEVERSE_FRONT_v12.3.png');act=weights>0;crit=np.asarray(sorted(set(base.FACE_OVAL+base.LEFT_EYE+base.RIGHT_EYE+base.NOSE+base.LIPS)),np.int64);delta=idf-id0
    rep={'version':'AINA FaceVerse v12.3 Similarity-Normalized Dense Fit','weighted_rmse':float(np.sqrt(np.sum(weights[act]*e[act]**2)/np.sum(weights[act]))),'critical_rmse':float(np.sqrt(np.mean(e[crit]**2))),'analytic_scale':asc,'id_initial_rms':float(np.sqrt(np.mean(id0**2))),'id_final_rms':float(np.sqrt(np.mean(idf**2))),'id_delta_rms':float(np.sqrt(np.mean(delta**2))),'id_delta_abs_max':float(np.max(np.abs(delta))),'eye_expression_14_15':exf[14:16].tolist(),'metric_scale':float(ms),'history':hist,'identity_lock':False,'acceptance_note':'Visual Clay likeness is the only pass gate.'};(args.out/'AINA_FACEVERSE_v12.3_REPORT.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
