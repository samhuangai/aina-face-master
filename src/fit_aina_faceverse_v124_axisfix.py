#!/usr/bin/env python3
"""AINA v12.4 — FaceVerse native-axis 478-point fit.

Critical correction: FaceVerse mesh Y is positive downward in its canonical
front view. v12.1-v12.3 fed targets with Y positive upward, so camera fitting
collapsed scale or coefficient fitting fought an impossible reflection. v12.4
uses the same MediaPipe 478 ordering but flips target Y back to FaceVerse's
native convention before similarity-normalized identity fitting.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import fit_aina_faceverse_v121_dense as base
import fit_aina_faceverse_v123_similarity as sim
from faceversev4 import FaceVerseModel_torch

ROOT=Path.cwd().resolve();FVROOT=(ROOT/'vendor/faceverse-onnx').resolve()


def overlay_native(rgb,target,pred,path):
    h,w=rgb.shape[:2];fig,ax=plt.subplots(figsize=(6,6),dpi=180);ax.imshow(rgb)
    # target/pred are both FaceVerse-native: +Y goes down.
    tx=(target[:,0]+.5)*w;ty=(target[:,1]+.5)*h;px=(pred[:,0]+.5)*w;py=(pred[:,1]+.5)*h
    ax.scatter(tx,ty,s=4,label='AINA target');ax.scatter(px,py,s=4,marker='+',label='FaceVerse fit');ax.legend(loc='lower right',fontsize=7);ax.axis('off');fig.tight_layout(pad=0);fig.savefig(path,bbox_inches='tight',pad_inches=0);plt.close(fig)


def compare(a,b,o):
    x=Image.open(a).convert('RGB');y=Image.open(b).convert('RGB');H=max(x.height,y.height);xw=int(x.width*H/x.height);yw=int(y.width*H/y.height);s=Image.new('RGB',(xw+yw,H),'white');s.paste(x.resize((xw,H)),(0,0));s.paste(y.resize((yw,H)),(xw,0));s.save(o)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--front',type=Path,required=True);ap.add_argument('--three-quarter',type=Path,required=True);ap.add_argument('--front-landmarks',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_faceverse_v124'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);qa=args.out/'QA';qa.mkdir(exist_ok=True)
    front=base.load_rgb(args.front);q=base.load_rgb(args.three_quarter)
    target=base.detect_mp478(front);target[:,1]*=-1.0  # +Y DOWN, matching FaceVerse canonical mesh
    weights=base.make_weights();weights[np.asarray(base.LEFT_EYE+base.RIGHT_EYE)]=7.5;weights[np.asarray(base.FACE_OVAL)]=5.0
    sess=ort.InferenceSession(str(FVROOT/'data/faceverse_resnet50_float32.onnx'),providers=['CPUExecutionProvider']);cf=base.infer(sess,front,base.bbox_from_68(args.front_landmarks,front.shape));cq=base.infer(sess,q,base.heuristic_bbox(q))
    model=FaceVerseModel_torch(device=torch.device('cpu'),facevrsepath=str(FVROOT/'data/faceverse_v4_2.npy'),camera_distance=10,focal=1000,center=128);idd=int(model.id_dims);expd=int(model.exp_dims);id0=.72*cf[:idd]+.28*cq[:idd]
    mpinds=np.asarray(model.fvd['keypoints_mediapipe']).reshape(-1).astype(np.int64);mean=np.asarray(model.fvd['meanshape'],np.float32)[mpinds]/100.;idb=base.selected_basis(model.fvd,mpinds,'idBase',idd);expb=base.selected_basis(model.fvd,mpinds,'exBase',expd)
    idf,exf,pred,e,asc,hist=sim.optimize(mean,idb,expb,id0.astype(np.float32),target,weights,steps=520)
    coeff=np.concatenate([idf,exf,np.zeros(int(model.tex_dims),np.float32),np.zeros(27,np.float32),np.zeros(3,np.float32),np.zeros(3,np.float32),np.zeros(4,np.float32)])[None]
    with torch.no_grad():res=model.run(torch.from_numpy(coeff).float(),only_lms=False,use_color=False)
    verts=np.asarray(res['vertices'][0].cpu(),np.float64);faces=np.asarray(model.tri.cpu(),np.int64);metric,ms=sim.norm_metric(verts);mesh=trimesh.Trimesh(vertices=metric,faces=faces,process=False);mesh.export(args.out/'AINA_FACEVERSE_FULL_v12.4_AXISFIX.obj');mesh.export(args.out/'AINA_FACEVERSE_FULL_v12.4_AXISFIX.glb');mesh.export(args.out/'AINA_FACEVERSE_FULL_v12.4_AXISFIX.ply');np.save(args.out/'AINA_FACEVERSE_IDENTITY_156_v12.4.npy',idf.astype(np.float32));np.save(args.out/'AINA_FACEVERSE_EXPRESSION_177_v12.4.npy',exf.astype(np.float32));np.save(args.out/'AINA_TARGET_MP478_NATIVE_v12.4.npy',target.astype(np.float32))
    overlay_native(front,target,pred,qa/'AINA_MP478_NATIVE_OVERLAY_v12.4.png');views=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p=qa/f'AINA_FACEVERSE_CLAY_{label}_v12.4.png';sim.render(metric,faces,yaw,p,f'AINA FaceVerse v12.4 {label}');views.append(p)
    ims=[Image.open(x).convert('RGB') for x in views];H=max(x.height for x in ims);W=max(x.width for x in ims);sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_FACEVERSE_CLAY_5VIEW_v12.4.png');compare(args.front,qa/'AINA_FACEVERSE_CLAY_front_v12.4.png',qa/'AINA_REFERENCE_VS_FACEVERSE_FRONT_v12.4.png')
    act=weights>0;crit=np.asarray(sorted(set(base.FACE_OVAL+base.LEFT_EYE+base.RIGHT_EYE+base.NOSE+base.LIPS)),np.int64);delta=idf-id0
    rep={'version':'AINA FaceVerse v12.4 Native-Axis Dense Fit','coordinate_fix':'MediaPipe target Y flipped to FaceVerse +Y-down convention','weighted_rmse':float(np.sqrt(np.sum(weights[act]*e[act]**2)/np.sum(weights[act]))),'critical_rmse':float(np.sqrt(np.mean(e[crit]**2))),'analytic_scale':asc,'id_initial_rms':float(np.sqrt(np.mean(id0**2))),'id_final_rms':float(np.sqrt(np.mean(idf**2))),'id_delta_rms':float(np.sqrt(np.mean(delta**2))),'id_delta_abs_max':float(np.max(np.abs(delta))),'eye_expression_14_15':exf[14:16].tolist(),'metric_scale':float(ms),'history':hist,'identity_lock':False,'acceptance_note':'First geometrically valid 478-to-478 FaceVerse fit. Visual Clay is still the pass gate.'};(args.out/'AINA_FACEVERSE_v12.4_REPORT.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
