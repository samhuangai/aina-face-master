#!/usr/bin/env python3
"""AINA v12.0 — FaceVerse V4 identity reconstruction experiment.

Uses the public FaceVerse-ONNX release to infer a higher-detail identity from
approved AINA art. Identity coefficients are blended from front + 3/4; all
expression, eye pose, head pose and translation are reset to neutral before
export. This is a geometry-base experiment, not an identity lock.
"""
from __future__ import annotations
import json, math, sys
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

ROOT=Path.cwd().resolve()
FVROOT=(ROOT/'vendor/faceverse-onnx').resolve()
sys.path.insert(0,str(FVROOT))
from faceversev4 import FaceVerseModel_torch


def load_rgb(path: Path):
    bgr=cv2.imread(str(path))
    if bgr is None: raise RuntimeError(f'missing {path}')
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)


def bbox_from_68(path: Path, image_shape, pad=.33):
    d=json.loads(path.read_text())
    pts=np.asarray(d['landmarks_xy'],np.float64)
    lo=pts.min(0);hi=pts.max(0);c=.5*(lo+hi);size=max(float((hi-lo).max()),1.)*(1+2*pad)
    h,w=image_shape[:2]
    x1=max(0.,c[0]-size*.5);x2=min(float(w),c[0]+size*.5)
    y1=max(0.,c[1]-size*.53);y2=min(float(h),c[1]+size*.47)
    return np.array([x1,y1,x2,y2],np.float32)


def heuristic_bbox(img, scale=.86):
    h,w=img.shape[:2];s=min(h,w)*scale;cx=w*.5;cy=h*.50
    return np.array([cx-s*.5,cy-s*.52,cx+s*.5,cy+s*.48],np.float32)


def preprocess(img,bbox):
    x1,y1,x2,y2=np.round(bbox).astype(int);x1=max(0,x1);y1=max(0,y1);x2=min(img.shape[1],x2);y2=min(img.shape[0],y2)
    crop=img[y1:y2,x1:x2]
    if crop.size==0: raise RuntimeError(f'empty crop {bbox}')
    r=cv2.resize(crop,(256,256),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.
    return np.transpose(r,(2,0,1))[None]


def infer(session,img,bbox):
    inp=session.get_inputs()[0].name;out=session.get_outputs()[0].name
    return np.asarray(session.run([out],{inp:preprocess(img,bbox)})[0][0],np.float32)


def normalize_mesh(v):
    v=np.asarray(v,np.float64).copy();c=np.median(v,axis=0);v-=c
    # FaceVerse units are already metric-ish after model /100. Make robust face
    # height about 180 mm for DCC inspection without changing proportions.
    span=float(np.percentile(v[:,1],99)-np.percentile(v[:,1],1));s=.180/max(span,1e-9)
    return v*s,s,c


def render(v,f,yaw,path,title):
    a=math.radians(yaw);c=math.cos(a);s=math.sin(a)
    # yaw around model Y; x/z rotate together
    p=v.copy();x=c*p[:,0]+s*p[:,2];z=-s*p[:,0]+c*p[:,2];p[:,0]=x;p[:,2]=z
    tri=p[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
    order=np.argsort(tri[:,:,2].mean(1));tri2=p[f[order],:2];nn=n[order]
    dif=np.clip(np.abs(nn[:,2]),0,1);side=np.clip(-.25*nn[:,0]-.18*nn[:,1]+.72*nn[:,2],0,1);it=np.clip(.67+.21*dif+.09*side,.54,.98)
    col=np.stack([it*.96,it*.975,it],1);xy=p[:,:2];lo=np.percentile(xy,1.5,0);hi=np.percentile(xy,98.5,0);ctr=.5*(lo+hi);ext=max(float((hi-lo).max()),1e-6)*.57
    fig,ax=plt.subplots(figsize=(5,5),dpi=190);ax.add_collection(PolyCollection(tri2,facecolors=col,edgecolors='none'))
    ax.set_xlim(ctr[0]-ext,ctr[0]+ext);ax.set_ylim(ctr[1]-ext,ctr[1]+ext);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10);fig.tight_layout(pad=.12);fig.savefig(path,bbox_inches='tight',pad_inches=.02);plt.close(fig)


def side_by_side(ref,act,out):
    a=Image.open(ref).convert('RGB');b=Image.open(act).convert('RGB');H=max(a.height,b.height);aw=int(a.width*H/a.height);bw=int(b.width*H/b.height);s=Image.new('RGB',(aw+bw,H),'white');s.paste(a.resize((aw,H)),(0,0));s.paste(b.resize((bw,H)),(aw,0));s.save(out)


def main():
    out=ROOT/'output_faceverse_v120';qa=out/'QA';out.mkdir(exist_ok=True);qa.mkdir(exist_ok=True)
    front_path=ROOT/'references/AINA_APPROVED_FRONT.jpg';q_path=ROOT/'references/AINA_APPROVED_3Q.jpg';target68=ROOT/'references/AINA_TARGET_3DDFA_SPARSE_68.json'
    front=load_rgb(front_path);q=load_rgb(q_path)
    sess=ort.InferenceSession(str(FVROOT/'data/faceverse_resnet50_float32.onnx'),providers=['CPUExecutionProvider'])
    bf=bbox_from_68(target68,front.shape,pad=.31);bq=heuristic_bbox(q,.88)
    cf=infer(sess,front,bf);cq=infer(sess,q,bq)

    dev=torch.device('cpu');model=FaceVerseModel_torch(device=dev,facevrsepath=str(FVROOT/'data/faceverse_v4_2.npy'),camera_distance=10,focal=1000,center=128)
    iddim=int(model.id_dims);expdim=int(model.exp_dims);texdim=int(model.tex_dims);alldim=int(model.all_dims)
    # Identity only: front dominates, 3Q helps side-depth identity signal.
    identity=.72*cf[:iddim]+.28*cq[:iddim]
    neutral=np.zeros_like(cf);neutral[:iddim]=identity
    # Keep neutral lighting irrelevant to geometry. Expression / pose / eyes = 0.
    coeff=torch.from_numpy(neutral[None]).float()
    with torch.no_grad():
        result=model.run(coeff,only_lms=False,use_color=False)
    verts=np.asarray(result['vertices'][0].cpu(),np.float64)
    faces=np.asarray(model.tri.cpu(),np.int64)
    if faces.min()==1:faces-=1

    # Export full FaceVerse topology plus skin-only surface for fair clay QA.
    metric,metric_scale,center=normalize_mesh(verts)
    full=trimesh.Trimesh(vertices=metric,faces=faces,process=False)
    full.export(out/'AINA_FACEVERSE_FULL_v12.0.obj');full.export(out/'AINA_FACEVERSE_FULL_v12.0.glb')
    skin_mask=np.asarray(model.fvd['parsing']['skin']).reshape(-1)>0
    good=skin_mask[faces].all(axis=1)
    skin=full.submesh([np.flatnonzero(good)],append=True,repair=False);skin.remove_unreferenced_vertices()
    skin.export(out/'AINA_FACEVERSE_SKIN_CLAY_v12.0.obj');skin.export(out/'AINA_FACEVERSE_SKIN_CLAY_v12.0.glb');skin.export(out/'AINA_FACEVERSE_SKIN_CLAY_v12.0.ply')
    np.save(out/'AINA_FACEVERSE_IDENTITY_156_v12.0.npy',identity.astype(np.float32))

    sv=np.asarray(skin.vertices);sf=np.asarray(skin.faces);views=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p=qa/f'AINA_FACEVERSE_CLAY_{label}_v12.0.png';render(sv,sf,yaw,p,f'AINA FaceVerse v12.0 {label}');views.append(p)
    ims=[Image.open(x).convert('RGB') for x in views];H=max(x.height for x in ims);W=max(x.width for x in ims);sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_FACEVERSE_CLAY_5VIEW_v12.0.png');side_by_side(front_path,qa/'AINA_FACEVERSE_CLAY_front_v12.0.png',qa/'AINA_REFERENCE_VS_FACEVERSE_FRONT_v12.0.png')
    report={'version':'AINA FaceVerse Base v12.0','source':'FaceVerse V4 ONNX public release','identity_dims':iddim,'expression_dims':expdim,'texture_dims':texdim,'vertices_full':int(len(verts)),'faces_full':int(len(faces)),'vertices_skin':int(len(sv)),'faces_skin':int(len(sf)),'identity_blend':{'front':.72,'three_quarter':.28},'expression_neutralized':True,'pose_neutralized':True,'eyes_neutralized':True,'metric_scale':float(metric_scale),'identity_lock':False,'acceptance_note':'Use only if actual clay visually beats the v11.2/v11.4 GNM route.'}
    (out/'AINA_FACEVERSE_v12.0_REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
