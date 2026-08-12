#!/usr/bin/env python3
"""AINA dense target reconstruction using 3DDFA_V2 / BFM.

This experiment deliberately leaves the GNM identity space. It estimates the
same AINA identity from the approved front, 3/4 and side effect-art references,
blends the BFM shape coefficients, removes expression, and exports a canonical
dense face mesh plus clay QA views.
"""
from __future__ import annotations

import json, os, sys, math
from pathlib import Path

import cv2
import yaml
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image

workspace=Path.cwd().resolve()
repo=(workspace/'vendor/3DDFA_V2').resolve()
sys.path.insert(0,str(repo))
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.tddfa_util import _parse_param


def load_img(path: Path):
    im=cv2.imread(str(path))
    if im is None: raise RuntimeError(f'missing image: {path}')
    return im


def run_one(tddfa,face_boxes,img):
    boxes=face_boxes(img)
    if not boxes:
        h,w=img.shape[:2]; boxes=[[0.08*w,0.05*h,0.92*w,0.96*h,1.0]]
    params,rois=tddfa(img,boxes)
    # choose the candidate closest to image centre when detector returns >1
    if len(params)>1:
        h,w=img.shape[:2]; c=np.array([w*.5,h*.5])
        centers=[np.array([(r[0]+r[2])*.5,(r[1]+r[3])*.5]) for r in rois]
        k=int(np.argmin([np.linalg.norm(x-c) for x in centers]))
        params=[params[k]]; rois=[rois[k]]
    sparse=tddfa.recon_vers(params,rois,dense_flag=False)[0].T
    dense=tddfa.recon_vers(params,rois,dense_flag=True)[0].T
    R,offset,shp,exp=_parse_param(params[0])
    return {'param':params[0],'roi':np.asarray(rois[0]),'sparse':sparse,'dense':dense,'R':np.asarray(R),'shape':np.asarray(shp).reshape(-1),'exp':np.asarray(exp).reshape(-1)}


def ortho_from_affine(R):
    # 3DDFA pose rows carry scale; polar decomposition removes it for QA camera.
    U,_,Vt=np.linalg.svd(np.asarray(R,dtype=np.float64))
    Q=U@Vt
    if np.linalg.det(Q)<0:
        U[:,-1]*=-1; Q=U@Vt
    return Q


def render_clay(vertices,faces,R0,yaw_deg,path,title):
    # camera-coordinate yaw around local vertical axis
    p0=vertices@R0.T
    a=math.radians(yaw_deg)
    cy,sy=math.cos(a),math.sin(a)
    # x/z yaw; y untouched
    x=cy*p0[:,0]+sy*p0[:,2]
    z=-sy*p0[:,0]+cy*p0[:,2]
    p=np.column_stack([x,p0[:,1],z])
    tri=p[faces]
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
    depth=tri[:,:,2].mean(axis=1)
    order=np.argsort(depth)
    tri2=p[faces[order],:2]; nn=n[order]
    diffuse=np.clip(np.abs(nn[:,2]),0,1)
    side=np.clip(-.30*nn[:,0]-.15*nn[:,1]-.72*nn[:,2],0,1)
    intensity=np.clip(.70+.17*diffuse+.09*side,.58,.98)
    colors=np.stack([intensity*.97,intensity*.98,intensity],axis=1)
    xy=p[:,:2]
    lo=np.percentile(xy,1.5,axis=0); hi=np.percentile(xy,98.5,axis=0)
    ctr=.5*(lo+hi); ext=max(float((hi-lo).max()),1e-6)*.57
    fig,ax=plt.subplots(figsize=(5,5),dpi=190)
    ax.add_collection(PolyCollection(tri2,facecolors=colors,edgecolors='none',closed=True))
    ax.set_xlim(ctr[0]-ext,ctr[0]+ext); ax.set_ylim(ctr[1]+ext,ctr[1]-ext)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title,fontsize=10)
    fig.tight_layout(pad=.12); fig.savefig(path,bbox_inches='tight',pad_inches=.02); plt.close(fig)


def render_image_pose(dense,faces,path,title):
    p=np.asarray(dense,float); tri=p[faces]
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
    order=np.argsort(tri[:,:,2].mean(axis=1))
    tri2=p[faces[order],:2]; nn=n[order]
    diffuse=np.clip(np.abs(nn[:,2]),0,1); inten=np.clip(.68+.25*diffuse,.55,.97)
    colors=np.stack([inten*.97,inten*.98,inten],axis=1)
    xy=p[:,:2]; lo=np.percentile(xy,1,axis=0);hi=np.percentile(xy,99,axis=0);ctr=.5*(lo+hi);ext=max(float((hi-lo).max()),1e-6)*.56
    fig,ax=plt.subplots(figsize=(5,5),dpi=190);ax.add_collection(PolyCollection(tri2,facecolors=colors,edgecolors='none'))
    ax.set_xlim(ctr[0]-ext,ctr[0]+ext);ax.set_ylim(ctr[1]+ext,ctr[1]-ext);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10)
    fig.tight_layout(pad=.12);fig.savefig(path,bbox_inches='tight',pad_inches=.02);plt.close(fig)


def normalize_metric(v):
    v=np.asarray(v,np.float64).copy()
    center=np.median(v,axis=0); v-=center
    # map robust face height to ~180 mm for convenient DCC scale
    h=float(np.percentile(v[:,1],99)-np.percentile(v[:,1],1)); s=.180/max(h,1e-9)
    return v*s,s,center


def side_by_side(ref_path,actual_path,out):
    ref=Image.open(ref_path).convert('RGB'); act=Image.open(actual_path).convert('RGB')
    H=max(ref.height,act.height);rw=int(ref.width*H/ref.height);aw=int(act.width*H/act.height)
    sheet=Image.new('RGB',(rw+aw,H),'white');sheet.paste(ref.resize((rw,H)),(0,0));sheet.paste(act.resize((aw,H)),(rw,0));sheet.save(out)


def main():
    out=workspace/'output_dense_3ddfa';qa=out/'QA';out.mkdir(exist_ok=True);qa.mkdir(exist_ok=True)
    paths={
      'front':workspace/'references/AINA_APPROVED_FRONT.jpg',
      'three_quarter':workspace/'references/AINA_APPROVED_3Q.jpg',
      'side':workspace/'references/AINA_APPROVED_SIDE.jpg',
    }
    imgs={k:load_img(v) for k,v in paths.items()}

    os.chdir(repo)
    cfg=yaml.load(Path('configs/mb1_120x120.yml').read_text(),Loader=yaml.SafeLoader);cfg['gpu_mode']=False
    fb=FaceBoxes(); tddfa=TDDFA(**cfg)
    rec={k:run_one(tddfa,fb,im) for k,im in imgs.items()}

    weights={'front':.62,'three_quarter':.30,'side':.08}
    shape=sum(weights[k]*rec[k]['shape'] for k in weights)
    zero_exp=np.zeros_like(rec['front']['exp'])
    raw=(tddfa.bfm.u+tddfa.bfm.w_shp@shape.reshape(-1,1)+tddfa.bfm.w_exp@zero_exp.reshape(-1,1)).reshape(3,-1,order='F').T
    tri=np.asarray(tddfa.tri,dtype=np.int64)
    if tri.ndim!=2: raise RuntimeError(f'unexpected tri shape {tri.shape}')
    if tri.shape[0]==3: tri=tri.T
    if tri.min()==1: tri=tri-1

    metric,metric_scale,metric_center=normalize_metric(raw)
    mesh=trimesh.Trimesh(vertices=metric,faces=tri,process=False)
    mesh.export(out/'AINA_DENSE_BFM_NEUTRAL_v11.0.obj');mesh.export(out/'AINA_DENSE_BFM_NEUTRAL_v11.0.glb');mesh.export(out/'AINA_DENSE_BFM_NEUTRAL_v11.0.ply')
    np.save(out/'AINA_BFM_SHAPE_40_v11.0.npy',shape.astype(np.float32))

    # Front image-space reconstruction tests whether the dense model actually
    # follows the approved effect-art face before any topology transfer.
    render_image_pose(rec['front']['dense'],tri,qa/'AINA_3DDFA_DENSE_FRONT_IMAGEPOSE.png','AINA dense 3DDFA front reconstruction')
    side_by_side(paths['front'],qa/'AINA_3DDFA_DENSE_FRONT_IMAGEPOSE.png',qa/'AINA_REFERENCE_VS_3DDFA_DENSE_FRONT.png')

    R0=ortho_from_affine(rec['front']['R'])
    views=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p=qa/f'AINA_DENSE_CLAY_{label}_v11.0.png';render_clay(raw,tri,R0,yaw,p,f'AINA dense v11.0 {label}');views.append(p)
    ims=[Image.open(x).convert('RGB') for x in views];H=max(i.height for i in ims);W=max(i.width for i in ims);sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_DENSE_CLAY_5VIEW_v11.0.png')

    report={'version':'AINA Dense Reconstruction v11.0','source':'3DDFA_V2 BFM noneck v3','topology':'BFM dense',
      'vertices':int(len(raw)),'triangles':int(len(tri)),'shape_blend_weights':weights,
      'shape_coefficients':shape.tolist(),'expression_neutralized':True,'metric_scale':float(metric_scale),
      'identity_lock':False,'next_gate':'Visual dense clay must first resemble approved AINA before any GNM topology transfer.'}
    (out/'AINA_DENSE_v11.0_REPORT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='shape_coefficients'},indent=2))

if __name__=='__main__':main()
