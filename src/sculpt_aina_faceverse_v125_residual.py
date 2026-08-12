#!/usr/bin/env python3
"""AINA v12.5 — dense residual sculpt on coherent v12.4 FaceVerse geometry.

The statistical identity fit is deliberately stopped at v12.4. Remaining AINA
front-view residuals from all 468 skin MediaPipe correspondences are transferred
as a smooth thin-plate-spline field onto only the largest connected FaceVerse
head/skin shell. Eyeballs, oral components and other disconnected geometry are
left untouched. A front-depth gate prevents the residual field from moving the
rear skull.
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

import fit_aina_faceverse_v121_dense as base


def detect_native(rgb):
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=.2) as fm:
        res=fm.process(rgb)
    if not res.multi_face_landmarks: raise RuntimeError('MediaPipe target detection failed')
    lm=res.multi_face_landmarks[0].landmark
    if len(lm)!=478: raise RuntimeError(f'expected 478 target points, got {len(lm)}')
    return np.asarray([[x.x-.5,x.y-.5] for x in lm],np.float64)  # FaceVerse +Y down


def components(nv,faces):
    e=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);a=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(nv,nv));a=(a+a.T).tocsr();n,lab=connected_components(a,directed=False);return [np.flatnonzero(lab==i) for i in range(n)]


def similarity(x,y,w):
    ww=w[:,None];sw=max(float(ww.sum()),1e-12);mx=(x*ww).sum(0)/sw;my=(y*ww).sum(0)/sw;xc=x-mx;yc=y-my;s=float((ww*xc*yc).sum()/max(float((ww*xc*xc).sum()),1e-12));return s,mx,my,s*xc+my


def clamp_rows(a,cap):
    out=a.copy();n=np.linalg.norm(out,axis=1);m=n>cap
    if np.any(m):out[m]*=(cap/n[m])[:,None]
    return out


def tri_area(v,f):
    t=v[f];return .5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)


def render(v,f,yaw,path,title):
    a=math.radians(yaw);c=math.cos(a);s=math.sin(a);p=v.copy();x=c*p[:,0]+s*p[:,2];z=-s*p[:,0]+c*p[:,2];p[:,0]=x;p[:,2]=z;tri=p[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9);order=np.argsort(-tri[:,:,2].mean(1));tri2=p[f[order],:2];nn=n[order];it=np.clip(.66+.22*np.abs(nn[:,2])+.08*np.clip(-.25*nn[:,0]-.18*nn[:,1]+.72*nn[:,2],0,1),.52,.98);col=np.stack([it*.96,it*.975,it],1);xy=p[:,:2];lo=np.percentile(xy,1.5,0);hi=np.percentile(xy,98.5,0);ctr=.5*(lo+hi);ext=max(float((hi-lo).max()),1e-6)*.57;fig,ax=plt.subplots(figsize=(5,5),dpi=190);ax.add_collection(PolyCollection(tri2,facecolors=col,edgecolors='none'));ax.set_xlim(ctr[0]-ext,ctr[0]+ext);ax.set_ylim(ctr[1]+ext,ctr[1]-ext);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10);fig.tight_layout(pad=.12);fig.savefig(path,bbox_inches='tight',pad_inches=.02);plt.close(fig)


def overlay(rgb,target,pred,path):
    h,w=rgb.shape[:2];fig,ax=plt.subplots(figsize=(6,6),dpi=180);ax.imshow(rgb);ax.scatter((target[:,0]+.5)*w,(target[:,1]+.5)*h,s=4,label='AINA target');ax.scatter((pred[:,0]+.5)*w,(pred[:,1]+.5)*h,s=4,marker='+',label='v12.5');ax.legend(loc='lower right',fontsize=7);ax.axis('off');fig.tight_layout(pad=0);fig.savefig(path,bbox_inches='tight',pad_inches=0);plt.close(fig)

def compare(a,b,o):
    x=Image.open(a).convert('RGB');y=Image.open(b).convert('RGB');H=max(x.height,y.height);xw=int(x.width*H/x.height);yw=int(y.width*H/y.height);s=Image.new('RGB',(xw+yw,H),'white');s.paste(x.resize((xw,H)),(0,0));s.paste(y.resize((yw,H)),(xw,0));s.save(o)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-full',type=Path,required=True);ap.add_argument('--front',type=Path,required=True);ap.add_argument('--faceverse-data',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_faceverse_v125'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);qa=args.out/'QA';qa.mkdir(exist_ok=True)
    mesh=trimesh.load(args.base_full,process=False,maintain_order=True);v=np.asarray(mesh.vertices,np.float64);f=np.asarray(mesh.faces,np.int64);v0=v.copy();rgb=cv2.cvtColor(cv2.imread(str(args.front)),cv2.COLOR_BGR2RGB);target=detect_native(rgb)
    fvd=np.load(args.faceverse_data,allow_pickle=True).item();mpinds=np.asarray(fvd['keypoints_mediapipe']).reshape(-1).astype(np.int64)
    if len(mpinds)!=478:raise RuntimeError(f'FaceVerse MP map has {len(mpinds)} points')
    weights=base.make_weights().astype(np.float64);active=np.flatnonzero(weights>0);ctrl_ids=mpinds[active];ctrl=v[ctrl_ids,:2];tgt=target[active]

    comps=components(len(v),f);head_ids=max(comps,key=len);head_set=np.zeros(len(v),bool);head_set[head_ids]=True
    if not np.all(head_set[ctrl_ids]):
        bad=int(np.sum(~head_set[ctrl_ids]));raise RuntimeError(f'{bad} active MediaPipe controls are not on largest head shell')
    head_faces=np.flatnonzero(head_set[f].all(axis=1));hf=f[head_faces]
    a0=tri_area(v,hf)

    passes=[]
    for pass_i,gain in enumerate((.92,.62)):
        lm=v[ctrl_ids,:2];s,mx,my,pred=similarity(lm,tgt,weights[active]);err=tgt-pred;res=err/max(abs(s),1e-9)
        # Cap each semantic request before interpolation. First pass allows more
        # silhouette motion; second pass is a polish.
        cap=.0062 if pass_i==0 else .0032;res=clamp_rows(res,cap)
        rbf=RBFInterpolator(lm,res,kernel='thin_plate_spline',smoothing=1.2e-6,degree=1)
        hp=v[head_ids];field=rbf(hp[:,:2])
        # Semantic-face support in XY plus a front-depth gate. In FaceVerse the
        # visible face has low Z while rear skull extends to much larger +Z.
        tree=cKDTree(lm);dxy,_=tree.query(hp[:,:2],k=1);xy_gate=np.exp(-.5*(dxy/.020)**4)
        face_z=float(np.median(v[ctrl_ids,2]));z_gate=np.exp(-.5*((hp[:,2]-face_z)/.042)**4);gate=np.clip(xy_gate*z_gate,0.,1.)
        delta=np.zeros_like(hp);delta[:,:2]=gain*field*gate[:,None];delta=clamp_rows(delta,.0065 if pass_i==0 else .0035)
        v[head_ids]+=delta
        lm2=v[ctrl_ids,:2];s2,mx2,my2,pred2=similarity(lm2,tgt,weights[active]);e=np.linalg.norm(pred2-tgt,axis=1)
        passes.append({'pass':pass_i,'gain':gain,'similarity_scale':s2,'weighted_rmse':float(np.sqrt(np.sum(weights[active]*e*e)/np.sum(weights[active]))),'median':float(np.median(e)),'p90':float(np.percentile(e,90)),'max_head_vertex_shift_this_pass_m':float(np.max(np.linalg.norm(delta,axis=1)))})
        print(json.dumps(passes[-1]))

    a1=tri_area(v,hf);ratio=a1/np.maximum(a0,1e-12);q01=float(np.percentile(ratio,1));q99=float(np.percentile(ratio,99));maxshift=float(np.max(np.linalg.norm(v-v0,axis=1)))
    if q01<.20 or q99>3.5:raise RuntimeError(f'v12.5 surface quality failed p01={q01:.4f}, p99={q99:.4f}')
    outm=trimesh.Trimesh(vertices=v,faces=f,process=False);outm.export(args.out/'AINA_FACEVERSE_FULL_v12.5_DENSE_RESIDUAL.obj');outm.export(args.out/'AINA_FACEVERSE_FULL_v12.5_DENSE_RESIDUAL.glb');outm.export(args.out/'AINA_FACEVERSE_FULL_v12.5_DENSE_RESIDUAL.ply')
    lm=v[ctrl_ids,:2];ss,mmx,mmy,pred=similarity(lm,tgt,weights[active]);e=np.linalg.norm(pred-tgt,axis=1);overlay(rgb,tgt,pred,qa/'AINA_MP468_RESIDUAL_OVERLAY_v12.5.png');views=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p=qa/f'AINA_FACEVERSE_CLAY_{label}_v12.5.png';render(v,f,yaw,p,f'AINA FaceVerse v12.5 {label}');views.append(p)
    ims=[Image.open(x).convert('RGB') for x in views];H=max(x.height for x in ims);W=max(x.width for x in ims);sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_FACEVERSE_CLAY_5VIEW_v12.5.png');compare(args.front,qa/'AINA_FACEVERSE_CLAY_front_v12.5.png',qa/'AINA_REFERENCE_VS_FACEVERSE_FRONT_v12.5.png')
    rep={'version':'AINA FaceVerse v12.5 Dense Residual Sculpt','base':'v12.4 native-axis coherent FaceVerse fit','topology_changed':False,'largest_head_component_vertices':int(len(head_ids)),'head_faces':int(len(head_faces)),'disconnected_components_untouched':int(len(comps)-1),'final_weighted_rmse':float(np.sqrt(np.sum(weights[active]*e*e)/np.sum(weights[active]))),'max_total_vertex_shift_m':maxshift,'triangle_area_ratio_p01':q01,'triangle_area_ratio_p99':q99,'passes':passes,'identity_lock':False,'acceptance_note':'This is the first direct dense residual geometry transfer; pass only if front/45/profile visually match AINA.'};(args.out/'AINA_FACEVERSE_v12.5_REPORT.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
