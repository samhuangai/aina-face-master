#!/usr/bin/env python3
"""AINA Face Master v10.2.1 — topology-safe Laplacian identity sculpt.

Solves a smooth displacement field on the GNM skin topology. Approved front,
3/4 and profile 68-point landmarks act as 2D camera-space constraints. Scalp,
ears and neck are automatically anchored. This avoids the RBF extrapolation
spikes seen in the experimental v10.2 pass while allowing identity changes that
GNM's 170D PCA space cannot express.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import cv2, face_alignment
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree
import trimesh
from gnm.shape import gnm_numpy, gnm_landmarks

VIEWS=('front','three_quarter','side')
VIEW_WEIGHT={'front':1.0,'three_quarter':0.38,'side':0.12}

def read_img(p): return np.asarray(Image.open(p).convert('RGB'))
def detect(fa,img):
 h,w=img.shape[:2];s=max(1.,720./max(h,w));work=cv2.resize(img,None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC) if s>1 else img
 ps=fa.get_landmarks_from_image(work)
 if not ps: raise RuntimeError('no face')
 ctr=np.array([work.shape[1]/2,work.shape[0]/2]);p=min(ps,key=lambda q:np.linalg.norm(np.asarray(q)[:,:2].mean(0)-ctr))
 return np.asarray(p,dtype=np.float64)[:,:2]/s
def norm_target(p,shape):
 h,w=shape[:2];s=.5*max(w,h);return (p-np.array([w*.5,h*.5]))/s
def lm_from(v,idx,bw): return (v[idx]*bw[...,None]).sum(-2)
def project(p,cam):
 r=np.asarray(cam['rotation_rows']);return float(cam['scale'])*(p@r.T)[:,:2]+np.asarray(cam['translation'])
def ids_for(view):
 if view=='front': return np.arange(68)
 if view=='three_quarter': return np.arange(68)
 return np.concatenate([np.arange(0,17),np.arange(27,36),np.arange(36,42),np.arange(48,60)])
def feat(i):
 if 36<=i<48:return 1.60
 if 27<=i<36:return 1.35
 if 48<=i<68:return 1.35
 if i<17:return 1.30
 if 17<=i<27:return .72
 return 1.0

def build_adjacency(n,faces):
 adj=[set() for _ in range(n)]
 for a,b,c in faces:
  adj[a].update((b,c));adj[b].update((a,c));adj[c].update((a,b))
 return adj

def solve_displacement(vertices,skin_ids,skin_faces_local,lm_idx_global,lm_bw,targets,cams):
 n=len(skin_ids);g2l={int(g):i for i,g in enumerate(skin_ids)}
 lm_local=np.vectorize(g2l.get)(lm_idx_global)
 sv=vertices[skin_ids];lm=lm_from(vertices,lm_idx_global,lm_bw)
 adj=build_adjacency(n,skin_faces_local)
 rows=[];cols=[];data=[];rhs=[];row=0
 def add(entries,b,weight):
  nonlocal row
  for c,val in entries:
   rows.append(row);cols.append(c);data.append(val*weight)
  rhs.append(float(b)*weight);row+=1
 # Smooth displacement field: d_i - mean(neighbors)=0, each xyz.
 lap_w=3.2
 for i,nbr in enumerate(adj):
  if not nbr:continue
  inv=1.0/len(nbr)
  for c in range(3):
   ent=[(3*i+c,1.0)]+[(3*j+c,-inv) for j in nbr]
   add(ent,0.0,lap_w)
 # Small global deformation prior.
 reg_w=.16
 for i in range(n):
  for c in range(3): add([(3*i+c,1.0)],0.0,reg_w)
 # Automatic rigid anchors far from facial controls.
 tree=cKDTree(lm)
 d,_=tree.query(sv,k=1)
 anchor=np.where(d>.064)[0]
 # Spatial thinning is enough; each anchor has all 3 coordinates fixed.
 if len(anchor)>950: anchor=anchor[::max(1,len(anchor)//950)][:950]
 for i in anchor:
  for c in range(3): add([(3*i+c,1.0)],0.0,24.0)
 # Multi-view projection constraints on barycentric GNM landmarks.
 control_base=145.0
 diagnostic={}
 for view in VIEWS:
  cam=cams[view];r=np.asarray(cam['rotation_rows'],dtype=np.float64);scale=float(cam['scale']);pred=project(lm,cam);ids=ids_for(view)
  err=np.linalg.norm(pred[ids]-targets[view][ids],axis=1);diagnostic[view]={'before_rmse':float(np.sqrt(np.mean(err**2))),'before_median':float(np.median(err))}
  for k in ids:
   delta=targets[view][k]-pred[k]
   weight=control_base*math.sqrt(VIEW_WEIGHT[view]*feat(int(k)))
   for axis in range(2):
    ent=[]
    for j,bary in zip(lm_local[k],lm_bw[k]):
     for c in range(3): ent.append((3*int(j)+c,float(bary)*scale*r[axis,c]))
    add(ent,float(delta[axis]),weight)
 A=coo_matrix((data,(rows,cols)),shape=(row,3*n)).tocsr();b=np.asarray(rhs)
 sol=lsqr(A,b,atol=2e-7,btol=2e-7,iter_lim=700,show=False)
 disp=sol[0].reshape(n,3)
 lengths=np.linalg.norm(disp,axis=1);max_allowed=.014
 # Smooth safety clamp; prevents any isolated vertex from becoming a spike.
 bad=lengths>max_allowed
 if np.any(bad): disp[bad]*=(max_allowed/lengths[bad])[:,None]
 out=vertices.copy();out[skin_ids]+=disp
 lm2=lm_from(out,lm_idx_global,lm_bw)
 for view in VIEWS:
  ids=ids_for(view);e=np.linalg.norm(project(lm2,cams[view])[ids]-targets[view][ids],axis=1)
  diagnostic[view].update(after_rmse=float(np.sqrt(np.mean(e**2))),after_median=float(np.median(e)),after_p90=float(np.percentile(e,90)))
 diagnostic['solver']={'istop':int(sol[1]),'iterations':int(sol[2]),'residual_norm':float(sol[3]),'anchors':int(len(anchor)),'max_displacement_m':float(np.linalg.norm(disp,axis=1).max()),'rms_displacement_m':float(np.sqrt(np.mean(disp**2)))}
 return out,diagnostic

def overlay(img,target_px,pred,out,title):
 h,w=img.shape[:2];s=.5*max(w,h);px=pred*s+np.array([w*.5,h*.5]);fig,ax=plt.subplots(figsize=(6,6),dpi=160);ax.imshow(img);ax.scatter(target_px[:,0],target_px[:,1],s=10,label='reference 68');ax.scatter(px[:,0],px[:,1],s=9,marker='x',label='v10.2.1 mesh');ax.axis('off');ax.set_title(title);ax.legend(loc='lower right',fontsize=7);fig.tight_layout(pad=.2);fig.savefig(out,bbox_inches='tight');plt.close(fig)
def render(v,f,r0,yaw,out,title):
 right,up,forward=r0[0],r0[1],r0[2];a=math.radians(yaw);right2=math.cos(a)*right+math.sin(a)*forward;forward2=-math.sin(a)*right+math.cos(a)*forward;R=np.stack([right2,up,forward2]);p=v@R.T;xy=p[:,:2];tri=p[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9);depth=tri[:,:,2].mean(1);order=np.argsort(depth)[::-1];ff=f[order];nn=n[order];tri2=xy[ff];dif=np.clip(np.abs(nn[:,2]),0,1);side=np.clip(-.3*nn[:,0]-.2*nn[:,1]-.7*nn[:,2],0,1);I=np.clip(.66+.21*dif+.10*side,.5,.98);col=np.stack([I*.96,I*.97,I],1);lo=np.percentile(xy,1.5,0);hi=np.percentile(xy,98.5,0);ct=(lo+hi)/2;ex=max((hi-lo).max(),1e-6)*.57;fig,ax=plt.subplots(figsize=(5,5),dpi=190);ax.add_collection(PolyCollection(tri2,facecolors=col,edgecolors='none'));ax.set_xlim(ct[0]-ex,ct[0]+ex);ax.set_ylim(ct[1]+ex,ct[1]-ex);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10);fig.tight_layout(pad=.15);fig.savefig(out,bbox_inches='tight',pad_inches=.02);plt.close(fig)

def main():
 ap=argparse.ArgumentParser();
 for n in ('front','three-quarter','side'):ap.add_argument('--'+n,type=Path,required=True)
 ap.add_argument('--identity',type=Path,required=True);ap.add_argument('--cameras',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_v1021'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);qa=args.out/'QA';qa.mkdir(exist_ok=True)
 refs={'front':read_img(args.front),'three_quarter':read_img(args.three_quarter),'side':read_img(args.side)};fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cpu',face_detector='sfd');tpx={k:detect(fa,refs[k]) for k in VIEWS};targets={k:norm_target(tpx[k],refs[k].shape) for k in VIEWS};cams=json.loads(args.cameras.read_text())
 g=gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,variant=gnm_numpy.GNMVariant.HEAD);ident=np.load(args.identity).reshape(1,-1);v=np.asarray(g(identity=ident))[0].astype(np.float64);tri=np.asarray(g.triangles,dtype=np.int64);sti=np.asarray(g.triangle_indices_for_group('skin'),dtype=np.int64);sfg=tri[sti];skin_ids=np.unique(sfg.reshape(-1));g2l={int(x):i for i,x in enumerate(skin_ids)};sf=np.vectorize(g2l.get)(sfg)
 cfg=gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68);idx=np.asarray(cfg.indices,dtype=np.int64);bw=np.asarray(cfg.weights,dtype=np.float64)
 v2,diag=solve_displacement(v,skin_ids,sf,idx,bw,targets,cams);sv=v2[skin_ids];skin=trimesh.Trimesh(vertices=sv,faces=sf,process=False);skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.2.1.obj');skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.2.1.ply');skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.2.1.glb');full=trimesh.Trimesh(vertices=v2,faces=tri,process=False);full.export(args.out/'AINA_FACE_MASTER_GNM_v10.2.1_FULL_TOPOLOGY.obj');full.export(args.out/'AINA_FACE_MASTER_GNM_v10.2.1_FULL_TOPOLOGY.glb')
 lm2=lm_from(v2,idx,bw)
 for view in VIEWS:overlay(refs[view],tpx[view],project(lm2,cams[view]),qa/f'AINA_{view}_overlay_v10.2.1.png',f'AINA v10.2.1 {view}')
 r0=np.asarray(cams['front']['rotation_rows']);paths=[]
 for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:p=qa/f'AINA_CLAY_{label}_v10.2.1.png';render(sv,sf,r0,yaw,p,f'AINA v10.2.1 Clay {label.replace("_"," ")}');paths.append(p)
 ims=[Image.open(p).convert('RGB') for p in paths];H=max(x.height for x in ims);W=max(x.width for x in ims);sheet=Image.new('RGB',(W*5,H),'white');
 for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
 sheet.save(qa/'AINA_CLAY_5VIEW_v10.2.1.png')
 report={'version':'AINA Face Master v10.2.1','base':'GNM v3 + v10.1 fitted identity','method':'sparse multi-view landmark constraints + topology Laplacian displacement + rigid anchors','skin_vertices':int(len(sv)),'skin_triangles':int(len(sf)),'diagnostic':diag,'identity_lock':False,'note':'Production candidate only after visual clay review; no hair/material tricks.'};(args.out/'AINA_v10.2.1_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
