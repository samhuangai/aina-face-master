#!/usr/bin/env python3
"""AINA Face Master v10.4 — eyelid contour + V-jaw identity lock pass.

Takes the stable v10.3 full GNM topology and applies only targeted corrective
fields: true eyelid contour handles, lower-jaw silhouette handles, nose alar/tip
refinement, and coherent outer-lip recovery. The approved front art is first
symmetrized so detector noise cannot make the two eyes or jaw asymmetric.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import cv2,face_alignment
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from PIL import Image
import trimesh
from gnm.shape import gnm_numpy,gnm_landmarks


def img(p):return np.asarray(Image.open(p).convert('RGB'))
def detect(fa,im):
 h,w=im.shape[:2];s=max(1.,720./max(h,w));x=cv2.resize(im,None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC) if s>1 else im;ps=fa.get_landmarks_from_image(x)
 if not ps:raise RuntimeError('no face');ctr=np.array([x.shape[1]/2,x.shape[0]/2]);q=min(ps,key=lambda p:np.linalg.norm(np.asarray(p)[:,:2].mean(0)-ctr));return np.asarray(q,dtype=np.float64)[:,:2]/s
def norm(p,shape):h,w=shape[:2];s=.5*max(w,h);return(p-np.array([w*.5,h*.5]))/s
def lm(v,idx,bw):return(v[idx]*bw[...,None]).sum(-2)
def target_cam(t,cam):return(t-np.asarray(cam['translation']))/float(cam['scale'])
def proj(p,cam):r=np.asarray(cam['rotation_rows']);return float(cam['scale'])*(p@r.T)[:,:2]+np.asarray(cam['translation'])

def symmetrize(t):
 o=t.copy();cx=float(np.mean([t[27,0],t[30,0],t[33,0],t[8,0],t[51,0],t[57,0]]))
 pairs=[(0,16),(1,15),(2,14),(3,13),(4,12),(5,11),(6,10),(7,9),(17,26),(18,25),(19,24),(20,23),(21,22),(31,35),(32,34),(36,45),(37,44),(38,43),(39,42),(40,47),(41,46),(48,54),(49,53),(50,52),(55,59),(56,58),(60,64),(61,63),(65,67)]
 for a,b in pairs:
  da=abs(t[a,0]-cx);db=abs(t[b,0]-cx);d=.5*(da+db);y=.5*(t[a,1]+t[b,1]);o[a]=[cx-d,y];o[b]=[cx+d,y]
 for i in [8,27,28,29,30,33,51,57,62,66]:o[i,0]=cx
 return o

def field(p,seeds,deltas,radii,gain=1.):
 num=np.zeros_like(p);den=np.zeros(len(p));support=np.zeros(len(p));xy=p[:,:2]
 for s,d,r in zip(seeds,deltas,radii):
  dist=np.linalg.norm(xy-s[:2],axis=1);w=np.exp(-.5*(dist/max(r,1e-6))**4);num+=w[:,None]*d;den+=w;support=np.maximum(support,w)
 out=num/np.maximum(den[:,None],1e-12);return out*support[:,None]*gain

def local_affine(p,c,tc,sx,sy,rx,ry,gain=.7,zshift=0):
 q=p[:,:2]-c;desired=tc+q*np.array([sx,sy]);d=desired-p[:,:2];w=np.exp(-.5*(((p[:,0]-c[0])/rx)**2+((p[:,1]-c[1])/ry)**2)**1.2)*gain;p[:,:2]+=d*w[:,None];p[:,2]+=zshift*w

def smooth(d,f,it=3,a=.14):
 adj=[set() for _ in range(len(d))]
 for x,y,z in f:adj[x].update((y,z));adj[y].update((x,z));adj[z].update((x,y))
 r=d.copy()
 for _ in range(it):
  old=r.copy()
  for i,n in enumerate(adj):
   if n:r[i]=(1-a)*old[i]+a*old[list(n)].mean(0)
 return .82*r+.18*d

def render(v,f,R0,yaw,path,title):
 right,up,forward=R0[0],R0[1],R0[2];a=math.radians(yaw);R=np.stack([math.cos(a)*right+math.sin(a)*forward,up,-math.sin(a)*right+math.cos(a)*forward]);p=v@R.T;xy=p[:,:2];tr=p[f];n=np.cross(tr[:,1]-tr[:,0],tr[:,2]-tr[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9);order=np.argsort(tr[:,:,2].mean(1))[::-1];ff=f[order];nn=n[order];tri2=xy[ff];I=np.clip(.64+.23*np.abs(nn[:,2])+.10*np.clip(-.3*nn[:,0]-.2*nn[:,1]-.7*nn[:,2],0,1),.48,.98);col=np.stack([I*.96,I*.97,I],1);lo=np.percentile(xy,1.5,0);hi=np.percentile(xy,98.5,0);ct=(lo+hi)/2;ex=max((hi-lo).max(),1e-6)*.57;fig,ax=plt.subplots(figsize=(5,5),dpi=190);ax.add_collection(PolyCollection(tri2,facecolors=col,edgecolors='none'));ax.set_xlim(ct[0]-ex,ct[0]+ex);ax.set_ylim(ct[1]+ex,ct[1]-ex);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10);fig.tight_layout(pad=.12);fig.savefig(path,bbox_inches='tight',pad_inches=.02);plt.close(fig)
def overlay(im,tp,pn,path):
 h,w=im.shape[:2];s=.5*max(w,h);pp=pn*s+np.array([w*.5,h*.5]);fig,ax=plt.subplots(figsize=(6,6),dpi=160);ax.imshow(im);ax.scatter(tp[:,0],tp[:,1],s=10,label='reference');ax.scatter(pp[:,0],pp[:,1],s=8,marker='x',label='v10.4');ax.axis('off');ax.legend(fontsize=7);fig.tight_layout(pad=.2);fig.savefig(path,bbox_inches='tight');plt.close(fig)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base-full',type=Path,required=True);ap.add_argument('--front',type=Path,required=True);ap.add_argument('--cameras',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_v104'));a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);qa=a.out/'QA';qa.mkdir(exist_ok=True)
 ref=img(a.front);fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cpu',face_detector='sfd');tp=detect(fa,ref);tn=norm(tp,ref.shape);cams=json.loads(a.cameras.read_text());cam=cams['front'];R=np.asarray(cam['rotation_rows'],dtype=np.float64);tc=symmetrize(target_cam(tn,cam))
 g=gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,variant=gnm_numpy.GNMVariant.HEAD);m=trimesh.load(a.base_full,process=False);v=np.asarray(m.vertices,dtype=np.float64);tri=np.asarray(g.triangles,dtype=np.int64)
 if len(v)!=len(g.template_vertex_positions):raise RuntimeError(f'vertex order/count mismatch {len(v)}')
 sti=np.asarray(g.triangle_indices_for_group('skin'),dtype=np.int64);sfg=tri[sti];skin_ids=np.unique(sfg.reshape(-1));g2l={int(x):i for i,x in enumerate(skin_ids)};sf=np.vectorize(g2l.get)(sfg);cfg=gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68);idx=np.asarray(cfg.indices,dtype=np.int64);bw=np.asarray(cfg.weights,dtype=np.float64);L=lm(v,idx,bw);LC=L@R.T;sv=v[skin_ids];p=sv@R.T;orig=p.copy()
 # 1) true eyelid contour handles, symmetric target, compact support so upper/lower lids separate instead of moving together.
 eye_ids=[36,37,38,39,40,41,42,43,44,45,46,47];se=[];de=[];ra=[]
 for i in eye_ids:
  d2=(tc[i]-LC[i,:2])*.88;se.append(LC[i]);de.append(np.array([d2[0],d2[1],0.]));ra.append(.0048 if i not in (36,39,42,45) else .0062)
 p+=field(p,se,de,ra,1.0)
 # extra coherent aperture opening based on average target/current eyelid height.
 for ids in ([36,37,38,39,40,41],[42,43,44,45,46,47]):
  c=LC[ids,:2].mean(0);up=ids[1:3];lo=ids[4:6];ch=abs(LC[up,1].mean()-LC[lo,1].mean());th=abs(tc[up,1].mean()-tc[lo,1].mean());extra=max(0.,min(.0032,(th-ch)*.36));
  if extra>0:
   su=[LC[i] for i in up];sl=[LC[i] for i in lo];du=[np.array([0.,-extra,0.]) for _ in up];dl=[np.array([0.,extra*.72,0.]) for _ in lo];p+=field(p,su+sl,du+dl,[.0048]*4,1.)
 # 2) V-jaw silhouette handles. Strong near chin, modest near upper mandibular angle.
 Lnow=lm_from_cam=None
 jaw_ids=list(range(3,14));gains={3:.25,4:.42,5:.62,6:.82,7:.90,8:.92,9:.90,10:.82,11:.62,12:.42,13:.25};se=[];de=[];ra=[]
 for i in jaw_ids:
  d2=tc[i]-LC[i,:2];gg=gains[i];se.append(LC[i]);de.append(np.array([d2[0]*gg,d2[1]*gg*.55,0.]));ra.append(.014 if i in (6,7,8,9,10) else .017)
 p+=field(p,se,de,ra,1.)
 # 3) narrower alar/tip and more delicate nose projection.
 nose_ids=[30,31,32,33,34,35];se=[];de=[]
 for i in nose_ids:
  d2=tc[i]-LC[i,:2];se.append(LC[i]);de.append(np.array([d2[0]*.72,d2[1]*.55,0.]))
 p+=field(p,se,de,[.007]*len(se),1.)
 nc=LC[30:36,:2].mean(0);nw=np.exp(-.5*(((p[:,0]-nc[0])/.018)**2+((p[:,1]-nc[1])/.024)**2)**1.25);p[:,2]+=.00135*nw
 # 4) outer lips as one coherent form; restore target width without inner-lip puckering.
 mc=LC[48:60,:2].mean(0);mt=tc[48:60].mean(0);cw=abs(LC[54,0]-LC[48,0]);tw=abs(tc[54,0]-tc[48,0]);ch=abs(LC[[50,51,52],1].mean()-LC[[56,57,58],1].mean());th=abs(tc[[50,51,52],1].mean()-tc[[56,57,58],1].mean());local_affine(p,mc,mt,float(np.clip(tw/max(cw,1e-8),.94,1.16)),float(np.clip(th/max(ch,1e-8),.94,1.20)),max(cw*1.35,.04),max(ch*2.6,.032),.72,-.00045)
 raw=p-orig;cap=.0048;n=np.linalg.norm(raw,axis=1);bad=n>cap
 if np.any(bad):raw[bad]*=(cap/n[bad])[:,None]
 d=smooth(raw,sf,3,.13);p=orig+d;sv2=p@R;v2=v.copy();v2[skin_ids]=sv2
 skin=trimesh.Trimesh(vertices=sv2,faces=sf,process=False);skin.export(a.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.4.obj');skin.export(a.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.4.ply');skin.export(a.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.4.glb');full=trimesh.Trimesh(vertices=v2,faces=tri,process=False);full.export(a.out/'AINA_FACE_MASTER_GNM_v10.4_FULL_TOPOLOGY.obj');full.export(a.out/'AINA_FACE_MASTER_GNM_v10.4_FULL_TOPOLOGY.glb')
 L2=lm(v2,idx,bw);pred=proj(L2,cam);overlay(ref,tp,pred,qa/'AINA_front_overlay_v10.4.png')
 # render skin and full; full view includes eyeballs so eye aperture can be judged truthfully.
 for meshv,meshf,prefix in [(sv2,sf,'SKIN'),(v2,tri,'FULL')]:
  paths=[]
  for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:q=qa/f'AINA_{prefix}_CLAY_{label}_v10.4.png';render(meshv,meshf,R,yaw,q,f'AINA v10.4 {prefix} {label.replace("_"," ")}');paths.append(q)
  ims=[Image.open(q).convert('RGB') for q in paths];H=max(x.height for x in ims);W=max(x.width for x in ims);sheet=Image.new('RGB',(W*5,H),'white')
  for k,im in enumerate(ims):sheet.paste(im,(k*W+(W-im.width)//2,(H-im.height)//2))
  sheet.save(qa/f'AINA_{prefix}_CLAY_5VIEW_v10.4.png')
 # reference vs actual full front QA.
 actual=Image.open(qa/'AINA_FULL_CLAY_front_v10.4.png').convert('RGB');refim=Image.open(a.front).convert('RGB');h=max(refim.height,actual.height);rw=int(refim.width*h/refim.height);aw=int(actual.width*h/actual.height);canvas=Image.new('RGB',(rw+aw,h),'white');canvas.paste(refim.resize((rw,h)),(0,0));canvas.paste(actual.resize((aw,h)),(rw,0));canvas.save(qa/'AINA_REFERENCE_VS_ACTUAL_FULL_FRONT_v10.4.png')
 def wd(x,a,b):return float(abs(x[b,0]-x[a,0]))
 lc2=L2@R.T;metrics={'eye_L_width_ratio_target_over_final':wd(tc,36,39)/max(wd(lc2,36,39),1e-9),'eye_R_width_ratio_target_over_final':wd(tc,42,45)/max(wd(lc2,42,45),1e-9),'nose_width_ratio_target_over_final':wd(tc,31,35)/max(wd(lc2,31,35),1e-9),'mouth_width_ratio_target_over_final':wd(tc,48,54)/max(wd(lc2,48,54),1e-9),'jaw_low_ratio_target_over_final':wd(tc,6,10)/max(wd(lc2,6,10),1e-9),'max_new_displacement_m':float(np.linalg.norm(d,axis=1).max())};report={'version':'AINA Face Master v10.4','base':'v10.3 stable GNM topology','method':'symmetric eyelid contour handles + V-jaw silhouette handles + nose/lip semantic correction','skin_vertices':int(len(sv2)),'skin_triangles':int(len(sf)),'metrics':metrics,'identity_lock':False,'note':'Actual full-head and skin five-view renders generated. Identity remains unlocked until visual comparison passes.'};(a.out/'AINA_v10.4_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
