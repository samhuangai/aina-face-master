#!/usr/bin/env python3
import numpy as np, trimesh, json, argparse
from pathlib import Path
from scipy import sparse
from scipy.sparse.csgraph import connected_components
K=np.array([1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,13695,12973,13422,6543,6537],dtype=np.int64)
def comps(nv,f):
 e=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]); a=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(nv,nv)); a=(a+a.T).tocsr(); n,lab=connected_components(a,directed=False); return [np.flatnonzero(lab==i) for i in range(n)]
def ell(p,c,r,inner=.45,outer=1.45):
 c=np.asarray(c);r=np.asarray(r);q=np.sqrt(np.sum(((p-c)/r)**2,axis=1));w=np.zeros(len(p));w[q<=inner]=1;m=(q>inner)&(q<outer);t=(q[m]-inner)/(outer-inner);w[m]=.5*(1+np.cos(np.pi*t));return w
def region_affine(p,c,r,s=(1,1,1),shift=(0,0,0),inner=.45,outer=1.45):
 w=ell(p,c,r,inner,outer)[:,None]; c=np.asarray(c); s=np.asarray(s); sh=np.asarray(shift); tgt=c+(p-c)*s+sh; p += w*(tgt-p)
def rbf(p,ctrl,disp,sigma=.006,zsigma=.012,strength=.5):
 ctrl=np.asarray(ctrl);disp=np.asarray(disp);s2=sigma*sigma;zs2=zsigma*zsigma
 for st in range(0,len(p),4096):
  pp=p[st:st+4096]; dx=pp[:,None,0]-ctrl[None,:,0];dy=pp[:,None,1]-ctrl[None,:,1];dz=pp[:,None,2]-ctrl[None,:,2];w=np.exp(-(dx*dx+dy*dy)/(2*s2)-dz*dz/(2*zs2));sw=w.sum(1);val=(w@disp)/(sw[:,None]+1e-12);env=1-np.exp(-1.15*sw);p[st:st+len(pp),:2]+=strength*val*env[:,None]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 m=trimesh.load(a.base,process=False,maintain_order=True);v0=np.asarray(m.vertices,float);f=np.asarray(m.faces,np.int64);v=v0.copy();cs=comps(len(v),f);head=max(cs,key=len);hm=np.zeros(len(v),bool);hm[head]=1;g={int(q):i for i,q in enumerate(head)};kl=np.array([g[int(q)] for q in K]);p=v[head].copy();lm=p[kl].copy();p0=p.copy()
 # head/face silhouette: preserve upper cranium, feminine narrower mid/lower face
 y=p[:,1];x=np.abs(p[:,0]);front=1/(1+np.exp((p[:,2]-.030)/.008)); side=1/(1+np.exp((x-.082)/.006))
 # cheek taper begins below eyes, stronger at jaw/chin
 wmid=np.clip((y+.010)/.055,0,1)*front*side; wlow=np.clip((y-.020)/.050,0,1)*front*side
 p[:,0]*=(1-.045*wmid)*(1-.090*wlow)
 # round / shorten chin slightly
 lm=p[kl].copy();chin=lm[8];region_affine(p,chin,[.040,.035,.045],s=(.91,.84,1.0),shift=(0,-.0023,.0008),inner=.35,outer=1.45)
 # eyes: clean almond, modestly wider, slightly less upturned outer tails
 lm=p[kl].copy()
 for ids,outerid in ((np.arange(36,42),36),(np.arange(42,48),45)):
  c=lm[ids].mean(0); region_affine(p,c,[.031,.020,.024],s=(1.07,1.025,1.0),inner=.38,outer=1.30)
  lm2=p[kl].copy(); disp=np.zeros((1,2));disp[0,1]=.00075;rbf(p,lm2[[outerid]],disp,sigma=.0045,zsigma=.008,strength=.65)
 # nose: narrow and retract lower nose, shorten lower axis, preserve delicate bridge
 lm=p[kl].copy();bc=lm[27:31].mean(0);region_affine(p,bc,[.020,.036,.025],s=(.90,.98,.88),shift=(0,0,.0010),inner=.4,outer=1.35)
 lm=p[kl].copy();nc=lm[30:36].mean(0);region_affine(p,nc,[.026,.025,.030],s=(.82,.90,.78),shift=(0,-.0018,.0025),inner=.35,outer=1.35)
 # small tip, subtle upward tip rotation via y-up and depth retraction
 lm=p[kl].copy();tip=lm[30];region_affine(p,tip,[.015,.017,.018],s=(.90,.90,.84),shift=(0,-.0008,.0012),inner=.35,outer=1.25)
 # lips: small soft mouth, slightly wider but much thinner, move up and retract
 lm=p[kl].copy();mc=lm[48:60].mean(0);region_affine(p,mc,[.038,.024,.030],s=(1.13,.84,.91),shift=(0,-.00035,.0007),inner=.35,outer=1.38)
 # cheeks: soften apple cheek width, keep gentle forward volume
 for sg in (-1,1):
  c=np.array([sg*.035,.010,.005]);region_affine(p,c,[.040,.038,.042],s=(.975,.985,.96),shift=(sg*.00035,0,-.00115),inner=.3,outer=1.35)
 # soften brow ridge and long bridge before rebuilding smooth feminine orbital mask
 lm=p[kl].copy()
 for ids in (np.arange(17,22),np.arange(22,27)):
  bc=lm[ids].mean(0); p[:,2]+=.0021*ell(p,bc,[.030,.022,.025],.38,1.28)
 br=lm[27:30].mean(0); p[:,2]+=.00125*ell(p,br,[.018,.030,.021],.40,1.28)
 # soften recessed orbital mask toward target's smooth feminine midface
 lm=p[kl].copy()
 for ids in (np.arange(36,42),np.arange(42,48)):
  ec=lm[ids].mean(0); wo=ell(p,ec,[.030,.024,.029],.38,1.30); wi=ell(p,ec,[.018,.011,.017],.62,1.12); p[:,2]-=.00115*wo*(1-.72*wi)
 # mild local Laplacian surface relaxation on z in face center, preserving silhouette
 # build head adjacency
 hf=f[hm[f].all(1)]; gl=-np.ones(len(v),int);gl[head]=np.arange(len(head));lf=gl[hf];rows=np.concatenate([lf[:,0],lf[:,1],lf[:,2],lf[:,1],lf[:,2],lf[:,0]]);cols=np.concatenate([lf[:,1],lf[:,2],lf[:,0],lf[:,0],lf[:,1],lf[:,2]]);A=sparse.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(head),len(head))).tocsr();deg=np.asarray(A.sum(1)).ravel();z=p[:,2].copy();lm=p[kl].copy();facec=np.array([0,.005,lm[30,2]+.010]);rel=ell(p,facec,[.073,.085,.050],.2,1.0)
 # exclude exact keypoint neighborhoods from strong smoothing
 for _ in range(3):
  av=(A@z)/np.maximum(deg,1);z=z+.14*rel*(av-z)
 p[:,2]=z
 v[head]=p
 # eyes components recenter behind current lids and preserve original spher sizes
 lm=v[K];eyes=sorted([q for q in cs if 650<len(q)<900],key=lambda q:v0[q].mean(0)[0]);
 for ids,el in zip(eyes,(lm[36:42],lm[42:48])):
  c=v[ids].mean(0);rim=el.mean(0);tc=np.array([rim[0],rim[1],rim[2]+.0070]);v[ids]+=tc-c
 # other disconnected oral components follow mouth center shift
 old=v0[K];new=v[K];shift=new[48:60].mean(0)-old[48:60].mean(0)
 for ids in cs: 
  if np.array_equal(ids,head) or any(np.array_equal(ids,e) for e in eyes):continue
  v[ids]+=shift
 out=trimesh.Trimesh(vertices=v,faces=f,process=False)
 for ext in ('obj','glb','ply'):out.export(a.out/f'AINA_FACEVERSE_FULL_v15.5_CLEAN_IDENTITY.{ext}')
 keep=hm.copy();[keep.__setitem__(e,True) for e in eyes];fid=np.flatnonzero(keep[f].all(1));clay=out.submesh([fid],append=True,repair=False)
 for ext in ('obj','glb','ply'):clay.export(a.out/f'AINA_FACEVERSE_IDENTITY_CLAY_v15.5.{ext}')
 rep={'version':'AINA Face Master v15.5 Clean Base Identity Rebuild','base':str(a.base),'topology_changed':False,'identity_lock':False,'candidate':True,'qa_gate':'actual naked-clay front + calibrated shallow 3Q (-15/-20/-25) + correctly oriented left-facing profile (+90) vs approved AINA references','reset_reason':'v15.4 accumulated depth deformation was rejected; v15.5 restarts from the verified smooth v12.5 topology/identity-fit mesh','note':'No new AINA effect/reference image is generated. Only actual OBJ/GLB/PLY geometry is edited.','max_head_delta_m':float(np.linalg.norm(p-p0,axis=1).max()),'rms_head_delta_m':float(np.sqrt(np.mean(np.sum((p-p0)**2,axis=1))))};(a.out/'AINA_FACEVERSE_v15.5_REPORT.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
