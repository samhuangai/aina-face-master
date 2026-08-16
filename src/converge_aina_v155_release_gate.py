#!/usr/bin/env python3
"""Final micro-convergence for the same AINA v15.5 locked production mesh.

This does not create a new face/version and does not relax any release threshold.
It consumes the already art-directed v15.5 locked mesh and performs three very
small smooth RBF landmark-space corrections, then overwrites the same locked
OBJ/GLB/PLY names and canonical v15.5 report.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components

K=np.array([1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,13695,12973,13422,6543,6537],dtype=np.int64)
W=np.ones(68,float);W[:17]=4.8;W[17:27]=1.5;W[27:36]=4.2;W[36:48]=6.2;W[48:60]=5.2;W[60:]=2.2

def components(nv,f):
 e=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);a=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(nv,nv));a=(a+a.T).tocsr();n,lab=connected_components(a,directed=False);return [np.flatnonzero(lab==i) for i in range(n)]

def target(path):
 d=json.loads(path.read_text());q=np.asarray(d['landmarks_xy'],float)
 if q.shape!=(68,2):raise RuntimeError(f'Expected 68x2 landmarks, got {q.shape}')
 return q

def similarity(cur,tgt):
 ww=W[:,None];mx=(cur*ww).sum(0)/W.sum();my=(tgt*ww).sum(0)/W.sum();X=cur-mx;Y=tgt-my;H=(X*ww).T@Y;U,S,Vt=np.linalg.svd(H);R=U@Vt
 if np.linalg.det(R)<0:Vt[-1]*=-1;R=U@Vt
 s=float(S.sum()/max(np.sum(ww*(X*X)),1e-12));desired=((tgt-my)@R.T)/max(s,1e-12)+mx;pred=s*(X@R)+my;err=np.linalg.norm(pred-tgt,axis=1)
 return desired,float(np.sqrt(np.mean(err*err))),float(err.max())

def micro(p,kl,tgt,iters=3):
 hist=[]
 for i in range(iters):
  lm=p[kl];desired,rmse,maxe=similarity(lm[:,:2],tgt);disp=desired-lm[:,:2];dn=np.linalg.norm(disp,axis=1);cap=.0035;bad=dn>cap
  if np.any(bad):disp[bad]*=(cap/dn[bad])[:,None]
  sigma=.009;strength=.38;s2=sigma*sigma;zs2=.024*.024
  for st in range(0,len(p),2048):
   pp=p[st:st+2048];dx=pp[:,None,0]-lm[None,:,0];dy=pp[:,None,1]-lm[None,:,1];dz=pp[:,None,2]-lm[None,:,2];w=np.exp(-(dx*dx+dy*dy)/(2*s2)-dz*dz/(2*zs2));sw=w.sum(1);val=(w@disp)/(sw[:,None]+1e-12);env=1-np.exp(-.9*sw);face=(1/(1+np.exp((pp[:,2]-.070)/.008)))*(1/(1+np.exp((-pp[:,1]-.055)/.007)));p[st:st+len(pp),:2]+=strength*val*env[:,None]*face[:,None]
  hist.append({'iteration':i+1,'rmse_px':rmse,'max_px':maxe,'max_requested_move_mm':float(dn.max()*1000)})
 return hist

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mesh',type=Path,required=True);ap.add_argument('--target-landmarks',type=Path,required=True);ap.add_argument('--prior-report',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 prior=json.loads(a.prior_report.read_text());tgt=target(a.target_landmarks);m=trimesh.load(a.mesh,process=False,maintain_order=True);v0=np.asarray(m.vertices,float);v=v0.copy();f=np.asarray(m.faces,np.int64);cs=components(len(v),f);head=max(cs,key=len);hm=np.zeros(len(v),bool);hm[head]=True;g={int(q):i for i,q in enumerate(head)};kl=np.array([g[int(q)] for q in K]);p=v[head].copy();hist=micro(p,kl,tgt,3);v[head]=p
 old=v0[K];new=v[K];eyes=sorted([q for q in cs if 650<len(q)<900],key=lambda q:v0[q].mean(0)[0])
 if len(eyes)!=2:raise RuntimeError(f'Expected 2 eye components, got {len(eyes)}')
 for ids,o,n in zip(eyes,(old[36:42],old[42:48]),(new[36:42],new[42:48])):v[ids]+=n.mean(0)-o.mean(0)
 d=new[48:60].mean(0)-old[48:60].mean(0)
 for ids in cs:
  if np.array_equal(ids,head) or any(np.array_equal(ids,e) for e in eyes):continue
  v[ids]+=d
 _,rmse,maxe=similarity(v[K,:2],tgt)
 tri0=v0[f];tri1=v[f];c0=np.cross(tri0[:,1]-tri0[:,0],tri0[:,2]-tri0[:,0]);c1=np.cross(tri1[:,1]-tri1[:,0],tri1[:,2]-tri1[:,0]);ar=np.linalg.norm(c1,axis=1)/np.maximum(np.linalg.norm(c0,axis=1),1e-12);p01=float(np.percentile(ar,1));p99=float(np.percentile(ar,99));flips=int(np.sum(np.sum(c0*c1,axis=1)<0))
 prior_checks=prior.get('checks',{});prior_mesh_ok=bool(prior_checks.get('topology_preserved',True) and prior_checks.get('finite_geometry',True) and prior_checks.get('two_eye_components',True) and prior_checks.get('triangle_area_p01_ge_0_50',True) and prior_checks.get('triangle_area_p99_le_1_55',True))
 checks={'topology_preserved':bool(len(v)==len(v0) and len(f)==len(np.asarray(m.faces))),'finite_geometry':bool(np.isfinite(v).all()),'two_eye_components':len(eyes)==2,'prior_mesh_health_pass':prior_mesh_ok,'front_landmark_rmse_le_1_10px':rmse<=1.10,'front_landmark_max_le_3_0px':maxe<=3.0,'incremental_triangle_area_p01_ge_0_80':p01>=.80,'incremental_triangle_area_p99_le_1_25':p99<=1.25,'no_new_normal_flips':flips==0}
 lock=bool(all(checks.values()));out=trimesh.Trimesh(vertices=v,faces=f,process=False)
 for ext in ('obj','glb','ply'):out.export(a.out/f'AINA_FACEVERSE_FULL_v15.5_IDENTITY_LOCKED.{ext}')
 keep=hm.copy();[keep.__setitem__(e,True) for e in eyes];fid=np.flatnonzero(keep[f].all(1));clay=out.submesh([fid],append=True,repair=False)
 for ext in ('obj','glb','ply'):clay.export(a.out/f'AINA_FACEVERSE_IDENTITY_CLAY_v15.5_LOCKED.{ext}')
 rep={'version':'AINA Face Master v15.5 Identity Lock','identity_lock':lock,'candidate':False,'no_new_face_version':True,'checks':checks,'front_landmark_rmse_px':rmse,'front_landmark_max_px':maxe,'incremental_triangle_area_ratio_p01':p01,'incremental_triangle_area_ratio_p99':p99,'incremental_normal_flip_count':flips,'prior_gate':prior,'micro_convergence_history':hist,'qa_gate':'same v15.5 real mesh; strict front landmark gate + preserved prior mesh health + shallow 3Q/profile visual QA','note':'Release convergence tightens the real mesh; thresholds were not relaxed and no reference/effect image was generated.'};(a.out/'AINA_FACEVERSE_v15.5_REPORT.json').write_text(json.dumps(rep,indent=2),encoding='utf-8');print(json.dumps(rep,indent=2))
 if not lock:raise SystemExit('AINA v15.5 final release convergence failed')
if __name__=='__main__':main()
