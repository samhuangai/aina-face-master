#!/usr/bin/env python3
"""Art-direct final AINA identity on the clean continuous MPFB2 female topology.

The rejected FaceVerse geometry is not used. This keeps the native continuous
MPFB body/head vertex order and rig topology, and only performs smooth bounded
semantic deformation. identity_lock stays false until the actual Blender naked
head passes front, shallow-3Q and profile visual review.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy import sparse

def ell(p,c,r,inner=.28,outer=1.20):
 c=np.asarray(c,float);r=np.asarray(r,float);q=np.sqrt(np.sum(((p-c)/r)**2,axis=1));w=np.zeros(len(p));w[q<=inner]=1;m=(q>inner)&(q<outer)
 if np.any(m):t=(q[m]-inner)/(outer-inner);w[m]=.5*(1+np.cos(np.pi*t))
 return w

def affine(p,c,r,s=(1,1,1),shift=(0,0,0),inner=.28,outer=1.20,mask=None):
 w=ell(p,c,r,inner,outer)
 if mask is not None:w*=mask
 c=np.asarray(c,float);target=c+(p-c)*np.asarray(s,float)+np.asarray(shift,float);p+=w[:,None]*(target-p)

def adjacency(n,faces):
 e=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);A=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(n,n));A=(A+A.T).tocsr();return A,np.asarray(A.sum(1)).ravel()

def write_obj(path,v,f):
 with path.open('w',encoding='utf-8') as h:
  for x,y,z in v:h.write(f'v {x:.9f} {y:.9f} {z:.9f}\n')
  for a,b,c in f:h.write(f'f {int(a)+1} {int(b)+1} {int(c)+1}\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--topology',type=Path,required=True);ap.add_argument('--qa',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 dat=np.load(a.topology);v0=np.asarray(dat['vertices_local'],float);faces=np.asarray(dat['faces'],np.int32);v=v0.copy();qa=json.loads(a.qa.read_text());info=qa['interesting_head_face_groups'];head=np.asarray(info['head']['strong_indices'],np.int64);lips=np.asarray(info['lips']['strong_indices'],np.int64);ears=np.asarray(info['ears']['strong_indices'],np.int64)
 if len(v)!=19158 or len(head)<4500 or len(lips)<300:raise RuntimeError('Unexpected MPFB topology/group layout')
 p=v[head].copy();eye_z=1.565;eye_x=.0355;mouth_z=float(v[lips,2].mean());front=np.clip((-p[:,1]-.070)/.090,0,1)

 # Stronger AINA V-line than the first MPFB review: preserve skull/neck, narrow
 # the visible mid/lower face continuously instead of pinching local landmarks.
 t=np.clip((1.555-p[:,2])/.145,0,1);p[:,0]*=(1-.185*(t**1.18)*front)
 chinband=np.clip((1.490-p[:,2])/.075,0,1);p[:,0]*=(1-.115*chinband*front)
 lower=np.clip((mouth_z+.004-p[:,2])/.075,0,1)*front;p[:,2]+=.0115*lower
 # Retract the heavy lower profile, strongest around chin/jaw but zero at neck.
 p[:,1]+=.0060*np.clip((1.505-p[:,2])/.085,0,1)*front

 # Youthful forehead/temple: smooth front plane and slightly compact lateral temples.
 temple=np.exp(-.5*((p[:,2]-1.585)/.052)**2)*np.clip((np.abs(p[:,0])-.052)/.050,0,1)*front;p[:,0]*=(1-.050*temple)
 p[:,1]-=.0022*ell(p,[0,-.112,1.616],[.078,.052,.058],.20,1.08)

 # AINA eye sockets: visibly larger almond aperture, width dominant over height,
 # outer tail lifted a little. The region is broad enough to avoid lid ripples.
 for sg in (-1,1):
  ec=np.array([sg*eye_x,-.145,eye_z]);ef=np.clip((-p[:,1]-.100)/.052,0,1)
  affine(p,ec,[.038,.031,.026],s=(1.22,1.0,1.34),inner=.20,outer=1.12,mask=ef)
  outer=np.array([sg*.054,-.144,1.566]);p[:,2]+=.0024*ell(p,outer,[.017,.025,.016],.16,1.10)*ef
  # soften adult supraorbital plane
  p[:,1]+=.0022*ell(p,[sg*.034,-.123,1.592],[.037,.029,.024],.22,1.15)
  # gentle apple cheek volume below eye
  p[:,1]-=.0030*ell(p,[sg*.038,-.134,1.523],[.039,.032,.032],.20,1.14)

 # Nose: markedly shorter/narrower and less projected, matching the approved
 # small delicate nose rather than the MakeHuman default adult nose.
 affine(p,[0,-.143,1.548],[.023,.035,.043],s=(.74,.94,.88),shift=(0,.0018,.0020),inner=.24,outer=1.18,mask=front)
 affine(p,[0,-.158,1.515],[.030,.030,.029],s=(.62,.74,.68),shift=(0,.0050,.0060),inner=.22,outer=1.16,mask=front)

 # Perioral shell follows the compact AINA mouth before exact native lips are set.
 mc0=v0[lips].mean(0);affine(p,mc0,[.043,.037,.028],s=(.84,.93,.86),shift=(0,.0020,.0035),inner=.22,outer=1.15,mask=front)

 # Rounded small chin: narrower, shorter and retracted.
 affine(p,[0,-.126,1.438],[.047,.052,.048],s=(.68,.78,.74),shift=(0,.0070,.0080),inner=.22,outer=1.18,mask=front)

 v[head]=p
 # Exact lips after shell update: clearly smaller mouth, soft neutral closure.
 mc=v0[lips].mean(0);q=v0[lips].copy();q=mc+(q-mc)*np.array([.74,.94,.82]);q[:,1]+=.0022;q[:,2]+=.0040;v[lips]=q
 # Small close ears.
 for sg in (-1,1):
  ids=ears[np.sign(v0[ears,0])==sg];c=v0[ids].mean(0);v[ids]=c+(v0[ids]-c)*np.array([.70,.78,.76]);v[ids,0]-=sg*.0040

 # Very light depth-only relaxation on the face shell. Exact eyes/nose/lips are protected.
 A,deg=adjacency(len(v),faces);y=v[:,1].copy();region=np.zeros(len(v));region[head]=np.clip((-v[head,1]-.070)/.090,0,1)*np.exp(-.5*((v[head,2]-1.525)/.105)**2);protect=np.zeros(len(v));protect[lips]=1
 for sg in (-1,1):protect=np.maximum(protect,ell(v,[sg*eye_x,-.145,eye_z],[.034,.028,.024],.20,1.0))
 protect=np.maximum(protect,ell(v,[0,-.153,1.526],[.026,.030,.038],.22,1.0));region*=1-.86*protect
 for _ in range(2):av=(A@y)/np.maximum(deg,1);y+=.08*region*(av-y)
 v[:,1]=y
 # exact lips once more after relaxation
 q=v0[lips].copy();q=mc+(q-mc)*np.array([.74,.94,.82]);q[:,1]+=.0022;q[:,2]+=.0040;v[lips]=q

 write_obj(a.out/'AINA_MPFB_FULL_v15.5_IDENTITY_CANDIDATE.obj',v,faces);np.savez_compressed(a.out/'AINA_MPFB_FULL_v15.5_IDENTITY_CANDIDATE.npz',vertices=v,faces=faces,head_indices=head,lips_indices=lips,ears_indices=ears)
 d=np.linalg.norm(v[head]-v0[head],axis=1);rep={'version':'AINA v15.5 MPFB Continuous Female Identity Candidate','identity_lock':False,'visual_review_required':True,'topology_changed':False,'continuous_body':True,'vertices':int(len(v)),'triangles':int(len(faces)),'head_vertices':int(len(head)),'lip_vertices':int(len(lips)),'ear_vertices':int(len(ears)),'max_head_delta_m':float(d.max()),'rms_head_delta_m':float(np.sqrt(np.mean(d*d))),'semantic_targets':['stronger youthful V-line','larger soft almond eye sockets','short narrow delicate nose','smaller soft lips','apple cheeks','short rounded retracted chin','small close ears'],'note':'Same native MPFB body/head vertex order and rig topology. Lock remains false until naked-head Blender visual QA passes.'};(a.out/'AINA_MPFB_v15.5_IDENTITY_REPORT.json').write_text(json.dumps(rep,indent=2),encoding='utf-8');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
