#!/usr/bin/env python3
"""AINA v14.0 art-directed semantic identity rebuild from smooth v12.5."""
import argparse,json
from pathlib import Path
import numpy as np,trimesh
from rebuild_aina_v133_landmark_laplacian import components
K=np.array([1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,13695,12973,13422,6543,6537])
def G(p,c,r):return np.exp(-.5*np.sum(((p-np.asarray(c))/np.asarray(r))**2,1))
def main():
 a=argparse.ArgumentParser();a.add_argument('--base-full',type=Path,required=True);a.add_argument('--out',type=Path,default=Path('output_v140'));x=a.parse_args();x.out.mkdir(parents=True,exist_ok=True)
 m=trimesh.load(x.base_full,process=False,maintain_order=True);v0=np.asarray(m.vertices,float);f=np.asarray(m.faces,int);v=v0.copy();cs=components(len(v),f);h=max(cs,key=len);hm=np.zeros(len(v),bool);hm[h]=1;mp={int(q):i for i,q in enumerate(h)};kl=np.array([mp[int(q)] for q in K]);p=v[h].copy()
 lat=np.clip((abs(p[:,0])-.045)/.04,0,1);up=np.clip((-p[:,1]+.01)/.12,0,1);rear=np.clip((p[:,2]+.005)/.08,0,1);p[:,0]*=1-.055*lat*(.45+.55*np.maximum(up,rear))
 for s in (-1.,1.):
  c=[s*.078,-.006,.027];w=G(p,c,[.022,.042,.042]);p[:,0]-=s*.0065*w;p[:,1]+=w*(c[1]-p[:,1])*.16;p[:,2]+=w*(c[2]-p[:,2])*.08
 lm=p[kl].copy()
 for ids in (np.arange(36,42),np.arange(42,48)):
  c=lm[ids].mean(0);w=G(p,c,[.032,.018,.025]);p[:,1]+=w*(p[:,1]-c[1])*.24;p[:,0]+=w*(p[:,0]-c[0])*.055;e=lm[ids];o=ids[np.argmin(e[:,0])] if c[0]<0 else ids[np.argmax(e[:,0])];w=G(p,lm[o],[.015,.01,.018]);p[:,1]-=.0021*w;p[:,2]+=.0004*w
 lm=p[kl];c=lm[17:27].mean(0);p[:,2]+=.0013*G(p,c,[.055,.022,.03])
 lm=p[kl];br=lm[27:31].mean(0);lo=lm[31:36].mean(0);no=lm[27:36].mean(0);p[:,2]+=.0028*G(p,no,[.027,.045,.034]);w=G(p,br,[.018,.035,.025]);p[:,0]+=w*(br[0]-p[:,0])*.12;p[:,2]+=.001*w;w=G(p,lo,[.023,.023,.025]);p[:,0]+=w*(lo[0]-p[:,0])*.22;p[:,1]-=.0014*w;p[:,2]+=.0024*w
 for s in (-1.,1.):
  c=[s*.035,.004,-.001];w=G(p,c,[.035,.032,.035]);p[:,0]+=w*(s*.031-p[:,0])*.025;p[:,2]-=.00115*w
 lm=p[kl];c=lm[48:68].mean(0);w=G(p,c,[.035,.025,.028]);p[:,0]+=w*(c[0]-p[:,0])*.085;p[:,1]+=w*(c[1]-p[:,1])*.1;p[:,2]+=.0017*w;p[:,1]-=.0008*G(p,c,[.028,.02,.022])
 w=np.clip((p[:,1]-.005)/.065,0,1)*np.exp(-.5*((p[:,2]-.01)/.06)**2);p[:,0]*=1-.07*w;p[:,1]-=.0017*w;lm=p[kl];c=lm[8];w=G(p,c,[.035,.025,.032]);p[:,0]+=w*(c[0]-p[:,0])*.15;p[:,1]-=.0022*w;p[:,2]+=.0016*w
 v[h]=p;old=v0[K];new=v[K];eyes=sorted([q for q in cs if 650<len(q)<900],key=lambda q:v0[q].mean(0)[0])
 for q,s in zip(eyes,[new[36:42].mean(0)-old[36:42].mean(0),new[42:48].mean(0)-old[42:48].mean(0)]):v[q]+=s
 s=new[48:60].mean(0)-old[48:60].mean(0)
 for q in cs:
  if np.array_equal(q,h) or any(np.array_equal(q,e) for e in eyes):continue
  v[q]+=s
 out=trimesh.Trimesh(v,f,process=False)
 for e in ('obj','glb','ply'):out.export(x.out/f'AINA_FACEVERSE_FULL_v14.0_ART_DIRECTED.{e}')
 keep=hm.copy()
 for q in eyes:keep[q]=1
 clay=out.submesh([np.flatnonzero(keep[f].all(1))],append=True,repair=False)
 for e in ('obj','glb','ply'):clay.export(x.out/f'AINA_FACEVERSE_IDENTITY_CLAY_v14.0.{e}')
 d=v[h]-v0[h];rep={'version':'AINA v14.0 Art-Directed Semantic Rebuild','base':'smooth v12.5','topology_changed':False,'max_head_delta_m':float(np.linalg.norm(d,axis=1).max()),'rms_head_delta_m':float(np.sqrt(np.mean(np.sum(d*d,1)))),'identity_lock':False,'qa_gate':'visual front + 45 + profile vs approved effect art','note':'v13.3 sparse-68 fit rejected as identity source; visual likeness has priority.'};(x.out/'AINA_FACEVERSE_v14.0_REPORT.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
