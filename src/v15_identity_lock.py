#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np,trimesh
from rebuild_aina_v133_landmark_laplacian import components
from sculpt_aina_v140_art_directed import K,G

def polish(p,kl):
 lm=p[kl]; ey=lm[36:48,1].mean(); dy=p[:,1]-ey
 w=np.exp(-.5*((p[:,2]-.010)/.075)**2)*np.exp(-.5*(p[:,0]/.085)**2)*np.clip(dy/.09,0,1)
 p[:,1]-=.115*dy*w
 lm=p[kl]; mid=lm[29:36].mean(0); lo=lm[31:36].mean(0)
 w=G(p,mid,[.025,.036,.032]); p[:,2]+=.003*w; p[:,0]+=w*(mid[0]-p[:,0])*.06
 w=G(p,lo,[.023,.023,.026]); p[:,2]+=.0022*w; p[:,1]-=.0008*w; p[:,0]+=w*(lo[0]-p[:,0])*.10
 lm=p[kl]; mc=lm[48:68].mean(0); w=G(p,mc,[.036,.024,.026])
 p[:,2]+=.003*w; p[:,0]+=w*(p[:,0]-mc[0])*.08; p[:,1]+=w*(mc[1]-p[:,1])*.08
 lower=np.clip((p[:,1]-ey)/.085,0,1)*np.exp(-.5*((p[:,2]-.012)/.075)**2); p[:,0]*=1-.035*lower
 return p

def main():
 a=argparse.ArgumentParser(); a.add_argument('--base-full',type=Path,required=True); a.add_argument('--out',type=Path,default=Path('output_v150')); x=a.parse_args(); x.out.mkdir(parents=True,exist_ok=True)
 m=trimesh.load(x.base_full,process=False,maintain_order=True); v=np.asarray(m.vertices,float); f=np.asarray(m.faces,int); cs=components(len(v),f); h=max(cs,key=len); hm=np.zeros(len(v),bool); hm[h]=1; mp={int(q):i for i,q in enumerate(h)}; kl=np.array([mp[int(q)] for q in K]); p=v[h].copy(); v[h]=polish(p,kl)
 out=trimesh.Trimesh(v,f,process=False); out.export(x.out/'AINA_v15_candidate.obj'); (x.out/'AINA_v15.0_REPORT.json').write_text(json.dumps({'version':'AINA v15.0 candidate','identity_lock':False,'candidate':True},indent=2))
if __name__=='__main__':main()
