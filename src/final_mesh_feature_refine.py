#!/usr/bin/env python3
"""Final direct AINA mesh refinement pass.
Applies smooth vertex-space edits to the real skin mesh only. Topology is unchanged.
Focus: shorter delicate nose, integrated lips, smaller rounded chin, soft V jaw,
and controlled apple-cheek volume without widening the face.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import trimesh


def gauss(p,cx,cy,rx,ry):
    q=((p[:,0]-cx)/rx)**2+((p[:,1]-cy)/ry)**2
    return np.exp(-0.5*q*q)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mesh',type=Path,required=True)
    ap.add_argument('--cameras',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    m=trimesh.load(a.mesh,process=False)
    if not isinstance(m,trimesh.Trimesh): raise RuntimeError('expected skin Trimesh')
    R=np.asarray(json.loads(a.cameras.read_text())['front']['rotation_rows'],float)
    v=np.asarray(m.vertices,float); p=v@R.T; d=np.zeros_like(p)
    # Face-front weighting. z is camera depth in this production coordinate frame.
    front=np.exp(-0.5*((p[:,2]+0.035)/0.080)**4)
    cx=0.0055
    # Nose: retain visible bridge, reduce bulb/ala width, shorten and soften tip.
    nose=gauss(p,cx,-0.213,0.030,0.046)*front
    lower=1/(1+np.exp(-(p[:,1]+0.228)/0.010))
    d[:,0]+=-(p[:,0]-cx)*0.075*nose*lower
    d[:,1]+=-0.0018*nose*lower
    bridge=gauss(p,cx,-0.238,0.018,0.028)*front
    d[:,2]+=-0.0011*bridge
    tip=gauss(p,cx,-0.204,0.016,0.014)*front
    d[:,2]+=-0.0019*tip
    # Lips: avoid pasted-on oval by softening perimeter, keeping center volume.
    lip=gauss(p,cx,-0.177,0.048,0.021)*front
    center=gauss(p,cx,-0.177,0.029,0.016)*front
    d[:,0]+=-(p[:,0]-cx)*0.035*lip
    d[:,2]+=0.00055*lip-0.00135*center
    upper=gauss(p,cx,-0.184,0.028,0.008)*front
    lowerlip=gauss(p,cx,-0.170,0.031,0.010)*front
    d[:,2]+=-0.00055*upper-0.00090*lowerlip
    cupid=gauss(p,cx,-0.184,0.009,0.005)*front
    d[:,1]+=-0.00075*cupid
    # Chin and jaw: shorter lower third, smaller rounded chin, V without pinching.
    chin=gauss(p,cx,-0.120,0.045,0.040)*front
    d[:,0]+=-(p[:,0]-cx)*0.070*chin
    d[:,1]+=0.0020*chin
    d[:,2]+=0.0007*chin
    jaw=gauss(p,cx,-0.135,0.095,0.075)*front
    side=np.clip(np.abs(p[:,0]-cx)/0.090,0,1)
    d[:,0]+=-(p[:,0]-cx)*0.055*jaw*(0.35+0.65*side)
    # Apple cheeks: subtle central lift, suppress lateral ballooning.
    for x in (cx-0.040,cx+0.040):
        cheek=gauss(p,x,-0.158,0.040,0.038)*front
        d[:,2]+=-0.00085*cheek
        d[:,0]+=-(p[:,0]-x)*0.020*cheek
    # Gentle attenuation at face boundary.
    d*=np.clip(front[:,None],0,1)
    newp=p+d
    m.vertices=newp@R
    a.out.parent.mkdir(parents=True,exist_ok=True)
    m.export(a.out)
    print('vertices',len(m.vertices),'faces',len(m.faces),'max_displacement_m',float(np.linalg.norm(d,axis=1).max()))
if __name__=='__main__': main()
