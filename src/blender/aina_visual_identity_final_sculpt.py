#!/usr/bin/env python3
"""AINA final real-mesh visual sculpt patch.

Edits the existing v15.5 mesh; it does not generate or substitute a reference image.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import aina_visual_identity_assembly as visual


def _w(coords,c,r,inner=0.0,outer=1.0):
    c=np.asarray(c,float); r=np.asarray(r,float)
    q=np.sqrt(np.sum(((coords-c)/r)**2,axis=1)); a=np.zeros(len(coords),float); a[q<=inner]=1.0
    m=(q>inner)&(q<outer)
    if np.any(m):
        t=(q[m]-inner)/(outer-inner+1e-12); a[m]=.5*(1+np.cos(np.pi*t))
    return a

def _sh(arr,c,r,d,inner=0.0,outer=1.0):
    arr += _w(arr,c,r,inner,outer)[:,None]*np.asarray(d,float)

def _sc(arr,c,r,s,inner=0.0,outer=1.0):
    ww=_w(arr,c,r,inner,outer)[:,None]; c=np.asarray(c,float)
    arr += ww*(c+(arr-c)*np.asarray(s,float)-arr)

def final_polish(mapped,head_ids,eye_groups):
    out=mapped.copy(); h=np.asarray(head_ids,np.int64); p=out[h].copy(); lm=out[visual.K].copy()
    # AINA silhouette: narrower lower third, shorter chin, restrained forehead.
    z=p[:,2]; t=np.clip((1.575-z)/.085,0,1); p[:,0]*=(1-.115*t**1.35); out[h]=p
    p=out[h].copy(); z=p[:,2]; t=np.clip((z-1.625)/.095,0,1); p[:,0]*=(1-.035*t); out[h]=p
    chin=lm[8]; p=out[h].copy(); _sc(p,chin,(.044,.045,.042),(.82,.94,.88),0,1.12); _sh(p,chin,(.046,.048,.044),(0,-.001,.003),0,1.05); out[h]=p
    # Eyes: larger visible almond aperture, shallower brow/orbit, slightly forward eye plane.
    p=out[h].copy(); lm=out[visual.K].copy()
    for rr in (range(36,42),range(42,48)):
        c=lm[list(rr)].mean(0)
        _sc(p,c,(.045,.034,.026),(1.20,1.02,1.10),.05,1.16)
        side=-1 if c[0]<0 else 1
        _sh(p,(c[0]+side*.010,c[1],c[2]),(.020,.022,.018),(0,.0005,.0010),0,1.08)
        _sh(p,(c[0],c[1]+.006,c[2]+.006),(.044,.034,.028),(0,.0035,.0010),0,1.10)
    # Nose: visible fine bridge and small rounded tip without lengthening the face.
    lm=out[visual.K].copy(); bridge=lm[27:31].mean(0); tip=lm[30]; base=lm[31:36].mean(0)
    _sc(p,base,(.031,.030,.030),(.72,1.0,.86),0,1.15); _sh(p,base,(.032,.032,.034),(0,.0030,.0010),0,1.12)
    _sh(p,bridge,(.025,.032,.050),(0,.0040,.0015),0,1.10); _sc(p,tip,(.024,.027,.027),(.82,1.0,.90),0,1.05); _sh(p,tip,(.023,.028,.028),(0,.0030,.0015),0,1.05)
    # Mouth: smaller width, real volume, pushed back relative to nose; soften corners.
    mouth=lm[48:60].mean(0); _sc(p,mouth,(.046,.034,.028),(.90,1.0,.78),0,1.18); _sh(p,mouth,(.048,.034,.030),(0,.0050,.0020),0,1.12)
    for idx in (48,54): _sh(p,lm[idx],(.018,.020,.015),(0,.0010,.0010),0,1.04)
    # Midface: soft cheek support, no forward bulge.
    for c in ((lm[40]+lm[31]+lm[48])/3,(lm[46]+lm[35]+lm[54])/3): _sh(p,c,(.044,.040,.040),(0,-.0010,.0010),0,1.10)
    # Ears: small and close to skull.
    for idx in (0,16):
        c=lm[idx]; _sc(p,c,(.034,.043,.055),(.74,.80,.74),0,1.12); _sh(p,c,(.035,.044,.055),((.004 if c[0]<0 else -.004),.003,0),0,1.05)
    out[h]=p
    # Real eyeball placement: slightly larger, forward enough to read as eyes, but behind lids.
    for ids in eye_groups:
        ids=np.asarray(ids,np.int64); c=out[ids].mean(0); target=lm[36:42].mean(0) if c[0]<0 else lm[42:48].mean(0)
        out[ids] += np.array([target[0]-c[0],.0070,target[2]-c[2]])
        c2=out[ids].mean(0); out[ids]=c2+(out[ids]-c2)*np.array([1.12,1.04,1.06])
    return out

visual.polish_real_face=final_polish


def final_material(name,color,metallic=0.0,roughness=.48,emission=None):
    overrides={
      'AINA_Skin':((.78,.62,.60,1),0.0,.40,None),
      'AINA_EyeWhite':((.93,.96,1.0,1),0.0,.22,None),
      'AINA_Iris':((.18,.40,.55,1),.02,.15,None),
      'AINA_Pupil':((.004,.008,.016,1),0.0,.18,None),
      'AINA_Hair_Silver':((.76,.80,.88,1),.10,.23,None),
      'AINA_Suit_Pearl':((.70,.76,.86,1),.12,.26,None),
      'AINA_Teeth':((.96,.94,.90,1),0.0,.27,None),
      'AINA_MouthInner':((.26,.045,.06,1),0.0,.42,None),
    }
    if name in overrides: color,metallic,roughness,emission=overrides[name]
    return visual.ORIGINAL_MAKE_MATERIAL(name,color,metallic,roughness,emission)
visual.polished_make_material=final_material

if __name__=='__main__':
    import aina_visual_identity_runtime as runtime
    runtime.main()
