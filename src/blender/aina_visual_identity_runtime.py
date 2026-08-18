#!/usr/bin/env python3
"""Runtime wrapper for AINA real-3D visual identity assembly."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))

from aina_vrm_addon_runtime import ensure_vrm_addon


def main():
    root=Path.cwd(); ensure_vrm_addon(root)
    import aina_final_vrm_release as release
    import aina_visual_identity_assembly as visual
    import aina_visual_eye_system as eye_system

    def already_enabled(_root:Path):
        import bpy
        if 'io_scene_vrm' not in bpy.context.preferences.addons:
            raise RuntimeError('VRM Addon preferences disappeared before visual assembly')
        return None,None,None

    # Correct in-place region helpers. Advanced indexing returns a copy, so the
    # visual module's semantic sculpt is replaced here with an explicit indexed
    # implementation before any geometry is created.
    def w(coords,c,r,inner=0.0,outer=1.0):
        c=np.asarray(c,float);r=np.asarray(r,float)
        q=np.sqrt(np.sum(((coords-c)/r)**2,axis=1));a=np.zeros(len(coords),float);a[q<=inner]=1.0
        m=(q>inner)&(q<outer)
        if np.any(m):
            t=(q[m]-inner)/(outer-inner+1e-12);a[m]=.5*(1+np.cos(np.pi*t))
        return a
    def sh(arr,ids,c,r,d,inner=0.0,outer=1.0):
        ids=np.asarray(ids,np.int64);p=arr[ids].copy();p+=w(p,c,r,inner,outer)[:,None]*np.asarray(d,float);arr[ids]=p
    def sc(arr,ids,c,r,s,inner=0.0,outer=1.0):
        ids=np.asarray(ids,np.int64);p=arr[ids].copy();c=np.asarray(c,float);ww=w(p,c,r,inner,outer)[:,None];tar=c+(p-c)*np.asarray(s,float);arr[ids]=p+ww*(tar-p)

    def corrected_polish(mapped,head_ids,eye_groups):
        out=mapped.copy();h=np.asarray(head_ids,np.int64)
        z=out[h,2];t=np.clip((1.605-z)/.10,0,1);p=out[h].copy();p[:,0]*=(1-.18*(t**1.45));out[h]=p
        z=out[h,2];m=z<1.58;ids=h[m];ww=np.clip((1.58-z[m])/.08,0,1);p=out[ids].copy();p[:,2]+=.009*(ww**1.15);out[ids]=p
        z=out[h,2];wt=np.clip((z-1.61)/.09,0,1);p=out[h].copy();p[:,0]*=(1-.045*wt);out[h]=p

        lm=out[visual.K].copy()
        for rr in (range(36,42),range(42,48)):
            c=lm[list(rr)].mean(0);sc(out,h,c,(.040,.024,.020),(1.16,1,1.12),.10,1.18);side=-1 if c[0]<0 else 1
            sh(out,h,(c[0]+side*.016,c[1],c[2]),(.018,.020,.016),(0,.0005,.0015),0,1.10)
            sh(out,h,(c[0],c[1]+.010,c[2]+.015),(.040,.040,.030),(0,.0020,.0015),0,1.05)
        lm=out[visual.K].copy()
        for rr in (range(17,22),range(22,27)):
            c=lm[list(rr)].mean(0);sh(out,h,c,(.042,.032,.026),(0,.0025,.0035),0,1.18)

        lm=out[visual.K].copy();bridge=lm[27:31].mean(0);nbase=lm[31:36].mean(0)
        sc(out,h,nbase,(.026,.028,.032),(.76,1,.88),0,1.15);sh(out,h,nbase,(.028,.030,.030),(0,.0042,.0052),0,1.15)
        sh(out,h,bridge,(.023,.030,.045),(0,.0023,.0023),0,1.10);sc(out,h,lm[27],(.024,.028,.028),(.86,1,.96),0,1.05)
        for c in ((-.032,-.002,1.577),(.032,-.002,1.577)):sh(out,h,c,(.044,.042,.040),(0,-.0028,.0018),0,1.12)

        lm=out[visual.K].copy();mouth=lm[48:60].mean(0)
        sc(out,h,mouth,(.043,.032,.026),(.82,1,.60),0,1.20);sh(out,h,mouth,(.045,.032,.030),(0,.0018,.0032),0,1.10)
        for corner in (lm[48],lm[54]):sh(out,h,corner,(.018,.020,.014),(0,.0005,.0010),0,1.05)
        lm=out[visual.K].copy();chin=lm[8]
        sc(out,h,chin,(.050,.050,.050),(.78,.92,.86),0,1.15);sh(out,h,chin,(.052,.050,.048),(0,.0022,.0040),0,1.05)
        lm=out[visual.K].copy()
        for ii in (0,16):
            c=lm[ii];sc(out,h,c,(.032,.040,.052),(.70,.80,.72),0,1.15);sh(out,h,c,(.034,.042,.055),((.006 if c[0]<0 else -.006),.003,0),0,1.05)

        # Original FaceVerse eye components are not used for final visual eyes,
        # but keep their coordinates coherent for topology/debug exports.
        lm=out[visual.K].copy()
        for ids in eye_groups:
            ids=np.asarray(ids,np.int64);c=out[ids].mean(0);target=lm[36:42].mean(0) if c[0]<0 else lm[42:48].mean(0)
            p=out[ids].copy();p+=np.array([target[0]-c[0],.0090,target[2]-c[2]]);c2=p.mean(0);p=c2+(p-c2)*np.array([1.04,1.00,.94]);out[ids]=p
        return out

    visual.polish_real_face=corrected_polish
    visual.base.enable_addons=already_enabled
    visual.base.create_body=release.create_native_body
    # Replace protruding FaceVerse eyeballs with the real 3D almond eye system,
    # including blink/wide/squint/look shape-key bindings.
    eye_system.install(visual,release)
    visual.main()

if __name__=='__main__': main()
