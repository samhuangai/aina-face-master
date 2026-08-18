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
        """Preserve locked front X/Z; fix actual 3D depth and soft-tissue identity."""
        out=mapped.copy();h=np.asarray(head_ids,np.int64)

        # Only a restrained lower-third taper. Avoid destroying the already fitted
        # 68-point front silhouette.
        z=out[h,2];t=np.clip((1.565-z)/.060,0,1)
        p=out[h].copy();p[:,0]*=(1-.055*(t**1.35));out[h]=p
        # Smaller central chin without pulling jaw landmarks around.
        lm=out[visual.K].copy();chin=lm[8]
        sc(out,h,chin,(.040,.042,.037),(.86,.98,.96),0,1.08)
        sh(out,h,chin,(.042,.042,.040),(0,-.0012,.0010),0,1.02)

        # Mild upper-cranium narrowing only; final silver hair carries silhouette.
        z=out[h,2];wt=np.clip((z-1.635)/.080,0,1)
        p=out[h].copy();p[:,0]*=(1-.025*wt);out[h]=p

        # Eye/orbit: target has soft shallow sockets, not hard adult brow ridges.
        lm=out[visual.K].copy()
        for rr in (range(36,42),range(42,48)):
            c=lm[list(rr)].mean(0)
            sh(out,h,(c[0],c[1]+.008,c[2]+.013),(.040,.038,.028),(0,.0045,.0005),0,1.15)
        lm=out[visual.K].copy()
        for rr in (range(17,22),range(22,27)):
            c=lm[list(rr)].mean(0);sh(out,h,c,(.040,.032,.025),(0,.0045,.0010),0,1.15)
        sh(out,h,lm[27],(.032,.032,.040),(0,.0035,0),0,1.10)

        # Nose: preserve length, make alae/tip more delicate and just slightly
        # less projected. The old model was not mainly wrong in X/Z here.
        lm=out[visual.K].copy();nbase=lm[31:36].mean(0);tip=lm[30]
        sc(out,h,nbase,(.028,.026,.028),(.84,1.0,.96),0,1.15)
        sh(out,h,nbase,(.030,.030,.032),(0,.0020,.0005),0,1.10)
        sh(out,h,tip,(.022,.026,.026),(0,.0010,.0003),0,1.05)

        # Major profile correction: old lips projected farther forward than the
        # nose. Move mouth/perioral volume back while keeping its fitted front
        # width and a soft youthful lip thickness.
        lm=out[visual.K].copy();mouth=lm[48:60].mean(0)
        sh(out,h,mouth,(.052,.040,.038),(0,.0070,0),0,1.20)
        sc(out,h,mouth,(.044,.032,.025),(.96,1.0,.86),0,1.15)
        sh(out,h,lm[51],(.025,.030,.025),(0,.0020,0),0,1.00)
        sh(out,h,lm[57],(.027,.030,.025),(0,.0015,.0006),0,1.00)

        # Soft apple cheeks, small forward volume only.
        lm=out[visual.K].copy()
        for c in ((lm[40]+lm[31]+lm[48])/3,(lm[46]+lm[35]+lm[54])/3):
            sh(out,h,c,(.042,.040,.040),(0,-.0015,.0005),0,1.10)

        # Smaller tucked ears, a persistent mismatch in front/3Q views.
        lm=out[visual.K].copy()
        for ii in (0,16):
            c=lm[ii];sc(out,h,c,(.034,.043,.055),(.78,.82,.78),0,1.12)
            sh(out,h,c,(.034,.043,.055),((.004 if c[0]<0 else -.004),.004,0),0,1.05)

        # Original FaceVerse eye components are retained only as hidden coherent
        # source topology. The visible eyes are replaced by the real almond system.
        lm=out[visual.K].copy()
        for ids in eye_groups:
            ids=np.asarray(ids,np.int64);c=out[ids].mean(0);target=lm[36:42].mean(0) if c[0]<0 else lm[42:48].mean(0)
            p=out[ids].copy();p+=np.array([target[0]-c[0],.0100,target[2]-c[2]]);c2=p.mean(0);p=c2+(p-c2)*np.array([1.02,1.00,.92]);out[ids]=p
        return out

    # Pale luminous materials under controlled lights, matching AINA's visual
    # language without the old overexposed white-world setup.
    original_make=visual.ORIGINAL_MAKE_MATERIAL
    def better_material(name,color,metallic=0.0,roughness=.48,emission=None):
        overrides={
          'AINA_Skin':((.78,.61,.58,1),0.0,.43,None),
          'AINA_EyeWhite':((.92,.95,.99,1),0.0,.24,None),
          'AINA_Iris':((.20,.48,.70,1),.02,.18,None),
          'AINA_Pupil':((.006,.010,.020,1),0.0,.20,None),
          'AINA_Hair_Silver':((.70,.75,.84,1),.08,.26,None),
          'AINA_Suit_Pearl':((.66,.72,.82,1),.14,.28,None),
          'AINA_Teeth':((.94,.92,.88,1),0.0,.30,None),
          'AINA_MouthInner':((.30,.06,.08,1),0.0,.46,None),
        }
        if name in overrides:color,metallic,roughness,emission=overrides[name]
        return original_make(name,color,metallic,roughness,emission)

    visual.polish_real_face=corrected_polish
    visual.base.enable_addons=already_enabled
    visual.base.create_body=release.create_native_body
    visual.base.make_material=better_material
    eye_system.install(visual,release)
    visual.main()

if __name__=='__main__': main()
