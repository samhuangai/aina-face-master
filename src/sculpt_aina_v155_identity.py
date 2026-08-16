#!/usr/bin/env python3
"""AINA Face Master v15.5 — final same-line identity sculpt.

No new face version is created. This starts from the verified smooth v12.5
FaceVerse topology and applies only broad semantic deformations selected by
visual comparison against the approved AINA effect-art. Sparse 68-point pulling
is intentionally avoided because it previously produced a numerically good but
visually wrong face.
"""
import argparse, json
from pathlib import Path
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components

K=np.array([1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,13695,12973,13422,6543,6537],dtype=np.int64)

def components(nv,f):
    e=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    a=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(nv,nv)); a=(a+a.T).tocsr()
    n,lab=connected_components(a,directed=False)
    return [np.flatnonzero(lab==i) for i in range(n)]

def ell(p,c,r,inner=.25,outer=1.25):
    c=np.asarray(c,float); r=np.asarray(r,float)
    q=np.sqrt(np.sum(((p-c)/r)**2,axis=1)); w=np.zeros(len(p)); w[q<=inner]=1.0
    m=(q>inner)&(q<outer)
    if np.any(m):
        t=(q[m]-inner)/(outer-inner); w[m]=.5*(1+np.cos(np.pi*t))
    return w

def affine(p,c,r,s=(1,1,1),shift=(0,0,0),inner=.25,outer=1.25):
    w=ell(p,c,r,inner,outer)[:,None]; c=np.asarray(c,float)
    target=c+(p-c)*np.asarray(s,float)+np.asarray(shift,float)
    p += w*(target-p)

def shift(p,c,r,d,inner=.25,outer=1.25):
    p += ell(p,c,r,inner,outer)[:,None]*np.asarray(d,float)

def health_and_identity_metrics(v0,v,f,hm):
    lm=v[K]; eye_h=(np.ptp(lm[36:42,1])+np.ptp(lm[42:48,1]))*.5
    metrics={
        'jaw_width_m':float(np.ptp(lm[:17,0])),
        'face_height_eye_to_chin_m':float(lm[8,1]-np.mean(lm[36:48,1])),
        'eye_height_mean_m':float(eye_h),
        'eye_width_right_m':float(np.ptp(lm[36:42,0])),
        'eye_width_left_m':float(np.ptp(lm[42:48,0])),
        'interocular_center_m':float(abs(lm[42:48,0].mean()-lm[36:42,0].mean())),
        'eye_outer_inner_slope_right_m':float(lm[36,1]-lm[39,1]),
        'eye_outer_inner_slope_left_m':float(lm[45,1]-lm[42,1]),
        'nose_width_m':float(np.ptp(lm[31:36,0])),
        'nose_length_m':float(lm[33,1]-lm[27,1]),
        'nose_projection_m':float(lm[27,2]-lm[30,2]),
        'mouth_width_m':float(abs(lm[54,0]-lm[48,0])),
        'mouth_height_m':float(np.ptp(lm[48:60,1])),
        'lip_to_chin_m':float(lm[8,1]-lm[57,1]),
    }
    pairs=[(i,16-i) for i in range(8)]+[(17,26),(18,25),(19,24),(20,23),(21,22)]+[(36,45),(37,44),(38,43),(39,42),(40,47),(41,46)]+[(31,35),(32,34)]+[(48,54),(49,53),(50,52),(55,59),(56,58),(60,64),(61,63),(65,67)]
    err=[abs(lm[a,0]+lm[b,0])+abs(lm[a,1]-lm[b,1])+abs(lm[a,2]-lm[b,2]) for a,b in pairs]
    metrics['symmetry_mean_m']=float(np.mean(err)); metrics['symmetry_max_m']=float(np.max(err))
    hf=f[hm[f].all(1)]; a=v0[hf]; b=v[hf]
    c0=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]); c1=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    ar=np.linalg.norm(c1,axis=1)/np.maximum(np.linalg.norm(c0,axis=1),1e-12)
    metrics.update({
        'triangle_area_ratio_min':float(ar.min()),
        'triangle_area_ratio_p01':float(np.percentile(ar,1)),
        'triangle_area_ratio_p05':float(np.percentile(ar,5)),
        'triangle_area_ratio_p99':float(np.percentile(ar,99)),
        'triangle_normal_flip_count':int(np.sum(np.sum(c0*c1,axis=1)<0)),
    })
    checks={
        'jaw_width':.130<=metrics['jaw_width_m']<=.137,
        'face_height':.085<=metrics['face_height_eye_to_chin_m']<=.092,
        'eye_height':.0085<=metrics['eye_height_mean_m']<=.0105,
        'eye_widths':.0275<=metrics['eye_width_right_m']<=.0305 and .0275<=metrics['eye_width_left_m']<=.0305,
        'eye_slope':-.0028<=metrics['eye_outer_inner_slope_right_m']<=-.0014 and -.0028<=metrics['eye_outer_inner_slope_left_m']<=-.0014,
        'nose_width':.0155<=metrics['nose_width_m']<=.0180,
        'nose_length':.032<=metrics['nose_length_m']<=.0365,
        'nose_projection':.0080<=metrics['nose_projection_m']<=.0110,
        'mouth_width':.039<=metrics['mouth_width_m']<=.043,
        'mouth_height':.015<=metrics['mouth_height_m']<=.019,
        'lip_to_chin':.021<=metrics['lip_to_chin_m']<=.026,
        'symmetry':metrics['symmetry_mean_m']<=.0030 and metrics['symmetry_max_m']<=.0050,
        'mesh_area':metrics['triangle_area_ratio_min']>=.20 and metrics['triangle_area_ratio_p05']>=.43 and metrics['triangle_area_ratio_p99']<=1.50 and metrics['triangle_normal_flip_count']==0,
    }
    return metrics,checks,bool(all(checks.values()))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--target-landmarks',type=Path); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    m=trimesh.load(a.base,process=False,maintain_order=True); v0=np.asarray(m.vertices,float); v=v0.copy(); f=np.asarray(m.faces,np.int64)
    cs=components(len(v),f); head=max(cs,key=len); hm=np.zeros(len(v),bool); hm[head]=True
    g={int(q):i for i,q in enumerate(head)}; kl=np.array([g[int(q)] for q in K]); p=v[head].copy(); p_start=p.copy(); lm=p[kl].copy()

    # Smaller/recessed ears so the naked-clay silhouette does not dominate.
    for sg in (-1,1): affine(p,[sg*.075,-.002,.055],[.028,.044,.040],s=(.72,.76,.74),shift=(-sg*.0055,-.001,.003),inner=.20,outer=1.12)
    # Youthful lower-face taper, restrained enough to retain the approved soft jaw width.
    front=1/(1+np.exp((p[:,2]-.043)/.010)); low=np.clip((p[:,1]+.008)/.060,0,1); p[:,0]*=1-.060*low*front
    # Recede adult brow ridge without pulling sparse landmarks.
    lm=p[kl].copy()
    for ids in (np.arange(17,22),np.arange(22,27)): shift(p,lm[ids].mean(0),[.034,.027,.030],[0,0,.0038],.22,1.18)
    # Soften orbital depth and reduce excessive outer-corner lift.
    lm=p[kl].copy()
    for ids in (np.arange(36,42),np.arange(42,48)): shift(p,lm[ids].mean(0),[.032,.022,.027],[0,0,.0008],.25,1.15)
    lm=p[kl].copy()
    for i in (36,45): shift(p,lm[i],[.008,.008,.013],[0,.0013,.0002],.12,1.08)
    # Delicate AINA nose: smaller bridge/lower nose and restrained projection.
    lm=p[kl].copy(); bridge=lm[27:31].mean(0); affine(p,bridge,[.020,.038,.030],s=(.78,.88,.84),shift=(0,-.0010,.0030),inner=.28,outer=1.24)
    lm=p[kl].copy(); lower=lm[30:36].mean(0); affine(p,lower,[.025,.026,.032],s=(.72,.82,.72),shift=(0,-.0017,.0025),inner=.24,outer=1.20)
    # Preserve natural lips while reducing the over-projected perioral region.
    lm=p[kl].copy(); mc=lm[48:60].mean(0); affine(p,mc,[.040,.024,.031],s=(1.00,.97,.96),shift=(0,-.0004,-.0005),inner=.26,outer=1.22)
    # Rounded tapered chin and subtle apple-cheek volume.
    lm=p[kl].copy(); chin=lm[8]; affine(p,chin,[.038,.033,.042],s=(.82,.90,.97),shift=(0,-.0025,-.0070),inner=.24,outer=1.28)
    lm=p[kl].copy(); shift(p,lm[8],[.030,.024,.034],[0,-.0005,-.0035],.16,1.14)
    for sg in (-1,1): shift(p,[sg*.036,.004,.003],[.038,.038,.043],[0,-.0005,-.0014],.18,1.17)

    # Final visual polish on the same v15.5 mesh: compact softer eyes, less harsh orbit,
    # slightly smaller nose, softer mouth, less projected chin and less prominent ears.
    lm=p[kl].copy()
    for ids,outerid,innerid in ((np.arange(36,42),36,39),(np.arange(42,48),45,42)):
        c=lm[ids].mean(0); affine(p,c,[.030,.018,.024],s=(.88,.82,1.0),shift=(0,0,.0005),inner=.28,outer=1.22)
        lm=p[kl].copy(); shift(p,lm[outerid],[.008,.008,.012],[0,.0013,.0002],.10,1.05); shift(p,lm[innerid],[.007,.007,.011],[0,-.0002,0],.10,1.05)
    lm=p[kl].copy()
    for ids in (np.arange(17,22),np.arange(22,27)):
        c=lm[ids].mean(0); shift(p,c,[.034,.028,.032],[0,.0003,.0016],.20,1.18)
    for ids in (np.arange(36,42),np.arange(42,48)):
        c=lm[ids].mean(0); shift(p,[c[0],c[1]-.010,c[2]],[.032,.023,.030],[0,0,.0010],.20,1.12)
    lm=p[kl].copy(); bridge=lm[27:31].mean(0); affine(p,bridge,[.019,.035,.028],s=(.94,.96,.96),shift=(0,-.0004,.0006),inner=.25,outer=1.18)
    lm=p[kl].copy(); lower=lm[30:36].mean(0); affine(p,lower,[.023,.023,.028],s=(.90,.92,.92),shift=(0,-.0005,.0007),inner=.24,outer=1.16)
    lm=p[kl].copy(); mc=lm[48:60].mean(0); affine(p,mc,[.039,.023,.029],s=(1.06,.98,.98),shift=(0,-.0003,-.0002),inner=.26,outer=1.20)
    front=1/(1+np.exp((p[:,2]-.043)/.010)); low=np.clip((p[:,1]-.010)/.055,0,1); p[:,0]*=1-.020*low*front
    lm=p[kl].copy(); chin=lm[8]; affine(p,chin,[.035,.030,.038],s=(.90,1.01,.96),shift=(0,.0013,.0030),inner=.24,outer=1.20)
    for sg in (-1,1): affine(p,[sg*.071,-.002,.057],[.026,.040,.038],s=(.86,.90,.88),shift=(-sg*.0015,0,.0015),inner=.18,outer=1.08)
    for sg in (-1,1): shift(p,[sg*.034,.004,.003],[.036,.034,.040],[0,-.0002,-.0007],.18,1.12)

    v[head]=p
    # Move detached eyeballs and oral components coherently with the final face.
    lmnew=v[K]; eyes=sorted([q for q in cs if 650<len(q)<900],key=lambda q:v0[q].mean(0)[0])
    for ids,ring,oldring in zip(eyes,(lmnew[36:42],lmnew[42:48]),(v0[K][36:42],v0[K][42:48])):
        v[ids]+=ring.mean(0)-oldring.mean(0)
    d=v[K][48:60].mean(0)-v0[K][48:60].mean(0)
    for ids in cs:
        if np.array_equal(ids,head) or any(np.array_equal(ids,e) for e in eyes): continue
        v[ids]+=d

    out=trimesh.Trimesh(vertices=v,faces=f,process=False)
    for ext in ('obj','glb','ply'): out.export(a.out/f'AINA_FACEVERSE_FULL_v15.5_CLEAN_IDENTITY.{ext}')
    keep=hm.copy(); [keep.__setitem__(e,True) for e in eyes]; fid=np.flatnonzero(keep[f].all(1)); clay=out.submesh([fid],append=True,repair=False)
    for ext in ('obj','glb','ply'): clay.export(a.out/f'AINA_FACEVERSE_IDENTITY_CLAY_v15.5.{ext}')

    metrics,checks,locked=health_and_identity_metrics(v0,v,f,hm)
    report={
        'version':'AINA Face Master v15.5 Final Identity Lock',
        'base':str(a.base),'topology_changed':False,'identity_lock':locked,'candidate':not locked,'no_new_face_version':True,
        'identity_method':'verified smooth v12.5 + restrained art-directed semantic sculpt + final visual polish; no sparse-landmark hard pull',
        'qa_gate':'actual naked-clay front + shallow 3Q + correct profile vs approved AINA references, plus mesh-health and bounded feature geometry',
        'visual_review_target':'approved AINA effect-art; reference is never regenerated or replaced',
        'next_stage':'final VRM assembly' if locked else 'identity sculpt blocked by gate',
        'max_head_delta_m':float(np.linalg.norm(p-p_start,axis=1).max()),
        'rms_head_delta_m':float(np.sqrt(np.mean(np.sum((p-p_start)**2,axis=1)))),
        'metrics':metrics,'checks':checks,
    }
    (a.out/'AINA_FACEVERSE_v15.5_REPORT.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
    if not locked: raise SystemExit('AINA v15.5 identity gate failed; final VRM assembly is blocked')

if __name__=='__main__': main()
