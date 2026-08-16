#!/usr/bin/env python3
"""AINA Face Master v15.5 — final identity lock sculpt.

This intentionally keeps the v15.5 identity line instead of creating more face
version numbers.  It starts from the verified smooth v12.5 FaceVerse topology,
applies the clean v15.5 sculpt, then a final art-directed correction pass that
was visually reviewed against the approved front / shallow 3Q / side AINA art.
The report only sets identity_lock=true when topology/mesh-health and bounded
feature-proportion gates all pass.
"""
import argparse, json
from pathlib import Path
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components

K=np.array([1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,13695,12973,13422,6543,6537],dtype=np.int64)

def comps(nv,f):
    e=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    a=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(nv,nv))
    a=(a+a.T).tocsr(); n,lab=connected_components(a,directed=False)
    return [np.flatnonzero(lab==i) for i in range(n)]

def ell(p,c,r,inner=.45,outer=1.45):
    c=np.asarray(c); r=np.asarray(r)
    q=np.sqrt(np.sum(((p-c)/r)**2,axis=1))
    w=np.zeros(len(p)); w[q<=inner]=1
    m=(q>inner)&(q<outer)
    if np.any(m):
        t=(q[m]-inner)/(outer-inner); w[m]=.5*(1+np.cos(np.pi*t))
    return w

def region_affine(p,c,r,s=(1,1,1),shift=(0,0,0),inner=.45,outer=1.45):
    w=ell(p,c,r,inner,outer)[:,None]
    c=np.asarray(c); s=np.asarray(s); sh=np.asarray(shift)
    tgt=c+(p-c)*s+sh
    p += w*(tgt-p)

def local_shift(p,c,r,shift,inner=.20,outer=1.25):
    p += ell(p,c,r,inner,outer)[:,None]*np.asarray(shift)

def rbf(p,ctrl,disp,sigma=.006,zsigma=.012,strength=.5):
    ctrl=np.asarray(ctrl); disp=np.asarray(disp)
    s2=sigma*sigma; zs2=zsigma*zsigma
    for st in range(0,len(p),4096):
        pp=p[st:st+4096]
        dx=pp[:,None,0]-ctrl[None,:,0]; dy=pp[:,None,1]-ctrl[None,:,1]; dz=pp[:,None,2]-ctrl[None,:,2]
        w=np.exp(-(dx*dx+dy*dy)/(2*s2)-dz*dz/(2*zs2)); sw=w.sum(1)
        val=(w@disp)/(sw[:,None]+1e-12); env=1-np.exp(-1.15*sw)
        p[st:st+len(pp),:2]+=strength*val*env[:,None]

def metric_gate(lm, v0, v, f, head, hm):
    eye_h=(np.ptp(lm[36:42,1])+np.ptp(lm[42:48,1]))*.5
    metrics={
        'jaw_width_m':float(np.ptp(lm[:17,0])),
        'face_height_eye_to_chin_m':float(lm[8,1]-np.mean(lm[36:48,1])),
        'eye_height_mean_m':float(eye_h),
        'nose_width_m':float(np.ptp(lm[31:36,0])),
        'nose_length_m':float(lm[33,1]-lm[27,1]),
        'nose_projection_m':float(lm[27,2]-lm[30,2]),
        'mouth_width_m':float(abs(lm[54,0]-lm[48,0])),
        'mouth_height_m':float(np.ptp(lm[48:60,1])),
        'lip_to_chin_m':float(lm[8,1]-lm[57,1]),
    }
    pairs=[(i,16-i) for i in range(8)] + [(17,26),(18,25),(19,24),(20,23),(21,22)] \
          + [(36,45),(37,44),(38,43),(39,42),(40,47),(41,46)] \
          + [(31,35),(32,34)] + [(48,54),(49,53),(50,52),(55,59),(56,58),(60,64),(61,63),(65,67)]
    errs=[]
    for a,b in pairs:
        errs.append(abs(lm[a,0]+lm[b,0])+abs(lm[a,1]-lm[b,1])+abs(lm[a,2]-lm[b,2]))
    metrics['symmetry_mean_m']=float(np.mean(errs)); metrics['symmetry_max_m']=float(np.max(errs))
    hf=f[hm[f].all(1)]
    tri0=v0[hf]; tri1=v[hf]
    c0=np.cross(tri0[:,1]-tri0[:,0],tri0[:,2]-tri0[:,0]); c1=np.cross(tri1[:,1]-tri1[:,0],tri1[:,2]-tri1[:,0])
    a0=.5*np.linalg.norm(c0,axis=1); a1=.5*np.linalg.norm(c1,axis=1)
    ar=a1/np.maximum(a0,1e-12)
    metrics['triangle_area_ratio_min']=float(np.min(ar)); metrics['triangle_area_ratio_p01']=float(np.percentile(ar,1)); metrics['triangle_area_ratio_p05']=float(np.percentile(ar,5)); metrics['triangle_area_ratio_p99']=float(np.percentile(ar,99)); metrics['triangle_normal_flip_count']=int(np.sum(np.sum(c0*c1,axis=1)<0))
    checks={
        'jaw_width': .125 <= metrics['jaw_width_m'] <= .142,
        'face_height': .072 <= metrics['face_height_eye_to_chin_m'] <= .086,
        'eye_height': .0105 <= metrics['eye_height_mean_m'] <= .0165,
        'nose_width': .012 <= metrics['nose_width_m'] <= .019,
        'nose_length': .028 <= metrics['nose_length_m'] <= .038,
        'nose_projection': .0045 <= metrics['nose_projection_m'] <= .0090,
        'mouth_width': .036 <= metrics['mouth_width_m'] <= .047,
        'mouth_height': .008 <= metrics['mouth_height_m'] <= .014,
        'lip_to_chin': .016 <= metrics['lip_to_chin_m'] <= .026,
        'symmetry': metrics['symmetry_mean_m'] <= .0030 and metrics['symmetry_max_m'] <= .0050,
        'mesh_area': metrics['triangle_area_ratio_min'] >= .15 and metrics['triangle_area_ratio_p05'] >= .45 and metrics['triangle_area_ratio_p99'] <= 1.85 and metrics['triangle_normal_flip_count'] == 0,
    }
    return metrics, checks, bool(all(checks.values()))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    m=trimesh.load(a.base,process=False,maintain_order=True)
    v0=np.asarray(m.vertices,float); f=np.asarray(m.faces,np.int64); v=v0.copy()
    cs=comps(len(v),f); head=max(cs,key=len); hm=np.zeros(len(v),bool); hm[head]=True
    g={int(q):i for i,q in enumerate(head)}; kl=np.array([g[int(q)] for q in K])
    p=v[head].copy(); p0=p.copy(); lm=p[kl].copy()

    y=p[:,1]; x=np.abs(p[:,0]); front=1/(1+np.exp((p[:,2]-.030)/.008)); side=1/(1+np.exp((x-.082)/.006))
    wmid=np.clip((y+.010)/.055,0,1)*front*side; wlow=np.clip((y-.020)/.050,0,1)*front*side
    p[:,0]*=(1-.045*wmid)*(1-.090*wlow)
    lm=p[kl].copy(); chin=lm[8]
    region_affine(p,chin,[.040,.035,.045],s=(.91,.84,1.0),shift=(0,-.0023,.0008),inner=.35,outer=1.45)
    lm=p[kl].copy()
    for ids,outerid in ((np.arange(36,42),36),(np.arange(42,48),45)):
        c=lm[ids].mean(0); region_affine(p,c,[.031,.020,.024],s=(1.07,1.025,1.0),inner=.38,outer=1.30)
        lm2=p[kl].copy(); disp=np.zeros((1,2)); disp[0,1]=.00075
        rbf(p,lm2[[outerid]],disp,sigma=.0045,zsigma=.008,strength=.65)
    lm=p[kl].copy(); bc=lm[27:31].mean(0)
    region_affine(p,bc,[.020,.036,.025],s=(.90,.98,.88),shift=(0,0,.0010),inner=.4,outer=1.35)
    lm=p[kl].copy(); nc=lm[30:36].mean(0)
    region_affine(p,nc,[.026,.025,.030],s=(.82,.90,.78),shift=(0,-.0018,.0025),inner=.35,outer=1.35)
    lm=p[kl].copy(); tip=lm[30]
    region_affine(p,tip,[.015,.017,.018],s=(.90,.90,.84),shift=(0,-.0008,.0012),inner=.35,outer=1.25)
    lm=p[kl].copy(); mc=lm[48:60].mean(0)
    region_affine(p,mc,[.038,.024,.030],s=(1.13,.84,.91),shift=(0,-.00035,.0007),inner=.35,outer=1.38)
    for sg in (-1,1):
        c=np.array([sg*.035,.010,.005]); region_affine(p,c,[.040,.038,.042],s=(.975,.985,.96),shift=(sg*.00035,0,-.00115),inner=.3,outer=1.35)
    lm=p[kl].copy()
    for ids in (np.arange(17,22),np.arange(22,27)):
        bc=lm[ids].mean(0); p[:,2]+=.0021*ell(p,bc,[.030,.022,.025],.38,1.28)
    br=lm[27:30].mean(0); p[:,2]+=.00125*ell(p,br,[.018,.030,.021],.40,1.28)
    lm=p[kl].copy()
    for ids in (np.arange(36,42),np.arange(42,48)):
        ec=lm[ids].mean(0); wo=ell(p,ec,[.030,.024,.029],.38,1.30); wi=ell(p,ec,[.018,.011,.017],.62,1.12)
        p[:,2]-=.00115*wo*(1-.72*wi)
    hf=f[hm[f].all(1)]; gl=-np.ones(len(v),int); gl[head]=np.arange(len(head)); lf=gl[hf]
    rows=np.concatenate([lf[:,0],lf[:,1],lf[:,2],lf[:,1],lf[:,2],lf[:,0]])
    cols=np.concatenate([lf[:,1],lf[:,2],lf[:,0],lf[:,0],lf[:,1],lf[:,2]])
    A=sparse.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(head),len(head))).tocsr(); deg=np.asarray(A.sum(1)).ravel()
    z=p[:,2].copy(); lm=p[kl].copy(); facec=np.array([0,.005,lm[30,2]+.010]); rel=ell(p,facec,[.073,.085,.050],.2,1.0)
    for _ in range(3):
        av=(A@z)/np.maximum(deg,1); z=z+.14*rel*(av-z)
    p[:,2]=z

    lm=p[kl].copy(); front=1/(1+np.exp((p[:,2]-.035)/.009))
    p[:,0] *= 1-.035*front
    eye_y=float(np.mean(lm[36:48,1])); w=np.clip((p[:,1]-eye_y)/.090,0,1)*front
    p[:,1]=eye_y+(p[:,1]-eye_y)*(1-.065*w)
    low=np.clip((p[:,1]-0.002)/.065,0,1)*front; p[:,0]*=1-.11*low
    lm=p[kl].copy(); chin=lm[8]
    region_affine(p,chin,[.038,.030,.040],s=(.80,.84,.96),shift=(0,-.0028,-.0011),inner=.30,outer=1.35)
    for sg in (-1,1):
        region_affine(p,[sg*.057,-.020,.020],[.035,.055,.050],s=(.94,1,1),shift=(-sg*.0012,0,0),inner=.25,outer=1.25)
        region_affine(p,[sg*.078,-.002,.028],[.025,.040,.040],s=(.82,.88,.88),shift=(-sg*.0025,0,.0012),inner=.25,outer=1.20)
    lm=p[kl].copy()
    for ids in (np.arange(17,22),np.arange(22,27)):
        local_shift(p,lm[ids].mean(0),[.032,.024,.026],[0,0,.0032],.28,1.30)
    lm=p[kl].copy()
    for ids in (np.arange(36,42),np.arange(42,48)):
        region_affine(p,lm[ids].mean(0),[.030,.018,.022],s=(1.035,1.035,1.0),inner=.30,outer=1.18)
    lm=p[kl].copy()
    for i in [37,38,43,44]: local_shift(p,lm[i],[.007,.006,.010],[0,-.00055,-.00015],.12,1.15)
    for i in [40,41,46,47]: local_shift(p,lm[i],[.007,.006,.010],[0,.00050,-.00005],.12,1.15)
    for i in [36,45]:
        sg=np.sign(lm[i,0]); local_shift(p,lm[i],[.006,.007,.010],[sg*.0009,-.00010,0],.12,1.15)
    lm=p[kl].copy(); bridge=lm[27:31].mean(0)
    region_affine(p,bridge,[.018,.034,.027],s=(.86,.98,.90),shift=(0,-.0004,.0008),inner=.35,outer=1.25)
    lm=p[kl].copy(); nose=lm[30:36].mean(0)
    region_affine(p,nose,[.024,.023,.027],s=(.80,.86,.82),shift=(0,-.0016,.0018),inner=.30,outer=1.28)
    lm=p[kl].copy(); tip=lm[30]
    region_affine(p,tip,[.014,.015,.017],s=(.90,.86,.92),shift=(0,-.0015,-.0004),inner=.28,outer=1.18)
    for sg in (-1,1):
        local_shift(p,[sg*.033,.002,.002],[.032,.030,.034],[-sg*.0002,-.00025,-.0008],.20,1.20)
    lm=p[kl].copy(); mc=lm[48:60].mean(0)
    region_affine(p,mc,[.036,.021,.027],s=(1.13,.76,.90),shift=(0,-.0010,.0010),inner=.30,outer=1.30)
    lm=p[kl].copy(); phil=(lm[33]+lm[51])/2
    region_affine(p,phil,[.023,.026,.026],s=(1,.92,.96),shift=(0,-.0005,.0003),inner=.20,outer=1.20)
    lm=p[kl].copy(); chin=lm[8]
    local_shift(p,chin,[.028,.022,.028],[0,-.0006,-.0012],.20,1.15)

    v[head]=p
    lm=v[K]
    eyes=sorted([q for q in cs if 650<len(q)<900],key=lambda q:v0[q].mean(0)[0])
    for ids,el in zip(eyes,(lm[36:42],lm[42:48])):
        c=v[ids].mean(0); rim=el.mean(0); tc=np.array([rim[0],rim[1],rim[2]+.0070]); v[ids]+=tc-c
    old=v0[K]; new=v[K]; mouth_shift=new[48:60].mean(0)-old[48:60].mean(0)
    for ids in cs:
        if np.array_equal(ids,head) or any(np.array_equal(ids,e) for e in eyes): continue
        v[ids]+=mouth_shift

    out=trimesh.Trimesh(vertices=v,faces=f,process=False)
    for ext in ('obj','glb','ply'): out.export(a.out/f'AINA_FACEVERSE_FULL_v15.5_CLEAN_IDENTITY.{ext}')
    keep=hm.copy(); [keep.__setitem__(e,True) for e in eyes]
    fid=np.flatnonzero(keep[f].all(1)); clay=out.submesh([fid],append=True,repair=False)
    for ext in ('obj','glb','ply'): clay.export(a.out/f'AINA_FACEVERSE_IDENTITY_CLAY_v15.5.{ext}')

    metrics,checks,gate_pass=metric_gate(v[K],v0,v,f,head,hm)
    rep={
        'version':'AINA Face Master v15.5 Final Identity Lock',
        'base':str(a.base),
        'topology_changed':False,
        'identity_lock':gate_pass,
        'candidate':not gate_pass,
        'qa_gate':'actual naked-clay front + calibrated shallow 3Q (-15/-20/-25) + correctly oriented left-facing profile (+90) vs approved AINA references, plus bounded geometry/mesh-health gate',
        'visual_review_target':'approved AINA effect-art identity; no generated replacement reference',
        'no_new_face_version':True,
        'next_stage':'final VRM assembly' if gate_pass else 'identity sculpt blocked by gate',
        'max_head_delta_m':float(np.linalg.norm(p-p0,axis=1).max()),
        'rms_head_delta_m':float(np.sqrt(np.mean(np.sum((p-p0)**2,axis=1)))),
        'metrics':metrics,
        'checks':checks,
    }
    (a.out/'AINA_FACEVERSE_v15.5_REPORT.json').write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
    if not gate_pass:
        raise SystemExit('AINA v15.5 identity gate failed; VRM assembly must not start')

if __name__=='__main__': main()
