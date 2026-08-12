#!/usr/bin/env python3
"""AINA v11.2 — effect-art TPS control-cage sculpt.

This stage intentionally stops asking a statistical face identity space to
produce AINA. A stable semantic GNM head only supplies production topology.
The approved front reference defines the actual facial proportions: corrected
68-point controls drive a thin-plate-spline deformation in the frontal camera
plane, with smooth spatial/depth gating so scalp, ears, rear skull and neck stay
stable. Depth remains a separate millimetre-scale art-directed pass.
"""
from __future__ import annotations

import argparse,json,math
from pathlib import Path
import face_alignment
import numpy as np
import trimesh
from scipy.interpolate import RBFInterpolator
from PIL import Image

import fit_aina_v101 as core
from gnm.shape import gnm_numpy,gnm_landmarks
from gnm.shape.semantic_sampler import IdentitySampler,Gender,Ethnicity

GNM_TO_STANDARD=np.array([0,1,6,5,4,3,2,7,8,9,10,11,12,13,14,15,16,*range(17,68)],dtype=np.int64)
RNG_SEED=20260812; BASE_A=264;BASE_B=290;WA=.65;WB=.35


def smoothstep01(x):
    x=np.clip(x,0.,1.);return x*x*(3.-2.*x)


def gauss(p,cx,cy,rx,ry,pow=2.0):
    q=((p[:,0]-cx)/max(rx,1e-9))**2+((p[:,1]-cy)/max(ry,1e-9))**2
    return np.exp(-.5*np.power(q,pow/2.))


def clampv(d,cap):
    o=d.copy();n=np.linalg.norm(o,axis=1);m=n>cap
    if np.any(m):o[m]*=(cap/n[m])[:,None]
    return o


def identity():
    sampler=IdentitySampler();rng=np.random.default_rng(RNG_SEED)
    ids=np.asarray(sampler.sample_identity(Gender.FEMALE,Ethnicity.ASIAN,num_samples=384,rng=rng),np.float64)
    return WA*ids[BASE_A]+WB*ids[BASE_B]


def controls(v,idx,bw):return (v[idx]*bw[...,None]).sum(-2)


def fw():
    w=np.ones(68);w[:17]=3.2;w[17:27]=1.2;w[27:36]=3.5;w[36:48]=5.;w[48:60]=4.2;w[60:68]=2.5;return w


def load_targets(args):
    refs={'front':core.load_image_rgb(args.front),'three_quarter':core.load_image_rgb(args.three_quarter),'side':core.load_image_rgb(args.side)}
    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cpu',face_detector='sfd')
    fj=json.loads(args.front_landmarks.read_text())
    px={'front':np.asarray(fj['landmarks_xy'],np.float32),'three_quarter':core.detect_68(fa,refs['three_quarter']),'side':core.detect_68(fa,refs['side'])}
    norm={k:core.normalize_target(px[k],refs[k].shape) for k in core.VIEW_ORDER}
    return refs,px,norm


def landmark_caps(res):
    cap=np.full(68,.010,np.float64)
    cap[:17]=.014;cap[17:27]=.007;cap[27:36]=.009;cap[36:48]=.010;cap[48:68]=.009
    mag=np.linalg.norm(res,axis=1);s=np.minimum(1.,cap/np.maximum(mag,1e-12));return res*s[:,None]


def face_gate(p,cp):
    # Semantic-point support union: 1 at every control, smooth decay away from
    # the actual face. Different sigmas allow broad jaw flow but crisp eyes.
    g=np.zeros(len(p),np.float64)
    sig=np.full(68,.020);sig[:17]=.031;sig[17:27]=.023;sig[27:36]=.020;sig[36:48]=.018;sig[48:68]=.020
    for i in range(68):
        d=np.linalg.norm(p[:,:2]-cp[i,:2],axis=1);w=np.exp(-.5*(d/sig[i])**4);g=np.maximum(g,w)
    zc=float(np.median(cp[:,2]));zg=np.exp(-.5*((p[:,2]-zc)/.085)**4)
    return np.clip(g*zg,0.,1.)


def depth_pass(p,cp):
    d=np.zeros_like(p)
    eye_y=float(.5*(cp[36:48,1].mean()+cp[17:27,1].mean()))
    nose_c=cp[27:36,:2].mean(0);mouth_c=cp[48:60,:2].mean(0);chin=cp[8,:2]
    nw=max(float(np.linalg.norm(cp[31,:2]-cp[35,:2])),.012);mw=max(float(np.linalg.norm(cp[48,:2]-cp[54,:2])),.025)
    # smaller/less projecting nose
    nose=gauss(p,float(nose_c[0]),float(nose_c[1]),1.65*nw,.042,2.2);tip=gauss(p,float(cp[33,0]),float(cp[33,1]),.014,.014,2.4)
    d[:,2]+=.0028*nose+.0008*tip
    # young apple cheeks: local forward volume
    for ex in (float(cp[36:42,0].mean()),float(cp[42:48,0].mean())):
        cheek_y=.53*float(cp[36:48,1].mean())+.47*float(mouth_c[1]);cheek=gauss(p,ex,cheek_y,.031,.030,2.2);d[:,2]-=.00145*cheek
    # soft lip volume, lower lip slightly fuller
    lip=gauss(p,float(mouth_c[0]),float(mouth_c[1]),.72*mw,.017,2.3);low=gauss(p,float(mouth_c[0]),float(mouth_c[1])+.004,.58*mw,.010,2.4)
    d[:,2]-=.00065*lip+.00075*low
    # chin slightly forward but shorter in profile
    ch=gauss(p,float(chin[0]),float(chin[1]),.036,.030,2.2);d[:,2]-=.00045*ch;d[:,1]-=.0008*ch
    return clampv(d,.0035)


def triangle_area(v,f):
    t=v[f];return .5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)


def compare(ref,act,out):
    a=Image.open(ref).convert('RGB');b=Image.open(act).convert('RGB');H=max(a.height,b.height);aw=int(a.width*H/a.height);bw=int(b.width*H/b.height)
    s=Image.new('RGB',(aw+bw,H),'white');s.paste(a.resize((aw,H)),(0,0));s.paste(b.resize((bw,H)),(aw,0));s.save(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--front',type=Path,required=True);ap.add_argument('--three-quarter',type=Path,required=True);ap.add_argument('--side',type=Path,required=True);ap.add_argument('--front-landmarks',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_v1120'));args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);qa=args.out/'QA';qa.mkdir(exist_ok=True)
    refs,px,target=load_targets(args)
    g=gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,variant=gnm_numpy.GNMVariant.HEAD)
    ident=identity();full_v=np.asarray(g(identity=ident[None,:]))[0].astype(np.float64);full_f=np.asarray(g.triangles,np.int64)
    cfg=gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68);idx=np.asarray(cfg.indices,np.int64)[GNM_TO_STANDARD];bw=np.asarray(cfg.weights,np.float64)[GNM_TO_STANDARD]
    c0=controls(full_v,idx,bw)
    skin_ti=np.asarray(g.triangle_indices_for_group('skin'),np.int64);skin_f=full_f[skin_ti];skin_ids=np.unique(skin_f);n_skin=len(skin_ids)
    if not np.array_equal(skin_ids,np.arange(n_skin)):raise RuntimeError('skin not contiguous prefix')
    skin_v=full_v[:n_skin].copy()

    R,scale,trans=core.scaled_ortho_init(c0,target['front'],fw());p=skin_v@R.T;cp=c0@R.T
    desired=(target['front']-trans)/scale;res=landmark_caps(desired-cp[:,:2])
    # Tiny smoothing regularizer only; TPS remains driven by the approved art.
    tps=RBFInterpolator(cp[:,:2],res,kernel='thin_plate_spline',smoothing=2.5e-6,degree=1)
    xy=tps(p[:,:2]);gate=face_gate(p,cp);raw=np.zeros_like(p);raw[:,:2]=xy*gate[:,None]
    raw=clampv(raw,.0145);raw+=depth_pass(p,cp);raw=clampv(raw,.0150)
    p2=p+raw;skin_v2=p2@R;full_v2=full_v.copy();full_v2[:n_skin]=skin_v2
    if not np.isfinite(full_v2).all():raise RuntimeError('non-finite TPS output')

    a0=triangle_area(skin_v,skin_f);a1=triangle_area(skin_v2,skin_f);ratio=a1/np.maximum(a0,1e-12)
    q01=float(np.percentile(ratio,1));q99=float(np.percentile(ratio,99));maxshift=float(np.max(np.linalg.norm(skin_v2-skin_v,axis=1)))
    if q01<.08 or q99>8.0:raise RuntimeError(f'TPS mesh quality failed: area p01={q01:.4f}, p99={q99:.4f}')

    full=trimesh.Trimesh(vertices=full_v2,faces=full_f,process=False);skin=trimesh.Trimesh(vertices=skin_v2,faces=skin_f,process=False)
    full.export(args.out/'AINA_FACE_MASTER_GNM_v11.2_TPS.obj');full.export(args.out/'AINA_FACE_MASTER_GNM_v11.2_TPS.glb');skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v11.2_TPS.obj');skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v11.2_TPS.glb');skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v11.2_TPS.ply')

    cf=controls(full_v2,idx,bw);fixed=core.project_np(cf,R,scale,trans);e=np.linalg.norm(fixed-target['front'],axis=1)
    errors={'front_fixed':{'rmse':float(np.sqrt(np.mean(e**2))),'median':float(np.median(e)),'p90':float(np.percentile(e,90))}}
    core.save_overlay(refs['front'],px['front'],fixed,qa/'AINA_front_overlay_v11.2.png','AINA v11.2 TPS front')
    for name in ('three_quarter','side'):
        cam=core.scaled_ortho_init(cf,target[name],np.ones(68));pred=core.project_np(cf,*cam);ee=np.linalg.norm(pred-target[name],axis=1);errors[name]={'rmse':float(np.sqrt(np.mean(ee**2))),'median':float(np.median(ee)),'p90':float(np.percentile(ee,90))};core.save_overlay(refs[name],px[name],pred,qa/f'AINA_{name}_overlay_v11.2.png',f'AINA v11.2 {name}')

    views=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        path=qa/f'AINA_FULL_CLAY_{label}_v11.2.png';core.render_mesh_ortho(full_v2,full_f,R,yaw,path,f'AINA v11.2 {label}');views.append(path)
    ims=[Image.open(x).convert('RGB') for x in views];H=max(i.height for i in ims);W=max(i.width for i in ims);sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_FULL_CLAY_5VIEW_v11.2.png');compare(args.front,qa/'AINA_FULL_CLAY_front_v11.2.png',qa/'AINA_REFERENCE_VS_ACTUAL_FRONT_v11.2.png')

    report={'version':'AINA Face Master v11.2 TPS Reference Cage','base':'deterministic FEMALE/ASIAN semantic GNM topology','topology_changed':False,'global_identity_solver_used':False,'deformation':'front 68-point thin-plate spline + smooth face/depth gate + bounded profile depth polish','max_vertex_shift_m':maxshift,'triangle_area_ratio_p01':q01,'triangle_area_ratio_p99':q99,'raw_landmark_residual_max_m':float(np.max(np.linalg.norm(desired-cp[:,:2],axis=1))),'errors':errors,'identity_lock':False,'acceptance_note':'Visual clay comparison remains the acceptance gate.'}
    (args.out/'AINA_v11.2_REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
