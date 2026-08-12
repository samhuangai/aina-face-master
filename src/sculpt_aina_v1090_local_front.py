#!/usr/bin/env python3
"""AINA v10.9.0 — bounded local front-reference sculpt.

Design goal: visual likeness without global mesh instability.

The v10.7/v10.8 experiments showed that sparse multi-view 3D triangulation can
lower landmark RMSE while destroying anatomy. v10.9 deliberately avoids any
global deformation solve. It starts from a deterministic FEMALE/ASIAN GNM
identity, fits one stable frontal camera, and transfers only *local* 2D
landmark residuals into five bounded facial regions (jaw, brows, eyes, nose,
mouth). Profile depth is then adjusted with millimetre-scale art-directed masks.

Original Google GNM topology and component ordering are preserved.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import face_alignment
import numpy as np
import trimesh
from PIL import Image

import fit_aina_v101 as core
from gnm.shape import gnm_numpy, gnm_landmarks
from gnm.shape.semantic_sampler import IdentitySampler, Gender, Ethnicity

GNM_TO_STANDARD = np.array([
    0,1,6,5,4,3,2,7,8,9,10,11,12,13,14,15,16,
    *range(17,68)
], dtype=np.int64)

RNG_SEED = 20260812
BASE_SAMPLE_A = 264
BASE_SAMPLE_B = 290
BASE_BLEND_A = 0.65
BASE_BLEND_B = 0.35


def smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3.0-2.0*x)


def gaussian_ellipse(p: np.ndarray, cx: float, cy: float, rx: float, ry: float, power: float=2.0) -> np.ndarray:
    q=((p[:,0]-cx)/max(rx,1e-9))**2+((p[:,1]-cy)/max(ry,1e-9))**2
    return np.exp(-0.5*np.power(q,power/2.0))


def build_adjacency(n: int, faces: np.ndarray) -> list[np.ndarray]:
    nbr=[set() for _ in range(n)]
    for a,b,c in faces:
        a=int(a);b=int(b);c=int(c)
        nbr[a].update((b,c)); nbr[b].update((a,c)); nbr[c].update((a,b))
    return [np.fromiter(x,dtype=np.int64) if x else np.empty(0,np.int64) for x in nbr]


def gentle_smooth(raw: np.ndarray, faces: np.ndarray, alpha: float=0.035) -> np.ndarray:
    nbr=build_adjacency(len(raw),faces)
    avg=np.zeros_like(raw)
    for i,ids in enumerate(nbr):
        avg[i]=raw[ids].mean(axis=0) if len(ids) else raw[i]
    sm=(1.0-alpha)*raw+alpha*avg
    return 0.96*raw+0.04*sm


def clamp_vectors(d: np.ndarray, cap: float) -> np.ndarray:
    out=d.copy(); n=np.linalg.norm(out,axis=1); over=n>cap
    if np.any(over): out[over]*=(cap/n[over])[:,None]
    return out


def semantic_identity() -> np.ndarray:
    sampler=IdentitySampler()
    rng=np.random.default_rng(RNG_SEED)
    identities=np.asarray(sampler.sample_identity(Gender.FEMALE,Ethnicity.ASIAN,num_samples=384,rng=rng),dtype=np.float64)
    return BASE_BLEND_A*identities[BASE_SAMPLE_A]+BASE_BLEND_B*identities[BASE_SAMPLE_B]


def sparse_controls(vertices: np.ndarray, idx_std: np.ndarray, bw_std: np.ndarray) -> np.ndarray:
    return (vertices[idx_std]*bw_std[...,None]).sum(axis=-2)


def front_weights() -> np.ndarray:
    w=np.ones(68,np.float64)
    w[:17]=2.6; w[17:27]=1.0; w[27:36]=3.0; w[36:48]=4.3; w[48:60]=3.7; w[60:68]=2.2
    return w


def local_residual_field(p: np.ndarray, controls_p: np.ndarray, residual_xy: np.ndarray, ids: np.ndarray,
                         sigma: float, support: float, cap: float, strength: float=1.0) -> np.ndarray:
    """Transfer landmark camera-plane residuals only to nearby skin vertices."""
    num=np.zeros((len(p),2),np.float64); den=np.zeros(len(p),np.float64)
    for li in ids:
        diff=p[:,:2]-controls_p[li,:2]
        dist=np.linalg.norm(diff,axis=1)
        m=dist<support
        if not np.any(m): continue
        z=dist[m]/max(sigma,1e-9)
        w=np.exp(-0.5*z*z)
        num[m]+=w[:,None]*residual_xy[li]
        den[m]+=w
    active=den>1e-7
    out=np.zeros((len(p),3),np.float64)
    out[active,:2]=strength*num[active]/den[active,None]
    # Fade at low aggregate support so a lone distant landmark never drags skin.
    conf=np.clip(den/1.5,0.0,1.0)
    out[:,:2]*=conf[:,None]
    return clamp_vectors(out,cap)


def detect_targets(args):
    refs={
      'front':core.load_image_rgb(args.front),
      'three_quarter':core.load_image_rgb(args.three_quarter),
      'side':core.load_image_rgb(args.side),
    }
    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cpu',face_detector='sfd')
    front_json=json.loads(args.front_landmarks.read_text())
    target_px={
      'front':np.asarray(front_json['landmarks_xy'],dtype=np.float32),
      'three_quarter':core.detect_68(fa,refs['three_quarter']),
      'side':core.detect_68(fa,refs['side']),
    }
    target={name:core.normalize_target(target_px[name],refs[name].shape) for name in core.VIEW_ORDER}
    return refs,target_px,target


def region_centers(cp: np.ndarray) -> dict:
    eye_l=cp[36:42,:2].mean(axis=0); eye_r=cp[42:48,:2].mean(axis=0)
    mouth=cp[48:60,:2].mean(axis=0); nose=cp[27:36,:2].mean(axis=0)
    chin=cp[8,:2]
    eye_width=0.5*(np.linalg.norm(cp[36,:2]-cp[39,:2])+np.linalg.norm(cp[42,:2]-cp[45,:2]))
    mouth_width=np.linalg.norm(cp[48,:2]-cp[54,:2])
    nose_width=np.linalg.norm(cp[31,:2]-cp[35,:2])
    return {'eye_l':eye_l,'eye_r':eye_r,'mouth':mouth,'nose':nose,'chin':chin,
            'eye_width':float(eye_width),'mouth_width':float(mouth_width),'nose_width':float(nose_width)}


def depth_polish(p: np.ndarray, cp: np.ndarray) -> tuple[np.ndarray,dict]:
    c=region_centers(cp); raw=np.zeros_like(p)
    cx=float(cp[27:36,0].mean())
    # Camera looks toward +Z; negative Z is toward camera. AINA needs a smaller,
    # less projecting nose but soft forward cheeks/lips.
    nose_c=c['nose']; nw=max(c['nose_width'],0.010)
    nose=gaussian_ellipse(p,float(nose_c[0]),float(nose_c[1]),1.75*nw,0.040,2.2)
    tip=gaussian_ellipse(p,float(cp[33,0]),float(cp[33,1]),0.014,0.014,2.3)
    raw[:,2]+=0.0021*nose+0.0007*tip

    # High youthful apple-cheek volume, kept local and symmetric around the eye-mouth band.
    cheek_y=0.58*float((c['eye_l'][1]+c['eye_r'][1])*0.5)+0.42*float(c['mouth'][1])
    for ex in (float(c['eye_l'][0]),float(c['eye_r'][0])):
        cheek=gaussian_ellipse(p,ex,cheek_y,0.030,0.029,2.2)
        raw[:,2]-=0.00125*cheek
        # slight lateral softening; never a global face squeeze
        raw[:,0]+=((cx+(p[:,0]-cx)*0.985)-p[:,0])*cheek

    mouth_c=c['mouth']; mw=max(c['mouth_width'],0.020)
    lips=gaussian_ellipse(p,float(mouth_c[0]),float(mouth_c[1]),0.72*mw,0.016,2.3)
    lower=gaussian_ellipse(p,float(mouth_c[0]),float(mouth_c[1])+0.004,0.60*mw,0.010,2.4)
    raw[:,2]-=0.00055*lips+0.00065*lower

    # Softer, slightly shorter chin projection.
    chin=gaussian_ellipse(p,float(c['chin'][0]),float(c['chin'][1]),0.036,0.030,2.2)
    raw[:,2]-=0.00045*chin
    raw[:,1]-=0.00075*chin

    return clamp_vectors(raw,0.0028), {'centers':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()}}


def render_reference_comparison(ref_path: Path, actual_path: Path, out: Path) -> None:
    ref=Image.open(ref_path).convert('RGB'); act=Image.open(actual_path).convert('RGB')
    H=max(ref.height,act.height)
    rw=int(round(ref.width*H/ref.height)); aw=int(round(act.width*H/act.height))
    sheet=Image.new('RGB',(rw+aw,H),'white')
    sheet.paste(ref.resize((rw,H)),(0,0)); sheet.paste(act.resize((aw,H)),(rw,0)); sheet.save(out)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--front',type=Path,required=True)
    ap.add_argument('--three-quarter',type=Path,required=True)
    ap.add_argument('--side',type=Path,required=True)
    ap.add_argument('--front-landmarks',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('output_v1090'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); qa=args.out/'QA'; qa.mkdir(exist_ok=True)

    refs,target_px,target=detect_targets(args)
    g=gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,variant=gnm_numpy.GNMVariant.HEAD)
    identity=semantic_identity()
    full_v=np.asarray(g(identity=identity[None,:]))[0].astype(np.float64)
    full_f=np.asarray(g.triangles,dtype=np.int64)

    cfg=gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
    idx=np.asarray(cfg.indices,dtype=np.int64)[GNM_TO_STANDARD]
    bw=np.asarray(cfg.weights,dtype=np.float64)[GNM_TO_STANDARD]
    controls=sparse_controls(full_v,idx,bw)

    skin_ti=np.asarray(g.triangle_indices_for_group('skin'),dtype=np.int64)
    skin_faces_global=full_f[skin_ti]
    skin_ids=np.unique(skin_faces_global)
    n_skin=len(skin_ids)
    if not np.array_equal(skin_ids,np.arange(n_skin,dtype=np.int64)):
        raise RuntimeError('GNM skin vertices are not a contiguous prefix; safe local sculpt aborted')
    skin_v=full_v[:n_skin].copy(); skin_f=skin_faces_global.copy()

    # One stable front camera only. It is never refit while the local residuals
    # are transferred, preventing camera/shape feedback loops.
    R,scale,trans=core.scaled_ortho_init(controls,target['front'],front_weights())
    p=skin_v@R.T; cp=controls@R.T
    desired_xy=(target['front']-trans)/scale
    residual_xy=desired_xy-cp[:,:2]

    # Clip raw landmark wishes before spatial propagation. These limits are in
    # metres in GNM model space and are intentionally small.
    per_lm=np.linalg.norm(residual_xy,axis=1)
    lm_cap=np.full(68,0.0030,np.float64)
    lm_cap[:17]=0.0040; lm_cap[17:27]=0.0022; lm_cap[27:36]=0.0030; lm_cap[36:48]=0.0032; lm_cap[48:68]=0.0030
    s=np.minimum(1.0,lm_cap/np.maximum(per_lm,1e-12)); residual_xy*=s[:,None]

    region_parts={}
    region_parts['jaw']=local_residual_field(p,cp,residual_xy,np.arange(0,17),sigma=0.014,support=0.034,cap=0.0035,strength=0.92)
    region_parts['brows']=local_residual_field(p,cp,residual_xy,np.arange(17,27),sigma=0.0085,support=0.021,cap=0.0018,strength=0.70)
    region_parts['eyes']=local_residual_field(p,cp,residual_xy,np.arange(36,48),sigma=0.0058,support=0.0155,cap=0.0027,strength=1.00)
    region_parts['nose']=local_residual_field(p,cp,residual_xy,np.arange(27,36),sigma=0.0065,support=0.017,cap=0.0024,strength=0.92)
    region_parts['mouth']=local_residual_field(p,cp,residual_xy,np.arange(48,68),sigma=0.0065,support=0.018,cap=0.0025,strength=0.95)

    raw=sum(region_parts.values(),np.zeros_like(p))
    depth,depth_stats=depth_polish(p,cp); raw+=depth
    raw=clamp_vectors(raw,0.0046)
    d=gentle_smooth(raw,skin_f,alpha=0.035)
    d=clamp_vectors(d,0.0048)
    p2=p+d; skin_v2=p2@R

    full_v2=full_v.copy(); full_v2[:n_skin]=skin_v2
    if not np.isfinite(full_v2).all(): raise RuntimeError('Non-finite vertices after local sculpt')
    max_shift=float(np.max(np.linalg.norm(full_v2-full_v,axis=1)))
    if max_shift>0.0050: raise RuntimeError(f'v10.9 safety cap violated: {max_shift:.6f} m')

    full=trimesh.Trimesh(vertices=full_v2,faces=full_f,process=False)
    skin=trimesh.Trimesh(vertices=skin_v2,faces=skin_f,process=False)
    full.export(args.out/'AINA_FACE_MASTER_GNM_v10.9.0_LOCAL_REFERENCE.obj')
    full.export(args.out/'AINA_FACE_MASTER_GNM_v10.9.0_LOCAL_REFERENCE.glb')
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.9.0_LOCAL_REFERENCE.obj')
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.9.0_LOCAL_REFERENCE.glb')
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.9.0_LOCAL_REFERENCE.ply')
    np.save(args.out/'AINA_FEMALE_BASE_IDENTITY_v10.9.0.npy',identity.astype(np.float32))

    # QA: fixed-front-camera error tells us whether local sculpt actually moved
    # toward the approved art. Other views get independent QA cameras only.
    final_controls=sparse_controls(full_v2,idx,bw)
    fixed_pred=core.project_np(final_controls,R,scale,trans)
    fixed_e=np.linalg.norm(fixed_pred-target['front'],axis=1)
    errors={'front_fixed_camera':{'rmse':float(np.sqrt(np.mean(fixed_e**2))),'median':float(np.median(fixed_e)),'p90':float(np.percentile(fixed_e,90))}}
    core.save_overlay(refs['front'],target_px['front'],fixed_pred,qa/'AINA_front_overlay_v10.9.0.png','AINA v10.9.0 front local sculpt')
    cameras={'front':(R,scale,trans)}
    for name in ('three_quarter','side'):
        cam=core.scaled_ortho_init(final_controls,target[name],np.ones(68,np.float64)); cameras[name]=cam
        pred=core.project_np(final_controls,*cam); e=np.linalg.norm(pred-target[name],axis=1)
        errors[name]={'rmse':float(np.sqrt(np.mean(e**2))),'median':float(np.median(e)),'p90':float(np.percentile(e,90))}
        core.save_overlay(refs[name],target_px[name],pred,qa/f'AINA_{name}_overlay_v10.9.0.png',f'AINA v10.9.0 {name}')

    paths=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        path=qa/f'AINA_FULL_CLAY_{label}_v10.9.0.png'
        core.render_mesh_ortho(full_v2,full_f,R,yaw,path,f'AINA v10.9.0 {label}'); paths.append(path)
    ims=[Image.open(x).convert('RGB') for x in paths]; H=max(x.height for x in ims); W=max(x.width for x in ims)
    sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims): sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_FULL_CLAY_5VIEW_v10.9.0.png')
    render_reference_comparison(args.front,qa/'AINA_FULL_CLAY_front_v10.9.0.png',qa/'AINA_REFERENCE_VS_ACTUAL_FRONT_v10.9.0.png')

    stats={name:float(np.max(np.linalg.norm(part,axis=1))) for name,part in region_parts.items()}
    report={
      'version':'AINA Face Master v10.9.0 Bounded Local Front-Reference Sculpt',
      'base':{'semantic':'FEMALE / ASIAN','rng_seed':RNG_SEED,'sample_indices':[BASE_SAMPLE_A,BASE_SAMPLE_B],'blend_weights':[BASE_BLEND_A,BASE_BLEND_B]},
      'topology_changed':False,'global_solver_used':False,'full_vertices':int(len(full_v2)),'skin_vertices':int(n_skin),
      'max_vertex_shift_m':max_shift,'region_max_requested_shift_m':stats,'depth_polish':depth_stats,
      'errors':errors,'identity_lock':False,
      'acceptance_note':'Identity stays unlocked until the actual front/45/profile clay visually matches the approved effect-art face; landmark RMSE alone is not acceptance.'
    }
    (args.out/'AINA_v10.9.0_REPORT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
