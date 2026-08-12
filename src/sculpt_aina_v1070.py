#!/usr/bin/env python3
"""AINA v10.7.0 — nonlinear effect-art sculpt on top of corrected GNM identity.

The statistical identity space is deliberately bypassed after v10.6.5. Sparse
landmarks are triangulated back into 3D from the approved front / 3Q / profile
references and the residual deformation is propagated locally over the original
GNM topology. This keeps topology and UV compatibility while allowing the face
to leave the adult-average GNM identity manifold.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np
import trimesh
import face_alignment
from PIL import Image

import fit_aina_v101 as core
from gnm.shape import gnm_numpy, gnm_landmarks

GNM_TO_STANDARD = np.array([
    0,1,6,5,4,3,2,7,8,9,10,11,12,13,14,15,16,
    *range(17,68)
], dtype=np.int64)


def load_cameras(path: Path):
    raw=json.loads(path.read_text())
    out={}
    for name in core.VIEW_ORDER:
        d=raw[name]
        out[name]=(np.asarray(d['rotation_rows'],np.float64),float(d['scale']),np.asarray(d['translation'],np.float64))
    return out


def view_weights(name: str) -> np.ndarray:
    w=np.ones(68,np.float64)
    if name=='front':
        w[0:17]=4.4; w[17:27]=1.5; w[27:36]=4.1; w[36:48]=5.4; w[48:68]=4.8
        return 3.8*w
    if name=='three_quarter':
        w[0:17]=2.3; w[17:27]=0.8; w[27:36]=2.7; w[36:48]=2.8; w[48:68]=2.5
        return 1.45*w
    w[:]=0.10; w[0:17]=0.78; w[5:12]=1.45; w[27:36]=2.1; w[48:60]=0.85
    w[36:42]=0.30
    return 0.72*w


def triangulate_controls(current: np.ndarray, target: dict[str,np.ndarray], cameras) -> np.ndarray:
    desired=np.zeros_like(current)
    # Tiny prior: enough to reject inconsistent hidden-side detector points,
    # but weak enough for the approved effect art to dominate.
    prior_lambda=0.018
    for li in range(68):
        A=[]; b=[]
        for name in core.VIEW_ORDER:
            r,s,t=cameras[name]
            vw=float(view_weights(name)[li])
            if vw <= 1e-8: continue
            sw=math.sqrt(vw)
            for d in range(2):
                A.append(sw*s*r[d])
                b.append(sw*(target[name][li,d]-t[d]))
        for d in range(3):
            row=np.zeros(3); row[d]=math.sqrt(prior_lambda)
            A.append(row); b.append(math.sqrt(prior_lambda)*current[li,d])
        x,*_=np.linalg.lstsq(np.asarray(A),np.asarray(b),rcond=None)
        desired[li]=x
    return desired


def control_points(vertices: np.ndarray, idx: np.ndarray, bw: np.ndarray) -> np.ndarray:
    return (vertices[idx]*bw[...,None]).sum(axis=-2)[GNM_TO_STANDARD]


def landmark_sigmas(face_scale: float) -> np.ndarray:
    sig=np.full(68,0.050*face_scale,np.float64)
    sig[0:17]=0.090*face_scale       # jaw / cheeks need broad silhouette flow
    sig[17:27]=0.055*face_scale
    sig[27:36]=0.050*face_scale
    sig[36:48]=0.043*face_scale      # eyes stay local and crisp
    sig[48:60]=0.052*face_scale
    sig[60:68]=0.042*face_scale
    return sig


def propagate(vertices: np.ndarray, controls: np.ndarray, residual: np.ndarray, sigmas: np.ndarray, gain: float) -> np.ndarray:
    v=vertices.copy()
    num=np.zeros_like(v); den=np.zeros((len(v),1),np.float64)
    # Gaussian compact support. Limiting every landmark to 2.6 sigma keeps the
    # back of skull, ears and neck from being dragged by facial edits.
    for i in range(68):
        d=np.linalg.norm(v-controls[i],axis=1)
        mask=d < 2.6*sigmas[i]
        if not np.any(mask): continue
        z=d[mask]/max(sigmas[i],1e-9)
        w=np.exp(-0.5*z*z)[:,None]
        num[mask]+=w*residual[i]
        den[mask]+=w
    active=den[:,0] > 1e-10
    delta=np.zeros_like(v)
    delta[active]=num[active]/den[active]
    # Soft confidence fade prevents isolated far vertices from moving as much
    # as the facial core even if they barely enter one support sphere.
    conf=np.clip(den[:,0]/2.2,0.0,1.0)[:,None]
    return v + gain*conf*delta


def sculpt(vertices, idx, bw, target, cameras, iterations=7):
    v=vertices.copy()
    start=control_points(v,idx,bw)
    face_scale=max(float(np.linalg.norm(np.ptp(start,axis=0))),1e-6)
    sigmas=landmark_sigmas(face_scale)
    history=[]
    for it in range(iterations):
        cur=control_points(v,idx,bw)
        desired=triangulate_controls(cur,target,cameras)
        residual=desired-cur
        # Strong early movement then convergence polish.
        gain=[0.88,0.78,0.66,0.56,0.48,0.40,0.34][min(it,6)]
        v=propagate(v,cur,residual,sigmas,gain)
        after=control_points(v,idx,bw)
        rec={'iteration':it,'control_rms_3d':float(np.sqrt(np.mean((desired-after)**2))),
             'max_control_move':float(np.max(np.linalg.norm(after-cur,axis=1)))}
        for name in core.VIEW_ORDER:
            r,s,t=cameras[name]
            pred=core.project_np(after,r,s,t)
            e=np.linalg.norm(pred-target[name],axis=1)
            rec[f'{name}_rmse']=float(np.sqrt(np.mean(e**2)))
        print(json.dumps(rec)); history.append(rec)
    return v,history


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-full',type=Path,required=True)
    ap.add_argument('--front',type=Path,required=True)
    ap.add_argument('--three-quarter',type=Path,required=True)
    ap.add_argument('--side',type=Path,required=True)
    ap.add_argument('--front-landmarks',type=Path,required=True)
    ap.add_argument('--cameras',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('output_v1070'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); qa=args.out/'QA'; qa.mkdir(exist_ok=True)

    refs={'front':core.load_image_rgb(args.front),'three_quarter':core.load_image_rgb(args.three_quarter),'side':core.load_image_rgb(args.side)}
    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cpu',face_detector='sfd')
    target_px={
      'front':np.asarray(json.loads(args.front_landmarks.read_text())['landmarks_xy'],dtype=np.float32),
      'three_quarter':core.detect_68(fa,refs['three_quarter']),
      'side':core.detect_68(fa,refs['side'])
    }
    target={name:core.normalize_target(target_px[name],refs[name].shape) for name in core.VIEW_ORDER}
    cameras=load_cameras(args.cameras)

    g=gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,variant=gnm_numpy.GNMVariant.HEAD)
    cfg=gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
    idx=np.asarray(cfg.indices,dtype=np.int64); bw=np.asarray(cfg.weights,dtype=np.float64)
    base=trimesh.load(args.base_full,process=False,maintain_order=True)
    vertices=np.asarray(base.vertices,dtype=np.float64)
    if len(vertices)!=len(g.template_vertex_positions):
        raise RuntimeError(f'Base mesh vertex count {len(vertices)} does not match GNM {len(g.template_vertex_positions)}')

    sculpted,history=sculpt(vertices,idx,bw,target,cameras,iterations=7)
    triangles=np.asarray(g.triangles,dtype=np.int64)
    full=trimesh.Trimesh(vertices=sculpted,faces=triangles,process=False)
    full.export(args.out/'AINA_FACE_MASTER_GNM_v10.7.0_EFFECT_SCULPT.obj')
    full.export(args.out/'AINA_FACE_MASTER_GNM_v10.7.0_EFFECT_SCULPT.glb')
    skin_ti=np.asarray(g.triangle_indices_for_group('skin'),dtype=np.int64)
    skin=full.submesh([skin_ti],append=True,repair=False); skin.remove_unreferenced_vertices()
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.7.0_EFFECT_SCULPT.obj')
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.7.0_EFFECT_SCULPT.glb')
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.7.0_EFFECT_SCULPT.ply')

    lm=control_points(sculpted,idx,bw)
    errors={}
    for name in core.VIEW_ORDER:
        r,s,t=cameras[name]; pred=core.project_np(lm,r,s,t); e=np.linalg.norm(pred-target[name],axis=1)
        errors[name]={'rmse':float(np.sqrt(np.mean(e**2))),'median':float(np.median(e)),'p90':float(np.percentile(e,90))}
        core.save_overlay(refs[name],target_px[name],pred,qa/f'AINA_{name}_overlay_v10.7.0.png',f'AINA v10.7.0 nonlinear sculpt {name}')

    R=cameras['front'][0]; sv=np.asarray(skin.vertices); sf=np.asarray(skin.faces); paths=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p=qa/f'AINA_CLAY_{label}_v10.7.0.png'; core.render_mesh_ortho(sv,sf,R,yaw,p,f'AINA v10.7.0 {label}'); paths.append(p)
    ims=[Image.open(p).convert('RGB') for p in paths]; H=max(x.height for x in ims); W=max(x.width for x in ims)
    sheet=Image.new('RGB',(W*5,H),'white')
    for i,im in enumerate(ims): sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_CLAY_5VIEW_v10.7.0.png')

    report={'version':'AINA Face Master v10.7.0 Effect-Art Nonlinear Sculpt','base':'v10.6.5 corrected identity',
      'topology_changed':False,'method':'multi-view 3D landmark triangulation + localized smooth mesh deformation',
      'errors':errors,'history':history,'identity_lock':False,
      'note':'Nonlinear sculpt intentionally leaves the bounded GNM identity manifold while preserving the original topology. Lock only after visual 5-view likeness passes.'}
    (args.out/'AINA_v10.7.0_REPORT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
