#!/usr/bin/env python3
"""AINA Face Master v10.6 — art-directed identity rebuild.

The v10.5 fit proved that landmark detectors under-estimate the stylized AINA eye
aperture. This pass therefore uses the approved artwork as an art-direction
constraint instead of treating detected landmark height as ground truth.

Changes are deliberately large but smooth:
* eye fissure height ~2x with GNM-native first-200 eye expression basis,
* overall face width reduced into a youthful heart/V silhouette,
* lower face shortened,
* nose alar/tip narrowed and projection reduced,
* lips softened/fullened without forcing inner-mouth landmarks,
* cheek apples receive slight forward volume.

GNM topology is preserved exactly.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path

import cv2
import face_alignment
import numpy as np
from PIL import Image
from scipy.optimize import lsq_linear
import trimesh

from gnm.shape import gnm_numpy, gnm_landmarks
import identity_lock_v104 as h


def detect68(fa, im):
    hh, ww = im.shape[:2]
    s = max(1.0, 720.0 / max(hh, ww))
    work = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else im
    preds = fa.get_landmarks_from_image(work)
    if not preds:
        raise RuntimeError('No face detected')
    ctr = np.array([work.shape[1]*.5, work.shape[0]*.5], dtype=np.float64)
    q = min(preds, key=lambda p: np.linalg.norm(np.asarray(p)[:, :2].mean(0)-ctr))
    return np.asarray(q, dtype=np.float64)[:, :2] / s


def norm_target(p, shape):
    hh, ww = shape[:2]
    s = .5 * max(ww, hh)
    return (p - np.array([ww*.5, hh*.5])) / s


def solve_art_eye_delta(g, identity, vertices, idx, bw, R, target_cam):
    lm = h.lm(vertices, idx, bw)
    lc = lm @ R.T
    desired = lc.copy()

    # Art-directed eye target: preserve target center/slant, but open the fissure
    # far beyond the detector's stylized-eye estimate.
    for ids in ([36,37,38,39,40,41], [42,43,44,45,46,47]):
        ids = np.asarray(ids)
        c = lc[ids,:2].mean(0)
        tc = target_cam[ids].mean(0)
        # use eye-corner axis from detected artwork, which is reliable for slant
        a, b = int(ids[0]), int(ids[3])
        tv = target_cam[b] - target_cam[a]
        tw = max(float(np.linalg.norm(tv)), 1e-8)
        te1 = tv / tw
        te2 = np.array([-te1[1], te1[0]])
        cv = lc[b,:2] - lc[a,:2]
        cw = max(float(np.linalg.norm(cv)), 1e-8)
        ce1 = cv / cw
        ce2 = np.array([-ce1[1], ce1[0]])
        # Width is modestly enlarged; vertical aperture is intentionally ~2x.
        sx = 1.08
        sy = 2.05
        for i in ids:
            rel = lc[i,:2] - c
            x = float(rel @ ce1)
            y = float(rel @ ce2)
            desired[i,:2] = tc + te1*(x*sx) + te2*(y*sy)
        # slight elegant outer-corner lift
        desired[int(ids[0]),1] -= .00045
        desired[int(ids[3]),1] -= .00020

    expr_basis = np.asarray(g.expression_basis[:200], dtype=np.float64)
    lm_basis = (expr_basis[:,idx,:] * bw[None,...,None]).sum(axis=-2)
    lm_basis_cam = np.einsum('elc,dc->eld', lm_basis, R)

    rows, rhs = [], []
    for i in range(36,48):
        corner = i in (36,39,42,45)
        for ax in (0,1):
            w = (3.6 if not corner else 1.7) if ax == 1 else (1.3 if corner else .45)
            rows.append(lm_basis_cam[:,i,ax] * w)
            rhs.append((desired[i,ax] - lc[i,ax]) * w)
    A = np.stack(rows)
    b = np.asarray(rhs)
    row_scale = max(float(np.median(np.linalg.norm(A,axis=1))), 1e-8)
    lam = (row_scale**2) * .018
    A2 = np.vstack([A, np.sqrt(lam)*np.eye(200)])
    b2 = np.concatenate([b, np.zeros(200)])
    sol = lsq_linear(A2, b2, bounds=(-2.5,2.5), method='trf', tol=1e-9, max_iter=400)
    coeff = sol.x
    expr = np.zeros((1,g.expression_dim), dtype=np.float64)
    expr[0,:200] = coeff
    neutral = np.asarray(g(identity=identity))[0]
    expressed = np.asarray(g(identity=identity, expression=expr))[0]
    delta = expressed - neutral
    # keep this pass eye-local in practice; expression basis already has semantic locality
    return delta, coeff, desired


def compact_gauss(p, center, rx, ry, power=2.2):
    q = ((p[:,0]-center[0])/max(rx,1e-6))**2 + ((p[:,1]-center[1])/max(ry,1e-6))**2
    return np.exp(-.5*np.power(q,power/2.0))


def render_set(v, f, R, qa, kind, version):
    paths=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p=qa/f'AINA_{kind}_CLAY_{label}_{version}.png'
        h.render(v,f,R,yaw,p,f'AINA {version} {kind} {label.replace("_"," ")}')
        paths.append(p)
    ims=[Image.open(p).convert('RGB') for p in paths]
    H=max(x.height for x in ims); W=max(x.width for x in ims)
    sheet=Image.new('RGB',(W*5,H),'white')
    for k,im in enumerate(ims):
        sheet.paste(im,(k*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/f'AINA_{kind}_CLAY_5VIEW_{version}.png')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-full',type=Path,required=True)
    ap.add_argument('--front',type=Path,required=True)
    ap.add_argument('--identity',type=Path,required=True)
    ap.add_argument('--cameras',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('output_v106'))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    qa=args.out/'QA'; qa.mkdir(exist_ok=True)

    ref=np.asarray(Image.open(args.front).convert('RGB'))
    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cpu',face_detector='sfd')
    tp=detect68(fa,ref)
    tn=norm_target(tp,ref.shape)
    cams=json.loads(args.cameras.read_text())
    cam=cams['front']
    R=np.asarray(cam['rotation_rows'],dtype=np.float64)
    tc=h.symmetrize(h.target_cam(tn,cam))

    g=gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,variant=gnm_numpy.GNMVariant.HEAD)
    mesh=trimesh.load(args.base_full,process=False)
    v=np.asarray(mesh.vertices,dtype=np.float64)
    tri=np.asarray(g.triangles,dtype=np.int64)
    if len(v)!=len(g.template_vertex_positions):
        raise RuntimeError(f'GNM vertex count mismatch {len(v)}')
    identity=np.load(args.identity).astype(np.float64).reshape(1,-1)
    cfg=gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
    idx=np.asarray(cfg.indices,dtype=np.int64); bw=np.asarray(cfg.weights,dtype=np.float64)

    # 1. Topology-native art-directed eye opening.
    eye_delta, coeff, desired_eye = solve_art_eye_delta(g,identity,v,idx,bw,R,tc)
    # Safety: expression displacement should stay within plausible eyelid range.
    ed=np.linalg.norm(eye_delta,axis=1)
    scale=min(1.0, .0085/max(float(ed.max()),1e-9))
    v1=v+eye_delta*scale

    # Skin topology.
    sti=np.asarray(g.triangle_indices_for_group('skin'),dtype=np.int64)
    sfg=tri[sti]
    skin_ids=np.unique(sfg.reshape(-1))
    g2l={int(x):i for i,x in enumerate(skin_ids)}
    sf=np.vectorize(g2l.get)(sfg)
    sv=v1[skin_ids]
    p=sv@R.T
    orig=p.copy()
    L=h.lm(v1,idx,bw)
    lc=L@R.T

    cx=float(np.mean([lc[27,0],lc[30,0],lc[33,0],lc[8,0]]))
    eye_y=float(np.mean(lc[36:48,1])); nose_y=float(lc[33,1]); mouth_y=float(lc[57,1]); chin_y=float(lc[8,1])
    # Sort landmarks vertically in case camera axis is flipped; interpolate by normalized anatomical fraction.
    denom=max(abs(chin_y-eye_y),1e-8)
    t=(p[:,1]-eye_y)/(chin_y-eye_y if abs(chin_y-eye_y)>1e-8 else 1.0)
    # face-width scale: youthful heart shape, stronger toward lower jaw
    s=np.ones(len(p))
    s=np.where(t<0, .96, s)
    s=np.where((t>=0)&(t<.35), .92, s)
    s=np.where((t>=.35)&(t<.62), .87, s)
    s=np.where((t>=.62)&(t<.82), .80, s)
    s=np.where(t>=.82, .74, s)
    s=np.clip(s,.72,1.0)
    # apply only to the facial envelope; ears/cranium/back of head remain stable
    half=max(abs(lc[14,0]-lc[2,0])*.72,.055)
    radial=np.exp(-.5*((p[:,0]-cx)/half)**6)
    face_z=float(np.median(lc[0:68,2]))
    depth_gate=np.exp(-.5*((p[:,2]-face_z)/.070)**4)
    vertical_gate=np.exp(-.5*((t-.52)/.70)**6)
    w=np.clip(radial*depth_gate*vertical_gate,0,1)
    desired_x=cx+s*(p[:,0]-cx)
    p[:,0]+=(desired_x-p[:,0])*w*.92

    # 2. Shorter lower face / smaller rounded chin.
    lower=(p[:,1]-nose_y)/(chin_y-nose_y if abs(chin_y-nose_y)>1e-8 else 1.0)
    lower_w=np.clip(lower,0,1)**1.3 * radial * depth_gate
    desired_y=nose_y + .90*(p[:,1]-nose_y)
    p[:,1]+=(desired_y-p[:,1])*lower_w*.86

    # 3. Delicate short nose: narrower alar, smaller tip, less projection.
    nc=lc[30:36,:2].mean(0)
    nw=compact_gauss(p,nc,.020,.028,2.2)
    p[:,0]=np.where(nw>1e-6, cx+(p[:,0]-cx)*(1-.12*nw), p[:,0])
    p[:,1]+=(nc[1]+.94*(p[:,1]-nc[1])-p[:,1])*nw*.50
    p[:,2]+=.0030*nw

    # 4. Soft, slightly fuller lips without inner-mouth point forcing.
    mc=lc[48:60,:2].mean(0)
    mw=compact_gauss(p,mc,.042,.027,2.2)
    p[:,1]+=(mc[1]+1.14*(p[:,1]-mc[1])-p[:,1])*mw*.64
    p[:,0]+=(mc[0]+.98*(p[:,0]-mc[0])-p[:,0])*mw*.45
    p[:,2]-=.00065*mw

    # 5. Subtle apple-cheek volume after narrowing the face.
    for eye_ids,alar,corner in [([36,39],31,48),([42,45],35,54)]:
        c=np.array([(lc[eye_ids,0].mean()+lc[alar,0]+lc[corner,0])/3,
                    (lc[eye_ids,1].mean()+lc[alar,1]+lc[corner,1])/3])
        cw=compact_gauss(p,c,.032,.036,2.3)
        p[:,2]-=.0010*cw

    # Smooth only the semantic correction relative to the eye-basis result.
    raw=p-orig
    raw=h.smooth(raw,sf,4,.14)
    # preserve large but smooth jaw narrowing; cap only local 3D displacement spikes
    rn=np.linalg.norm(raw,axis=1)
    over=rn>.014
    if np.any(over): raw[over]*=(.014/rn[over])[:,None]
    p=orig+raw
    v2=v1.copy(); v2[skin_ids]=p@R

    skin=trimesh.Trimesh(vertices=v2[skin_ids],faces=sf,process=False)
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.6.obj')
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.6.ply')
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.6.glb')
    full=trimesh.Trimesh(vertices=v2,faces=tri,process=False)
    full.export(args.out/'AINA_FACE_MASTER_GNM_v10.6_FULL_TOPOLOGY.obj')
    full.export(args.out/'AINA_FACE_MASTER_GNM_v10.6_FULL_TOPOLOGY.glb')
    np.save(args.out/'AINA_ART_EYE_BAKE_COEFFICIENTS_v10.6.npy',coeff.astype(np.float32))

    render_set(v2[skin_ids],sf,R,qa,'SKIN','v10.6')
    render_set(v2,tri,R,qa,'FULL','v10.6')

    actual=Image.open(qa/'AINA_FULL_CLAY_front_v10.6.png').convert('RGB')
    refim=Image.open(args.front).convert('RGB')
    H=max(refim.height,actual.height); rw=int(refim.width*H/refim.height); aw=int(actual.width*H/actual.height)
    comp=Image.new('RGB',(rw+aw,H),'white'); comp.paste(refim.resize((rw,H)),(0,0)); comp.paste(actual.resize((aw,H)),(rw,0))
    comp.save(qa/'AINA_REFERENCE_VS_ACTUAL_FULL_FRONT_v10.6.png')

    lf=h.lm(v2,idx,bw)@R.T
    def width(x,a,b): return float(abs(x[b,0]-x[a,0]))
    def eheight(x,ids): return float(abs(x[ids[1:3],1].mean()-x[ids[4:6],1].mean()))
    metrics={
        'eye_L_height_multiplier_vs_v10.5': eheight(lf,[36,37,38,39,40,41])/max(eheight(lc,[36,37,38,39,40,41]),1e-9),
        'eye_R_height_multiplier_vs_v10.5': eheight(lf,[42,43,44,45,46,47])/max(eheight(lc,[42,43,44,45,46,47]),1e-9),
        'jaw_low_width_multiplier_vs_v10.5': width(lf,6,10)/max(width(lc,6,10),1e-9),
        'nose_width_multiplier_vs_v10.5': width(lf,31,35)/max(width(lc,31,35),1e-9),
        'mouth_width_multiplier_vs_v10.5': width(lf,48,54)/max(width(lc,48,54),1e-9),
        'eye_basis_scale': float(scale),
        'eye_coeff_rms': float(np.sqrt(np.mean(coeff**2))),
        'max_semantic_displacement_m': float(np.linalg.norm(raw,axis=1).max()),
    }
    report={
        'version':'AINA Face Master v10.6',
        'base':'v10.5 stable GNM topology',
        'method':'art-directed eye aperture + heart/V face proportion rebuild + short delicate nose + shortened lower face',
        'skin_vertices':int(len(skin_ids)),
        'skin_triangles':int(len(sf)),
        'metrics':metrics,
        'identity_lock':False,
        'note':'Detector eye-height is intentionally overridden. Acceptance is visual reference-vs-full-clay comparison.'
    }
    (args.out/'AINA_v10.6_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
