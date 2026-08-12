#!/usr/bin/env python3
"""AINA v10.6.5 — corrected-landmark identity rebuild.

Fixes the GNM jaw-landmark ordering mismatch before multi-view fitting and uses
3DDFA-derived front reference landmarks for the approved effect-art face. The
underlying GNM v3 topology is unchanged.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import trimesh
import face_alignment

import fit_aina_v101 as core
from gnm.shape import gnm_numpy, gnm_landmarks

# GNM HEAD_SPARSE_68 jaw samples are not stored in the conventional left-to-
# right 68-landmark order. Other regions already follow the standard ordering.
GNM_TO_STANDARD = np.array([
    0,1,6,5,4,3,2,7,8,9,10,11,12,13,14,15,16,
    *range(17,68)
], dtype=np.int64)


def weights_for_view(name: str) -> np.ndarray:
    w=np.ones(68,np.float64)
    if name=='front':
        w[0:17]=2.80; w[17:27]=1.15; w[27:36]=2.65
        w[36:48]=3.35; w[48:60]=3.15; w[60:68]=1.80
    elif name=='three_quarter':
        w[0:17]=1.65; w[17:27]=0.90; w[27:36]=2.20
        w[36:48]=2.15; w[48:60]=2.10; w[60:68]=1.25
    else:
        w[:]=0.08; w[0:17]=0.42; w[5:12]=1.00
        w[27:36]=2.10; w[48:60]=0.95
        w[36:42]=0.32; w[17:22]=0.18
    return w


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--front',type=Path,required=True)
    ap.add_argument('--three-quarter',type=Path,required=True)
    ap.add_argument('--side',type=Path,required=True)
    ap.add_argument('--front-landmarks',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('output_v1065'))
    ap.add_argument('--outer-steps',type=int,default=18)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); qa=args.out/'QA'; qa.mkdir(exist_ok=True)

    refs={'front':core.load_image_rgb(args.front),'three_quarter':core.load_image_rgb(args.three_quarter),'side':core.load_image_rgb(args.side)}
    fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cpu',face_detector='sfd')
    target_px={
      'front':np.asarray(json.loads(args.front_landmarks.read_text())['landmarks_xy'],dtype=np.float32),
      'three_quarter':core.detect_68(fa,refs['three_quarter']),
      'side':core.detect_68(fa,refs['side'])
    }
    target={name:core.normalize_target(target_px[name],refs[name].shape) for name in core.VIEW_ORDER}

    g=gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,variant=gnm_numpy.GNMVariant.HEAD)
    cfg=gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
    idx=np.asarray(cfg.indices,dtype=np.int64); bw=np.asarray(cfg.weights,dtype=np.float64)
    tv=np.asarray(g.template_vertex_positions,dtype=np.float64)
    ib=np.asarray(g.vertex_identity_basis,dtype=np.float64)
    lm_native=(tv[idx]*bw[...,None]).sum(axis=-2)
    basis_native=(ib[:170,idx,:]*bw[None,...,None]).sum(axis=-2)
    template_lm=lm_native[GNM_TO_STANDARD]
    basis=basis_native[:,GNM_TO_STANDARD,:]

    # Patch the shared robust solver with weights appropriate to the corrected
    # semantics; front identity dominates, profile only stabilizes depth.
    core.weights_for_view=weights_for_view
    core.VIEW_GLOBAL={'front':2.80,'three_quarter':1.00,'side':0.25}
    core.ROBUST_DELTA={'front':0.026,'three_quarter':0.034,'side':0.045}
    identity_head,cameras,history=core.fit_alternating(template_lm,basis,target,args.outer_steps)

    identity=np.zeros(g.identity_dim,dtype=np.float64); identity[:170]=identity_head
    vertices=np.asarray(g(identity=identity[None,:]))[0]
    triangles=np.asarray(g.triangles,dtype=np.int64)
    full=trimesh.Trimesh(vertices=vertices,faces=triangles,process=False)
    full.export(args.out/'AINA_FACE_MASTER_GNM_v10.6.5_CORRECTED.glb')
    full.export(args.out/'AINA_FACE_MASTER_GNM_v10.6.5_CORRECTED.obj')
    skin_ti=np.asarray(g.triangle_indices_for_group('skin'),dtype=np.int64)
    skin=full.submesh([skin_ti],append=True,repair=False); skin.remove_unreferenced_vertices()
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.6.5_CORRECTED.obj')
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.6.5_CORRECTED.glb')
    skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.6.5_CORRECTED.ply')
    np.save(args.out/'AINA_identity_coefficients_v10.6.5.npy',identity.astype(np.float32))
    np.save(args.out/'AINA_GNM_ORDERED_VERTICES_v10.6.5.npy',vertices.astype(np.float32))

    final_lm=template_lm+np.einsum('i,ilc->lc',identity_head,basis)
    errors={}; cameras_json={}
    for name in core.VIEW_ORDER:
        r,s,t=cameras[name]; pred=core.project_np(final_lm,r,s,t); e=np.linalg.norm(pred-target[name],axis=1)
        errors[name]={'rmse':float(np.sqrt(np.mean(e**2))),'median':float(np.median(e)),'p90':float(np.percentile(e,90))}
        cameras_json[name]={'rotation_rows':r.tolist(),'scale':float(s),'translation':t.tolist(),**errors[name]}
        core.save_overlay(refs[name],target_px[name],pred,qa/f'AINA_{name}_overlay_v10.6.5.png',f'AINA v10.6.5 corrected {name}')
    R=cameras['front'][0]
    sv=np.asarray(skin.vertices); sf=np.asarray(skin.faces)
    paths=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p=qa/f'AINA_CLAY_{label}_v10.6.5.png'; core.render_mesh_ortho(sv,sf,R,yaw,p,f'AINA v10.6.5 {label}'); paths.append(p)
    from PIL import Image
    ims=[Image.open(p).convert('RGB') for p in paths]; H=max(x.height for x in ims); W=max(x.width for x in ims)
    sheet=Image.new('RGB',(W*5,H),'white')
    for i,im in enumerate(ims): sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_CLAY_5VIEW_v10.6.5.png')

    report={
      'version':'AINA Face Master v10.6.5 Corrected Landmark Identity Rebuild',
      'base':'Google GNM v3 HEAD', 'topology_changed':False,
      'jaw_landmark_permutation':GNM_TO_STANDARD[:17].tolist(),
      'front_target':'3DDFA sparse-68 extracted from approved effect-art front',
      'identity_dimensions_optimized':170,
      'vertices':int(len(vertices)),'triangles':int(len(triangles)),
      'skin_vertices':int(len(sv)),'skin_triangles':int(len(sf)),
      'errors':errors,'history':history,'identity_lock':False,
      'note':'This rebuild corrects the landmark semantic mismatch before fitting. Identity lock remains false until visual clay QA passes.'
    }
    (args.out/'AINA_v10.6.5_REPORT.json').write_text(json.dumps(report,indent=2))
    (args.out/'AINA_CAMERAS_v10.6.5.json').write_text(json.dumps(cameras_json,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
