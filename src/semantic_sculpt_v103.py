#!/usr/bin/env python3
"""AINA Face Master v10.3 — semantic soft-tissue identity sculpt.

Unlike v10.2/v10.2.1, this pass does NOT force every 68-point landmark onto the
reference. It treats eyes, nose, lips, cheeks and jaw as coherent anatomical
regions, derives only their global scale/orientation/position from the approved
AINA art, then applies smooth bounded deformations to the clean v10.1 GNM skin.
The goal is visual likeness with preserved facial flow, not minimum point error.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
import cv2
import face_alignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from PIL import Image
import trimesh

from gnm.shape import gnm_numpy, gnm_landmarks

VIEWS = ("front", "three_quarter", "side")


def read_img(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def detect68(fa, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(1.0, 720.0 / max(h, w))
    work = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else img
    preds = fa.get_landmarks_from_image(work)
    if not preds:
        raise RuntimeError("No face detected")
    ctr = np.array([work.shape[1] * .5, work.shape[0] * .5], dtype=np.float64)
    best = min(preds, key=lambda q: np.linalg.norm(np.asarray(q)[:, :2].mean(0) - ctr))
    p = np.asarray(best, dtype=np.float64)[:, :2] / s
    if p.shape != (68, 2):
        raise RuntimeError(f"Expected 68, got {p.shape}")
    return p


def normalize_target(p: np.ndarray, shape) -> np.ndarray:
    h, w = shape[:2]
    s = .5 * max(w, h)
    return (p - np.array([w * .5, h * .5])) / s


def lm_from(v, idx, bw):
    return (v[idx] * bw[..., None]).sum(axis=-2)


def target_to_camera(target_norm, cam):
    return (target_norm - np.asarray(cam["translation"], dtype=np.float64)) / float(cam["scale"])


def project(points, cam):
    r = np.asarray(cam["rotation_rows"], dtype=np.float64)
    return float(cam["scale"]) * (points @ r.T)[:, :2] + np.asarray(cam["translation"], dtype=np.float64)


def gaussian_weight(xy, center, rx, ry, power=2.0):
    rx = max(float(rx), 1e-5); ry = max(float(ry), 1e-5)
    q = ((xy[:, 0] - center[0]) / rx) ** 2 + ((xy[:, 1] - center[1]) / ry) ** 2
    return np.exp(-0.5 * np.power(q, power / 2.0))


def eye_transform(curr, target, ids):
    # ids are in contour order: corner, upper2, corner, lower2.
    i0, i1, i2, i3, i4, i5 = ids
    cc = curr[ids].mean(axis=0); tc = target[ids].mean(axis=0)
    cv = curr[i3] - curr[i0]; tv = target[i3] - target[i0]
    cw = np.linalg.norm(cv); tw = np.linalg.norm(tv)
    ce1 = cv / max(cw, 1e-8); te1 = tv / max(tw, 1e-8)
    ce2 = np.array([-ce1[1], ce1[0]]); te2 = np.array([-te1[1], te1[0]])
    ch = abs(((curr[[i1, i2]].mean(0) - curr[[i4, i5]].mean(0)) @ ce2))
    th = abs(((target[[i1, i2]].mean(0) - target[[i4, i5]].mean(0)) @ te2))
    sx = float(np.clip(tw / max(cw, 1e-8), .94, 1.20))
    sy = float(np.clip(th / max(ch, 1e-8), 1.02, 1.55))
    Bc = np.stack([ce1, ce2], axis=1)
    Bt = np.stack([te1, te2], axis=1)
    A = Bt @ np.diag([sx, sy]) @ Bc.T
    return cc, tc, A, cw, max(ch, .012), {"width_scale": sx, "height_scale": sy}


def apply_local_affine(p, center, target_center, A, rx, ry, strength=1.0, depth_shift=0.0, power=2.2):
    xy = p[:, :2]
    local = xy - center
    desired = target_center + local @ A.T
    dxy = desired - xy
    w = gaussian_weight(xy, center, rx, ry, power=power) * strength
    p[:, :2] += dxy * w[:, None]
    if depth_shift:
        p[:, 2] += depth_shift * w
    return w


def apply_vertical_midface_warp(p, curr, target, face_center_x, face_rx):
    eye_y = .5 * (curr[36:42, 1].mean() + curr[42:48, 1].mean())
    nose_y = curr[33, 1]; mouth_y = curr[48:60, 1].mean(); chin_y = curr[8, 1]
    t_eye_y = .5 * (target[36:42, 1].mean() + target[42:48, 1].mean())
    t_nose_y = target[33, 1]; t_mouth_y = target[48:60, 1].mean(); t_chin_y = target[8, 1]
    ys = np.array([eye_y, nose_y, mouth_y, chin_y], dtype=np.float64)
    yt = np.array([t_eye_y, t_nose_y, t_mouth_y, t_chin_y], dtype=np.float64)
    order = np.argsort(ys); ys = ys[order]; yt = yt[order]
    desired_y = np.interp(p[:, 1], ys, yt, left=p[:, 1], right=p[:, 1])
    inside = (p[:, 1] >= ys[0]) & (p[:, 1] <= ys[-1])
    xfade = np.exp(-0.5 * ((p[:, 0] - face_center_x) / max(face_rx, 1e-6)) ** 4)
    w = inside.astype(np.float64) * xfade * .48
    p[:, 1] += (desired_y - p[:, 1]) * w
    return {"eye_y_scale_target": float((t_nose_y-t_eye_y)/max(nose_y-eye_y,1e-8)), "lower_face_y_scale_target": float((t_chin_y-t_nose_y)/max(chin_y-nose_y,1e-8))}


def apply_jaw_warp(p, curr, target):
    cx = .5 * (curr[0, 0] + curr[16, 0])
    tcx = .5 * (target[0, 0] + target[16, 0])
    pair_ids = [(2,14),(4,12),(6,10),(7,9)]
    cy=[]; ratio=[]; shift=[]
    for a,b in pair_ids:
        cwidth=abs(curr[b,0]-curr[a,0]); twidth=abs(target[b,0]-target[a,0])
        cy.append(.5*(curr[a,1]+curr[b,1])); ratio.append(float(np.clip(twidth/max(cwidth,1e-8),.78,1.05))); shift.append(.5*(target[a,0]+target[b,0])-cx)
    cy=np.asarray(cy);ratio=np.asarray(ratio);shift=np.asarray(shift);order=np.argsort(cy);cy=cy[order];ratio=ratio[order];shift=shift[order]
    nosebase=curr[33,1]; chin=curr[8,1]; denom=max(chin-nosebase,1e-8)
    t=np.clip((p[:,1]-nosebase)/denom,0,1)
    r=np.interp(p[:,1],cy,ratio,left=1.0,right=ratio[-1]); sh=np.interp(p[:,1],cy,shift,left=0.0,right=shift[-1])
    face_half=max(abs(curr[14,0]-curr[2,0])*.62,.04)
    radial=np.exp(-0.5*((p[:,0]-cx)/face_half)**6)
    w=(t**1.35)*radial*.84
    desired_x=tcx+sh+r*(p[:,0]-cx)
    p[:,0]+=(desired_x-p[:,0])*w
    # small rounded-chin vertical correction, applied as one soft region.
    chin_target=target[8,1]; chin_w=gaussian_weight(p[:,:2],np.array([cx,chin]),face_half*.55,abs(chin-nosebase)*.28,power=2.2)*.52
    p[:,1]+=(chin_target-chin)*chin_w
    return {"jaw_scale_low":float(ratio[-1]),"jaw_scale_upper":float(ratio[0])}


def smooth_displacement(raw, faces, iterations=6, alpha=.22):
    n=len(raw); adj=[set() for _ in range(n)]
    for a,b,c in faces:
        adj[a].update((b,c));adj[b].update((a,c));adj[c].update((a,b))
    d=raw.copy()
    active=np.linalg.norm(raw,axis=1)>1e-6
    for _ in range(iterations):
        old=d.copy(); new=d.copy()
        for i,nbr in enumerate(adj):
            if not nbr: continue
            mean=old[list(nbr)].mean(axis=0)
            a=alpha if active[i] else alpha*.38
            new[i]=(1-a)*old[i]+a*mean
        d=new
    return .72*d+.28*raw


def feature_metrics(lm, target):
    def dist(a,b):return float(np.linalg.norm(lm[a]-lm[b]))
    def tdist(a,b):return float(np.linalg.norm(target[a]-target[b]))
    out={}
    for name,ids in [('eye_L',(36,39)),('eye_R',(42,45)),('nose',(31,35)),('mouth',(48,54)),('jaw_mid',(4,12)),('jaw_low',(6,10))]:
        out[name]={"current":dist(*ids),"target":tdist(*ids),"ratio_target_over_current":tdist(*ids)/max(dist(*ids),1e-9)}
    return out


def save_overlay(img, target_px, pred_norm, path, title):
    h,w=img.shape[:2];s=.5*max(w,h);pp=pred_norm*s+np.array([w*.5,h*.5]);fig,ax=plt.subplots(figsize=(6,6),dpi=160);ax.imshow(img);ax.scatter(target_px[:,0],target_px[:,1],s=10,label='reference');ax.scatter(pp[:,0],pp[:,1],s=8,marker='x',label='v10.3');ax.axis('off');ax.set_title(title);ax.legend(loc='lower right',fontsize=7);fig.tight_layout(pad=.2);fig.savefig(path,bbox_inches='tight');plt.close(fig)


def render(v,f,R0,yaw,path,title):
    right,up,forward=R0[0],R0[1],R0[2];a=math.radians(yaw);R=np.stack([math.cos(a)*right+math.sin(a)*forward,up,-math.sin(a)*right+math.cos(a)*forward]);p=v@R.T;xy=p[:,:2];tri=p[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9);order=np.argsort(tri[:,:,2].mean(1))[::-1];ff=f[order];nn=n[order];tri2=xy[ff];dif=np.clip(np.abs(nn[:,2]),0,1);side=np.clip(-.3*nn[:,0]-.2*nn[:,1]-.7*nn[:,2],0,1);I=np.clip(.66+.21*dif+.10*side,.50,.98);col=np.stack([I*.96,I*.97,I],1);lo=np.percentile(xy,1.5,axis=0);hi=np.percentile(xy,98.5,axis=0);ct=(lo+hi)/2;ex=max((hi-lo).max(),1e-6)*.57;fig,ax=plt.subplots(figsize=(5,5),dpi=190);ax.add_collection(PolyCollection(tri2,facecolors=col,edgecolors='none'));ax.set_xlim(ct[0]-ex,ct[0]+ex);ax.set_ylim(ct[1]+ex,ct[1]-ex);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10);fig.tight_layout(pad=.15);fig.savefig(path,bbox_inches='tight',pad_inches=.02);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--front',type=Path,required=True);ap.add_argument('--three-quarter',type=Path,required=True);ap.add_argument('--side',type=Path,required=True);ap.add_argument('--identity',type=Path,required=True);ap.add_argument('--cameras',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_v103'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);qa=args.out/'QA';qa.mkdir(exist_ok=True)
    refs={'front':read_img(args.front),'three_quarter':read_img(args.three_quarter),'side':read_img(args.side)};fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cpu',face_detector='sfd');tpx={k:detect68(fa,refs[k]) for k in VIEWS};tn={k:normalize_target(tpx[k],refs[k].shape) for k in VIEWS};cams=json.loads(args.cameras.read_text());fcam=cams['front'];R=np.asarray(fcam['rotation_rows'],dtype=np.float64);target_cam=target_to_camera(tn['front'],fcam)
    g=gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,variant=gnm_numpy.GNMVariant.HEAD);ident=np.load(args.identity).reshape(1,-1);v=np.asarray(g(identity=ident))[0].astype(np.float64);tri=np.asarray(g.triangles,dtype=np.int64);sti=np.asarray(g.triangle_indices_for_group('skin'),dtype=np.int64);sfg=tri[sti];skin_ids=np.unique(sfg.reshape(-1));g2l={int(x):i for i,x in enumerate(skin_ids)};sf=np.vectorize(g2l.get)(sfg);sv=v[skin_ids];cfg=gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68);idx=np.asarray(cfg.indices,dtype=np.int64);bw=np.asarray(cfg.weights,dtype=np.float64);lm=lm_from(v,idx,bw);lmcam=lm@R.T
    p=sv@R.T;orig=p.copy(); semantic={}
    face_cx=.5*(lmcam[0,0]+lmcam[16,0]);face_rx=abs(lmcam[14,0]-lmcam[2,0])*.65
    semantic['midface']=apply_vertical_midface_warp(p,lmcam[:,:2],target_cam,face_cx,face_rx)
    semantic['jaw']=apply_jaw_warp(p,lmcam[:,:2],target_cam)
    # Eyes: coherent similarity/anisotropic aperture transform, with subtle recessed socket support.
    for label,ids in [('eye_L',[36,37,38,39,40,41]),('eye_R',[42,43,44,45,46,47])]:
        cc,tc,A,w,h,m=eye_transform(lmcam[:,:2],target_cam,ids);apply_local_affine(p,cc,tc,A,w*1.35,max(h*3.2,w*.42),strength=.90,depth_shift=.00035,power=2.4);semantic[label]=m
    # Nose as one soft unit: narrow/shorten in front and reduce projection slightly.
    nc=lmcam[27:36,:2].mean(0);nt=target_cam[27:36].mean(0);nw=abs(lmcam[35,0]-lmcam[31,0]);tw=abs(target_cam[35,0]-target_cam[31,0]);nh=abs(lmcam[33,1]-lmcam[27,1]);th=abs(target_cam[33,1]-target_cam[27,1]);nsx=float(np.clip(tw/max(nw,1e-8),.76,1.02));nsy=float(np.clip(th/max(nh,1e-8),.86,1.06));A=np.diag([nsx,nsy]);apply_local_affine(p,nc,nt,A,max(nw*1.75,.035),max(nh*.78,.045),strength=.86,depth_shift=.00155,power=2.0);semantic['nose']={'width_scale':nsx,'height_scale':nsy,'depth_back_m':.00155}
    # Mouth uses outer lip as a single soft form; inner lip landmarks are deliberately ignored.
    mc=lmcam[48:60,:2].mean(0);mt=target_cam[48:60].mean(0);mw=abs(lmcam[54,0]-lmcam[48,0]);tmw=abs(target_cam[54,0]-target_cam[48,0]);mh=abs(lmcam[[50,51,52],1].mean()-lmcam[[56,57,58],1].mean());tmh=abs(target_cam[[50,51,52],1].mean()-target_cam[[56,57,58],1].mean());msx=float(np.clip(tmw/max(mw,1e-8),.82,1.04));msy=float(np.clip(tmh/max(mh,1e-8),.92,1.20));A=np.diag([msx,msy]);apply_local_affine(p,mc,mt,A,max(mw*1.45,.045),max(mh*3.0,.038),strength=.76,depth_shift=-.00085,power=2.1);semantic['mouth']={'width_scale':msx,'height_scale':msy,'lip_forward_m':.00085}
    # Apple cheeks: soft forward volume, not point fitting.
    for side_sign,eye_ids,alar,corner in [(-1,[36,39],31,48),(1,[42,45],35,54)]:
        c=np.array([(lmcam[eye_ids,0].mean()+lmcam[alar,0]+lmcam[corner,0])/3,(lmcam[eye_ids,1].mean()+lmcam[alar,1]+lmcam[corner,1])/3]);rx=abs(lmcam[39,0]-lmcam[36,0])*1.25;ry=abs(lmcam[48:60,1].mean()-lmcam[36:48,1].mean())*.58;wgt=gaussian_weight(p[:,:2],c,max(rx,.036),max(ry,.036),power=2.2)*.78;p[:,2]-=.00120*wgt;p[:,0]+=side_sign*.00032*wgt
    semantic['cheeks']={'forward_m':.00120,'lateral_m':.00032}
    # Rounded small chin: modest forward support after lower-face shortening.
    chin=np.array([lmcam[8,0],lmcam[8,1]]);cw=gaussian_weight(p[:,:2],chin,max(abs(lmcam[10,0]-lmcam[6,0])*.72,.036),max(abs(lmcam[8,1]-lmcam[57,1])*.75,.035),power=2.2)*.58;p[:,2]-=.00065*cw;semantic['chin']={'forward_m':.00065}
    raw=p-orig;raw_norm=np.linalg.norm(raw,axis=1);cap=.0065;bad=raw_norm>cap
    if np.any(bad):raw[bad]*=(cap/raw_norm[bad])[:,None]
    smooth=smooth_displacement(raw,sf,iterations=6,alpha=.21);sn=np.linalg.norm(smooth,axis=1);bad=sn>cap
    if np.any(bad):smooth[bad]*=(cap/sn[bad])[:,None]
    p=orig+smooth;sv2=p@R;v2=v.copy();v2[skin_ids]=sv2
    skin=trimesh.Trimesh(vertices=sv2,faces=sf,process=False);skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.3.obj');skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.3.ply');skin.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v10.3.glb');full=trimesh.Trimesh(vertices=v2,faces=tri,process=False);full.export(args.out/'AINA_FACE_MASTER_GNM_v10.3_FULL_TOPOLOGY.obj');full.export(args.out/'AINA_FACE_MASTER_GNM_v10.3_FULL_TOPOLOGY.glb')
    lm2=lm_from(v2,idx,bw);lm2cam=lm2@R.T;metrics={'before':feature_metrics(lmcam[:,:2],target_cam),'after':feature_metrics(lm2cam[:,:2],target_cam)}
    errors={}
    for view in VIEWS:
        pred=project(lm2,cams[view]);e=np.linalg.norm(pred-tn[view],axis=1);errors[view]={'rmse_all68':float(np.sqrt(np.mean(e**2))),'median_all68':float(np.median(e))};save_overlay(refs[view],tpx[view],pred,qa/f'AINA_{view}_overlay_v10.3.png',f'AINA v10.3 {view}')
    paths=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        q=qa/f'AINA_CLAY_{label}_v10.3.png';render(sv2,sf,R,yaw,q,f'AINA v10.3 Clay {label.replace("_"," ")}');paths.append(q)
    ims=[Image.open(q).convert('RGB') for q in paths];H=max(i.height for i in ims);W=max(i.width for i in ims);sheet=Image.new('RGB',(W*5,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_CLAY_5VIEW_v10.3.png')
    report={'version':'AINA Face Master v10.3','base':'clean GNM v10.1 identity','method':'semantic coherent-region sculpt (eyes/nose/lips/cheeks/jaw) + bounded topology diffusion','skin_vertices':int(len(sv2)),'skin_triangles':int(len(sf)),'max_displacement_m':float(np.linalg.norm(smooth,axis=1).max()),'rms_displacement_m':float(np.sqrt(np.mean(smooth**2))),'semantic_controls':semantic,'feature_metrics':metrics,'landmark_diagnostics_only':errors,'identity_lock':False,'note':'Landmark error is diagnostic only; visual five-view likeness is the acceptance criterion.'};(args.out/'AINA_v10.3_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
