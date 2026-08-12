#!/usr/bin/env python3
"""AINA v11.3 — identity-feature sculpt on stable v11.2 TPS topology.

Only perceptual identity regions are edited: eyelid fissures/canthi, brow and
glabella mass, frontal temple/forehead width, nose tip/alae, lip volume, apple
cheeks and chin. Every edit is analytic, local and millimetre-bounded.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from PIL import Image

import fit_aina_v101 as core
from gnm.shape import gnm_numpy,gnm_landmarks

GNM_TO_STANDARD=np.array([0,1,6,5,4,3,2,7,8,9,10,11,12,13,14,15,16,*range(17,68)],dtype=np.int64)


def smoothstep01(x):
    x=np.clip(x,0.,1.);return x*x*(3.-2.*x)

def gauss(p,cx,cy,rx,ry,pow=2.0):
    q=((p[:,0]-cx)/max(rx,1e-9))**2+((p[:,1]-cy)/max(ry,1e-9))**2
    return np.exp(-.5*np.power(q,pow/2.))

def clampv(d,cap):
    o=d.copy();n=np.linalg.norm(o,axis=1);m=n>cap
    if np.any(m):o[m]*=(cap/n[m])[:,None]
    return o

def controls(v,idx,bw):return (v[idx]*bw[...,None]).sum(-2)

def fw():
    w=np.ones(68);w[:17]=3.2;w[17:27]=1.1;w[27:36]=3.4;w[36:48]=5.;w[48:60]=4.;w[60:68]=2.2;return w


def find_eye_components(full_mesh,R,cp):
    comps=[]
    for comp in full_mesh.split(only_watertight=False):
        nv=len(comp.vertices)
        if 300<=nv<=500:
            cc=np.asarray(comp.vertices).mean(0)@R.T
            comps.append((comp,cc))
    if len(comps)<2:raise RuntimeError(f'Only {len(comps)} candidate eye components')
    targets=[cp[36:42].mean(0),cp[42:48].mean(0)]
    chosen=[];used=set()
    for t in targets:
        ranked=sorted(enumerate(comps),key=lambda it:float(np.linalg.norm(it[1][1][:2]-t[:2])))
        for k,(comp,cc) in ranked:
            if k not in used:used.add(k);chosen.append((comp,cc));break
    return chosen


def eye_open_delta(p,skin_world,eye_mesh,eye_cp):
    ew=np.asarray(eye_mesh.vertices,np.float64);dist3,_=cKDTree(ew).query(skin_world,k=1)
    # Adaptive rim band around the eyeball surface.
    rim=np.where(dist3<0.00145)[0]
    if len(rim)<25:rim=np.argsort(dist3)[:80]
    rp=p[rim];x0=float(eye_cp[:,0].mean());cy=float(eye_cp[:,1].mean());xmin,xmax=np.percentile(rp[:,0],[3,97]);half=max(.5*(xmax-xmin),.008)
    side=-1. if x0<np.median(p[:,0]) else 1.
    handle=np.zeros((len(rim),3),np.float64)
    for k,i in enumerate(rim):
        u=float(np.clip((p[i,0]-x0)/half,-1,1));arch=math.sqrt(max(0.,1-u*u));outer=np.clip(side*u,0.,1.);inner=np.clip(-side*u,0.,1.)
        upper=p[i,1]<=cy
        if upper:
            # AINA: larger almond opening with lifted outer tail, not round doll eyes.
            handle[k,1]=-0.00185*arch-0.00072*smoothstep01(outer)+0.00010*smoothstep01(inner)
        else:
            handle[k,1]=+0.00068*arch-0.00012*smoothstep01(outer)
        handle[k,0]=side*0.00034*smoothstep01(outer)-side*0.00008*smoothstep01(inner)
        handle[k,2]=-0.00018*arch
    tree=cKDTree(rp[:,:2]);dd,near=tree.query(p[:,:2],k=1);w=np.exp(-.5*(dd/.0048)**4)
    rz=float(np.median(rp[:,2]));w*=np.exp(-.5*((p[:,2]-rz)/.013)**4)
    return handle[near]*w[:,None],{'rim_vertices':int(len(rim)),'max_handle_m':float(np.max(np.linalg.norm(handle,axis=1)))}


def triangle_area(v,f):
    t=v[f];return .5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)

def compare(ref,act,out):
    a=Image.open(ref).convert('RGB');b=Image.open(act).convert('RGB');H=max(a.height,b.height);aw=int(a.width*H/a.height);bw=int(b.width*H/b.height);s=Image.new('RGB',(aw+bw,H),'white');s.paste(a.resize((aw,H)),(0,0));s.paste(b.resize((bw,H)),(aw,0));s.save(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-full',type=Path,required=True);ap.add_argument('--base-skin',type=Path,required=True);ap.add_argument('--front',type=Path,required=True);ap.add_argument('--front-landmarks',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_v1130'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);qa=args.out/'QA';qa.mkdir(exist_ok=True)
    full=trimesh.load(args.base_full,process=False,maintain_order=True);skin=trimesh.load(args.base_skin,process=False,maintain_order=True)
    fv=np.asarray(full.vertices,np.float64);ff=np.asarray(full.faces,np.int64);sv=np.asarray(skin.vertices,np.float64);sf=np.asarray(skin.faces,np.int64);ns=len(sv)
    if not np.allclose(fv[:ns],sv,atol=2e-6):raise RuntimeError('v11.2 full mesh no longer begins with skin block')

    g=gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,variant=gnm_numpy.GNMVariant.HEAD);cfg=gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68);idx=np.asarray(cfg.indices,np.int64)[GNM_TO_STANDARD];bw=np.asarray(cfg.weights,np.float64)[GNM_TO_STANDARD]
    cpw=controls(fv,idx,bw);ref=core.load_image_rgb(args.front);target_px=np.asarray(json.loads(args.front_landmarks.read_text())['landmarks_xy'],np.float32);target=core.normalize_target(target_px,ref.shape);R,scale,trans=core.scaled_ortho_init(cpw,target,fw());p=sv@R.T;cp=cpw@R.T
    raw=np.zeros_like(p);stats={}

    # 1) EYES — explicitly open the actual eyelid rim around the fixed eyeballs.
    for ei,(eye,cc) in enumerate(find_eye_components(full,R,cp)):
        eye_cp=cp[36:42] if ei==0 else cp[42:48];d,st=eye_open_delta(p,sv,eye,eye_cp);raw+=d;stats[f'eye_{ei}']=st

    midx=float(cp[27:36,0].mean());eye_y=float(cp[36:48,1].mean());eye_l=float(cp[36:42,0].mean());eye_r=float(cp[42:48,0].mean())
    front_z=float(np.median(cp[:,2]));front_gate=np.exp(-.5*((p[:,2]-front_z)/.080)**4)

    # 2) BROW/GLABELLA — remove adult ridge, lift/soften upper orbital transition.
    for ex in (eye_l,eye_r):
        brow=gauss(p,ex,eye_y-.033,.038,.026,2.2)*front_gate;under=gauss(p,ex,eye_y+.018,.034,.022,2.2)*front_gate
        raw[:,2]+=0.0038*brow;raw[:,1]-=0.00065*brow;raw[:,2]-=0.00055*under
    glab=gauss(p,midx,eye_y-.028,.029,.034,2.2)*front_gate;raw[:,2]+=0.0024*glab

    # 3) FRONTAL TEMPLE/FOREHEAD — narrower youthful frontal shell only; rear skull untouched.
    fore=gauss(p,midx,eye_y-.078,.086,.088,2.0)*front_gate
    raw[:,0]+=((midx+(p[:,0]-midx)*.958)-p[:,0])*fore

    # 4) NOSE — delicate alae/tip and lower bridge, preserving the TPS landmark placement.
    nc=cp[27:36,:2].mean(0);nw=max(float(np.linalg.norm(cp[31,:2]-cp[35,:2])),.012)
    nose=gauss(p,float(nc[0]),float(nc[1]),1.70*nw,.041,2.3)*front_gate;lower=smoothstep01((p[:,1]-(float(nc[1])-.018))/.052);nwgt=nose*lower
    raw[:,0]+=((float(nc[0])+(p[:,0]-float(nc[0]))*.90)-p[:,0])*nwgt;raw[:,2]+=0.0023*nose
    tip=gauss(p,float(cp[33,0]),float(cp[33,1]),.014,.014,2.5)*front_gate;raw[:,2]+=0.0010*tip;raw[:,1]-=0.00055*tip

    # 5) LIPS — keep approved width, add youthful cupid/lower-lip volume instead of a flat seam.
    mc=cp[48:60,:2].mean(0);mw=max(float(np.linalg.norm(cp[48,:2]-cp[54,:2])),.025)
    lip=gauss(p,float(mc[0]),float(mc[1]),.70*mw,.018,2.4)*front_gate;upper=gauss(p,float(mc[0]),float(mc[1])-.004,.58*mw,.010,2.5)*front_gate;lowerlip=gauss(p,float(mc[0]),float(mc[1])+.005,.60*mw,.011,2.5)*front_gate;cupid=gauss(p,float(mc[0]),float(mc[1])-.006,.010,.0065,2.5)*front_gate
    raw[:,2]-=0.00085*upper+0.00155*lowerlip;raw[:,1]-=0.00055*cupid;raw[:,1]+=0.00022*lowerlip

    # 6) APPLE CHEEKS — forward high cheek volume, less lateral lower-cheek mass.
    cheek_y=.56*eye_y+.44*float(mc[1])
    for ex in (eye_l,eye_r):
        apple=gauss(p,ex,cheek_y,.033,.031,2.3)*front_gate;raw[:,2]-=0.00185*apple;raw[:,0]+=((midx+(p[:,0]-midx)*.982)-p[:,0])*apple
    lower_side=(gauss(p,eye_l,float(mc[1])+.010,.044,.042,2.0)+gauss(p,eye_r,float(mc[1])+.010,.044,.042,2.0))*front_gate
    raw[:,2]+=0.00085*lower_side

    # 7) CHIN — shorter, slightly narrower, less heavy in profile.
    ch=cp[8,:2];chin=gauss(p,float(ch[0]),float(ch[1]),.039,.032,2.3)*front_gate;raw[:,1]-=0.00115*chin;raw[:,0]+=((float(ch[0])+(p[:,0]-float(ch[0]))*.965)-p[:,0])*chin;raw[:,2]+=0.00035*chin

    raw=clampv(raw,.0062);p2=p+raw;sv2=p2@R;fv2=fv.copy();fv2[:ns]=sv2
    a0=triangle_area(sv,sf);a1=triangle_area(sv2,sf);ratio=a1/np.maximum(a0,1e-12);q01=float(np.percentile(ratio,1));q99=float(np.percentile(ratio,99));maxshift=float(np.max(np.linalg.norm(sv2-sv,axis=1)))
    if q01<.15 or q99>4.:raise RuntimeError(f'feature sculpt mesh quality p01={q01}, p99={q99}')
    fout=trimesh.Trimesh(vertices=fv2,faces=ff,process=False);sout=trimesh.Trimesh(vertices=sv2,faces=sf,process=False);fout.export(args.out/'AINA_FACE_MASTER_GNM_v11.3_IDENTITY_FEATURES.obj');fout.export(args.out/'AINA_FACE_MASTER_GNM_v11.3_IDENTITY_FEATURES.glb');sout.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v11.3_IDENTITY_FEATURES.obj');sout.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v11.3_IDENTITY_FEATURES.glb');sout.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v11.3_IDENTITY_FEATURES.ply')

    cfinal=controls(fv2,idx,bw);pred=core.project_np(cfinal,R,scale,trans);e=np.linalg.norm(pred-target,axis=1);core.save_overlay(ref,target_px,pred,qa/'AINA_front_overlay_v11.3.png','AINA v11.3 identity features')
    views=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        path=qa/f'AINA_FULL_CLAY_{label}_v11.3.png';core.render_mesh_ortho(fv2,ff,R,yaw,path,f'AINA v11.3 {label}');views.append(path)
    ims=[Image.open(x).convert('RGB') for x in views];H=max(i.height for i in ims);W=max(i.width for i in ims);sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_FULL_CLAY_5VIEW_v11.3.png');compare(args.front,qa/'AINA_FULL_CLAY_front_v11.3.png',qa/'AINA_REFERENCE_VS_ACTUAL_FRONT_v11.3.png')
    report={'version':'AINA Face Master v11.3 Identity Feature Sculpt','base':'v11.2 TPS Reference Cage','topology_changed':False,'max_additional_shift_m':maxshift,'triangle_area_ratio_p01':q01,'triangle_area_ratio_p99':q99,'front_fixed_rmse':float(np.sqrt(np.mean(e**2))),'eye_stats':stats,'identity_lock':False,'acceptance_note':'Visual likeness is the gate; v11.3 intentionally prioritizes eye/brow/nose/lip/cheek/chin identity over sparse-score optimization.'}
    (args.out/'AINA_v11.3_REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
