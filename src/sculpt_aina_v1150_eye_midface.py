#!/usr/bin/env python3
"""AINA v11.5 — absolute almond-eye / youthful midface sculpt.

Input is v11.4. The previous eye passes only added offsets and left the original
round GNM eye anatomy visually dominant. v11.5 instead finds both connected eye
shells per side, scales/recesses them as one globe group, and sets the eyelid rim
toward an explicit long-almond curve. It also performs a stronger but bounded
nose/lip/chin/skull polish against the approved AINA visual identity.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from PIL import Image
import fit_aina_v101 as core
from gnm.shape import gnm_numpy,gnm_landmarks

GNM_TO_STANDARD=np.array([0,1,6,5,4,3,2,7,8,9,10,11,12,13,14,15,16,*range(17,68)],dtype=np.int64)

def smoothstep01(x):x=np.clip(x,0.,1.);return x*x*(3.-2.*x)
def gauss(p,cx,cy,rx,ry,pow=2.):
    q=((p[:,0]-cx)/max(rx,1e-9))**2+((p[:,1]-cy)/max(ry,1e-9))**2
    return np.exp(-.5*np.power(q,pow/2.))
def clampv(d,cap):
    o=d.copy();n=np.linalg.norm(o,axis=1);m=n>cap
    if np.any(m):o[m]*=(cap/n[m])[:,None]
    return o
def controls(v,idx,bw):return (v[idx]*bw[...,None]).sum(-2)
def fw():
    w=np.ones(68);w[:17]=2.7;w[17:27]=1.;w[27:36]=3.;w[36:48]=4.5;w[48:60]=3.5;w[60:68]=2.;return w

def global_components(nv,faces):
    e=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);a=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(nv,nv));a=(a+a.T).tocsr();n,labels=connected_components(a,directed=False);return [np.flatnonzero(labels==i) for i in range(n)]

def eye_groups(fv,R,faces,cp):
    comps=global_components(len(fv),faces);small=[]
    for ids in comps:
        if 330<=len(ids)<=430:
            cc=(fv[ids]@R.T).mean(0);small.append((ids,cc))
    groups=[]
    for eye_cp in (cp[36:42],cp[42:48]):
        target=eye_cp.mean(0);rank=sorted(small,key=lambda z:float(np.linalg.norm(z[1][:2]-target[:2])))
        chosen=[]
        for ids,cc in rank:
            if np.linalg.norm(cc[:2]-target[:2])<.018:chosen.append(ids)
        if not chosen:
            chosen=[rank[0][0]]
        groups.append(np.unique(np.concatenate(chosen)))
    return groups

def explicit_almond_field(p,skin_world,eye_world,eye_cp):
    tree3=cKDTree(eye_world);d3,_=tree3.query(skin_world,k=1)
    rim=np.flatnonzero(d3<.00185)
    if len(rim)<50:rim=np.argsort(d3)[:130]
    rp=p[rim];x0=float(eye_cp[:,0].mean());cy=float(eye_cp[:,1].mean())
    lo,hi=np.percentile(rp[:,0],[2,98]);current_half=max(.5*(hi-lo),.011);target_half=max(current_half*1.10,.0150)
    # Approved art MediaPipe contour is about 2.5:1 width:height. The target
    # curve therefore preserves a meaningful lower lid instead of collapsing
    # into the over-thin slit produced by the first v11.5 draft.
    side=-1. if x0<0 else 1.
    handle=np.zeros((len(rim),3),np.float64)
    for k,i in enumerate(rim):
        rel=float(p[i,0]-x0);u=float(np.clip(rel/max(current_half,1e-9),-1,1));outer=np.clip(side*u,0.,1.);inner=np.clip(-side*u,0.,1.)
        uu=float(np.clip(rel/target_half,-1,1));arch=max(0.,1.-uu*uu)
        upper=p[i,1]<=cy
        if upper:
            ydes=cy-.00680*(arch**.58)-.00085*smoothstep01(outer)+.00012*smoothstep01(inner)
        else:
            ydes=cy+.00480*(arch**.68)-.00055*smoothstep01(outer)+.00005*smoothstep01(inner)
        xdes=x0+rel*(target_half/current_half)
        handle[k,0]=.88*(xdes-p[i,0]);handle[k,1]=.92*(ydes-p[i,1]);handle[k,2]=-.00010*(arch**.6)
    handle=clampv(handle,.0058)
    t=cKDTree(rp[:,:2]);dd,near=t.query(p[:,:2],k=1);w=np.exp(-.5*(dd/.0047)**4);rz=float(np.median(rp[:,2]));w*=np.exp(-.5*((p[:,2]-rz)/.015)**4)
    return handle[near]*w[:,None],{'rim_vertices':int(len(rim)),'current_half_width_m':current_half,'target_half_width_m':target_half,'target_aspect_ratio':2.5,'max_handle_m':float(np.max(np.linalg.norm(handle,axis=1)))}

def area(v,f):
    t=v[f];return .5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
def compare(a,b,o):
    x=Image.open(a).convert('RGB');y=Image.open(b).convert('RGB');H=max(x.height,y.height);xw=int(x.width*H/x.height);yw=int(y.width*H/y.height);s=Image.new('RGB',(xw+yw,H),'white');s.paste(x.resize((xw,H)),(0,0));s.paste(y.resize((yw,H)),(xw,0));s.save(o)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-full',type=Path,required=True);ap.add_argument('--base-skin',type=Path,required=True);ap.add_argument('--front',type=Path,required=True);ap.add_argument('--front-landmarks',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_v1150'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);qa=args.out/'QA';qa.mkdir(exist_ok=True)
    full=trimesh.load(args.base_full,process=False,maintain_order=True);skin=trimesh.load(args.base_skin,process=False,maintain_order=True);fv=np.asarray(full.vertices,np.float64);ff=np.asarray(full.faces,np.int64);sv=np.asarray(skin.vertices,np.float64);sf=np.asarray(skin.faces,np.int64);ns=len(sv)
    if not np.allclose(fv[:ns],sv,atol=2e-6):raise RuntimeError('skin prefix mismatch')
    g=gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,variant=gnm_numpy.GNMVariant.HEAD);cfg=gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68);idx=np.asarray(cfg.indices,np.int64)[GNM_TO_STANDARD];bw=np.asarray(cfg.weights,np.float64)[GNM_TO_STANDARD]
    cpw=controls(fv,idx,bw);ref=core.load_image_rgb(args.front);tpx=np.asarray(json.loads(args.front_landmarks.read_text())['landmarks_xy'],np.float32);tn=core.normalize_target(tpx,ref.shape);R,sc,tr=core.scaled_ortho_init(cpw,tn,fw());pfull=fv@R.T;p=sv@R.T;cp=cpw@R.T
    raw=np.zeros_like(p);mid=float(cp[27:36,0].mean());eye_y=float(cp[36:48,1].mean());mc=cp[48:60,:2].mean(0);chin=cp[8,:2];zface=float(np.median(cp[:,2]));fg=np.exp(-.5*((p[:,2]-zface)/.082)**4)

    groups=eye_groups(fv,R,ff,cp);eye_stats={};pfull2=pfull.copy()
    for i,ids in enumerate(groups):
        center=pfull[ids].mean(0);q=pfull[ids]-center;q[:,0]*=.925;q[:,1]*=.925;q[:,2]*=.90;pfull2[ids]=center+q;pfull2[ids,2]+=.00125
        ew=fv[ids]
        d,st=explicit_almond_field(p,sv,ew,cp[36:42] if i==0 else cp[42:48]);raw+=d;st['eye_component_vertices']=int(len(ids));eye_stats[str(i)]=st

    for ex in (float(cp[36:42,0].mean()),float(cp[42:48,0].mean())):
        brow=gauss(p,ex,eye_y-.032,.040,.026,2.2)*fg;under=gauss(p,ex,eye_y+.018,.038,.023,2.2)*fg
        raw[:,2]+=.0033*brow;raw[:,1]-=.00040*brow;raw[:,2]+=.00105*under
    gl=gauss(p,mid,eye_y-.027,.030,.035,2.2)*fg;raw[:,2]+=.0027*gl

    nc=cp[27:36,:2].mean(0);nw=max(float(np.linalg.norm(cp[31,:2]-cp[35,:2])),.012);nose=gauss(p,float(nc[0]),float(nc[1]),1.70*nw,.044,2.3)*fg;lower=smoothstep01((p[:,1]-(float(nc[1])-.020))/.052);bridge=gauss(p,float(nc[0]),float(nc[1])-.020,1.05*nw,.038,2.3)*fg
    raw[:,0]+=((float(nc[0])+(p[:,0]-float(nc[0]))*.68)-p[:,0])*nose*lower;raw[:,0]+=((float(nc[0])+(p[:,0]-float(nc[0]))*.82)-p[:,0])*bridge
    raw[:,2]+=.0034*nose;tip=gauss(p,float(cp[33,0]),float(cp[33,1]),.014,.015,2.5)*fg;raw[:,2]+=.0016*tip;raw[:,1]-=.0012*tip

    mw=max(float(np.linalg.norm(cp[48,:2]-cp[54,:2])),.025);lip=gauss(p,float(mc[0]),float(mc[1]),.76*mw,.020,2.4)*fg
    raw[:,0]+=((float(mc[0])+(p[:,0]-float(mc[0]))*1.105)-p[:,0])*lip
    up=gauss(p,float(mc[0]),float(mc[1])-.004,.61*mw,.0105,2.5)*fg;lo=gauss(p,float(mc[0]),float(mc[1])+.005,.64*mw,.012,2.5)*fg;cupid=gauss(p,float(mc[0]),float(mc[1])-.006,.010,.0062,2.5)*fg
    raw[:,2]-=.00125*up+.00195*lo;raw[:,1]-=.00070*cupid;raw[:,1]+=.00025*lo

    ch=gauss(p,float(chin[0]),float(chin[1]),.043,.036,2.3)*fg;raw[:,0]+=((float(chin[0])+(p[:,0]-float(chin[0]))*.90)-p[:,0])*ch;raw[:,1]-=.00155*ch;raw[:,2]+=.00045*ch

    upper=smoothstep01((eye_y-.038-p[:,1])/.090);side=smoothstep01((np.abs(p[:,0]-mid)-.020)/.060);skull=upper*(.45+.55*side)
    raw[:,0]+=((mid+(p[:,0]-mid)*.925)-p[:,0])*skull;raw[:,1]+=.0048*upper

    raw=clampv(raw,.0085);pskin=p+raw;sv2=pskin@R
    fv2=pfull2@R;fv2[:ns]=sv2
    a0=area(sv,sf);a1=area(sv2,sf);rat=a1/np.maximum(a0,1e-12);q01=float(np.percentile(rat,1));q99=float(np.percentile(rat,99));mx=float(np.max(np.linalg.norm(fv2-fv,axis=1)))
    if q01<.10 or q99>5.5:raise RuntimeError(f'mesh quality fail p01={q01} p99={q99}')
    fo=trimesh.Trimesh(vertices=fv2,faces=ff,process=False);so=trimesh.Trimesh(vertices=sv2,faces=sf,process=False);fo.export(args.out/'AINA_FACE_MASTER_GNM_v11.5_EYE_MIDFACE.obj');fo.export(args.out/'AINA_FACE_MASTER_GNM_v11.5_EYE_MIDFACE.glb');so.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v11.5_EYE_MIDFACE.obj');so.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v11.5_EYE_MIDFACE.glb');so.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v11.5_EYE_MIDFACE.ply')
    cf=controls(fv2,idx,bw);pred=core.project_np(cf,R,sc,tr);e=np.linalg.norm(pred-tn,axis=1);core.save_overlay(ref,tpx,pred,qa/'AINA_front_overlay_v11.5.png','AINA v11.5 eye + midface')
    views=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        path=qa/f'AINA_FULL_CLAY_{label}_v11.5.png';core.render_mesh_ortho(fv2,ff,R,yaw,path,f'AINA v11.5 {label}');views.append(path)
    ims=[Image.open(x).convert('RGB') for x in views];H=max(i.height for i in ims);W=max(i.width for i in ims);sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_FULL_CLAY_5VIEW_v11.5.png');compare(args.front,qa/'AINA_FULL_CLAY_front_v11.5.png',qa/'AINA_REFERENCE_VS_ACTUAL_FRONT_v11.5.png')
    rep={'version':'AINA Face Master v11.5 Eye/Midface','base':'v11.4 visual proportions','topology_changed':False,'max_additional_shift_m':mx,'area_ratio_p01':q01,'area_ratio_p99':q99,'front_fixed_rmse':float(np.sqrt(np.mean(e**2))),'eye_stats':eye_stats,'identity_lock':False,'acceptance':'actual visual clay only'};(args.out/'AINA_v11.5_REPORT.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
