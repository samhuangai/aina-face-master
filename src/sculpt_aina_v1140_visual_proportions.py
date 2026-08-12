#!/usr/bin/env python3
"""AINA v11.4 — visual-proportion sculpt.

Built from the stable v11.2 TPS base, not cumulatively from v11.3. This pass is
judged by the approved AINA art rather than sparse RMSE. It narrows lateral
cheek/jaw/chin and neck mass, creates a long almond eye fissure (upper-lid-led,
minimal lower-lid drop), makes the nose more delicate, and gives the lips a
wider/fuller youthful form. All fields are analytic and bounded.
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

def smoothstep01(x):x=np.clip(x,0.,1.);return x*x*(3.-2.*x)
def gauss(p,cx,cy,rx,ry,pow=2.):
    q=((p[:,0]-cx)/max(rx,1e-9))**2+((p[:,1]-cy)/max(ry,1e-9))**2;return np.exp(-.5*np.power(q,pow/2.))
def clampv(d,cap):
    o=d.copy();n=np.linalg.norm(o,axis=1);m=n>cap
    if np.any(m):o[m]*=(cap/n[m])[:,None]
    return o
def controls(v,idx,bw):return (v[idx]*bw[...,None]).sum(-2)
def fw():
    w=np.ones(68);w[:17]=2.8;w[17:27]=1.;w[27:36]=3.;w[36:48]=4.5;w[48:60]=3.5;w[60:68]=2.;return w

def eye_components(mesh,R,cp):
    comps=[]
    for c in mesh.split(only_watertight=False):
        if 300<=len(c.vertices)<=500:comps.append((c,np.asarray(c.vertices).mean(0)@R.T))
    targets=[cp[36:42].mean(0),cp[42:48].mean(0)];out=[];used=set()
    for t in targets:
        for k,item in sorted(enumerate(comps),key=lambda z:float(np.linalg.norm(z[1][1][:2]-t[:2]))):
            if k not in used:used.add(k);out.append(item);break
    if len(out)!=2:raise RuntimeError('eye component discovery failed')
    return out

def almond_eye(p,skin_world,eye,eye_cp,mid):
    ev=np.asarray(eye.vertices,np.float64);d3,_=cKDTree(ev).query(skin_world,k=1);rim=np.where(d3<.00145)[0]
    if len(rim)<25:rim=np.argsort(d3)[:80]
    rp=p[rim];x0=float(eye_cp[:,0].mean());cy=float(eye_cp[:,1].mean());xmin,xmax=np.percentile(rp[:,0],[3,97]);half=max(.5*(xmax-xmin),.008);side=-1. if x0<mid else 1.;h=np.zeros((len(rim),3))
    for k,i in enumerate(rim):
        u=float(np.clip((p[i,0]-x0)/half,-1,1));arch=math.sqrt(max(0.,1-u*u));outer=np.clip(side*u,0.,1.);inner=np.clip(-side*u,0.,1.);upper=p[i,1]<=cy
        # Target is a long almond: most opening comes from the upper lid; lower
        # lid stays nearly straight and outer corner rises.
        if upper:h[k,1]=-0.00175*arch-0.00100*smoothstep01(outer)+0.00005*smoothstep01(inner)
        else:h[k,1]=+0.00018*arch-0.00028*smoothstep01(outer)+0.00002*smoothstep01(inner)
        # ~5% fissure widening plus a little extra outer-tail reach.
        h[k,0]=0.050*(p[i,0]-x0)+side*0.00028*smoothstep01(outer)
        h[k,2]=-0.00015*arch
    tree=cKDTree(rp[:,:2]);dd,near=tree.query(p[:,:2],k=1);w=np.exp(-.5*(dd/.0048)**4);rz=float(np.median(rp[:,2]));w*=np.exp(-.5*((p[:,2]-rz)/.013)**4)
    return h[near]*w[:,None],{'rim':int(len(rim)),'max_handle_m':float(np.max(np.linalg.norm(h,axis=1)))}
def area(v,f):
    t=v[f];return .5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
def compare(a,b,o):
    x=Image.open(a).convert('RGB');y=Image.open(b).convert('RGB');H=max(x.height,y.height);xw=int(x.width*H/x.height);yw=int(y.width*H/y.height);s=Image.new('RGB',(xw+yw,H),'white');s.paste(x.resize((xw,H)),(0,0));s.paste(y.resize((yw,H)),(xw,0));s.save(o)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-full',type=Path,required=True);ap.add_argument('--base-skin',type=Path,required=True);ap.add_argument('--front',type=Path,required=True);ap.add_argument('--front-landmarks',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_v1140'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);qa=args.out/'QA';qa.mkdir(exist_ok=True)
    full=trimesh.load(args.base_full,process=False,maintain_order=True);skin=trimesh.load(args.base_skin,process=False,maintain_order=True);fv=np.asarray(full.vertices,np.float64);ff=np.asarray(full.faces,np.int64);sv=np.asarray(skin.vertices,np.float64);sf=np.asarray(skin.faces,np.int64);ns=len(sv)
    if not np.allclose(fv[:ns],sv,atol=2e-6):raise RuntimeError('skin prefix mismatch')
    g=gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,variant=gnm_numpy.GNMVariant.HEAD);cfg=gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68);idx=np.asarray(cfg.indices,np.int64)[GNM_TO_STANDARD];bw=np.asarray(cfg.weights,np.float64)[GNM_TO_STANDARD]
    cpw=controls(fv,idx,bw);ref=core.load_image_rgb(args.front);tpx=np.asarray(json.loads(args.front_landmarks.read_text())['landmarks_xy'],np.float32);tn=core.normalize_target(tpx,ref.shape);R,sc,tr=core.scaled_ortho_init(cpw,tn,fw());p=sv@R.T;cp=cpw@R.T;raw=np.zeros_like(p)
    mid=float(cp[27:36,0].mean());eye_y=float(cp[36:48,1].mean());el=float(cp[36:42,0].mean());er=float(cp[42:48,0].mean());mc=cp[48:60,:2].mean(0);chin=cp[8,:2];zface=float(np.median(cp[:,2]));front_gate=np.exp(-.5*((p[:,2]-zface)/.082)**4)

    # 1. Long almond eyes.
    est={}
    for i,(eye,_) in enumerate(eye_components(full,R,cp)):
        d,s=almond_eye(p,sv,eye,cp[36:42] if i==0 else cp[42:48],mid);raw+=d;est[str(i)]=s

    # 2. Slim lateral face while preserving central eye/nose/mouth spacing.
    ax=np.abs(p[:,0]-mid)
    side=smoothstep01((ax-.038)/.034)
    face_band=np.exp(-.5*((p[:,1]-(eye_y+.045))/.105)**4)
    lateral=side*face_band*(.55+.45*front_gate)
    raw[:,0]+=((mid+(p[:,0]-mid)*.895)-p[:,0])*lateral

    # 3. Stronger V lower face/chin.
    lower=smoothstep01((p[:,1]-(float(mc[1])-.008))/.072)
    lower_side=smoothstep01((ax-.022)/.040)*lower*(.55+.45*front_gate)
    raw[:,0]+=((mid+(p[:,0]-mid)*.885)-p[:,0])*lower_side
    ch=gauss(p,float(chin[0]),float(chin[1]),.042,.035,2.2)*front_gate
    raw[:,0]+=((float(chin[0])+(p[:,0]-float(chin[0]))*.875)-p[:,0])*ch;raw[:,1]-=.0011*ch;raw[:,2]+=.00045*ch

    # 4. Narrower frontal forehead/temples; back skull largely unchanged by front gate.
    forehead=gauss(p,mid,eye_y-.076,.092,.090,2.0)*front_gate
    fside=smoothstep01((ax-.030)/.050)
    raw[:,0]+=((mid+(p[:,0]-mid)*.900)-p[:,0])*forehead*fside

    # 5. Soft brow/orbit with less masculine ridge.
    for ex in (el,er):
        brow=gauss(p,ex,eye_y-.032,.039,.026,2.2)*front_gate;under=gauss(p,ex,eye_y+.018,.037,.023,2.2)*front_gate
        raw[:,2]+=.0036*brow;raw[:,1]-=.00055*brow;raw[:,2]-=.00085*under
    gl=gauss(p,mid,eye_y-.027,.030,.034,2.2)*front_gate;raw[:,2]+=.0025*gl

    # 6. Delicate narrow nose: alae ~22% narrower locally, bridge ~10% narrower.
    nc=cp[27:36,:2].mean(0);nw=max(float(np.linalg.norm(cp[31,:2]-cp[35,:2])),.012)
    nose=gauss(p,float(nc[0]),float(nc[1]),1.65*nw,.043,2.3)*front_gate;low=smoothstep01((p[:,1]-(float(nc[1])-.020))/.050);bridge=gauss(p,float(nc[0]),float(nc[1])-.020,1.0*nw,.035,2.3)*front_gate
    raw[:,0]+=((float(nc[0])+(p[:,0]-float(nc[0]))*.78)-p[:,0])*nose*low;raw[:,0]+=((float(nc[0])+(p[:,0]-float(nc[0]))*.90)-p[:,0])*bridge
    raw[:,2]+=.0027*nose;tip=gauss(p,float(cp[33,0]),float(cp[33,1]),.014,.014,2.5)*front_gate;raw[:,2]+=.0010*tip;raw[:,1]-=.00055*tip

    # 7. Slightly wider, much fuller soft lips.
    mw=max(float(np.linalg.norm(cp[48,:2]-cp[54,:2])),.025);lip=gauss(p,float(mc[0]),float(mc[1]),.73*mw,.019,2.4)*front_gate
    raw[:,0]+=((float(mc[0])+(p[:,0]-float(mc[0]))*1.085)-p[:,0])*lip
    up=gauss(p,float(mc[0]),float(mc[1])-.004,.60*mw,.010,2.5)*front_gate;lo=gauss(p,float(mc[0]),float(mc[1])+.005,.62*mw,.0115,2.5)*front_gate;cupid=gauss(p,float(mc[0]),float(mc[1])-.006,.010,.006,2.5)*front_gate
    raw[:,2]-=.00105*up+.00175*lo;raw[:,1]-=.00055*cupid;raw[:,1]+=.00028*lo

    # 8. High apple cheeks forward, but side cheek mass back.
    cheek_y=.55*eye_y+.45*float(mc[1])
    for ex in (el,er):
        apple=gauss(p,ex,cheek_y,.032,.030,2.3)*front_gate;raw[:,2]-=.0017*apple
    sidecheek=lateral*gauss(p,mid,float(mc[1])-.010,.095,.065,2.0);raw[:,2]+=.0011*sidecheek

    # 9. Slender neck below chin. This only moves skin; facial components stay fixed.
    neck=smoothstep01((p[:,1]-(float(chin[1])+.018))/.070);neck*=1.-front_gate*.25
    raw[:,0]+=((mid+(p[:,0]-mid)*.84)-p[:,0])*neck

    raw=clampv(raw,.0115);p2=p+raw;sv2=p2@R;fv2=fv.copy();fv2[:ns]=sv2
    a0=area(sv,sf);a1=area(sv2,sf);rat=a1/np.maximum(a0,1e-12);q01=float(np.percentile(rat,1));q99=float(np.percentile(rat,99));mx=float(np.max(np.linalg.norm(sv2-sv,axis=1)))
    if q01<.12 or q99>5.:raise RuntimeError(f'quality fail p01={q01}, p99={q99}')
    fo=trimesh.Trimesh(vertices=fv2,faces=ff,process=False);so=trimesh.Trimesh(vertices=sv2,faces=sf,process=False);fo.export(args.out/'AINA_FACE_MASTER_GNM_v11.4_VISUAL_PROPORTIONS.obj');fo.export(args.out/'AINA_FACE_MASTER_GNM_v11.4_VISUAL_PROPORTIONS.glb');so.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v11.4_VISUAL_PROPORTIONS.obj');so.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v11.4_VISUAL_PROPORTIONS.glb');so.export(args.out/'AINA_FACE_MASTER_SKIN_CLAY_v11.4_VISUAL_PROPORTIONS.ply')
    cf=controls(fv2,idx,bw);pred=core.project_np(cf,R,sc,tr);e=np.linalg.norm(pred-tn,axis=1);core.save_overlay(ref,tpx,pred,qa/'AINA_front_overlay_v11.4.png','AINA v11.4 visual proportions')
    views=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        path=qa/f'AINA_FULL_CLAY_{label}_v11.4.png';core.render_mesh_ortho(fv2,ff,R,yaw,path,f'AINA v11.4 {label}');views.append(path)
    ims=[Image.open(x).convert('RGB') for x in views];H=max(i.height for i in ims);W=max(i.width for i in ims);sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_FULL_CLAY_5VIEW_v11.4.png');compare(args.front,qa/'AINA_FULL_CLAY_front_v11.4.png',qa/'AINA_REFERENCE_VS_ACTUAL_FRONT_v11.4.png')
    report={'version':'AINA Face Master v11.4 Visual Proportions','base':'v11.2 TPS stable topology','topology_changed':False,'max_additional_shift_m':mx,'area_ratio_p01':q01,'area_ratio_p99':q99,'front_fixed_rmse':float(np.sqrt(np.mean(e**2))),'eye_stats':est,'identity_lock':False,'acceptance':'visual match only'};(args.out/'AINA_v11.4_REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
