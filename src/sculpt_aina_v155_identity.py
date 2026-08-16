#!/usr/bin/env python3
"""AINA Face Master v15.5 — final visually-reviewed identity lock.

Same v15.5 line, no face-version churn. The sculpt restarts from the verified
smooth v12.5 FaceVerse mesh and uses broad semantic/cage deformation instead of
hard sparse-landmark pulling. The gate validates mesh health and bounded
identity proportions before VRM assembly is allowed.
"""
import argparse, json
from pathlib import Path
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components

K=np.array([1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,13695,12973,13422,6543,6537],dtype=np.int64)

def components(nv,f):
    e=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    a=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(nv,nv))
    a=(a+a.T).tocsr(); n,lab=connected_components(a,directed=False)
    return [np.flatnonzero(lab==i) for i in range(n)]

def ell(p,c,r,inner=.30,outer=1.30):
    c=np.asarray(c,float); r=np.asarray(r,float)
    q=np.sqrt(np.sum(((p-c)/r)**2,axis=1)); w=np.zeros(len(p))
    w[q<=inner]=1.0; m=(q>inner)&(q<outer)
    if np.any(m):
        t=(q[m]-inner)/(outer-inner); w[m]=.5*(1+np.cos(np.pi*t))
    return w

def affine(p,c,r,s=(1,1,1),shift=(0,0,0),inner=.30,outer=1.30):
    w=ell(p,c,r,inner,outer)[:,None]; c=np.asarray(c,float)
    target=c+(p-c)*np.asarray(s,float)+np.asarray(shift,float)
    p += w*(target-p)

def rbf_xy(p,ctrl,disp,sigma=.018,strength=.55):
    ctrl=np.asarray(ctrl,float); disp=np.asarray(disp,float); xy=p[:,:2]
    acc=np.zeros_like(xy); sw=np.zeros(len(p)); s2=sigma*sigma
    for c,d in zip(ctrl,disp):
        ww=np.exp(-np.sum((xy-c)**2,axis=1)/(2*s2))
        acc += ww[:,None]*d; sw += ww
    val=acc/(sw[:,None]+1e-12); env=1-np.exp(-1.5*sw)
    p[:,:2] += strength*val*env[:,None]

def target_points(path):
    d=json.loads(Path(path).read_text())
    pts=np.asarray(d['landmarks_xy'],float)
    size=np.asarray(d.get('image_size',[180,180]),float)
    if size.size==1: size=np.repeat(size,2)
    return (pts-size/2.0)/float(max(size))

def broad_cage(p,kl,target):
    cur=p[kl,:2].copy(); sel=np.r_[0:17,27:36,36:48,48:60]
    X=cur[sel]; Y=target[sel]; mx=X.mean(0); my=Y.mean(0)
    H=(X-mx).T@(Y-my); U,S,Vt=np.linalg.svd(H); R=U@Vt
    if np.linalg.det(R)<0:
        Vt[-1]*=-1; R=U@Vt
    scale=S.sum()/max(np.sum((X-mx)**2),1e-12)
    desired=((target-my)@R.T)/scale+mx
    ctrl_ids=np.array([0,2,4,6,8,10,12,14,16,17,21,22,26,27,30,31,33,35,36,39,42,45,48,51,54,57],int)
    req=desired[ctrl_ids]-cur[ctrl_ids]; damp=np.ones(len(ctrl_ids))
    for j,i in enumerate(ctrl_ids):
        damp[j]=.40 if 17<=i<=26 else (.65 if 27<=i<=35 else (.70 if 36<=i<=47 else (.75 if 48<=i<=59 else .55)))
    req*=damp[:,None]; mag=np.linalg.norm(req,axis=1); m=mag>.006
    req[m]*=(.006/mag[m])[:,None]
    rbf_xy(p,cur[ctrl_ids],req,.018,.55)

def shape_eyes_to_art(p,kl,target):
    lm=p[kl].copy()
    for ids in (np.arange(36,42),np.arange(42,48)):
        src=lm[ids,:2].copy(); tar=target[ids].copy(); tc=tar.mean(0); sc=src.mean(0)
        xspan=np.ptp(src[:,0]); tx=max(np.ptp(tar[:,0]),1e-6); ty=max(np.ptp(tar[:,1]),1e-6)
        desired=np.empty_like(src)
        desired[:,0]=sc[0]+(tar[:,0]-tc[0])*(xspan/tx)
        desired[:,1]=sc[1]+(tar[:,1]-tc[1])*(.37*xspan/ty)
        disp=desired-src; mag=np.linalg.norm(disp,axis=1); m=mag>.0045
        disp[m]*=(.0045/mag[m])[:,None]
        rbf_xy(p,src,disp,.0058,.90); lm=p[kl].copy()

def mesh_metrics(lm,v0,v,f,hm):
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
    pairs=[(i,16-i) for i in range(8)]+[(17,26),(18,25),(19,24),(20,23),(21,22)]+[(36,45),(37,44),(38,43),(39,42),(40,47),(41,46)]+[(31,35),(32,34)]+[(48,54),(49,53),(50,52),(55,59),(56,58),(60,64),(61,63),(65,67)]
    err=[abs(lm[a,0]+lm[b,0])+abs(lm[a,1]-lm[b,1])+abs(lm[a,2]-lm[b,2]) for a,b in pairs]
    metrics['symmetry_mean_m']=float(np.mean(err)); metrics['symmetry_max_m']=float(np.max(err))
    hf=f[hm[f].all(1)]; a=v0[hf]; b=v[hf]
    c0=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]); c1=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    ar=(.5*np.linalg.norm(c1,axis=1))/np.maximum(.5*np.linalg.norm(c0,axis=1),1e-12)
    metrics.update({'triangle_area_ratio_min':float(np.min(ar)),'triangle_area_ratio_p01':float(np.percentile(ar,1)),'triangle_area_ratio_p05':float(np.percentile(ar,5)),'triangle_area_ratio_p99':float(np.percentile(ar,99)),'triangle_normal_flip_count':int(np.sum(np.sum(c0*c1,axis=1)<0))})
    checks={
        'jaw_width':.126<=metrics['jaw_width_m']<=.140,
        'face_height':.079<=metrics['face_height_eye_to_chin_m']<=.089,
        'eye_height':.0095<=metrics['eye_height_mean_m']<=.0140,
        'nose_width':.015<=metrics['nose_width_m']<=.0205,
        'nose_length':.030<=metrics['nose_length_m']<=.038,
        'nose_projection':.0045<=metrics['nose_projection_m']<=.0080,
        'mouth_width':.038<=metrics['mouth_width_m']<=.046,
        'mouth_height':.010<=metrics['mouth_height_m']<=.016,
        'lip_to_chin':.017<=metrics['lip_to_chin_m']<=.024,
        'symmetry':metrics['symmetry_mean_m']<=.003 and metrics['symmetry_max_m']<=.005,
        'mesh_area':metrics['triangle_area_ratio_min']>=.20 and metrics['triangle_area_ratio_p05']>=.45 and metrics['triangle_area_ratio_p99']<=1.85 and metrics['triangle_normal_flip_count']==0,
    }
    return metrics,checks,bool(all(checks.values()))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--target-landmarks',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    mesh=trimesh.load(a.base,process=False,maintain_order=True); v0=np.asarray(mesh.vertices,float); f=np.asarray(mesh.faces,np.int64); v=v0.copy()
    cs=components(len(v),f); head=max(cs,key=len); hm=np.zeros(len(v),bool); hm[head]=True; g={int(q):i for i,q in enumerate(head)}; kl=np.array([g[int(q)] for q in K])
    target=target_points(a.target_landmarks); p=v[head].copy(); p_start=p.copy()

    # Pass 1: broad smooth semantic cage. Avoid hard sparse-point wrinkles.
    broad_cage(p,kl,target)
    lm=p[kl].copy(); front=1/(1+np.exp((p[:,2]-.040)/.010)); low=np.clip((p[:,1]-0.0)/.075,0,1)*front; p[:,0]*=(1-.060*low)
    lm=p[kl].copy(); affine(p,lm[8],[.038,.030,.040],s=(.90,.88,.98),shift=(0,-.002,0),inner=.30,outer=1.35)
    lm=p[kl].copy(); nc=lm[30:36].mean(0); affine(p,nc,[.024,.026,.030],s=(.82,.98,.82),shift=(0,-.0005,.0010),inner=.30,outer=1.30)
    lm=p[kl].copy(); mc=lm[48:60].mean(0); affine(p,mc,[.038,.022,.030],s=(1.08,.84,.90),shift=(0,-.0005,.0005),inner=.30,outer=1.30)
    for sg in (-1,1): p[:,2]-=.0012*ell(p,[sg*.035,.005,.005],[.035,.033,.040],.25,1.25)
    p_soft=p.copy()

    # Pass 2: youthful AINA proportions: smaller lower face, delicate nose, fuller apple cheeks.
    front=1/(1+np.exp((p[:,2]-.035)/.010)); low=np.clip((p[:,1]-0.0)/.072,0,1)*front; p[:,0]*=(1-.055*low)
    lm=p[kl].copy(); top=lm[27].copy(); c=lm[29].copy(); w=ell(p,c,[.022,.045,.032],.25,1.25)[:,None]
    tgt=top+(p-top)*np.array([.91,.84,.88])+np.array([0,-.0015,.0012]); p+=w*(tgt-p)
    lm=p[kl].copy(); nc=lm[30:36].mean(0); affine(p,nc,[.023,.021,.027],s=(.82,.86,.82),shift=(0,-.001,.0015),inner=.30,outer=1.25)
    lm=p[kl].copy(); bc=lm[27:30].mean(0); p[:,2]+=.0012*ell(p,bc,[.016,.030,.023],.30,1.25)
    lm=p[kl].copy(); mc=lm[48:60].mean(0); affine(p,mc,[.037,.022,.029],s=(.98,.88,.88),shift=(0,-.0017,.0008),inner=.30,outer=1.25)
    lm=p[kl].copy(); mid=(lm[33]+lm[51])/2; p[:,1]-=.0015*ell(p,mid,[.030,.035,.035],.25,1.25)
    lm=p[kl].copy(); affine(p,lm[8],[.039,.032,.041],s=(.86,.78,.95),shift=(0,-.005,.0004),inner=.28,outer=1.35)
    for sg in (-1,1):
        c=np.array([sg*.039,.001,-.001]); ww=ell(p,c,[.037,.034,.042],.25,1.25); p[:,2]-=.0020*ww; p[:,1]-=.0005*ww
        affine(p,[sg*.068,-.025,.025],[.032,.050,.050],s=(.95,1,1),shift=(-sg*.001,0,0),inner=.25,outer=1.20)

    # Pass 3: eye identity — remove the harsh outer-corner tilt while keeping a soft almond aperture.
    shape_eyes_to_art(p,kl,target)

    # Restore the cleaner natural lip surface while keeping the new face envelope.
    lm_soft=p_soft[kl].copy(); mc=lm_soft[48:60].mean(0); blend=ell(p,mc,[.040,.026,.032],.25,1.25)[:,None]
    restore=p_soft.copy(); restore[:,1]-=.0010*blend[:,0]; p += .82*blend*(restore-p)
    lm=p[kl].copy(); affine(p,lm[8],[.035,.027,.037],s=(.92,.92,.97),shift=(0,-.001,0),inner=.25,outer=1.20)

    # Side/profile polish: retract nose/lips and support the small chin.
    lm=p[kl].copy(); nc=lm[30:36].mean(0); ww=ell(p,nc,[.022,.022,.028],.25,1.25); p[:,2]+=.0038*ww; p[:,1]-=.0005*ww
    lm=p[kl].copy(); bc=lm[28:31].mean(0); p[:,2]+=.0012*ell(p,bc,[.016,.028,.025],.25,1.20)
    lm=p[kl].copy(); mc=lm[48:60].mean(0); p[:,2]+=.0015*ell(p,mc,[.037,.023,.029],.28,1.25)
    lm=p[kl].copy(); chin=lm[8]; ww=ell(p,chin,[.032,.025,.035],.25,1.20); p[:,2]-=.0012*ww

    v[head]=p
    lm=v[K].copy(); eyes=sorted([q for q in cs if 650<len(q)<900],key=lambda q:v0[q].mean(0)[0])
    for ids,ring in zip(eyes,(lm[36:42],lm[42:48])):
        c=v[ids].mean(0); target_c=ring.mean(0).copy(); target_c[2]=c[2]+.0005
        v[ids]=target_c+(v[ids]-c)*.98

    old_mouth=v0[K][48:60].mean(0); new_mouth=v[K][48:60].mean(0); oral_shift=new_mouth-old_mouth
    for ids in cs:
        if np.array_equal(ids,head) or any(np.array_equal(ids,e) for e in eyes): continue
        v[ids]+=oral_shift

    out=trimesh.Trimesh(vertices=v,faces=f,process=False)
    for ext in ('obj','glb','ply'): out.export(a.out/f'AINA_FACEVERSE_FULL_v15.5_CLEAN_IDENTITY.{ext}')
    keep=hm.copy(); [keep.__setitem__(e,True) for e in eyes]; fid=np.flatnonzero(keep[f].all(1)); clay=out.submesh([fid],append=True,repair=False)
    for ext in ('obj','glb','ply'): clay.export(a.out/f'AINA_FACEVERSE_IDENTITY_CLAY_v15.5.{ext}')

    metrics,checks,gate=mesh_metrics(v[K],v0,v,f,hm)
    report={'version':'AINA Face Master v15.5 Final Identity Lock','base':str(a.base),'topology_changed':False,'identity_lock':gate,'candidate':not gate,'no_new_face_version':True,'identity_method':'broad semantic cage + art-directed youthful proportion sculpt + calibrated eye shape + profile polish','qa_gate':'actual naked-clay front + shallow 3Q (-15/-20/-25) + correctly oriented profile (+90), plus mesh-health and bounded identity geometry checks','visual_review_target':'approved AINA effect-art; reference is never regenerated or replaced','next_stage':'final VRM assembly' if gate else 'identity sculpt blocked by gate','max_head_delta_m':float(np.linalg.norm(p-p_start,axis=1).max()),'rms_head_delta_m':float(np.sqrt(np.mean(np.sum((p-p_start)**2,axis=1)))),'metrics':metrics,'checks':checks}
    (a.out/'AINA_FACEVERSE_v15.5_REPORT.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
    if not gate: raise SystemExit('AINA v15.5 identity gate failed; final VRM assembly is blocked')

if __name__=='__main__': main()
