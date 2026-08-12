#!/usr/bin/env python3
"""AINA v11.1 — direct 468-point target mesh from approved effect art.

Unlike GNM/BFM identity fitting, this stage does not ask a statistical identity
space to invent AINA. MediaPipe Face Mesh provides the same 468 semantic points
for each reference view. Frontal X/Y is retained directly; aligned 3/4/profile
estimates contribute depth. The official MediaPipe canonical topology is then
reused to create a concrete AINA target surface for downstream retopology.
"""
from __future__ import annotations

import json, math
from pathlib import Path
import cv2
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image
import mediapipe as mp

ROOT=Path.cwd()


def parse_faces(path: Path) -> np.ndarray:
    faces=[]; nv=0
    for line in path.read_text().splitlines():
        if line.startswith('v '): nv+=1
        elif line.startswith('f '):
            ids=[int(x.split('/')[0])-1 for x in line.split()[1:]]
            if len(ids)==3: faces.append(ids)
            elif len(ids)>3:
                for k in range(1,len(ids)-1): faces.append([ids[0],ids[k],ids[k+1]])
    if nv!=468: raise RuntimeError(f'canonical model has {nv} vertices, expected 468')
    return np.asarray(faces,np.int64)


def detect(path: Path):
    bgr=cv2.imread(str(path))
    if bgr is None: raise RuntimeError(f'missing {path}')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=False,min_detection_confidence=0.25) as fm:
        res=fm.process(rgb)
    if not res.multi_face_landmarks: return None,(w,h)
    lms=res.multi_face_landmarks[0].landmark
    if len(lms)!=468: raise RuntimeError(f'got {len(lms)} landmarks')
    v=np.array([[(x.x-.5)*w,-(x.y-.5)*h,-x.z*w] for x in lms],np.float64)
    return v,(w,h)


def similarity_align(moving: np.ndarray, fixed: np.ndarray):
    """Umeyama similarity: returns moving transformed into fixed coordinates."""
    X=moving;Y=fixed
    mx=X.mean(0);my=Y.mean(0);Xc=X-mx;Yc=Y-my
    C=(Yc.T@Xc)/len(X)
    U,S,Vt=np.linalg.svd(C);D=np.eye(3)
    if np.linalg.det(U@Vt)<0:D[-1,-1]=-1
    R=U@D@Vt
    var=np.mean(np.sum(Xc*Xc,axis=1)); scale=float(np.trace(np.diag(S)@D)/max(var,1e-12))
    t=my-scale*(R@mx)
    return (scale*(R@X.T)).T+t,{'scale':scale,'R':R.tolist(),'t':t.tolist()}


def normalize_metric(v):
    v=v.copy(); c=np.median(v,axis=0);v-=c
    h=float(np.percentile(v[:,1],99)-np.percentile(v[:,1],1));s=.180/max(h,1e-9)
    return v*s,s,c


def render(vertices,faces,yaw,path,title):
    a=math.radians(yaw);c=math.cos(a);s=math.sin(a)
    x=c*vertices[:,0]+s*vertices[:,2];z=-s*vertices[:,0]+c*vertices[:,2];p=np.c_[x,vertices[:,1],z]
    tri=p[faces];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
    order=np.argsort(tri[:,:,2].mean(1));tri2=p[faces[order],:2];nn=n[order]
    dif=np.clip(np.abs(nn[:,2]),0,1);side=np.clip(-.3*nn[:,0]+.18*nn[:,1]+.7*nn[:,2],0,1);it=np.clip(.68+.20*dif+.08*side,.55,.97)
    col=np.stack([it*.97,it*.98,it],1)
    xy=p[:,:2];lo=np.percentile(xy,1,0);hi=np.percentile(xy,99,0);ctr=.5*(lo+hi);ext=max(float((hi-lo).max()),1e-6)*.57
    fig,ax=plt.subplots(figsize=(5,5),dpi=200);ax.add_collection(PolyCollection(tri2,facecolors=col,edgecolors='none'))
    ax.set_xlim(ctr[0]-ext,ctr[0]+ext);ax.set_ylim(ctr[1]-ext,ctr[1]+ext);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10)
    fig.tight_layout(pad=.1);fig.savefig(path,bbox_inches='tight',pad_inches=.02);plt.close(fig)


def compare(ref,actual,out):
    a=Image.open(ref).convert('RGB');b=Image.open(actual).convert('RGB');H=max(a.height,b.height);aw=int(a.width*H/a.height);bw=int(b.width*H/b.height)
    s=Image.new('RGB',(aw+bw,H),'white');s.paste(a.resize((aw,H)),(0,0));s.paste(b.resize((bw,H)),(aw,0));s.save(out)


def main():
    out=ROOT/'output_mediapipe468';qa=out/'QA';out.mkdir(exist_ok=True);qa.mkdir(exist_ok=True)
    refs={'front':ROOT/'references/AINA_APPROVED_FRONT.jpg','three_quarter':ROOT/'references/AINA_APPROVED_3Q.jpg','side':ROOT/'references/AINA_APPROVED_SIDE.jpg'}
    detected={};sizes={}
    for k,p in refs.items():
        detected[k],sizes[k]=detect(p)
    if detected['front'] is None: raise RuntimeError('MediaPipe did not detect approved front AINA reference')

    base=detected['front'];aligned={'front':base};align_info={}
    for k in ('three_quarter','side'):
        if detected[k] is not None:
            aligned[k],align_info[k]=similarity_align(detected[k],base)

    # Preserve the approved front image exactly in X/Y. Use other views only to
    # improve depth. Front depth remains the majority prior because strict side
    # synthetic art can yield noisy hidden-side landmarks.
    target=base.copy();z=.68*base[:,2];ws=.68
    if 'three_quarter' in aligned: z+=.24*aligned['three_quarter'][:,2];ws+=.24
    if 'side' in aligned: z+=.08*aligned['side'][:,2];ws+=.08
    target[:,2]=z/ws

    faces=parse_faces(ROOT/'vendor/canonical_face_model.obj')
    metric,scale,center=normalize_metric(target)
    mesh=trimesh.Trimesh(vertices=metric,faces=faces,process=False)
    mesh.export(out/'AINA_FACE_TARGET_MEDIAPIPE468_v11.1.obj');mesh.export(out/'AINA_FACE_TARGET_MEDIAPIPE468_v11.1.glb');mesh.export(out/'AINA_FACE_TARGET_MEDIAPIPE468_v11.1.ply')
    np.save(out/'AINA_MEDIAPIPE468_VERTICES_RAW_v11.1.npy',target.astype(np.float32))

    paths=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p=qa/f'AINA_MP468_CLAY_{label}_v11.1.png';render(metric,faces,yaw,p,f'AINA MP468 v11.1 {label}');paths.append(p)
    ims=[Image.open(p).convert('RGB') for p in paths];H=max(x.height for x in ims);W=max(x.width for x in ims);sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W+(W-im.width)//2,(H-im.height)//2))
    sheet.save(qa/'AINA_MP468_CLAY_5VIEW_v11.1.png')
    compare(refs['front'],qa/'AINA_MP468_CLAY_front_v11.1.png',qa/'AINA_REFERENCE_VS_MP468_FRONT_v11.1.png')

    report={'version':'AINA Face Target v11.1 MediaPipe 468','vertices':468,'triangles':int(len(faces)),'front_detected':True,
      'three_quarter_detected':detected['three_quarter'] is not None,'side_detected':detected['side'] is not None,'depth_weights':{'front':.68,'three_quarter':.24,'side':.08},
      'alignments':align_info,'metric_scale':float(scale),'topology':'official MediaPipe canonical face topology','identity_lock':False,
      'next_gate':'If visual face target matches the approved AINA art, transfer this 468-point surface into the production full-head topology.'}
    (out/'AINA_MP468_v11.1_REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
