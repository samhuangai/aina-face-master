#!/usr/bin/env python3
"""AINA v13.3 — clean FaceVerse landmark-constrained Laplacian rebuild.

Starts from smooth v12.5 FaceVerse geometry, not from any v13.x manual sculpt.
The approved effect-art 68-point target is mapped to FaceVerse's native 68
landmark vertices after a weighted 2D similarity fit. Landmark residuals become
soft handles in a topology-wide Laplacian displacement solve on the largest
head shell. Z/depth is preserved in this pass except for a very low-frequency
nose/cheek/chin semantic correction. Eyeballs and oral components are then moved
only coherently with their nearby facial region.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree
from PIL import Image
import vtk


def components(nv, faces):
    e=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    a=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(nv,nv));a=(a+a.T).tocsr()
    n,lab=connected_components(a,directed=False)
    return [np.flatnonzero(lab==i) for i in range(n)]


def target68(path: Path):
    d=json.loads(path.read_text())
    w,h=d['image_size']; p=np.asarray(d['landmarks_xy'],np.float64)
    # FaceVerse canonical +Y goes down. Center/scale to an image-normalized space.
    return np.c_[(p[:,0]-w*.5)/max(w,h),(p[:,1]-h*.5)/max(w,h)]


def category_weights():
    w=np.ones(68,np.float64)
    w[0:17]=4.8      # oval / jaw
    w[17:27]=1.5     # brows
    w[27:36]=4.2     # nose
    w[36:48]=6.2     # eyes
    w[48:60]=5.2     # outer lips
    w[60:68]=2.2     # inner lips: weaker so neutral closure stays smooth
    return w


def similarity(x,y,w):
    ww=w[:,None]; sw=float(ww.sum())
    mx=(x*ww).sum(0)/sw; my=(y*ww).sum(0)/sw
    X=x-mx; Y=y-my
    # full 2D similarity including small rotation, solved weighted Procrustes
    H=(X*ww).T@Y
    U,S,Vt=np.linalg.svd(H); R=U@Vt
    if np.linalg.det(R)<0:
        Vt[-1]*=-1; R=U@Vt
    denom=float(np.sum(ww*(X*X)))
    s=float(S.sum()/max(denom,1e-12))
    pred=s*(X@R)+my
    return s,R,mx,my,pred


def build_adjacency(n, faces_local):
    adj=[set() for _ in range(n)]
    for a,b,c in faces_local:
        a=int(a);b=int(b);c=int(c)
        adj[a].update((b,c));adj[b].update((a,c));adj[c].update((a,b))
    return adj


def solve_laplacian(head_v, faces_local, control_local, desired_xy, cweights, anchor_mask):
    n=len(head_v); adj=build_adjacency(n,faces_local)
    rows=[]; cols=[]; data=[]; bx=[]; by=[]; row=0
    def add(entries, tx, ty, wt):
        nonlocal row
        for j,val in entries:
            rows.append(row);cols.append(j);data.append(val*wt)
        bx.append(tx*wt);by.append(ty*wt);row+=1
    # Smooth displacement field.
    lap_w=2.7
    for i,nbr in enumerate(adj):
        if not nbr: continue
        inv=1.0/len(nbr)
        add([(i,1.0)]+[(j,-inv) for j in nbr],0.,0.,lap_w)
    # Small zero-displacement prior everywhere.
    for i in range(n): add([(i,1.0)],0.,0.,.22)
    # Scalp/back/ears/neck anchors. These remain rigid in the 2D identity solve.
    for i in np.flatnonzero(anchor_mask): add([(int(i),1.0)],0.,0.,17.0)
    # Landmark handles.
    for li,(idx,tgt,w) in enumerate(zip(control_local,desired_xy,cweights)):
        d=tgt-head_v[idx,:2]
        cw=55.0*math.sqrt(float(w))
        add([(int(idx),1.0)],float(d[0]),float(d[1]),cw)
    A=sparse.coo_matrix((data,(rows,cols)),shape=(row,n)).tocsr()
    dx=lsqr(A,np.asarray(bx),atol=2e-8,btol=2e-8,iter_lim=1000)[0]
    dy=lsqr(A,np.asarray(by),atol=2e-8,btol=2e-8,iter_lim=1000)[0]
    return np.c_[dx,dy]


def render_vtk(objpath, outpath, yaw=0, size=900):
    r=vtk.vtkOBJReader();r.SetFileName(str(objpath));r.Update()
    norm=vtk.vtkPolyDataNormals();norm.SetInputConnection(r.GetOutputPort());norm.ComputePointNormalsOn();norm.SplittingOff();norm.Update()
    mapper=vtk.vtkPolyDataMapper();mapper.SetInputConnection(norm.GetOutputPort());mapper.ScalarVisibilityOff()
    act=vtk.vtkActor();act.SetMapper(mapper);act.GetProperty().SetColor(.72,.73,.75);act.GetProperty().SetAmbient(.25);act.GetProperty().SetDiffuse(.75);act.GetProperty().SetSpecular(.04);act.RotateY(yaw)
    ren=vtk.vtkRenderer();ren.SetBackground(1,1,1);ren.AddActor(act)
    win=vtk.vtkRenderWindow();win.SetOffScreenRendering(1);win.SetSize(size,size);win.AddRenderer(ren);win.SetMultiSamples(4)
    b=act.GetBounds();cx=(b[0]+b[1])/2;cy=(b[2]+b[3])/2;cz=(b[4]+b[5])/2;scale=max(b[1]-b[0],b[3]-b[2])*.58
    cam=ren.GetActiveCamera();cam.SetFocalPoint(cx,cy,cz);cam.SetPosition(cx,cy,cz-.8);cam.SetViewUp(0,-1,0);cam.ParallelProjectionOn();cam.SetParallelScale(scale);ren.ResetCameraClippingRange()
    li=vtk.vtkLight();li.SetLightTypeToCameraLight();li.SetIntensity(.88);ren.AddLight(li)
    fill=vtk.vtkLight();fill.SetPosition(cx-.5,cy-.5,cz-.7);fill.SetFocalPoint(cx,cy,cz);fill.SetIntensity(.25);ren.AddLight(fill)
    win.Render();w=vtk.vtkWindowToImageFilter();w.SetInput(win);w.SetInputBufferTypeToRGB();w.ReadFrontBufferOff();w.Update();wr=vtk.vtkPNGWriter();wr.SetFileName(str(outpath));wr.SetInputConnection(w.GetOutputPort());wr.Write();win.Finalize()


def compare(refpath, renderpath, outpath):
    ref=Image.open(refpath).convert('RGB'); im=Image.open(renderpath).convert('RGB')
    H=max(ref.height,im.height); rw=int(ref.width*H/ref.height); iw=int(im.width*H/im.height)
    c=Image.new('RGB',(rw+iw,H),'white');c.paste(ref.resize((rw,H)),(0,0));c.paste(im.resize((iw,H)),(rw,0));c.save(outpath)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-full',type=Path,required=True);ap.add_argument('--faceverse-data',type=Path,required=True);ap.add_argument('--target68',type=Path,required=True);ap.add_argument('--front-ref',type=Path,required=True);ap.add_argument('--q3-ref',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('output_v133'));args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);qa=args.out/'QA';qa.mkdir(exist_ok=True)
    mesh=trimesh.load(args.base_full,process=False,maintain_order=True);v=np.asarray(mesh.vertices,np.float64);f=np.asarray(mesh.faces,np.int64);v0=v.copy()
    comps=components(len(v),f);head=max(comps,key=len);hm=np.zeros(len(v),bool);hm[head]=True;hfi=np.flatnonzero(hm[f].all(1));hf_global=f[hfi];g2l={int(g):i for i,g in enumerate(head)};hf=np.vectorize(g2l.get)(hf_global);hv=v[head].copy()
    fvd=np.load(args.faceverse_data,allow_pickle=True).item();k68=np.asarray(fvd['keypoints_68']).reshape(-1).astype(np.int64)
    if len(k68)<68: raise RuntimeError(f'FaceVerse keypoints_68 has {len(k68)} points')
    k68=k68[:68]
    if not np.all(hm[k68]): raise RuntimeError('Some 68 controls are not on largest head shell')
    kl=np.asarray([g2l[int(x)] for x in k68],np.int64)
    tgt=target68(args.target68); w=category_weights(); cur=hv[kl,:2]
    s,R,mx,my,pred=similarity(cur,tgt,w)
    desired=((tgt-my)@R.T)/max(s,1e-12)+mx
    initial=np.linalg.norm(pred-tgt,axis=1)
    # Keep desired movement bounded to prevent stylized target from destroying anatomy.
    req=desired-cur; reqn=np.linalg.norm(req,axis=1); cap=.012
    bad=reqn>cap
    if np.any(bad): req[bad]*=(cap/reqn[bad])[:,None]
    desired=cur+req
    # Anchors: rear skull, top scalp, neck boundary, and far ears.
    zface=float(np.median(hv[kl,2])); anchor=(hv[:,2]>zface+.055) | (hv[:,1]<np.percentile(hv[:,1],7)) | (np.abs(hv[:,0])>.074)
    disp2=solve_laplacian(hv,hf,kl,desired,w,anchor)
    dn=np.linalg.norm(disp2,axis=1);capall=.014;bad=dn>capall
    if np.any(bad): disp2[bad]*=(capall/dn[bad])[:,None]
    hv[:,:2]+=disp2
    # Low-frequency depth correction only: smaller nose projection, gentle apple cheeks, small rounded chin.
    def ell(cx,cy,cz,rx,ry,rz):
        q=((hv[:,0]-cx)/rx)**2+((hv[:,1]-cy)/ry)**2+((hv[:,2]-cz)/rz)**2
        return np.exp(-.5*q)
    # Locate features from corrected landmarks rather than hardcoded coordinates.
    lm=hv[kl]; nose_c=lm[27:36].mean(0); mouth_c=lm[48:60].mean(0); chin_c=lm[8]
    nose=ell(nose_c[0],nose_c[1],nose_c[2],.028,.035,.035);hv[:,2]+=0.0018*nose  # +Z is back in current metric convention
    for ids in ([36,39,31,48],[42,45,35,54]):
        c=lm[np.asarray(ids)].mean(0);cheek=ell(c[0],c[1]+.003,c[2],.030,.028,.035);hv[:,2]-=.00065*cheek
    chin=ell(chin_c[0],chin_c[1],chin_c[2],.032,.022,.028);hv[:,2]+=.00035*chin
    v[head]=hv
    # Move eyeballs with local eye-center displacement; move oral components with mouth center displacement.
    eye_comps=[];oral=[]
    for ids in comps:
        if np.array_equal(ids,head):continue
        c0=v0[ids].mean(0)
        if 650<len(ids)<900 and abs(c0[0])>.015 and c0[1]<.01: eye_comps.append(ids)
        else: oral.append(ids)
    eye_comps=sorted(eye_comps,key=lambda ids:v0[ids].mean(0)[0])
    old_lm=v0[k68];new_lm=v[k68]
    eye_shift=[new_lm[36:42].mean(0)-old_lm[36:42].mean(0),new_lm[42:48].mean(0)-old_lm[42:48].mean(0)]
    for ids,sh in zip(eye_comps,eye_shift):v[ids]+=sh
    mouth_shift=new_lm[48:60].mean(0)-old_lm[48:60].mean(0)
    for ids in oral:v[ids]+=mouth_shift
    out=trimesh.Trimesh(vertices=v,faces=f,process=False)
    for ext in ('obj','glb','ply'):out.export(args.out/f'AINA_FACEVERSE_FULL_v13.3.{ext}')
    # head + eyes identity clay
    hmesh=trimesh.Trimesh(vertices=v[head],faces=hf,process=False);parts=[hmesh]
    for ids in eye_comps:
        mm=np.zeros(len(v),bool);mm[ids]=True;fi=np.flatnonzero(mm[f].all(1));mp={int(g):i for i,g in enumerate(ids)};lf=np.vectorize(mp.get)(f[fi]);parts.append(trimesh.Trimesh(vertices=v[ids],faces=lf,process=False))
    identity=trimesh.util.concatenate(parts)
    for ext in ('obj','glb','ply'):identity.export(args.out/f'AINA_FACEVERSE_IDENTITY_CLAY_v13.3.{ext}')
    ip=args.out/'AINA_FACEVERSE_IDENTITY_CLAY_v13.3.obj';views=[]
    for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
        p=qa/f'AINA_VTK_{label}_v13.3.png';render_vtk(ip,p,yaw);views.append(p)
    ims=[Image.open(x).convert('RGB') for x in views];W,H=ims[0].size;sheet=Image.new('RGB',(5*W,H),'white')
    for i,im in enumerate(ims):sheet.paste(im,(i*W,0))
    sheet.save(qa/'AINA_VTK_5VIEW_v13.3.png');compare(args.front_ref,qa/'AINA_VTK_front_v13.3.png',qa/'AINA_REFERENCE_VS_VTK_FRONT_v13.3.png');compare(args.q3_ref,qa/'AINA_VTK_left_45_v13.3.png',qa/'AINA_REFERENCE_3Q_VS_VTK_45_v13.3.png')
    final_lm=v[k68,:2];sf,Rf,mxf,myf,pf=similarity(final_lm,tgt,w);err=np.linalg.norm(pf-tgt,axis=1)
    tri0=v0[hf_global];tri1=v[hf_global];a0=.5*np.linalg.norm(np.cross(tri0[:,1]-tri0[:,0],tri0[:,2]-tri0[:,0]),axis=1);a1=.5*np.linalg.norm(np.cross(tri1[:,1]-tri1[:,0],tri1[:,2]-tri1[:,0]),axis=1);ar=a1/np.maximum(a0,1e-12)
    rep={'version':'AINA FaceVerse v13.3 Landmark-Constrained Laplacian Rebuild','base':'clean v12.5 FaceVerse mesh','topology_changed':False,'full_vertices':int(len(v)),'full_triangles':int(len(f)),'head_vertices':int(len(head)),'identity_clay_vertices':int(len(identity.vertices)),'initial_68_rmse':float(np.sqrt(np.mean(initial**2))),'final_68_rmse':float(np.sqrt(np.mean(err**2))),'max_head_xy_shift_m':float(np.max(np.linalg.norm(disp2,axis=1))),'triangle_area_ratio_p01':float(np.percentile(ar,1)),'triangle_area_ratio_p99':float(np.percentile(ar,99)),'qa_renderer':'VTK true Z-buffer','identity_lock':False,'note':'Clean one-pass 68-point topology-wide solve from v12.5; no inherited v13 manual sculpt.'}
    (args.out/'AINA_FACEVERSE_v13.3_REPORT.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
