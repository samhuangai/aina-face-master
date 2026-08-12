#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import trimesh

from gnm.shape import gnm_numpy, gnm_landmarks
from gnm.shape.semantic_sampler import IdentitySampler, Gender, Ethnicity

GNM_TO_STANDARD=np.array([0,1,6,5,4,3,2,7,8,9,10,11,12,13,14,15,16,*range(17,68)],dtype=np.int64)

def norm_target(points,size):
    W,H=size; s=.5*max(W,H)
    return (np.asarray(points,float)-np.array([W*.5,H*.5]))/s

def weights():
    w=np.ones(68,float)
    w[:17]=1.35; w[17:27]=.55; w[27:36]=2.0; w[36:48]=2.45; w[48:68]=2.25
    return w

def fit_camera(P,Q,w):
    X=np.c_[P,np.ones(len(P))]; sw=np.sqrt(w)[:,None]
    beta=np.linalg.lstsq(X*sw,Q*sw,rcond=None)[0]
    A=beta[:3].T; b=beta[3]
    n1=np.linalg.norm(A[0]); n2=np.linalg.norm(A[1]); s=max(1e-8,.5*(n1+n2))
    r1=A[0]/max(n1,1e-9); v2=A[1]-np.dot(A[1],r1)*r1; r2=v2/max(np.linalg.norm(v2),1e-9); r3=np.cross(r1,r2); r3/=max(np.linalg.norm(r3),1e-9); r2=np.cross(r3,r1)
    R=np.stack([r1,r2,r3]);
    if np.linalg.det(R)<0:R[2]*=-1
    return R,s,b

def projected_score(lm,target):
    w=weights(); R,s,b=fit_camera(lm,target,w); pred=s*(lm@R.T)[:,:2]+b
    e=np.linalg.norm(pred-target,axis=1)
    base=float(np.sqrt(np.sum(w*e*e)/np.sum(w)))
    # Dimensionless feature ratios after projection; emphasize AINA's wide soft mouth,
    # small nose, balanced eye separation and non-knife lower jaw.
    def dist(a,b,x=pred): return float(np.linalg.norm(x[a]-x[b]))
    def tdist(a,b): return float(np.linalg.norm(target[a]-target[b]))
    ratios=[]
    for a,b,ww in [(36,39,1.2),(42,45,1.2),(31,35,1.8),(48,54,1.6),(0,16,.7),(6,10,1.0)]:
        ratios.append((math.log(max(dist(a,b),1e-6)/max(tdist(a,b),1e-6)),ww))
    ec=np.linalg.norm(pred[36:42].mean(0)-pred[42:48].mean(0)); tec=np.linalg.norm(target[36:42].mean(0)-target[42:48].mean(0)); ratios.append((math.log(ec/tec),1.3))
    ratio_pen=sum(ww*r*r for r,ww in ratios)/sum(ww for _,ww in ratios)
    return base+0.055*math.sqrt(ratio_pen),base,R,s,b,pred

def main():
    target_json=json.loads(Path('references/AINA_TARGET_3DDFA_SPARSE_68.json').read_text())
    target=norm_target(target_json['landmarks_xy'],target_json['image_size'])
    g=gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,variant=gnm_numpy.GNMVariant.HEAD)
    cfg=gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
    idx=np.asarray(cfg.indices,dtype=np.int64); bw=np.asarray(cfg.weights,dtype=np.float64)
    tv=np.asarray(g.template_vertex_positions,dtype=np.float64); ib=np.asarray(g.vertex_identity_basis,dtype=np.float64)
    tlm=(tv[idx]*bw[...,None]).sum(-2)[GNM_TO_STANDARD]
    lb=(ib[:,idx,:]*bw[None,...,None]).sum(-2)[:,GNM_TO_STANDARD,:]

    sampler=IdentitySampler()
    rng=np.random.default_rng(20260812)
    identities=sampler.sample_identity(Gender.FEMALE,Ethnicity.ASIAN,num_samples=384,rng=rng)
    out=Path('output_semantic_samples'); out.mkdir(exist_ok=True); (out/'models').mkdir(exist_ok=True)
    scores=[]
    for i,identity in enumerate(identities):
        lm=tlm+np.einsum('i,ilc->lc',identity,lb)
        score,base,R,s,b,pred=projected_score(lm,target)
        scores.append((score,base,i,R,s,b,pred))
    scores.sort(key=lambda x:x[0])
    top=scores[:16]
    triangles=np.asarray(g.triangles,dtype=np.int64); skin_ti=np.asarray(g.triangle_indices_for_group('skin'),dtype=np.int64)
    rank=[]
    top_ids=[]
    for rank_i,(score,base,i,R,s,b,pred) in enumerate(top):
        ident=np.asarray(identities[i],dtype=np.float64); top_ids.append(ident)
        vertices=np.asarray(g(identity=ident[None,:]))[0]
        full=trimesh.Trimesh(vertices=vertices,faces=triangles,process=False)
        name=f'S{rank_i:02d}'
        full.export(out/'models'/f'{name}.obj'); full.export(out/'models'/f'{name}.glb')
        skin=full.submesh([skin_ti],append=True,repair=False); skin.remove_unreferenced_vertices();skin.export(out/'models'/f'{name}_skin.obj')
        rank.append({'rank':rank_i,'sample_index':i,'score':float(score),'landmark_rmse':float(base),'camera_R':R.tolist(),'camera_scale':float(s),'camera_t':b.tolist()})
    np.save(out/'AINA_TOP16_FEMALE_ASIAN_IDENTITIES.npy',np.asarray(top_ids,dtype=np.float32))
    (out/'ranking.json').write_text(json.dumps(rank,indent=2))
    print(json.dumps(rank,indent=2))

if __name__=='__main__':main()
