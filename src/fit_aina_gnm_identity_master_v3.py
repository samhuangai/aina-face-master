#!/usr/bin/env python3
"""AINA Identity Master v3: cross-prior identity + baked neutral expression fit."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np,trimesh
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from gnm.shape import gnm_numpy
from gnm.shape.semantic_sampler import IdentitySampler,Gender,Ethnicity
import fit_aina_gnm_identity_master as v1
import fit_aina_gnm_identity_master_v2 as v2


def args():
 p=argparse.ArgumentParser();p.add_argument('--targets',type=Path,required=True);p.add_argument('--front-target',type=Path,required=True);p.add_argument('--front-ref',type=Path,required=True);p.add_argument('--q3-ref',type=Path,required=True);p.add_argument('--side-ref',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('output_identity_master_v3'));p.add_argument('--seed',type=int,default=20260821);return p.parse_args()

def lm_expression_basis(gnm,indices,weights):
 b=np.asarray(gnm.vertex_expression_basis,dtype=np.float64);return (b[:,indices,:]*weights[None,...,None]).sum(-2)

def solve_neutral_expression(base_lm,expr_basis,targets):
 e=np.zeros(expr_basis.shape[0],np.float64);hist=[]
 for it in range(8):
  lm=base_lm+np.einsum('i,ilc->lc',e,expr_basis);A=[];rhs=[];before={};cams={}
  for n in ('front','three_quarter','side'):
   t=targets[n]['points'];w=v1.landmark_weights(n).copy();w[36:48]*=1.65;w[48:68]*=1.15;cam=v1.fit_camera(lm,t,w);cams[n]=cam;before[n]=v1.weighted_rmse(v1.project(lm,cam),t,w);R,s,tr=cam;base=s*(base_lm@R.T)[:,:2]+tr;bb=s*np.einsum('ilc,dc->ild',expr_basis,R)[:,:,:2];M=bb.transpose(1,2,0).reshape(-1,len(e));sw=np.repeat(np.sqrt(w),2);A.append(M*sw[:,None]);rhs.append((t-base).reshape(-1)*sw)
  reg=6.0+1.0*it;A.append(np.eye(len(e))*math.sqrt(reg));rhs.append(np.zeros(len(e)));sol=np.linalg.lstsq(np.vstack(A),np.concatenate(rhs),rcond=1e-6)[0];sol=np.clip(sol,-1.4,1.4);norm=np.linalg.norm(sol)
  if norm>7.0:sol*=7.0/norm
  e=.30*e+.70*sol;after_lm=base_lm+np.einsum('i,ilc->lc',e,expr_basis);after={n:v1.weighted_rmse(v1.project(after_lm,cams[n]),targets[n]['points'],v1.landmark_weights(n)) for n in cams};hist.append({'iteration':it,'before':before,'after':after,'expression_norm':float(np.linalg.norm(e)),'max_coefficient':float(np.max(np.abs(e)))})
 lm=base_lm+np.einsum('i,ilc->lc',e,expr_basis);cams={};metrics={}
 for n in ('front','three_quarter','side'):cams[n]=v1.fit_camera(lm,targets[n]['points'],v1.landmark_weights(n));metrics[n]=v1.weighted_rmse(v1.project(lm,cams[n]),targets[n]['points'],v1.landmark_weights(n))
 return e,lm,cams,metrics,hist

def surface_score(vertices,lm,skin_ids):
 s=vertices[skin_ids];eye=float(lm[36:48,1].mean());mouth=float(lm[48:60,1].mean());nose=float(lm[30,2]);band=(s[:,1]>mouth)&(s[:,1]<eye)&(s[:,2]>nose-.105)
 if band.sum()<30:return 99.,{}
 sw=float(np.percentile(s[band,0],98)-np.percentile(s[band,0],2));fw=float(np.linalg.norm(lm[0]-lm[16]));jaw=float(np.linalg.norm(lm[4]-lm[12]));cheek=sw/max(fw,1e-6);taper=jaw/max(fw,1e-6);pen=1.6*(math.log(cheek/.80)**2)+1.0*(math.log(taper/.76)**2)
 return math.sqrt(pen/2.6),{'surface_cheek_over_face':cheek,'jaw_over_face':taper,'surface_width_m':sw}

def stronger_sculpt(v,li,lw,cam):
 out,health=v2.sculpt(v,li,lw,cam);R=cam[0];p=out@R.T;lm=v1.compute_landmarks(out,li,lw)@R.T;cx=float(lm[27:36,0].mean());eye=float(lm[36:48,1].mean());mouth=float(lm[48:60,1].mean());chin=float(lm[8,1]);nose_z=float(lm[30,2]);front=np.exp(-.5*((p[:,2]-nose_z)/.110)**4);mid=np.exp(-.5*((p[:,1]-(eye+mouth)/2)/.060)**4);lower=np.clip((mouth-p[:,1])/(mouth-chin+1e-6),0,1);factor=1-front*(.10*mid+.09*lower);p[:,0]=cx+(p[:,0]-cx)*factor
 for sx in (-.040,.040):p=v2.sh(p,(sx,(eye+mouth)/2,nose_z-.045),(.052,.055,.060),(0,0,-health['front_sign']*.0045),0,1.08)
 lm=v1.compute_landmarks(p@R,li,lw)@R.T
 for ids in (range(36,42),range(42,48)):
  c=lm[list(ids)].mean(0);p=v2.sc(p,c,(.043,.032,.035),(1.08,1.28,1.0),.02,1.08)
 d=p@R-v;health['v3_extra_max_m']=float(np.linalg.norm(d,axis=1).max());health['v3_extra_rms_m']=float(np.sqrt(np.mean(np.sum(d*d,axis=1))));return p@R,health

def export(gnm,v,out,name):
 full=trimesh.Trimesh(v,np.asarray(gnm.triangles),process=False)
 for ext in ('obj','glb','ply'):full.export(out/f'{name}_FULL.{ext}')
 skin=full.submesh([np.asarray(gnm.triangle_indices_for_group('skin'))],append=True,repair=False);skin.remove_unreferenced_vertices()
 for ext in ('obj','glb','ply'):skin.export(out/f'{name}_SKIN.{ext}')
 return skin

def render(v,faces,lm,R,path,title):
 p=v@R.T;l=lm@R.T;xy=p[:,:2];tr=p[faces];n=np.cross(tr[:,1]-tr[:,0],tr[:,2]-tr[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9);order=np.argsort(tr[:,:,2].mean(1))[::-1];nn=n[order];val=np.clip(.60+.26*np.abs(nn[:,2])+.13*np.clip(-.32*nn[:,0]-.18*nn[:,1]-.74*nn[:,2],0,1),.40,.98);col=np.stack([val*.95,val*.97,val],1);xl,xh=np.min(l[:17,0]),np.max(l[:17,0]);yl=min(np.min(l[17:27,1])-.092,np.min(l[:,1])-.030);yh=l[8,1]+.030;fw=xh-xl;fh=yh-yl;fig,ax=plt.subplots(figsize=(5.4,5.4),dpi=190);ax.add_collection(PolyCollection(xy[faces[order]],facecolors=col,edgecolors='none'));ax.set_xlim(xl-.20*fw,xh+.20*fw);ax.set_ylim(yh+.06*fh,yl-.06*fh);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10);fig.tight_layout(pad=.06);fig.savefig(path,bbox_inches='tight',pad_inches=.02);plt.close(fig)
def compare(ref,im,out,label):
 a=Image.open(ref).convert('RGB');b=Image.open(im).convert('RGB');h=max(a.height,b.height);aw=int(a.width*h/a.height);bw=int(b.width*h/b.height);c=Image.new('RGB',(aw+bw,h+34),'white');c.paste(a.resize((aw,h)),(0,0));c.paste(b.resize((bw,h)),(aw,0));d=ImageDraw.Draw(c);d.text((8,h+8),f'Approved {label}',fill='black');d.text((aw+8,h+8),'AINA Identity Master v3 real clay',fill='black');c.save(out)
def sheet(items,out):
 panels=[]
 for p,label in items:
  im=Image.open(p).convert('RGB');im.thumbnail((490,490));q=Image.new('RGB',(510,535),'white');q.paste(im,((510-im.width)//2,4));ImageDraw.Draw(q).text((8,510),label,fill='black');panels.append(q)
 s=Image.new('RGB',(510*len(panels),535),'white')
 for i,p in enumerate(panels):s.paste(p,(510*i,0))
 s.save(out)

def main():
 a=args();a.out.mkdir(parents=True,exist_ok=True);qa=a.out/'QA';qa.mkdir(exist_ok=True);targets=v1.load_targets(a.targets,a.front_target);g=gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,variant=gnm_numpy.GNMVariant.HEAD);li,lw,tpl,ib=v1.sparse_landmark_model(g);eb=lm_expression_basis(g,li,lw);sampler=IdentitySampler();rng=np.random.default_rng(a.seed);pools=[]
 for eth in (Ethnicity.ASIAN,Ethnicity.WHITE,Ethnicity.MIDDLE_EASTERN,Ethnicity.BLACK):pools.append((eth.name,sampler.sample_identity(Gender.FEMALE,eth,num_samples=1024,rng=rng).astype(float)))
 pools.append(('ASIAN_WHITE_BLEND',sampler.blend_identities({Gender.FEMALE:1.0},{Ethnicity.ASIAN:.55,Ethnicity.WHITE:.45},num_samples=2048,rng=rng).astype(float)));samples=np.concatenate([x[1] for x in pools]);labels=np.concatenate([[name]*len(x) for name,x in pools]);rank=[]
 for i,idv in enumerate(samples):lm=tpl+np.einsum('i,ilc->lc',idv,ib);sc,cams,details=v2.score(lm,targets);rank.append((sc,i,cams,details))
 rank.sort(key=lambda x:x[0]);top=rank[:96];top_ids=np.asarray([samples[x[1]] for x in top]);verts=np.asarray(g(identity=top_ids));skin_ids=np.unique(np.asarray(g.triangles)[np.asarray(g.triangle_indices_for_group('skin'))]);reranked=[]
 for j,item in enumerate(top):sc0,i,cams,details=item;lm=tpl+np.einsum('i,ilc->lc',samples[i],ib);sp,sd=surface_score(verts[j],lm,skin_ids);reranked.append((sc0+.085*sp,i,cams,details,sd))
 reranked.sort(key=lambda x:x[0]);best_score,best_i,best_cams,best_details,best_surface=reranked[0];best=samples[best_i];mean,axes,std=v1.build_female_prior(samples,components=112);fid,flm,fcams,fmetrics,fhist,z,bz=v2.solve(tpl,ib,targets,mean,axes,std,best);expr,elm,ecams,emetrics,ehist=solve_neutral_expression(flm,eb,targets);base=np.asarray(g(identity=fid[None,:],expression=expr[None,:]))[0].astype(float);styled,art=stronger_sculpt(base,li,lw,ecams['front']);final,rhist,rhealth=v2.reconverge(styled,np.asarray(g.triangles),li,lw,targets);export(g,base,a.out,'AINA_IDENTITY_MASTER_V3_EXPRESSION_FITTED');export(g,final,a.out,'AINA_IDENTITY_MASTER_V3_FINAL');np.save(a.out/'AINA_IDENTITY_MASTER_V3_IDENTITY.npy',fid.astype(np.float32));np.save(a.out/'AINA_IDENTITY_MASTER_V3_NEUTRAL_EXPRESSION.npy',expr.astype(np.float32));tri=np.asarray(g.triangles);lm=v1.compute_landmarks(final,li,lw);cams={};metrics={}
 for n in ('front','three_quarter','side'):cams[n]=v1.fit_camera(lm,targets[n]['points'],v1.landmark_weights(n));metrics[n]=v1.weighted_rmse(v1.project(lm,cams[n]),targets[n]['points'],v1.landmark_weights(n))
 pen,rr=v2.ratios(v1.project(lm,cams['front']),targets['front']['points']);refs={'front':a.front_ref,'three_quarter':a.q3_ref,'side':a.side_ref};items=[]
 for n in refs:p=qa/f'AINA_IDENTITY_MASTER_V3_{n.upper()}_CLAY.png';render(final,tri,lm,cams[n][0],p,f'AINA Identity Master v3 — {n}');compare(refs[n],p,qa/f'AINA_APPROVED_VS_IDENTITY_MASTER_V3_{n.upper()}.png','3Q' if n=='three_quarter' else n);items.append((p,n.replace('_',' ').title()))
 sheet(items,qa/'AINA_IDENTITY_MASTER_V3_3VIEW.png');five=[]
 for yaw,label in [(-90,'LEFT_PROFILE'),(-45,'LEFT_45'),(0,'FRONT'),(45,'RIGHT_45'),(90,'RIGHT_PROFILE')]:p=qa/f'AINA_IDENTITY_MASTER_V3_{label}.png';render(final,tri,lm,v1.yaw_rotation(cams['front'][0],yaw),p,f'AINA Identity Master v3 — {label}');five.append((p,label.replace('_',' ').title()))
 sheet(five,qa/'AINA_IDENTITY_MASTER_V3_5VIEW.png');t0=base[tri];t1=final[tri];a0=.5*np.linalg.norm(np.cross(t0[:,1]-t0[:,0],t0[:,2]-t0[:,0]),axis=1);a1=.5*np.linalg.norm(np.cross(t1[:,1]-t1[:,0],t1[:,2]-t1[:,0]),axis=1);ar=a1/np.maximum(a0,1e-12);report={'product':'AINA Identity Master Reconstruction v3','source_model':'Google GNM v3 HEAD','identity_pool':'female Asian, White, Middle Eastern, Black and Asian/White blend','sample_count':len(samples),'best_pool':str(labels[best_i]),'best_index':int(best_i),'best_score':float(best_score),'best_landmark_details':best_details,'best_surface_details':best_surface,'identity_dimension':g.identity_dim,'expression_dimension':g.expression_dim,'neutral_expression_baked':True,'neutral_expression_norm':float(np.linalg.norm(expr)),'neutral_expression_max_coefficient':float(np.max(np.abs(expr))),'identity_fit_metrics':fmetrics,'expression_fit_metrics':emetrics,'final_metrics':metrics,'final_ratio_penalty':pen,'final_ratios':rr,'identity_fit_history':fhist,'expression_fit_history':ehist,'art_health':art,'reconvergence_history':rhist,'reconvergence_health':rhealth,'mesh_health':{'max_m':float(np.linalg.norm(final-base,axis=1).max()),'rms_m':float(np.sqrt(np.mean(np.sum((final-base)**2,axis=1)))),'area_p01':float(np.percentile(ar,1)),'area_p99':float(np.percentile(ar,99)),'degenerate':int(np.sum(a1<1e-12))},'vertices':len(final),'triangles':len(tri),'topology_changed':False,'real_mesh_reconstructed':True,'new_reference_generated':False,'identity_lock':False,'visual_identity_lock':False,'candidate':True,'next_gate':'Inspect real front/3Q/side clay. If accepted, build the real eye globe/cornea/iris system and transfer expressions; otherwise continue neutral sculpt.'};(a.out/'AINA_IDENTITY_MASTER_V3_REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
