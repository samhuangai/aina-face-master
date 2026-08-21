#!/usr/bin/env python3
"""AINA Identity Master v2: best-prior centred GNM fit + real art-directed Mesh sculpt."""
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


def args():
 p=argparse.ArgumentParser();p.add_argument('--targets',type=Path,required=True);p.add_argument('--front-target',type=Path,required=True);p.add_argument('--front-ref',type=Path,required=True);p.add_argument('--q3-ref',type=Path,required=True);p.add_argument('--side-ref',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('output_identity_master_v2'));p.add_argument('--samples',type=int,default=4096);p.add_argument('--seed',type=int,default=20260821);return p.parse_args()

def dist(p,a,b):return float(np.linalg.norm(p[a]-p[b]))
def eye_h(p,left):return .5*(dist(p,43,47)+dist(p,44,46)) if left else .5*(dist(p,37,41)+dist(p,38,40))
def ratios(p,t):
 m={'face_w':(dist(p,0,16),dist(t,0,16),.6),'jaw_w':(dist(p,4,12),dist(t,4,12),1.3),'chin_w':(dist(p,6,10),dist(t,6,10),1.1),'eyeR_w':(dist(p,36,39),dist(t,36,39),1.8),'eyeL_w':(dist(p,42,45),dist(t,42,45),1.8),'eyeR_h':(eye_h(p,0),eye_h(t,0),1.5),'eyeL_h':(eye_h(p,1),eye_h(t,1),1.5),'nose_w':(dist(p,31,35),dist(t,31,35),1.8),'mouth_w':(dist(p,48,54),dist(t,48,54),1.7),'lower_h':(abs(p[8,1]-p[33,1]),abs(t[8,1]-t[33,1]),1.6)}
 out={k:max(a,1e-8)/max(b,1e-8) for k,(a,b,_) in m.items()};pen=math.sqrt(sum(w*math.log(out[k])**2 for k,(_,_,w) in m.items())/sum(w for _,_,w in m.values()));return pen,out

def score(lm,targets):
 total=0;cameras={};rm={};vw={'front':1.,'three_quarter':1.22,'side':1.12}
 for n in ('front','three_quarter','side'):
  w=v1.landmark_weights(n);cam=v1.fit_camera(lm,targets[n]['points'],w);cameras[n]=cam;rm[n]=v1.weighted_rmse(v1.project(lm,cam),targets[n]['points'],w);total+=vw[n]*rm[n]
 pen,rr=ratios(v1.project(lm,cameras['front']),targets['front']['points']);return total+.105*pen,cameras,{'rmse':rm,'ratio_penalty':pen,'ratios':rr}

def solve(tpl,basis,targets,mean,axes,std,best):
 bz=np.clip(v1.identity_to_latent(best,mean,axes,std),-3,3);dz=np.zeros_like(bz);bid=v1.latent_to_identity(bz,mean,axes,std);blm=tpl+np.einsum('i,ilc->lc',bid,basis);ib=axes*std[:,None];lb=np.einsum('ki,ilc->klc',ib,basis);hist=[]
 for it in range(9):
  lm=blm+np.einsum('k,klc->lc',dz,lb);A=[];b=[];cams={};before={}
  for n in ('front','three_quarter','side'):
   t=targets[n]['points'];w=v1.landmark_weights(n);cam=v1.fit_camera(lm,t,w);cams[n]=cam;before[n]=v1.weighted_rmse(v1.project(lm,cam),t,w);R,s,tr=cam;base=s*(blm@R.T)[:,:2]+tr;bb=s*np.einsum('klc,dc->kld',lb,R)[:,:,:2];M=bb.transpose(1,2,0).reshape(-1,len(dz));sw=np.repeat(np.sqrt(w),2);A.append(M*sw[:,None]);b.append((t-base).reshape(-1)*sw)
  reg=1.05+.12*it;A.append(np.eye(len(dz))*math.sqrt(reg));b.append(np.zeros(len(dz)));sol=np.clip(np.linalg.lstsq(np.vstack(A),np.concatenate(b),rcond=1e-6)[0],-1.35,1.35);dz=.28*dz+.72*sol;dz=np.clip(bz+dz,-3.15,3.15)-bz;alm=blm+np.einsum('k,klc->lc',dz,lb);after={n:v1.weighted_rmse(v1.project(alm,cams[n]),targets[n]['points'],v1.landmark_weights(n)) for n in cams};hist.append({'iteration':it,'before':before,'after':after,'best_z_norm':float(np.linalg.norm(bz)),'delta_z_norm':float(np.linalg.norm(dz))})
 z=np.clip(bz+dz,-3.15,3.15);ident=v1.latent_to_identity(z,mean,axes,std);lm=tpl+np.einsum('i,ilc->lc',ident,basis);cams={};metrics={}
 for n in ('front','three_quarter','side'):
  cams[n]=v1.fit_camera(lm,targets[n]['points'],v1.landmark_weights(n));metrics[n]=v1.weighted_rmse(v1.project(lm,cams[n]),targets[n]['points'],v1.landmark_weights(n))
 return ident,lm,cams,metrics,hist,z,bz

def cw(p,c,r,inner=0.,outer=1.):
 q=np.sqrt(np.sum(((p-np.asarray(c))/np.asarray(r))**2,axis=1));w=np.zeros(len(p));w[q<=inner]=1;m=(q>inner)&(q<outer)
 if np.any(m):t=(q[m]-inner)/(outer-inner+1e-12);w[m]=.5*(1+np.cos(np.pi*t))
 return w
def sc(p,c,r,s,inner=0.,outer=1.):
 c=np.asarray(c);w=cw(p,c,r,inner,outer)[:,None];return p+w*(c+(p-c)*np.asarray(s)-p)
def sh(p,c,r,d,inner=0.,outer=1.):return p+cw(p,c,r,inner,outer)[:,None]*np.asarray(d)

def sculpt(v,li,lw,cam):
 R=cam[0];p=v@R.T;base=p.copy();lm=v1.compute_landmarks(v,li,lw)@R.T;cx=float(lm[27:36,0].mean());brow=float(lm[17:27,1].mean());eye=float(lm[36:48,1].mean());mouth=float(lm[48:60,1].mean());chin=float(lm[8,1]);fs=1. if lm[30,2]>=np.mean(lm[[0,8,16],2]) else -1.
 fw=np.exp(-.5*((p[:,2]-lm[30,2])/.115)**4);lt=np.clip((p[:,1]-mouth)/(chin-mouth+1e-6),0,1);upper=.925-.03*np.clip((p[:,1]-brow)/(eye-brow+1e-6),0,1);lower=.875-.155*lt**1.2;xs=np.where(p[:,1]<=mouth,upper,lower);p[:,0]=cx+(p[:,0]-cx)*(1-(1-xs)*fw);p[:,1]-=.0105*lt**1.12*fw
 lm=v1.compute_landmarks(p@R,li,lw)@R.T;c=lm[8];p=sc(p,c,(.05,.045,.045),(.76,.88,.91),.02,1.08);p=sh(p,c,(.05,.046,.046),(0,-.003,-fs*.0012),0,1.04)
 lm=v1.compute_landmarks(p@R,li,lw)@R.T
 for ids,oi,ii in ((range(36,42),36,39),(range(42,48),45,42)):
  c=lm[list(ids)].mean(0);p=sc(p,c,(.04,.03,.032),(1.22,1.26,1.02),.04,1.12);p=sh(p,c,(.042,.032,.034),(0,.0003,fs*.0012),0,1.10);lm=v1.compute_landmarks(p@R,li,lw)@R.T;o=lm[oi];ss=-1 if o[0]<cx else 1;p=sh(p,o,(.017,.014,.02),(ss*.0012,-.0011,fs*.0004),.02,1.02);p=sh(p,lm[ii],(.015,.014,.018),(-ss*.00015,.00015,fs*.00025),.02,1.02);p=sh(p,c+(0,.012,-fs*.006),(.041,.03,.035),(0,-.001,fs*.0016),0,1.08);p=sh(p,c+(0,-.024,-fs*.004),(.043,.031,.036),(0,.001,-fs*.0014),0,1.08)
 lm=v1.compute_landmarks(p@R,li,lw)@R.T;br=lm[27:31].mean(0);nb=lm[31:36].mean(0);tip=lm[30];p=sc(p,br,(.026,.037,.04),(.80,.94,1),.03,1.12);p=sh(p,br,(.026,.037,.042),(0,-.0008,fs*.0027),.02,1.10);p=sc(p,nb,(.032,.028,.033),(.72,.86,.92),.03,1.10);p=sh(p,nb,(.032,.029,.034),(0,-.0026,fs*.0035),.02,1.08);p=sc(p,tip,(.022,.022,.025),(.84,.82,.92),.02,1.05);p=sh(p,tip,(.022,.023,.026),(0,-.0018,fs*.0044),.02,1.04)
 lm=v1.compute_landmarks(p@R,li,lw)@R.T;m=lm[48:60].mean(0);p=sc(p,m,(.049,.028,.03),(1.12,.80,.92),.06,1.10);p=sh(p,m,(.05,.031,.032),(0,-.0014,fs*.0014),.03,1.08);p=sh(p,lm[[49,50,51,52,53]].mean(0),(.035,.013,.02),(0,-.0009,fs*.0008),.01,1.04);p=sh(p,lm[[55,56,57,58,59]].mean(0),(.036,.014,.021),(0,-.0002,fs*.0016),.01,1.04)
 lm=v1.compute_landmarks(p@R,li,lw)@R.T
 for c in ((lm[40]+lm[31]+lm[48])/3,(lm[46]+lm[35]+lm[54])/3):p=sh(p,c,(.044,.038,.042),((-0.0012 if c[0]>cx else .0012),-.001,fs*.0022),.02,1.10)
 lm=v1.compute_landmarks(p@R,li,lw)@R.T
 for i in (0,16):c=lm[i];ss=-1 if c[0]<cx else 1;p=sc(p,c,(.04,.06,.055),(.76,.84,.82),0,1.10);p=sh(p,c,(.041,.061,.056),(-ss*.004,0,-fs*.0014),0,1.08)
 out=p@R;d=out-v;return out,{'front_sign':fs,'max_m':float(np.linalg.norm(d,axis=1).max()),'rms_m':float(np.sqrt(np.mean(np.sum(d*d,axis=1))))}

def reconverge(v,tri,li,lw,targets):
 base=v.copy();out=v.copy();hist=[];vs={'front':.46,'three_quarter':.36,'side':.34}
 for it,gs in enumerate((1.,.64,.40,.24)):
  lm=v1.compute_landmarks(out,li,lw);acc=np.zeros_like(out);den=np.zeros(len(out));rm={}
  for n in ('front','three_quarter','side'):
   t=targets[n]['points'];w=v1.landmark_weights(n);cam=v1.fit_camera(lm,t,w);pred=v1.project(lm,cam);rm[n]=v1.weighted_rmse(pred,t,w);R,s,_=cam;disp=np.c_[(t-pred)/max(s,1e-8),np.zeros(68)]@R;L=np.linalg.norm(disp,axis=1);disp*=np.minimum(1,.0036/np.maximum(L,1e-9))[:,None]*vs[n]*gs
   for j,c in enumerate(lm):r=v1.landmark_radius(j)*.82;dd=np.linalg.norm(out-c,axis=1);ww=np.exp(-.5*(dd/r)**4);ww[dd>r*1.35]=0;ww*=w[j];acc+=ww[:,None]*disp[j];den+=ww
  d=acc/np.maximum(den[:,None],1e-9);d[den<.12]=0;L=np.linalg.norm(d,axis=1);d*=np.minimum(1,.0019/np.maximum(L,1e-9))[:,None];out+=d;hist.append({'iteration':it,'rmse':rm,'max_step_m':float(np.linalg.norm(d,axis=1).max())})
 a0=.5*np.linalg.norm(np.cross(base[tri][:,1]-base[tri][:,0],base[tri][:,2]-base[tri][:,0]),axis=1);a1=.5*np.linalg.norm(np.cross(out[tri][:,1]-out[tri][:,0],out[tri][:,2]-out[tri][:,0]),axis=1);rr=a1/np.maximum(a0,1e-12);return out,hist,{'max_m':float(np.linalg.norm(out-base,axis=1).max()),'rms_m':float(np.sqrt(np.mean(np.sum((out-base)**2,axis=1)))),'area_p01':float(np.percentile(rr,1)),'area_p99':float(np.percentile(rr,99)),'degenerate':int(np.sum(a1<1e-12))}

def export(gnm,v,out,name):
 full=trimesh.Trimesh(v,np.asarray(gnm.triangles),process=False)
 for e in ('obj','glb','ply'):full.export(out/f'{name}_FULL.{e}')
 skin=full.submesh([np.asarray(gnm.triangle_indices_for_group('skin'))],append=True,repair=False);skin.remove_unreferenced_vertices()
 for e in ('obj','glb','ply'):skin.export(out/f'{name}_SKIN.{e}')
 return skin
def render(v,faces,lm,R,path,title):
 p=v@R.T;l=lm@R.T;xy=p[:,:2];tr=p[faces];n=np.cross(tr[:,1]-tr[:,0],tr[:,2]-tr[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9);order=np.argsort(tr[:,:,2].mean(1))[::-1];nn=n[order];val=np.clip(.61+.25*np.abs(nn[:,2])+.12*np.clip(-.35*nn[:,0]-.18*nn[:,1]-.72*nn[:,2],0,1),.42,.98);col=np.stack([val*.95,val*.97,val],1);xl,xh=np.min(l[:17,0]),np.max(l[:17,0]);yl=min(np.min(l[17:27,1])-.095,np.min(l[:,1])-.035);yh=l[8,1]+.035;fw=xh-xl;fh=yh-yl;fig,ax=plt.subplots(figsize=(5.4,5.4),dpi=190);ax.add_collection(PolyCollection(xy[faces[order]],facecolors=col,edgecolors='none'));ax.set_xlim(xl-.22*fw,xh+.22*fw);ax.set_ylim(yh+.07*fh,yl-.07*fh);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10);fig.tight_layout(pad=.08);fig.savefig(path,bbox_inches='tight',pad_inches=.02);plt.close(fig)
def compare(ref,im,out,label):
 a=Image.open(ref).convert('RGB');b=Image.open(im).convert('RGB');h=max(a.height,b.height);aw=int(a.width*h/a.height);bw=int(b.width*h/b.height);c=Image.new('RGB',(aw+bw,h+36),'white');c.paste(a.resize((aw,h)),(0,0));c.paste(b.resize((bw,h)),(aw,0));d=ImageDraw.Draw(c);d.text((8,h+9),f'Approved {label}',fill='black');d.text((aw+8,h+9),'AINA Identity Master v2 real clay',fill='black');c.save(out)
def sheet(items,out):
 panels=[]
 for p,label in items:
  im=Image.open(p).convert('RGB');im.thumbnail((500,500));q=Image.new('RGB',(520,545),'white');q.paste(im,((520-im.width)//2,4));ImageDraw.Draw(q).text((8,520),label,fill='black');panels.append(q)
 s=Image.new('RGB',(520*len(panels),545),'white')
 for i,p in enumerate(panels):s.paste(p,(520*i,0))
 s.save(out)

def main():
 a=args();a.out.mkdir(parents=True,exist_ok=True);qa=a.out/'QA';qa.mkdir(exist_ok=True);targets=v1.load_targets(a.targets,a.front_target);g=gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3,variant=gnm_numpy.GNMVariant.HEAD);li,lw,tpl,basis=v1.sparse_landmark_model(g);sampler=IdentitySampler();rng=np.random.default_rng(a.seed);samples=sampler.sample_identity(Gender.FEMALE,Ethnicity.ASIAN,num_samples=a.samples,rng=rng).astype(float);rank=[]
 for i,idv in enumerate(samples):lm=tpl+np.einsum('i,ilc->lc',idv,basis);rank.append((*score(lm,targets),i))
 rank.sort(key=lambda x:x[0]);best_score,best_cams,best_details,best_i=rank[0];best=samples[best_i];mean,axes,std=v1.build_female_prior(samples,components=96);fid,flm,fcams,fmetrics,hist,z,bz=solve(tpl,basis,targets,mean,axes,std,best);bv=np.asarray(g(identity=best[None,:]))[0].astype(float);fv=np.asarray(g(identity=fid[None,:]))[0].astype(float);sv,art=sculpt(fv,li,lw,fcams['front']);final,rhist,rhealth=reconverge(sv,np.asarray(g.triangles),li,lw,targets);export(g,bv,a.out,'AINA_IDENTITY_MASTER_V2_BEST_PRIOR');export(g,fv,a.out,'AINA_IDENTITY_MASTER_V2_FITTED_PRIOR');export(g,final,a.out,'AINA_IDENTITY_MASTER_V2_FINAL');np.save(a.out/'AINA_IDENTITY_MASTER_V2_IDENTITY.npy',fid.astype(np.float32));np.save(a.out/'AINA_IDENTITY_MASTER_V2_LATENT.npy',z.astype(np.float32));tri=np.asarray(g.triangles);blm=v1.compute_landmarks(bv,li,lw);final_lm=v1.compute_landmarks(final,li,lw);cams={};metrics={}
 for n in ('front','three_quarter','side'):cams[n]=v1.fit_camera(final_lm,targets[n]['points'],v1.landmark_weights(n));metrics[n]=v1.weighted_rmse(v1.project(final_lm,cams[n]),targets[n]['points'],v1.landmark_weights(n))
 pen,rr=ratios(v1.project(final_lm,cams['front']),targets['front']['points']);stages=[]
 for name,v,lm,cam in [('BEST_PRIOR',bv,blm,best_cams['front']),('FITTED_PRIOR',fv,flm,fcams['front']),('FINAL',final,final_lm,cams['front'])]:p=qa/f'AINA_IDENTITY_MASTER_V2_{name}_FRONT.png';render(v,tri,lm,cam[0],p,f'AINA Identity Master v2 — {name}');stages.append((p,name.replace('_',' ').title()))
 sheet(stages,qa/'AINA_IDENTITY_MASTER_V2_STAGE_CONTACT_SHEET.png');refs={'front':a.front_ref,'three_quarter':a.q3_ref,'side':a.side_ref}
 for n in refs:p=qa/f'AINA_IDENTITY_MASTER_V2_{n.upper()}_CLAY.png';render(final,tri,final_lm,cams[n][0],p,f'AINA Identity Master v2 — {n}');compare(refs[n],p,qa/f'AINA_APPROVED_VS_IDENTITY_MASTER_V2_{n.upper()}.png','3Q' if n=='three_quarter' else n)
 five=[]
 for yaw,label in [(-90,'LEFT_PROFILE'),(-45,'LEFT_45'),(0,'FRONT'),(45,'RIGHT_45'),(90,'RIGHT_PROFILE')]:p=qa/f'AINA_IDENTITY_MASTER_V2_{label}.png';render(final,tri,final_lm,v1.yaw_rotation(cams['front'][0],yaw),p,f'AINA Identity Master v2 — {label}');five.append((p,label.replace('_',' ').title()))
 sheet(five,qa/'AINA_IDENTITY_MASTER_V2_5VIEW.png');t0=fv[tri];t1=final[tri];a0=.5*np.linalg.norm(np.cross(t0[:,1]-t0[:,0],t0[:,2]-t0[:,0]),axis=1);a1=.5*np.linalg.norm(np.cross(t1[:,1]-t1[:,0],t1[:,2]-t1[:,0]),axis=1);ar=a1/np.maximum(a0,1e-12);report={'product':'AINA Identity Master Reconstruction v2','source_model':'Google GNM v3 HEAD','prior':'Female + Asian semantic prior','design_change':'best sampled identity remains solve centre; AINA ratio ranking and strong art-directed neutral sculpt','sample_count':a.samples,'best_prior_index':best_i,'best_prior_score':best_score,'best_prior_details':best_details,'identity_dimension':g.identity_dim,'expression_dimension_available':g.expression_dim,'vertices':len(final),'triangles':len(tri),'topology_changed':False,'new_reference_generated':False,'real_mesh_reconstructed':True,'best_z_norm':float(np.linalg.norm(bz)),'fitted_delta_z_norm':float(np.linalg.norm(z-bz)),'fit_rmse_before_art_sculpt':fmetrics,'fit_rmse_after':metrics,'final_ratio_penalty':pen,'final_ratios':rr,'fit_history':hist,'art_health':art,'reconvergence_history':rhist,'reconvergence_health':rhealth,'final_mesh_health':{'max_m':float(np.linalg.norm(final-fv,axis=1).max()),'rms_m':float(np.sqrt(np.mean(np.sum((final-fv)**2,axis=1)))),'area_p01':float(np.percentile(ar,1)),'area_p99':float(np.percentile(ar,99)),'degenerate':int(np.sum(a1<1e-12))},'identity_lock':False,'visual_identity_lock':False,'candidate':True,'next_gate':'Inspect approved front, 3Q, side and five-view real clay before expression transfer or VRM export.'};(a.out/'AINA_IDENTITY_MASTER_V2_REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
