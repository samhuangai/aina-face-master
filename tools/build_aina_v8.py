#!/usr/bin/env python3
from __future__ import annotations
import io,json,math,struct,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
import trimesh

DT={5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
NC={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}

def read_glb(p):
 d=Path(p).read_bytes();magic,ver,total=struct.unpack_from('<4sII',d,0);assert magic==b'glTF' and ver==2
 off=12;js=None;bb=b''
 while off<total:
  n,t=struct.unpack_from('<II',d,off);off+=8;c=d[off:off+n];off+=n
  if t==0x4E4F534A:js=c
  elif t==0x004E4942:bb=c
 return json.loads(js.decode().rstrip(' \0')),bytearray(bb)
def write_glb(p,g,bb):
 bb=bytes(bb);g['buffers'][0]['byteLength']=len(bb);j=json.dumps(g,separators=(',',':'),ensure_ascii=False).encode();j+=b' '*((-len(j))%4);bb+=b'\0'*((-len(bb))%4);total=12+8+len(j)+8+len(bb);Path(p).write_bytes(struct.pack('<4sII',b'glTF',2,total)+struct.pack('<II',len(j),0x4E4F534A)+j+struct.pack('<II',len(bb),0x004E4942)+bb)
def acc(g,b,i):
 a=g['accessors'][i];v=g['bufferViews'][a['bufferView']];dt=np.dtype(DT[a['componentType']]).newbyteorder('<');n=NC[a['type']];s=v.get('byteOffset',0)+a.get('byteOffset',0);st=v.get('byteStride',dt.itemsize*n)
 if st==dt.itemsize*n:return np.frombuffer(b,dtype=dt,count=a['count']*n,offset=s).reshape(a['count'],n).copy()
 return np.ndarray((a['count'],n),dtype=dt,buffer=b,offset=s,strides=(st,dt.itemsize)).copy()
def putacc(g,b,i,x):
 a=g['accessors'][i];v=g['bufferViews'][a['bufferView']];dt=np.dtype(DT[a['componentType']]).newbyteorder('<');n=NC[a['type']];x=np.asarray(x,dtype=dt).reshape(a['count'],n);s=v.get('byteOffset',0)+a.get('byteOffset',0);st=v.get('byteStride',dt.itemsize*n)
 if st==dt.itemsize*n:b[s:s+x.nbytes]=x.tobytes()
 else:
  for k,r in enumerate(x):b[s+k*st:s+k*st+r.nbytes]=r.tobytes()
 if np.issubdtype(dt,np.floating):a['min']=x.min(0).astype(float).tolist();a['max']=x.max(0).astype(float).tolist()
def imgraw(g,b,i):
 v=g['bufferViews'][g['images'][i]['bufferView']];s=v.get('byteOffset',0);return bytes(b[s:s+v['byteLength']])
def gauss(x,y,cx,cy,rx,ry):return np.exp(-.5*(((x-cx)/rx)**2+((y-cy)/ry)**2))

def project_reference_skin(base,ref):
 ref=ref.convert('RGBA').resize((699,687),Image.Resampling.LANCZOS)
 mask=Image.new('L',(260,315),0);d=ImageDraw.Draw(mask);d.polygon([(28,8),(232,8),(258,92),(248,190),(212,275),(164,315),(96,315),(48,275),(12,190),(2,92)],fill=255);mask=mask.resize((699,687),Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(25))
 layer=Image.new('RGBA',base.size,(0,0,0,0));layer.paste(ref,(163,275),mask)
 return Image.alpha_composite(base,layer)
def textures(g,b,refpath):
 rep={}
 def load(i):return Image.open(io.BytesIO(imgraw(g,b,i))).convert('RGBA')
 a=np.array(load(0)).astype(np.float32);rgb=a[...,:3];m=np.clip((rgb[...,0]-rgb[...,1]+35)/130,0,1)[...,None];rgb=rgb*(1-.5*m)+np.array([220,118,137],np.float32)*(.5*m);rep[0]=Image.fromarray(np.dstack([rgb,a[...,3]]).clip(0,255).astype(np.uint8),'RGBA')
 a=np.array(load(4)).astype(np.float32);l=a[...,:3].mean(2)/255;o=np.zeros_like(a);o[...,0]=24+75*l;o[...,1]=70+125*l;o[...,2]=92+148*l;o[...,3]=a[...,3];rep[4]=Image.fromarray(o.clip(0,255).astype(np.uint8),'RGBA')
 a=np.array(load(6)).astype(np.float32);rgb=a[...,:3];lum=rgb.mean(2,keepdims=True);rgb=.72*rgb+.28*lum;rgb=.58*rgb+.42*np.array([247,222,218],np.float32);skin=Image.fromarray(np.dstack([rgb,a[...,3]]).clip(0,255).astype(np.uint8),'RGBA')
 if Path(refpath).exists():skin=project_reference_skin(skin,Image.open(refpath))
 rep[6]=skin
 for i,c in [(11,(100,87,99)),(12,(48,43,54))]:
  a=np.array(load(i)).astype(np.float32);al=a[...,3:4]/255;a[...,:3]=a[...,:3]*(1-al)+np.array(c,np.float32)*al;rep[i]=Image.fromarray(a.clip(0,255).astype(np.uint8),'RGBA')
 a=np.array(load(13)).astype(np.float32);lum=a[...,:3].mean(2);dark=lum<100;o=np.zeros_like(a);detail=(lum-128)*.08;o[...,0]=230+detail;o[...,1]=234+detail;o[...,2]=243+detail;o[dark,0]=31;o[dark,1]=45;o[dark,2]=72;o[...,3]=a[...,3];rep[13]=Image.fromarray(o.clip(0,255).astype(np.uint8),'RGBA')
 a=np.array(load(15)).astype(np.float32);rgb=a[...,:3];l=.2126*rgb[...,0]+.7152*rgb[...,1]+.0722*rgb[...,2];o=np.zeros_like(a);o[...,0]=150+.47*l;o[...,1]=155+.50*l;o[...,2]=174+.55*l;o[...,3]=a[...,3];rep[15]=Image.fromarray(o.clip(0,255).astype(np.uint8),'RGBA')
 if Path(refpath).exists():
  r=Image.open(refpath).convert('RGB');side=min(r.size);r=r.crop(((r.width-side)//2,(r.height-side)//2,(r.width+side)//2,(r.height+side)//2)).resize((512,512),Image.Resampling.LANCZOS);rep[17]=r.convert('RGBA')
 for i in (0,4,5,10,11,12):
  src=load(i);rep[i]=Image.new('RGBA',src.size,(0,0,0,0))
 out={}
 for i,im in rep.items():
  q=io.BytesIO();im.save(q,'PNG',optimize=True);out[g['images'][i]['bufferView']]=q.getvalue()
 return out

def deform(g,b):
 pidx=g['meshes'][0]['primitives'][0]['attributes']['POSITION'];v=acc(g,b,pidx).astype(np.float64);sets=[np.unique(acc(g,b,p['indices']).reshape(-1).astype(np.int64)) for p in g['meshes'][0]['primitives']];mouth,iris,hi,skin,white,brow,line=sets;allv=np.unique(np.concatenate(sets));x,y,z=v.T
 t=np.clip((1.438-y)/(1.438-1.337),0,1);x[allv]*=1-.035*t[allv]**1.15;lo=np.clip((1.394-y)/(1.394-1.337),0,1);y[allv]+=.0048*lo[allv]**1.2;up=np.clip((y-1.49)/.085,0,1);x[allv]*=1-.035*up[allv];chin=allv[(y[allv]<1.370)&(np.abs(x[allv])<.060)];ct=np.clip((1.370-y[chin])/.033,0,1);x[chin]*=1+.14*ct
 ears=skin[np.abs(x[skin])>.077];x[ears]*=.66;y[ears]=1.425+(y[ears]-1.425)*.76;z[ears]=.5*z[ears]+.5*np.maximum(z[ears],-.005)
 for cx in (-.044,.044):w=gauss(x,y,cx,1.405,.034,.030);z[skin]-=.0025*w[skin]
 wb=gauss(x,y,0,1.435,.014,.029);wt=gauss(x,y,0,1.403,.017,.017);wa=gauss(np.abs(x),y,.014,1.393,.010,.011);z[skin]-=.0008*wb[skin]+.0010*wt[skin]+.0005*wa[skin]
 nr=skin[(np.abs(x[skin])<.020)&(y[skin]>1.385)&(y[skin]<1.425)];z[nr]=np.maximum(z[nr],-.0585)
 mcx=x[mouth].mean();mcy=y[mouth].mean();x[mouth]=mcx+(x[mouth]-mcx)*1.18;y[mouth]=mcy+(y[mouth]-mcy)*1.05+.001;z[mouth]-=.0003;z[mouth]=np.maximum(z[mouth],-.0615)
 for ids,sx,sy in [(white,.91,.72),(line,.91,.75),(iris,.79,.79),(hi,.79,.79)]:
  for sign in (-1,1):
   q=ids[x[ids]*sign>0];cx=x[q].mean();cy=y[q].mean();x[q]=cx+(x[q]-cx)*sx;y[q]=cy+(y[q]-cy)*sy+.0012
 for sign in (-1,1):q=brow[x[brow]*sign>0];cx=x[q].mean();x[q]=cx+(x[q]-cx)*.95;y[q]-=.002
 v[:,0]=x;v[:,1]=y;v[:,2]=z;putacc(g,b,pidx,v.astype(np.float32))
 bp=g['meshes'][1]['primitives'];bodypos=bp[0]['attributes']['POSITION'];bv=acc(g,b,bodypos).astype(np.float64);hair=np.unique(acc(g,b,bp[1]['indices']).reshape(-1).astype(np.int64));cx=0;cy=1.47;cz=.02;bv[hair,0]=cx+(bv[hair,0]-cx)*.88;bv[hair,1]=cy+(bv[hair,1]-cy)*.97;bv[hair,2]=cz+(bv[hair,2]-cz)*.80;putacc(g,b,bodypos,bv.astype(np.float32))
 return {'face_vertices':len(v),'hair_vertices':len(hair),'morph_targets':len(g['meshes'][0]['primitives'][0].get('targets',[]))}

def qmat(q):
 x,y,z,w=q;return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w,0],[2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w,0],[2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y,0],[0,0,0,1]],float)
def local(n):
 if 'matrix'in n:return np.array(n['matrix']).reshape(4,4).T
 M=np.eye(4);M[:3,3]=n.get('translation',[0,0,0]);return M@qmat(n.get('rotation',[0,0,0,1]))@np.diag(n.get('scale',[1,1,1])+[1])
def globals_(g):
 par={}
 for i,n in enumerate(g['nodes']):
  for c in n.get('children',[]):par[c]=i
 C={}
 def go(i):
  if i not in C:C[i]=go(par[i])@local(g['nodes'][i]) if i in par else local(g['nodes'][i])
  return C[i]
 return [go(i) for i in range(len(g['nodes']))]
def seg(a,b,r=.0014,n=8):return trimesh.creation.cylinder(radius=r,segment=np.array([a,b]),sections=n)
def ell(loc,sc,sub=2):
 m=trimesh.creation.icosphere(subdivisions=sub,radius=1);m.apply_scale(sc);m.apply_translation(loc);return m
def join(ms):
 V=[];N=[];F=[];o=0
 for m in ms:V.append(np.asarray(m.vertices,np.float32));N.append(np.asarray(m.vertex_normals,np.float32));F.append(np.asarray(m.faces,np.uint32)+o);o+=len(m.vertices)
 return np.vstack(V),np.vstack(N),np.vstack(F)
def accessory_specs(g):
 def material(name,col,emit=None):
  m={'name':name,'pbrMetallicRoughness':{'baseColorFactor':col,'metallicFactor':0,'roughnessFactor':.48},'doubleSided':True}
  if emit:m['emissiveFactor']=emit
  return m
 hm=len(g['materials']);g['materials'].append(material('AINA_Silver_Updo',[.82,.84,.93,1]));mm=len(g['materials']);g['materials'].append(material('AINA_Hairpins',[.34,.45,.68,1]));cm=len(g['materials']);g['materials'].append(material('AINA_Core',[.05,.48,1,1],[.1,.65,1]));wm=len(g['materials']);g['materials'].append(material('AINA_Uniform',[.93,.95,.99,1]))
 G=globals_(g);head=10;chest=4;ih=np.linalg.inv(G[head]);ic=np.linalg.inv(G[chest])
 hair=[ell((0,1.586,.060),(.033,.028,.032),2)]
 for k in range(8):
  a=2*math.pi*k/8;hair.append(ell((.021*math.cos(a),1.588+.014*math.sin(a),.061+.009*math.cos(a)),(.012,.011,.013),2))
 for sign in (-1,1):
  for j in range(4):
   a=(sign*(.025+.014*j),1.568,.035);b=(sign*(.040+.010*j),1.530,-.015);c=(sign*(.052+.007*j),1.485,-.052);d=(sign*(.050+.006*j),1.440,-.066)
   hair.extend([seg(a,b,.00145),seg(b,c,.00135),seg(c,d,.00115)])
  hair.extend([seg((sign*.078,1.505,-.015),(sign*.092,1.445,-.052),.00125),seg((sign*.092,1.445,-.052),(sign*.082,1.387,-.040),.00105)])
 for x0 in (-.035,-.017,0,.017,.035):hair.extend([seg((x0,1.570,.032),(x0*.90,1.525,-.030),.00125),seg((x0*.90,1.525,-.030),(x0*.70,1.485,-.070),.0010)])
 V,N,F=join(hair);V=(np.c_[V,np.ones(len(V))]@ih.T)[:,:3].astype(np.float32);N=(N@ih[:3,:3].T).astype(np.float32);N/=np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-8)
 metal=[seg((-.063,1.585,.035),(-.020,1.622,.067),.0012),seg((.063,1.585,.035),(.020,1.622,.067),.0012)]
 pts=[]
 for i in range(17):
  a=-1.05+2.10*i/16;pts.append((.100*math.sin(a),1.585+.015*math.cos(a),.025+.045*math.cos(a)))
 for a,b in zip(pts[:-1],pts[1:]):metal.append(seg(a,b,.0011,7))
 MV,MN,MF=join(metal);MV=(np.c_[MV,np.ones(len(MV))]@ih.T)[:,:3].astype(np.float32);MN=(MN@ih[:3,:3].T).astype(np.float32);MN/=np.maximum(np.linalg.norm(MN,axis=1,keepdims=True),1e-8)
 core=trimesh.creation.icosphere(subdivisions=1,radius=.030);core.apply_scale((.70,1.05,.40));core.apply_transform(trimesh.transformations.rotation_matrix(math.pi/4,[0,0,1]));core.apply_translation((0,1.245,-.127));CV,CN,CF=join([core]);coll=[]
 for s in (-1,1):
  q=trimesh.creation.box(extents=(.042,.078,.017));q.apply_transform(trimesh.transformations.rotation_matrix(s*.27,[0,0,1]));q.apply_translation((s*.032,1.292,-.099));coll.append(q)
 WV,WN,WF=join(coll);CV=(np.c_[CV,np.ones(len(CV))]@ic.T)[:,:3].astype(np.float32);CN=(CN@ic[:3,:3].T).astype(np.float32);WV=(np.c_[WV,np.ones(len(WV))]@ic.T)[:,:3].astype(np.float32);WN=(WN@ic[:3,:3].T).astype(np.float32)
 return [('AINA_Updo',head,[(V,N,F,hm),(MV,MN,MF,mm)]),('AINA_Core_Collar',chest,[(CV,CN,CF,cm),(WV,WN,WF,wm)])]

def rebuild(g,b,repl,specs):
 payload=[]
 for i,v in enumerate(g['bufferViews']):s=v.get('byteOffset',0);payload.append(repl.get(i,bytes(b[s:s+v['byteLength']])))
 nb=bytearray()
 for i,r in enumerate(payload):
  while len(nb)%4:nb.append(0)
  g['bufferViews'][i]['byteOffset']=len(nb);g['bufferViews'][i]['byteLength']=len(r);nb.extend(r)
 def view(raw,target=None):
  while len(nb)%4:nb.append(0)
  i=len(g['bufferViews']);d={'buffer':0,'byteOffset':len(nb),'byteLength':len(raw)}
  if target:d['target']=target
  g['bufferViews'].append(d);nb.extend(raw);return i
 def accessor(x,t,ct,target=None):
  x=np.asarray(x);i=len(g['accessors']);a={'bufferView':view(x.tobytes(),target),'componentType':ct,'count':len(x),'type':t}
  if np.issubdtype(x.dtype,np.floating):a['min']=x.min(0).astype(float).tolist();a['max']=x.max(0).astype(float).tolist()
  g['accessors'].append(a);return i
 for name,parent,prims in specs:
  m={'name':name,'primitives':[]}
  for V,N,F,mat in prims:m['primitives'].append({'attributes':{'POSITION':accessor(V.astype('<f4'),'VEC3',5126,34962),'NORMAL':accessor(N.astype('<f4'),'VEC3',5126,34962)},'indices':accessor(F.reshape(-1).astype('<u4'),'SCALAR',5125,34963),'material':mat,'mode':4})
  mi=len(g['meshes']);g['meshes'].append(m);ni=len(g['nodes']);g['nodes'].append({'name':name,'mesh':mi});g['nodes'][parent].setdefault('children',[]).append(ni)
 g['buffers'][0]['byteLength']=len(nb);return nb

def vrm1(g):
 old=g['extensions']['VRM'];bones={q['bone']:{'node':q['node']} for q in old['humanoid']['humanBones']};groups={q.get('presetName'):q for q in old['blendShapeMaster']['blendShapeGroups']};mp={'neutral':'neutral','a':'aa','i':'ih','u':'ou','e':'ee','o':'oh','blink':'blink','blink_l':'blinkLeft','blink_r':'blinkRight','angry':'angry','fun':'relaxed','joy':'happy','sorrow':'sad','unknown':'surprised'};pre={}
 for a,n in mp.items():
  q=groups.get(a)
  if q:pre[n]={'morphTargetBinds':[{'node':59,'index':z['index'],'weight':z.get('weight',100)/100} for z in q.get('binds',[])],'isBinary':False,'overrideBlink':'none','overrideLookAt':'none','overrideMouth':'none'}
 g['extensions']['VRMC_vrm']={'specVersion':'1.0','meta':{'name':'AINA','version':'1.0 V8','authors':['Shenzhen Uoon Technology Co., Ltd.'],'copyrightInformation':'AINA Digital Human','contactInformation':'','references':['AINA approved identity board'],'thirdPartyLicenses':'VRoid sample base provenance retained in package.','thumbnailImage':17,'avatarPermission':'onlyAuthor','allowExcessivelyViolentUsage':False,'allowExcessivelySexualUsage':False,'commercialUsage':'corporation','creditNotation':'required','allowRedistribution':False,'modification':'allowModificationRedistribution','otherLicenseUrl':'https://opensource.org/licenses/MIT'},'humanoid':{'humanBones':bones},'firstPerson':{'meshAnnotations':[{'node':59,'type':'auto'},{'node':60,'type':'auto'}]},'lookAt':{'offsetFromHeadBone':[0,.06,0],'type':'bone','rangeMapHorizontalInner':{'inputMaxValue':90,'outputScale':10},'rangeMapHorizontalOuter':{'inputMaxValue':90,'outputScale':10},'rangeMapVerticalDown':{'inputMaxValue':90,'outputScale':10},'rangeMapVerticalUp':{'inputMaxValue':90,'outputScale':10}},'expressions':{'preset':pre,'custom':{}}}
 g['extensionsUsed']=[x for x in g.get('extensionsUsed',[]) if x!='VRM'];g['extensionsUsed'].append('VRMC_vrm');g['extensionsRequired']=[x for x in g.get('extensionsRequired',[]) if x!='VRM'];g['extensionsRequired'].append('VRMC_vrm');return len(bones),len(pre)
def main():
 if len(sys.argv)<4:raise SystemExit('build_aina_v8.py INPUT REF OUTPUT_DIR')
 inp=Path(sys.argv[1]);ref=Path(sys.argv[2]);out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True);g,b=read_glb(inp)
 for m in g.get('materials',[]):
  if 'extensions' in m and 'KHR_materials_unlit' in m['extensions']:
   m['extensions'].pop('KHR_materials_unlit',None)
   if not m['extensions']:m.pop('extensions',None)
 for i in (0,1,2,4,5,6):
  m=g['materials'][i];m['alphaMode']='BLEND';m['pbrMetallicRoughness']['baseColorFactor']=[1,1,1,0]
 r=deform(g,b);repl=textures(g,b,ref);spec=accessory_specs(g);nb=rebuild(g,b,repl,spec);bn,ep=vrm1(g);g['asset']['generator']='AINA VRM Production V8';vrm=out/'AINA_VRM_CANDIDATE_V8.vrm';write_glb(vrm,g,nb);(out/'AINA_VISUAL_MASTER_V8.glb').write_bytes(vrm.read_bytes());report={'version':'V8','vrm1':True,'humanoid_bones':bn,'expression_presets':ep,'geometry':r,'accessories':['silver updo','hairpins','white collar','ice-blue core'],'identity_lock':False,'visual_identity_lock':False};(out/'AINA_V8_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
