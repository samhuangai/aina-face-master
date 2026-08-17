#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys,math
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector

K=np.array([1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,13695,12973,13422,6543,6537],dtype=np.int64)

def args():
 a=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [];p=argparse.ArgumentParser();p.add_argument('--face',type=Path,required=True);p.add_argument('--out',type=Path,required=True);return p.parse_args(a)

def read_obj(path):
 v=[];f=[]
 for line in path.read_text(errors='ignore').splitlines():
  if line.startswith('v '):q=line.split();v.append((float(q[1]),float(q[2]),float(q[3])))
  elif line.startswith('f '):
   ids=[int(x.split('/')[0])-1 for x in line.split()[1:]]
   for i in range(1,len(ids)-1):f.append((ids[0],ids[i],ids[i+1]))
 return np.asarray(v,float),np.asarray(f,np.int32)

def mapv(v):
 o=np.empty_like(v);o[:,0]=v[:,0]*1.08;o[:,1]=v[:,2]*1.08;o[:,2]=-v[:,1]*1.08;o[:,2]+=1.72-float(o[:,2].max());return o

def groups(n,faces):
 par=np.arange(n,dtype=np.int32)
 def find(x):
  while par[x]!=x:par[x]=par[par[x]];x=par[x]
  return int(x)
 def union(a,b):
  a=find(int(a));b=find(int(b));
  if a!=b:par[b]=a
 for a,b,c in faces:union(a,b);union(b,c);union(c,a)
 d={}
 for i in range(n):d.setdefault(find(i),[]).append(i)
 return [np.asarray(x,np.int32) for x in d.values()]

def mat(name,c,rough=.45,metal=.0):
 m=bpy.data.materials.new(name);m.diffuse_color=tuple(c);m.use_nodes=True;bs=m.node_tree.nodes.get('Principled BSDF')
 if bs:
  bs.inputs['Base Color'].default_value=tuple(c);bs.inputs['Roughness'].default_value=rough;bs.inputs['Metallic'].default_value=metal
 return m

def mesh(name,v,f,m):
 me=bpy.data.meshes.new(name+'Mesh');me.from_pydata(v.tolist(),[],f.tolist());me.update();o=bpy.data.objects.new(name,me);bpy.context.collection.objects.link(o);me.materials.append(m);return o

def curve(name,pts,r,m):
 cu=bpy.data.curves.new(name+'Curve','CURVE');cu.dimensions='3D';cu.bevel_depth=r;cu.bevel_resolution=2;cu.resolution_u=2;sp=cu.splines.new('BEZIER');sp.bezier_points.add(len(pts)-1)
 for bp,p in zip(sp.bezier_points,pts):bp.co=p;bp.handle_left_type='AUTO';bp.handle_right_type='AUTO'
 o=bpy.data.objects.new(name,cu);bpy.context.collection.objects.link(o);cu.materials.append(m);return o

def sphere(name,c,scale,m):
 bpy.ops.mesh.primitive_uv_sphere_add(segments=28,ring_count=14,radius=1,location=c);o=bpy.context.object;o.name=name;o.scale=scale;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);o.data.materials.append(m);return o

def main():
 a=args();a.out.mkdir(parents=True,exist_ok=True);bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
 raw,faces=read_obj(a.face);v=mapv(raw);gs=groups(len(v),faces);head=max(gs,key=len);hm=np.zeros(len(v),bool);hm[head]=True;hf=faces[hm[faces].all(1)];g={int(q):i for i,q in enumerate(head)};lf=np.asarray([[g[int(x)] for x in tri] for tri in hf],np.int32)
 skin=mat('AINA_Skin',(0.965,.84,.82,1),.52);white=mat('AINA_EyeWhite',(.97,.98,1,1),.28);iris=mat('AINA_Iris',(.34,.61,.78,1),.22,.02);pupil=mat('AINA_Pupil',(.012,.018,.030,1),.20);dark=mat('AINA_LashBrow',(.04,.05,.065,1),.38);hair=mat('AINA_Hair_Silver',(.80,.84,.91,1),.32,.03)
 mesh('AINA_Head',v[head],lf,skin)
 eye_groups=sorted([q for q in gs if 650<len(q)<900],key=lambda q:float(v[q,0].mean()))
 if len(eye_groups)!=2:raise RuntimeError(f'Expected two eyes, got {[len(q) for q in eye_groups]}')
 for side,q in zip(('R','L'),eye_groups):
  mp={int(x):i for i,x in enumerate(q)};mask=np.zeros(len(v),bool);mask[q]=True;efi=faces[mask[faces].all(1)];ef=np.asarray([[mp[int(x)] for x in tri] for tri in efi],np.int32);mesh('AINA_Eye_'+side,v[q],ef,white);c=v[q].mean(0);sphere('AINA_Iris_'+side,c+np.array([0,-.0112,0]),(.0084,.0012,.0084),iris);sphere('AINA_Pupil_'+side,c+np.array([0,-.0122,0]),(.0037,.0009,.0037),pupil)
 lm=v[K]
 for n,ids,lo in [('R',[36,37,38,39],[41,40,39]),('L',[42,43,44,45],[47,46,45])]:curve('UpperLash_'+n,[Vector(lm[i])+Vector((0,-.0022,0)) for i in ids],.0011,dark);curve('LowerLash_'+n,[Vector(lm[i])+Vector((0,-.0020,0)) for i in lo],.00042,dark)
 curve('Brow_R',[Vector(lm[i])+Vector((0,-.002,.003)) for i in [17,18,19,20,21]],.00155,dark);curve('Brow_L',[Vector(lm[i])+Vector((0,-.002,.003)) for i in [22,23,24,25,26]],.00155,dark)
 # compact silver center-part hair silhouette, kept off the face for identity judgement
 for i,pts in enumerate([
 [(-.085,-.015,1.70),(-.092,-.045,1.64),(-.080,-.055,1.56),(-.065,-.045,1.48)],[(.085,-.015,1.70),(.092,-.045,1.64),(.080,-.055,1.56),(.065,-.045,1.48)],
 [(-.060,-.070,1.71),(-.047,-.092,1.665),(-.030,-.103,1.615)],[(-.032,-.082,1.715),(-.022,-.101,1.675),(-.012,-.108,1.63)],
 [(.032,-.082,1.715),(.022,-.101,1.675),(.012,-.108,1.63)],[(.060,-.070,1.71),(.047,-.092,1.665),(.030,-.103,1.615)]
 ]):curve('Hair_'+str(i),[Vector(p) for p in pts],.0040 if i<2 else .0030,hair)
 scene=bpy.context.scene;scene.render.engine='BLENDER_EEVEE_NEXT';scene.render.image_settings.file_format='PNG';scene.render.resolution_x=640;scene.render.resolution_y=640;scene.render.resolution_percentage=100;scene.world.color=(.94,.95,.98)
 def area(name,loc,e,size):d=bpy.data.lights.new(name,'AREA');d.energy=e;d.size=size;o=bpy.data.objects.new(name,d);bpy.context.collection.objects.link(o);o.location=loc
 area('Key',(1.5,-2.2,2.3),600,3);area('Fill',(-1.5,-1.7,2.0),380,2.5);area('Rim',(0,1.4,2.2),450,2.4)
 cd=bpy.data.cameras.new('Cam');cam=bpy.data.objects.new('Cam',cd);bpy.context.collection.objects.link(cam);scene.camera=cam
 def shot(name,loc,target,lens=92):cam.location=loc;cam.data.lens=lens;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(a.out/name);bpy.ops.render.render(write_still=True)
 shot('AINA_HEAD_FRONT.png',(0,-1.75,1.615),(0,0,1.615));shot('AINA_HEAD_3Q.png',(.58,-1.65,1.62),(0,0,1.61));shot('AINA_HEAD_PROFILE.png',(1.65,0,1.62),(0,0,1.61),88)
 bpy.ops.wm.save_as_mainfile(filepath=str(a.out/'AINA_HEAD_VISUAL_GATE.blend'));print('[AINA_HEAD_GATE] rendered real Blender front/3Q/profile',flush=True)
if __name__=='__main__':main()
