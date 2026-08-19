#!/usr/bin/env python3
"""Real 3D AINA eye system for visual identity production."""
from __future__ import annotations
import math
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector


def _mesh(name,verts,faces,material):
    me=bpy.data.meshes.new(name+'_Mesh');me.from_pydata([tuple(v) for v in verts],[],[tuple(f) for f in faces]);me.update()
    ob=bpy.data.objects.new(name,me);bpy.context.collection.objects.link(ob)
    if material:me.materials.append(material)
    for p in me.polygons:p.use_smooth=True
    return ob


def _almond(name,center,material,side):
    c=np.asarray(center,float);rx=.0155;rz=.00545;n=56;boundary=[]
    for i in range(n+1):
        u=-1+2*i/n
        z=rz*(max(0.,1-u*u)**.58)+(0.00042*u if side=='L' else -0.00042*u)
        y=-.00925-.00115*(1-u*u);boundary.append((c[0]+rx*u,y,c[2]+z))
    for i in range(n+1):
        u=1-2*i/n
        z=-rz*.72*(max(0.,1-u*u)**.74)+(0.00022*u if side=='L' else -0.00022*u)
        y=-.00920-.00090*(1-u*u);boundary.append((c[0]+rx*u,y,c[2]+z))
    verts=[(c[0],-.01045,c[2])]+boundary;faces=[];m=len(boundary)
    for i in range(m):faces.append((0,1+i,1+((i+1)%m)))
    ob=_mesh(name,verts,faces,material);ob.shape_key_add(name='Basis')
    blink='eyeBlinkLeft' if side=='L' else 'eyeBlinkRight';wide='eyeWideLeft' if side=='L' else 'eyeWideRight';squint='eyeSquintLeft' if side=='L' else 'eyeSquintRight'
    kb=ob.shape_key_add(name=blink)
    for p in kb.data:p.co.y+=.030
    kw=ob.shape_key_add(name=wide)
    for p in kw.data:p.co.z=c[2]+(p.co.z-c[2])*1.15
    ks=ob.shape_key_add(name=squint)
    for p in ks.data:p.co.z=c[2]+(p.co.z-c[2])*.54
    return ob


def _disc(name,center,radius,material,side,pupil=False):
    c=np.asarray(center,float);n=56;verts=[tuple(c)];faces=[]
    for i in range(n):
        a=2*math.pi*i/n;x=radius*math.cos(a);z=radius*(1.04 if not pupil else 1.0)*math.sin(a);y=c[1]-.00024*(1-(x/max(radius,1e-6))**2)
        verts.append((c[0]+x,y,c[2]+z))
    for i in range(n):faces.append((0,1+i,1+((i+1)%n)))
    ob=_mesh(name,verts,faces,material);ob.shape_key_add(name='Basis')
    blink='eyeBlinkLeft' if side=='L' else 'eyeBlinkRight';kb=ob.shape_key_add(name=blink)
    for p in kb.data:p.co.y+=.030
    dirs={('L','eyeLookUpLeft'):(0,0,.0020),('R','eyeLookUpRight'):(0,0,.0020),('L','eyeLookDownLeft'):(0,0,-.0018),('R','eyeLookDownRight'):(0,0,-.0018),('L','eyeLookInLeft'):(-.0018,0,0),('R','eyeLookInRight'):(.0018,0,0),('L','eyeLookOutLeft'):(.0018,0,0),('R','eyeLookOutRight'):(-.0018,0,0)}
    for (s,key),d in dirs.items():
        if s!=side:continue
        k=ob.shape_key_add(name=key)
        for p in k.data:p.co+=Vector(d)
    return ob


def install(visual,release):
    base=visual.base;original_uv_sphere=base.create_uv_sphere;original_configure=base.configure_expressions
    def create_face_objects(face_path:Path,height,skin,eye_mat,teeth_mat,mouth_mat):
        raw,faces=base.read_obj(face_path);mapped=base.map_face_vertices(raw,height);roots,groups=base.component_data(len(raw),faces)
        head_root=max(groups,key=lambda r:len(groups[r]));eye_roots=[r for r,g in groups.items() if 650<len(g)<900];eye_roots=sorted(eye_roots,key=lambda r:float(mapped[groups[r],0].mean()))
        if len(eye_roots)!=2:raise RuntimeError('Expected two FaceVerse eye components')
        oral_roots=sorted([r for r in groups if r!=head_root and r not in eye_roots],key=lambda r:len(groups[r]),reverse=True)
        mapped=visual.polish_real_face(mapped,groups[head_root],[groups[r] for r in eye_roots])
        keep_mask=np.array([roots[int(f[0])] not in set(eye_roots) for f in faces],dtype=bool);head_faces=faces[keep_mask]
        head=base.mesh_object('AINA_Face_v15_5',mapped,head_faces);head.data.materials.append(skin);head.data.materials.append(teeth_mat);head.data.materials.append(mouth_mat)
        face_roots=[roots[int(f[0])] for f in faces[keep_mask]];oral_big=set(oral_roots[:2])
        for poly,r in zip(head.data.polygons,face_roots):poly.material_index=0 if r==head_root else (1 if r in oral_big else 2);poly.use_smooth=True
        lm=mapped[visual.K];centers={'R':lm[36:42].mean(0),'L':lm[42:48].mean(0)};eyes=[]
        for side in ('R','L'):
            c=centers[side].copy();c[1]=0.;eo=_almond('AINA_Eye_'+side,c,eye_mat,side);rid=eye_roots[0] if side=='R' else eye_roots[1];eyes.append((eo,np.asarray(groups[rid],np.int32),c))
        tongue_ids=groups[oral_roots[-1]] if oral_roots else np.array([],dtype=np.int32)
        return head,eyes,mapped,groups,head_root,oral_roots,tongue_ids
    def create_uv_sphere(name,location,scale,material,parent=None,rig=None):
        if name.startswith('AINA_Iris_') or name.startswith('AINA_Pupil_'):
            side=name.rsplit('_',1)[-1];loc=np.asarray(location,float);radius=.00465 if name.startswith('AINA_Iris_') else .00195;loc[1]=-.01115 if name.startswith('AINA_Iris_') else -.01152
            ob=_disc(name,loc,radius,material,side,pupil=name.startswith('AINA_Pupil_'))
            if parent and rig:base.bone_parent_preserve(ob,rig,parent)
            return ob
        return original_uv_sphere(name,location,scale,material,parent,rig)
    def configure_expressions(rig,head):
        configured=original_configure(rig,head)
        from io_scene_vrm.editor.extension import get_armature_extension
        preset=get_armature_extension(rig.data).vrm1.expressions.preset
        for pname,items in base.PRESET_BINDS.items():
            expr=getattr(preset,pname)
            for key,weight in items:
                for oname in ('AINA_Eye_L','AINA_Eye_R','AINA_Iris_L','AINA_Iris_R','AINA_Pupil_L','AINA_Pupil_R'):
                    o=bpy.data.objects.get(oname)
                    if not o or not o.data.shape_keys or key not in o.data.shape_keys.key_blocks:continue
                    b=expr.morph_target_binds.add();b.node.mesh_object_name=o.name;b.index=key;b.weight=float(weight)
        return configured
    def setup_render(out:Path):
        scene=bpy.context.scene;scene.render.engine='BLENDER_EEVEE_NEXT';scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False;scene.world.color=(.055,.065,.085)
        for o in scene.objects:
            if o.type=='MESH':
                for p in o.data.polygons:p.use_smooth=True
        for o in list(scene.objects):
            if o.type in {'LIGHT','CAMERA'}:bpy.data.objects.remove(o,do_unlink=True)
        def area(name,loc,energy,size):
            d=bpy.data.lights.new(name,'AREA');d.energy=energy;d.shape='DISK';d.size=size;o=bpy.data.objects.new(name,d);bpy.context.collection.objects.link(o);o.location=loc;o.rotation_euler=(Vector((0,0,1.58))-o.location).to_track_quat('-Z','Y').to_euler()
        area('AINA_Key',(1.7,-2.4,2.5),420,3.2);area('AINA_Fill',(-1.8,-1.8,2.0),210,2.8);area('AINA_Rim',(0,1.9,2.4),300,2.6)
        cd=bpy.data.cameras.new('AINA_Camera');cam=bpy.data.objects.new('AINA_Camera',cd);bpy.context.collection.objects.link(cam);scene.camera=cam;previews=out/'Preview';previews.mkdir(parents=True,exist_ok=True)
        def clear_all():
            for o in scene.objects:
                if o.type=='MESH' and o.data.shape_keys:
                    for kb in o.data.shape_keys.key_blocks:kb.value=0.
        def apply(vals):
            for o in scene.objects:
                if o.type!='MESH' or not o.data.shape_keys:continue
                for k,v in vals.items():
                    if k in o.data.shape_keys.key_blocks:o.data.shape_keys.key_blocks[k].value=float(v)
        def render(name,loc,target,vals,res=(768,768)):
            clear_all();apply(vals);cam.location=loc;cam.data.lens=78;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.resolution_x=res[0];scene.render.resolution_y=res[1];scene.render.resolution_percentage=100;scene.render.filepath=str(previews/name);bpy.ops.render.render(write_still=True)
        cases={'AINA_REAL_NEUTRAL_FRONT.png':{},'AINA_REAL_HAPPY_FRONT.png':{'mouthSmileLeft':.82,'mouthSmileRight':.82,'cheekSquintLeft':.30,'cheekSquintRight':.30},'AINA_REAL_SURPRISED_FRONT.png':{'browInnerUp':.55,'eyeWideLeft':.86,'eyeWideRight':.86,'jawOpen':.58},'AINA_REAL_BLINK_FRONT.png':{'eyeBlinkLeft':1.,'eyeBlinkRight':1.},'AINA_REAL_AA_FRONT.png':{'jawOpen':.72,'mouthFunnel':.20}}
        for name,vals in cases.items():render(name,(0,-1.22,1.615),(0,0,1.615),vals)
        render('AINA_REAL_NEUTRAL_3Q.png',(.43,-1.13,1.62),(0,0,1.605),{});render('AINA_REAL_FULL_BODY_FRONT.png',(0,-4.7,1.05),(0,0,.98),{},(1024,1536));clear_all();return [str(p) for p in sorted(previews.glob('AINA_REAL_*.png'))]
    base.create_face_objects=create_face_objects;base.create_uv_sphere=create_uv_sphere;base.configure_expressions=configure_expressions;base.setup_render=setup_render
