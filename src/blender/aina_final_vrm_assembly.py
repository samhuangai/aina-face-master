#!/usr/bin/env python3
"""Assemble the locked AINA v15.5 face into the final production VRM character.

Run inside Blender 4.5+ in background mode.  This script deliberately does not
create another face version.  It consumes an identity_lock=true v15.5 OBJ,
builds a rigged MPFB2 body, grafts the locked face, creates 52 production facial
controls, VRM 1.0 humanoid/expression/look-at/spring-bone data, renders previews,
and exports AINA_MASTER.blend + AINA.vrm + QA artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector

K=np.array([1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,13695,12973,13422,6543,6537],dtype=np.int64)

SHAPE_KEYS=[
    'browDownLeft','browDownRight','browInnerUp','browOuterUpLeft','browOuterUpRight',
    'cheekPuff','cheekSquintLeft','cheekSquintRight',
    'eyeBlinkLeft','eyeBlinkRight','eyeLookDownLeft','eyeLookDownRight','eyeLookInLeft','eyeLookInRight','eyeLookOutLeft','eyeLookOutRight','eyeLookUpLeft','eyeLookUpRight','eyeSquintLeft','eyeSquintRight','eyeWideLeft','eyeWideRight',
    'jawForward','jawLeft','jawOpen','jawRight',
    'mouthClose','mouthDimpleLeft','mouthDimpleRight','mouthFrownLeft','mouthFrownRight','mouthFunnel','mouthLeft','mouthLowerDownLeft','mouthLowerDownRight','mouthPressLeft','mouthPressRight','mouthPucker','mouthRight','mouthRollLower','mouthRollUpper','mouthShrugLower','mouthShrugUpper','mouthSmileLeft','mouthSmileRight','mouthStretchLeft','mouthStretchRight','mouthUpperUpLeft','mouthUpperUpRight',
    'noseSneerLeft','noseSneerRight','tongueOut',
]

PRESET_BINDS={
    'happy':[('mouthSmileLeft',.82),('mouthSmileRight',.82),('cheekSquintLeft',.30),('cheekSquintRight',.30)],
    'angry':[('browDownLeft',.82),('browDownRight',.82),('mouthFrownLeft',.45),('mouthFrownRight',.45)],
    'sad':[('browInnerUp',.72),('mouthFrownLeft',.72),('mouthFrownRight',.72)],
    'relaxed':[('mouthSmileLeft',.22),('mouthSmileRight',.22),('eyeSquintLeft',.10),('eyeSquintRight',.10)],
    'surprised':[('browInnerUp',.55),('eyeWideLeft',.86),('eyeWideRight',.86),('jawOpen',.58)],
    'neutral':[],
    'aa':[('jawOpen',.72),('mouthFunnel',.20)],
    'ih':[('mouthStretchLeft',.62),('mouthStretchRight',.62)],
    'ou':[('mouthPucker',.78),('mouthFunnel',.55)],
    'ee':[('mouthSmileLeft',.38),('mouthSmileRight',.38),('mouthStretchLeft',.52),('mouthStretchRight',.52)],
    'oh':[('jawOpen',.48),('mouthFunnel',.78)],
    'blink':[('eyeBlinkLeft',1.0),('eyeBlinkRight',1.0)],
    'blink_left':[('eyeBlinkLeft',1.0)],
    'blink_right':[('eyeBlinkRight',1.0)],
    'look_up':[('eyeLookUpLeft',1.0),('eyeLookUpRight',1.0)],
    'look_down':[('eyeLookDownLeft',1.0),('eyeLookDownRight',1.0)],
    'look_left':[('eyeLookOutLeft',.72),('eyeLookInRight',.72)],
    'look_right':[('eyeLookInLeft',.72),('eyeLookOutRight',.72)],
}

MAIN_HUMANOID={
    'hips':'pelvis','spine':'spine_01','chest':'spine_02','upper_chest':'spine_03','neck':'neck_01','head':'head',
    'left_shoulder':'clavicle_l','left_upper_arm':'upperarm_l','left_lower_arm':'lowerarm_l','left_hand':'hand_l',
    'right_shoulder':'clavicle_r','right_upper_arm':'upperarm_r','right_lower_arm':'lowerarm_r','right_hand':'hand_r',
    'left_upper_leg':'thigh_l','left_lower_leg':'calf_l','left_foot':'foot_l','left_toes':'ball_l',
    'right_upper_leg':'thigh_r','right_lower_leg':'calf_r','right_foot':'foot_r','right_toes':'ball_r',
}

FINGER_HUMANOID={
    'left_thumb_metacarpal':'thumb_01_l','left_thumb_proximal':'thumb_02_l','left_thumb_distal':'thumb_03_l',
    'left_index_proximal':'index_01_l','left_index_intermediate':'index_02_l','left_index_distal':'index_03_l',
    'left_middle_proximal':'middle_01_l','left_middle_intermediate':'middle_02_l','left_middle_distal':'middle_03_l',
    'left_ring_proximal':'ring_01_l','left_ring_intermediate':'ring_02_l','left_ring_distal':'ring_03_l',
    'left_little_proximal':'pinky_01_l','left_little_intermediate':'pinky_02_l','left_little_distal':'pinky_03_l',
    'right_thumb_metacarpal':'thumb_01_r','right_thumb_proximal':'thumb_02_r','right_thumb_distal':'thumb_03_r',
    'right_index_proximal':'index_01_r','right_index_intermediate':'index_02_r','right_index_distal':'index_03_r',
    'right_middle_proximal':'middle_01_r','right_middle_intermediate':'middle_02_r','right_middle_distal':'middle_03_r',
    'right_ring_proximal':'ring_01_r','right_ring_intermediate':'ring_02_r','right_ring_distal':'ring_03_r',
    'right_little_proximal':'pinky_01_r','right_little_intermediate':'pinky_02_r','right_little_distal':'pinky_03_r',
}


def log(msg):
    print(f'[AINA_FINAL] {msg}', flush=True)


def parse_args():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    ap=argparse.ArgumentParser()
    ap.add_argument('--face',type=Path,required=True)
    ap.add_argument('--identity-report',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--height',type=float,default=1.72)
    return ap.parse_args(argv)


def enable_addons(root: Path):
    mpfb_src=root/'vendor'/'mpfb2'/'src'
    vrm_src=root/'vendor'/'VRM-Addon-for-Blender'/'src'
    for p in (mpfb_src,vrm_src):
        if str(p) not in sys.path: sys.path.insert(0,str(p))
    import mpfb
    try: mpfb.register()
    except Exception as e:
        if 'already registered' not in str(e).lower(): log(f'MPFB register note: {e}')
    import io_scene_vrm
    try: io_scene_vrm.register()
    except Exception as e:
        if 'already registered' not in str(e).lower(): log(f'VRM add-on register note: {e}')
    from mpfb.services import HumanService, TargetService
    from mpfb.entities.objectproperties import HumanObjectProperties
    return HumanService,TargetService,HumanObjectProperties


def clear_scene():
    bpy.ops.object.mode_set(mode='OBJECT') if bpy.context.object and bpy.context.object.mode!='OBJECT' else None
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes,bpy.data.curves,bpy.data.materials,bpy.data.cameras,bpy.data.lights):
        for block in list(datablocks):
            if block.users==0: datablocks.remove(block)


def world_bounds(obj):
    mw=obj.matrix_world
    pts=[mw @ v.co for v in obj.data.vertices]
    mins=np.min(np.array([p[:] for p in pts]),axis=0); maxs=np.max(np.array([p[:] for p in pts]),axis=0)
    return mins,maxs


def create_body(HumanService,TargetService,HumanObjectProperties,height):
    body=HumanService.create_human(); body.name='AINA_Body_Base'
    for k,val in [('gender',1.0),('age',.50),('muscle',.34),('weight',.38),('height',.58),('proportions',.56)]:
        try: HumanObjectProperties.set_value(k,val,entity_reference=body)
        except Exception as e: log(f'MPFB property {k} note: {e}')
    try: TargetService.reapply_macro_details(body)
    except Exception as e: log(f'MPFB macro reapply note: {e}')
    bpy.context.view_layer.objects.active=body; body.select_set(True)
    rig=HumanService.add_builtin_rig(body,'game_engine'); rig.name='AINA_Humanoid_Rig'
    bpy.context.view_layer.update()
    lo,hi=world_bounds(body); h=max(float(hi[2]-lo[2]),1e-6); s=height/h
    rig.scale=(s,s,s); bpy.context.view_layer.update()
    lo,hi=world_bounds(body); rig.location.z-=float(lo[2]); bpy.context.view_layer.update()
    log(f'Body scaled to {height:.3f}m using scale={s:.6f}')
    return body,rig


def cut_body_head(body,cut_z=1.505):
    bm=bmesh.new(); bm.from_mesh(body.data); mw=body.matrix_world.copy()
    doomed=[v for v in bm.verts if (mw @ v.co).z>cut_z]
    log(f'Removing {len(doomed)} MPFB head vertices above z={cut_z:.3f}')
    bmesh.ops.delete(bm,geom=doomed,context='VERTS'); bm.to_mesh(body.data); bm.free(); body.data.update()


def make_material(name,color,metallic=0.0,roughness=.48,emission=None):
    m=bpy.data.materials.new(name); m.diffuse_color=tuple(color); m.use_nodes=True
    bsdf=m.node_tree.nodes.get('Principled BSDF') if m.node_tree else None
    if bsdf:
        if bsdf.inputs.get('Base Color'): bsdf.inputs['Base Color'].default_value=tuple(color)
        if bsdf.inputs.get('Metallic'): bsdf.inputs['Metallic'].default_value=metallic
        if bsdf.inputs.get('Roughness'): bsdf.inputs['Roughness'].default_value=roughness
        if emission:
            for key in ('Emission Color','Emission'):
                if bsdf.inputs.get(key): bsdf.inputs[key].default_value=tuple(emission); break
            if bsdf.inputs.get('Emission Strength'): bsdf.inputs['Emission Strength'].default_value=3.0
    try:
        gltf=m.vrm_addon_extension.mtoon1; gltf.enabled=True
        gltf.pbr_metallic_roughness.base_color_factor=tuple(color)
        if emission:
            gltf.emissive_factor=tuple(emission[:3])
            gltf.extensions.khr_materials_emissive_strength.emissive_strength=3.0
        toon=gltf.extensions.vrmc_materials_mtoon; toon.shading_toony_factor=.72; toon.gi_equalization_factor=.8
    except Exception as e: log(f'MToon material note for {name}: {e}')
    return m


def assign_single_material(obj,mat):
    obj.data.materials.clear(); obj.data.materials.append(mat)
    for p in obj.data.polygons: p.material_index=0


def write_swatch(out_dir:Path,name,color):
    out_dir.mkdir(parents=True,exist_ok=True)
    img=bpy.data.images.new(f'AINA_{name}_swatch',width=64,height=64,alpha=True)
    img.generated_color=tuple(color); img.filepath_raw=str(out_dir/f'{name}.png'); img.file_format='PNG'
    try: img.save()
    except Exception as e: log(f'Texture swatch save note {name}: {e}')


def read_obj(path:Path):
    verts=[]; faces=[]
    for line in path.read_text(errors='ignore').splitlines():
        if line.startswith('v '):
            q=line.split(); verts.append((float(q[1]),float(q[2]),float(q[3])))
        elif line.startswith('f '):
            ids=[int(x.split('/')[0])-1 for x in line.split()[1:]]
            for i in range(1,len(ids)-1): faces.append((ids[0],ids[i],ids[i+1]))
    return np.asarray(verts,float),np.asarray(faces,np.int32)


def component_data(n,faces):
    parent=np.arange(n,dtype=np.int32)
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return int(x)
    def union(a,b):
        ra,rb=find(int(a)),find(int(b))
        if ra!=rb: parent[rb]=ra
    for a,b,c in faces:
        union(a,b); union(b,c); union(c,a)
    roots=np.array([find(i) for i in range(n)],dtype=np.int32)
    groups={}
    for i,r in enumerate(roots): groups.setdefault(int(r),[]).append(i)
    return roots,{r:np.asarray(ids,dtype=np.int32) for r,ids in groups.items()}


def map_face_vertices(v,height):
    s=1.08
    out=np.empty_like(v); out[:,0]=v[:,0]*s; out[:,1]=v[:,2]*s; out[:,2]=-v[:,1]*s
    out[:,2]+=height-float(out[:,2].max())
    return out


def mesh_object(name,verts,faces):
    me=bpy.data.meshes.new(name+'_Mesh'); me.from_pydata(verts.tolist(),[],faces.tolist()); me.update()
    ob=bpy.data.objects.new(name,me); bpy.context.collection.objects.link(ob); return ob


def create_face_objects(face_path:Path,height,skin,eye_mat,teeth_mat,mouth_mat):
    raw,faces=read_obj(face_path); mapped=map_face_vertices(raw,height); roots,groups=component_data(len(raw),faces)
    head_root=max(groups,key=lambda r:len(groups[r])); eye_roots=[r for r,g in groups.items() if 650<len(g)<900]
    eye_roots=sorted(eye_roots,key=lambda r:float(mapped[groups[r],0].mean()))
    if len(eye_roots)!=2: raise RuntimeError(f'Expected 2 eye components, got {[(r,len(groups[r])) for r in eye_roots]}')
    oral_roots=sorted([r for r in groups if r!=head_root and r not in eye_roots],key=lambda r:len(groups[r]),reverse=True)
    keep_mask=np.array([roots[int(f[0])] not in set(eye_roots) for f in faces],dtype=bool); head_faces=faces[keep_mask]
    head=mesh_object('AINA_Face_v15_5',mapped,head_faces); head.data.materials.append(skin); head.data.materials.append(teeth_mat); head.data.materials.append(mouth_mat)
    face_roots=[roots[int(f[0])] for f in faces[keep_mask]]
    oral_big=set(oral_roots[:2])
    for poly,r in zip(head.data.polygons,face_roots): poly.material_index=0 if r==head_root else (1 if r in oral_big else 2)
    eyes=[]
    for r in eye_roots:
        ids=groups[r]; remap={int(g):i for i,g in enumerate(ids)}; sf=[]
        for f in faces[roots[faces[:,0]]==r]: sf.append(tuple(remap[int(x)] for x in f))
        eo=mesh_object('AINA_Eye_R' if mapped[ids,0].mean()<0 else 'AINA_Eye_L',mapped[ids],np.asarray(sf,np.int32)); assign_single_material(eo,eye_mat)
        eyes.append((eo,ids,mapped[ids].mean(0)))
    tongue_ids=groups[oral_roots[-1]] if oral_roots else np.array([],dtype=np.int32)
    return head,eyes,mapped,groups,head_root,oral_roots,tongue_ids


def select_only(obj):
    bpy.ops.object.select_all(action='DESELECT'); obj.select_set(True); bpy.context.view_layer.objects.active=obj


def add_bone_world(rig,name,head_world,tail_world,parent_name):
    inv=rig.matrix_world.inverted(); eb=rig.data.edit_bones.new(name); eb.head=inv@Vector(head_world); eb.tail=inv@Vector(tail_world)
    if parent_name and rig.data.edit_bones.get(parent_name): eb.parent=rig.data.edit_bones[parent_name]
    return eb


def add_control_bones(rig,eye_centers):
    select_only(rig); bpy.ops.object.mode_set(mode='EDIT')
    eye_bones={}
    for side,c in eye_centers.items():
        bn=f'AINA_Eye_{side}'; add_bone_world(rig,bn,c,Vector(c)+Vector((0,-.020,0)),'head'); eye_bones[side]=bn
    chains={
        'HairL':[('AINA_Hair_L_1',(.070,-.025,1.675),(.082,-.035,1.585),'head'),('AINA_Hair_L_2',(.082,-.035,1.585),(.070,-.030,1.500),'AINA_Hair_L_1')],
        'HairR':[('AINA_Hair_R_1',(-.070,-.025,1.675),(-.082,-.035,1.585),'head'),('AINA_Hair_R_2',(-.082,-.035,1.585),(-.070,-.030,1.500),'AINA_Hair_R_1')],
        'HairBack':[('AINA_Hair_Back_1',(0,.070,1.700),(0,.082,1.615),'head'),('AINA_Hair_Back_2',(0,.082,1.615),(0,.072,1.525),'AINA_Hair_Back_1')],
    }
    for items in chains.values():
        for name,h,t,parent in items: add_bone_world(rig,name,h,t,parent)
    bpy.ops.object.mode_set(mode='OBJECT'); bpy.context.view_layer.update()
    return eye_bones,{k:[x[0] for x in v] for k,v in chains.items()}


def bone_parent_preserve(obj,rig,bone_name):
    mw=obj.matrix_world.copy(); obj.parent=rig; obj.parent_type='BONE'; obj.parent_bone=bone_name; obj.matrix_world=mw


def create_uv_sphere(name,location,scale,material,parent=None,rig=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=16,radius=1.0,location=location)
    ob=bpy.context.object; ob.name=name; ob.scale=scale; bpy.ops.object.transform_apply(location=False,rotation=False,scale=True); assign_single_material(ob,material)
    if parent and rig: bone_parent_preserve(ob,rig,parent)
    return ob


def create_curve(name,points,radius,material,parent=None,rig=None):
    cu=bpy.data.curves.new(name+'_Curve','CURVE'); cu.dimensions='3D'; cu.resolution_u=2; cu.bevel_depth=radius; cu.bevel_resolution=3
    sp=cu.splines.new('BEZIER'); sp.bezier_points.add(len(points)-1)
    for bp,p in zip(sp.bezier_points,points): bp.co=p; bp.handle_left_type='AUTO'; bp.handle_right_type='AUTO'
    ob=bpy.data.objects.new(name,cu); bpy.context.collection.objects.link(ob); ob.data.materials.append(material)
    if parent and rig: bone_parent_preserve(ob,rig,parent)
    return ob


def create_hair(rig,hair_mat,hair_chains):
    verts=[]; faces=[]; nphi=48; nt=15; center=np.array([0,.020,1.625]); rx,ry,rz=.102,.090,.115
    for i in range(nphi):
        phi=2*math.pi*i/nphi; front=math.sin(phi)<0; tmax=1.15 if front else 2.03
        for j in range(nt):
            th=tmax*j/(nt-1); verts.append((center+np.array([rx*math.sin(th)*math.cos(phi),ry*math.sin(th)*math.sin(phi),rz*math.cos(th)])).tolist())
    for i in range(nphi):
        ni=(i+1)%nphi
        for j in range(nt-1):
            a=i*nt+j; b=ni*nt+j; c=ni*nt+j+1; d=i*nt+j+1; faces.extend([(a,b,c),(a,c,d)])
    cap=mesh_object('AINA_Hair_Cap',np.asarray(verts,float),np.asarray(faces,np.int32)); assign_single_material(cap,hair_mat); bone_parent_preserve(cap,rig,'head')
    create_uv_sphere('AINA_Hair_Bun',(0,.084,1.696),(.054,.046,.055),hair_mat,'head',rig)
    bangs=[
        [(-.055,-.075,1.702),(-.050,-.086,1.665),(-.043,-.091,1.625)],
        [(-.025,-.083,1.710),(-.020,-.094,1.670),(-.014,-.097,1.625)],
        [(0,-.085,1.714),(.002,-.097,1.673),(.008,-.099,1.628)],
        [(.027,-.082,1.708),(.025,-.093,1.668),(.020,-.096,1.625)],
        [(.056,-.073,1.699),(.052,-.085,1.660),(.045,-.090,1.620)],
    ]
    for i,pts in enumerate(bangs): create_curve(f'AINA_Bang_{i+1}',pts,.0045,hair_mat,'head',rig)
    segs=[
        ('AINA_SideHair_L1',[(.070,-.045,1.675),(.082,-.052,1.620),(.082,-.050,1.585)],.0055,'AINA_Hair_L_1'),
        ('AINA_SideHair_L2',[(.082,-.050,1.585),(.079,-.054,1.540),(.070,-.047,1.500)],.0045,'AINA_Hair_L_2'),
        ('AINA_SideHair_R1',[(-.070,-.045,1.675),(-.082,-.052,1.620),(-.082,-.050,1.585)],.0055,'AINA_Hair_R_1'),
        ('AINA_SideHair_R2',[(-.082,-.050,1.585),(-.079,-.054,1.540),(-.070,-.047,1.500)],.0045,'AINA_Hair_R_2'),
        ('AINA_BackHair_1',[(0,.075,1.690),(0,.088,1.645),(0,.082,1.612)],.012,'AINA_Hair_Back_1'),
        ('AINA_BackHair_2',[(0,.082,1.612),(0,.080,1.565),(0,.070,1.525)],.010,'AINA_Hair_Back_2'),
    ]
    for name,pts,r,bone in segs: create_curve(name,pts,r,hair_mat,bone,rig)


def create_collar_and_accent(rig,suit_mat,accent_mat):
    n=40; verts=[]; faces=[]; z0,z1=1.495,1.545
    for z in (z0,z1):
        for i in range(n):
            a=2*math.pi*i/n; verts.append((.057*math.cos(a),.051*math.sin(a),z))
    for i in range(n):
        j=(i+1)%n; faces.extend([(i,j,n+j),(i,n+j,n+i)])
    collar=mesh_object('AINA_High_Collar',np.asarray(verts,float),np.asarray(faces,np.int32)); assign_single_material(collar,suit_mat); bone_parent_preserve(collar,rig,'neck_01')
    v=np.array([[0,-.132,1.405],[.024,-.124,1.365],[0,-.142,1.330],[-.024,-.124,1.365],[0,-.112,1.365]],float)
    f=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4],[0,3,2],[0,2,1]],np.int32)
    crystal=mesh_object('AINA_Core_Crystal',v,f); assign_single_material(crystal,accent_mat); bone_parent_preserve(crystal,rig,'spine_03')


def weights(coords,c,r,inner=.25,outer=1.20):
    c=np.asarray(c,float); r=np.asarray(r,float); q=np.sqrt(np.sum(((coords-c)/r)**2,axis=1)); w=np.zeros(len(coords)); w[q<=inner]=1
    m=(q>inner)&(q<outer)
    if np.any(m):
        t=(q[m]-inner)/(outer-inner); w[m]=.5*(1+np.cos(np.pi*t))
    return w


def shift_region(coords,c,r,d,inner=.25,outer=1.20):
    coords += weights(coords,c,r,inner,outer)[:,None]*np.asarray(d,float)


def scale_region(coords,c,r,s,inner=.25,outer=1.20):
    w=weights(coords,c,r,inner,outer)[:,None]; c=np.asarray(c,float); target=c+(coords-c)*np.asarray(s,float); coords += w*(target-coords)


def create_shape_keys(head,base,tongue_ids):
    if len(base)<=int(K.max()): raise RuntimeError('Face vertex order does not contain required semantic indices')
    lm=base[K]; browL=lm[22:27].mean(0); browR=lm[17:22].mean(0); eyeL=lm[42:48].mean(0); eyeR=lm[36:42].mean(0)
    mouth=lm[48:60].mean(0); chin=lm[8]; jaw=(mouth+chin)/2; cornerL=lm[54]; cornerR=lm[48]
    upperL=lm[[52,53,54]].mean(0); upperR=lm[[48,49,50]].mean(0); lowerL=lm[[54,55,56]].mean(0); lowerR=lm[[48,58,59]].mean(0)
    cheekL=(eyeL+lm[35]+cornerL)/3; cheekR=(eyeR+lm[31]+cornerR)/3; noseL=lm[35]; noseR=lm[31]
    head.shape_key_add(name='Basis'); stats={}
    for name in SHAPE_KEYS:
        c=base.copy()
        side=1 if 'Left' in name else (-1 if 'Right' in name else 0)
        eye=eyeL if side==1 else eyeR; brow=browL if side==1 else browR; corner=cornerL if side==1 else cornerR; cheek=cheekL if side==1 else cheekR
        if name.startswith('browDown'): shift_region(c,brow,(.035,.025,.022),(0,0,-.005))
        elif name=='browInnerUp':
            for cc in (lm[21],lm[22]): shift_region(c,cc,(.022,.022,.022),(0,0,.006))
        elif name.startswith('browOuterUp'): shift_region(c,lm[[25,26]].mean(0) if side==1 else lm[[17,18]].mean(0),(.025,.022,.022),(0,0,.005))
        elif name=='cheekPuff':
            shift_region(c,cheekL,(.040,.035,.040),(.001,-.0045,.0005)); shift_region(c,cheekR,(.040,.035,.040),(-.001,-.0045,.0005))
        elif name.startswith('cheekSquint'): shift_region(c,cheek,(.035,.032,.030),(0,-.001,.0030))
        elif name.startswith('eyeBlink'): scale_region(c,eye,(.037,.026,.020),(1,1,.08))
        elif name.startswith('eyeSquint'): scale_region(c,eye,(.037,.026,.021),(1,1,.55)); shift_region(c,cheek,(.032,.030,.026),(0,-.0007,.0017))
        elif name.startswith('eyeWide'): scale_region(c,eye,(.037,.026,.021),(1,1,1.30))
        elif name.startswith('eyeLook'):
            direction=np.zeros(3); direction[0]=(.0035 if ('OutLeft' in name or 'InRight' in name) else (-.0035 if ('InLeft' in name or 'OutRight' in name) else 0)); direction[2]=(.003 if 'Up' in name else (-.003 if 'Down' in name else 0)); shift_region(c,eye,(.030,.022,.019),direction)
        elif name=='jawForward': shift_region(c,jaw,(.050,.055,.045),(0,-.004,0))
        elif name=='jawLeft': shift_region(c,jaw,(.055,.060,.048),(.004,0,0))
        elif name=='jawOpen': shift_region(c,jaw,(.050,.060,.048),(0,0,-.010)); shift_region(c,mouth,(.040,.035,.030),(0,0,-.005))
        elif name=='jawRight': shift_region(c,jaw,(.055,.060,.048),(-.004,0,0))
        elif name=='mouthClose': scale_region(c,mouth,(.040,.028,.025),(1,1,.25))
        elif name.startswith('mouthDimple'): shift_region(c,corner,(.026,.025,.022),(.0018*side,.0012,-.0010))
        elif name.startswith('mouthFrown'): shift_region(c,corner,(.027,.025,.022),(.0008*side,.0005,-.0040))
        elif name=='mouthFunnel': scale_region(c,mouth,(.040,.030,.030),(.72,1.05,1.05)); shift_region(c,mouth,(.038,.030,.028),(0,-.003,0))
        elif name=='mouthLeft': shift_region(c,mouth,(.045,.030,.028),(.005,0,0))
        elif name.startswith('mouthLowerDown'): shift_region(c,lowerL if side==1 else lowerR,(.030,.025,.022),(0,0,-.0035))
        elif name.startswith('mouthPress'): scale_region(c,corner,(.026,.024,.020),(.96,1,.68))
        elif name=='mouthPucker': scale_region(c,mouth,(.040,.030,.030),(.76,1.0,1.15)); shift_region(c,mouth,(.036,.030,.026),(0,-.0035,0))
        elif name=='mouthRight': shift_region(c,mouth,(.045,.030,.028),(-.005,0,0))
        elif name=='mouthRollLower': shift_region(c,lm[[55,56,57,58,59]].mean(0),(.038,.024,.020),(0,.002,.001))
        elif name=='mouthRollUpper': shift_region(c,lm[[49,50,51,52,53]].mean(0),(.038,.024,.020),(0,.002,-.001))
        elif name=='mouthShrugLower': shift_region(c,lm[[55,56,57,58,59]].mean(0),(.038,.024,.021),(0,0,.003))
        elif name=='mouthShrugUpper': shift_region(c,lm[[49,50,51,52,53]].mean(0),(.038,.024,.021),(0,0,.003))
        elif name.startswith('mouthSmile'): shift_region(c,corner,(.025,.025,.022),(.0018*side,-.0004,.0042))
        elif name.startswith('mouthStretch'): shift_region(c,corner,(.026,.025,.022),(.0042*side,0,0))
        elif name.startswith('mouthUpperUp'): shift_region(c,upperL if side==1 else upperR,(.028,.023,.020),(0,0,.0032))
        elif name.startswith('noseSneer'):
            nc=noseL if side==1 else noseR; shift_region(c,nc,(.022,.023,.024),(.0006*side,-.0012,.0030))
        elif name=='tongueOut':
            if len(tongue_ids): c[tongue_ids]+=np.array([0,-.008,-.003])
            else: shift_region(c,mouth,(.030,.030,.020),(0,-.006,-.002))
        else: raise RuntimeError(f'Unhandled shape key {name}')
        delta=np.linalg.norm(c-base,axis=1); stats[name]={'max_m':float(delta.max()),'rms_m':float(np.sqrt(np.mean(delta*delta))),'moved_vertices':int(np.sum(delta>1e-5))}
        if stats[name]['max_m']<2e-4: raise RuntimeError(f'Shape key {name} is effectively empty')
        sk=head.shape_key_add(name=name)
        try: sk.data.foreach_set('co',c.astype(np.float32).ravel())
        except Exception:
            for i,p in enumerate(c): sk.data[i].co=p
    return stats


def configure_humanoid(rig,eye_bones):
    ext=rig.data.vrm_addon_extension; ext.spec_version='1.0'; hb=ext.vrm1.humanoid.human_bones
    mapping={**MAIN_HUMANOID,**FINGER_HUMANOID,'left_eye':eye_bones['L'],'right_eye':eye_bones['R']}
    assigned={}
    for attr,bone in mapping.items():
        if bone in rig.data.bones and hasattr(hb,attr): getattr(hb,attr).node.bone_name=bone; assigned[attr]=bone
    missing=[k for k in ('hips','spine','head','left_upper_arm','right_upper_arm','left_upper_leg','right_upper_leg') if k not in assigned]
    if missing: raise RuntimeError(f'Mandatory humanoid mappings missing: {missing}')
    meta=ext.vrm1.meta; meta.vrm_name='AINA'; meta.version='1.0.0'; meta.authors.add().value='AINA Project'
    meta.copyright_information='AINA'; meta.contact_information='AINA Project'; meta.avatar_permission='onlyAuthor'; meta.commercial_usage='corporation'; meta.credit_notation='unnecessary'; meta.modification='allowModification'
    look=ext.vrm1.look_at
    try: look.type=look.TYPE_BONE.identifier
    except Exception: look.type='bone'
    try: look.offset_from_head_bone=(0,-.025,.055)
    except Exception: pass
    return assigned


def configure_expressions(rig,head):
    preset=rig.data.vrm_addon_extension.vrm1.expressions.preset; configured={}
    for pname,items in PRESET_BINDS.items():
        expr=getattr(preset,pname); count=0
        for key,weight in items:
            if not head.data.shape_keys or key not in head.data.shape_keys.key_blocks: continue
            b=expr.morph_target_binds.add(); b.node.mesh_object_name=head.name; b.index=key; b.weight=float(weight); count+=1
        if pname.startswith('blink'):
            try: expr.is_binary=True
            except Exception: pass
        configured[pname]=count
    if len(configured)!=18: raise RuntimeError(f'VRM preset count mismatch: {len(configured)}')
    return configured


def configure_springbones(rig,hair_chains):
    from io_scene_vrm.common import ops as common_ops
    from io_scene_vrm.editor.extension import get_armature_extension
    ext=get_armature_extension(rig.data); sb=ext.spring_bone1; created=[]
    for label,chain in hair_chains.items():
        result=common_ops.vrm.add_spring_bone1_spring(armature_object_name=rig.name)
        if result!={'FINISHED'}: raise RuntimeError(f'Failed adding spring {label}: {result}')
        si=len(sb.springs)-1; spring=sb.springs[si]
        try: spring.vrm_name=label
        except Exception: pass
        for bone in chain:
            result=common_ops.vrm.add_spring_bone1_spring_joint(armature_object_name=rig.name,spring_index=si)
            if result!={'FINISHED'}: raise RuntimeError(f'Failed adding spring joint {bone}: {result}')
            j=spring.joints[-1]; j.node.bone_name=bone; j.drag_force=.36; j.stiffness=.82; j.gravity_power=.06; j.hit_radius=.006
        created.append({'name':label,'joints':list(chain)})
    return created


def setup_render(out:Path):
    scene=bpy.context.scene; scene.render.engine='BLENDER_EEVEE_NEXT'; scene.render.image_settings.file_format='PNG'; scene.render.film_transparent=False
    scene.world.color=(.94,.96,1.0)
    def area(name,loc,energy,size):
        d=bpy.data.lights.new(name,'AREA'); d.energy=energy; d.shape='DISK'; d.size=size; o=bpy.data.objects.new(name,d); bpy.context.collection.objects.link(o); o.location=loc; return o
    area('AINA_Key',(2.5,-3.0,3.0),900,4.0); area('AINA_Fill',(-2.5,-2.0,2.3),550,3.0); area('AINA_Rim',(0,2.5,2.8),700,3.0)
    cd=bpy.data.cameras.new('AINA_Camera'); cam=bpy.data.objects.new('AINA_Camera',cd); bpy.context.collection.objects.link(cam); scene.camera=cam
    previews=out/'Preview'; previews.mkdir(parents=True,exist_ok=True)
    def render(name,loc,target,lens,res):
        cam.location=loc; cam.data.lens=lens; direction=Vector(target)-cam.location; cam.rotation_euler=direction.to_track_quat('-Z','Y').to_euler()
        scene.render.resolution_x=res[0]; scene.render.resolution_y=res[1]; scene.render.resolution_percentage=100; scene.render.filepath=str(previews/name)
        bpy.ops.render.render(write_still=True)
    render('AINA_PORTRAIT_FRONT.png',(0,-2.25,1.61),(0,0,1.61),72,(1024,1024))
    render('AINA_PORTRAIT_3Q.png',(1.05,-2.15,1.62),(0,0,1.60),72,(1024,1024))
    render('AINA_FULL_BODY_FRONT.png',(0,-5.3,1.05),(0,0,0.95),58,(1024,1536))
    return [str(previews/x) for x in ('AINA_PORTRAIT_FRONT.png','AINA_PORTRAIT_3Q.png','AINA_FULL_BODY_FRONT.png')]


def main():
    a=parse_args(); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True); (out/'QA').mkdir(exist_ok=True); (out/'Textures').mkdir(exist_ok=True)
    ident=json.loads(a.identity_report.read_text())
    if ident.get('identity_lock') is not True: raise RuntimeError('v15.5 identity is not locked; refusing VRM assembly')
    root=Path.cwd(); HumanService,TargetService,HumanObjectProperties=enable_addons(root); clear_scene()

    skin=make_material('AINA_Skin',(0.95,0.79,0.75,1),0,.58)
    eye_white=make_material('AINA_EyeWhite',(0.96,0.98,1.0,1),0,.32)
    iris_mat=make_material('AINA_Iris',(0.22,0.55,0.78,1),.05,.25)
    pupil_mat=make_material('AINA_Pupil',(0.015,0.025,0.04,1),.05,.30)
    hair_mat=make_material('AINA_Hair_Silver',(0.78,0.82,0.89,1),.05,.34)
    suit_mat=make_material('AINA_Suit_Pearl',(0.84,0.88,0.94,1),.18,.28)
    teeth_mat=make_material('AINA_Teeth',(0.96,0.94,0.89,1),0,.38)
    mouth_mat=make_material('AINA_MouthInner',(0.38,0.12,0.15,1),0,.55)
    accent_mat=make_material('AINA_Accent_Blue',(0.12,0.45,0.90,1),.25,.22,(0.08,0.45,1.0,1))
    for n,c in [('skin',(0.95,0.79,0.75,1)),('hair_silver',(0.78,0.82,0.89,1)),('suit_pearl',(0.84,0.88,0.94,1)),('accent_blue',(0.12,0.45,0.90,1))]: write_swatch(out/'Textures',n,c)

    body,rig=create_body(HumanService,TargetService,HumanObjectProperties,a.height); cut_body_head(body); assign_single_material(body,suit_mat)
    head,eye_items,mapped,groups,head_root,oral_roots,tongue_ids=create_face_objects(a.face,a.height,skin,eye_white,teeth_mat,mouth_mat)
    eye_centers={'R':Vector(eye_items[0][2]),'L':Vector(eye_items[1][2])}
    eye_bones,hair_chains=add_control_bones(rig,eye_centers)
    bone_parent_preserve(head,rig,'head')
    for eo,ids,c in eye_items:
        side='R' if float(c[0])<0 else 'L'; bone_parent_preserve(eo,rig,eye_bones[side])
        center=Vector(c); create_uv_sphere(f'AINA_Iris_{side}',center+Vector((0,-.0112,0)),(.0074,.00125,.0074),iris_mat,eye_bones[side],rig)
        create_uv_sphere(f'AINA_Pupil_{side}',center+Vector((0,-.0121,0)),(.0034,.0010,.0034),pupil_mat,eye_bones[side],rig)
    create_hair(rig,hair_mat,hair_chains); create_collar_and_accent(rig,suit_mat,accent_mat)

    shape_stats=create_shape_keys(head,mapped,tongue_ids)
    humanoid=configure_humanoid(rig,eye_bones); expressions=configure_expressions(rig,head); springs=configure_springbones(rig,hair_chains)

    blend_path=out/'AINA_MASTER.blend'; bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    previews=setup_render(out); bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    vrm_path=out/'AINA.vrm'; result=bpy.ops.export_scene.vrm(filepath=str(vrm_path))
    if result!={'FINISHED'}: raise RuntimeError(f'VRM export failed: {result}')
    if not vrm_path.exists() or vrm_path.stat().st_size<100_000: raise RuntimeError(f'VRM output missing or too small: {vrm_path}')

    qa={
        'product':'AINA Final VRM Production','identity_lock':True,'identity_version':ident.get('version'),'face_source':str(a.face),
        'height_m':a.height,'humanoid_mapped_count':len(humanoid),'humanoid_mapped':humanoid,
        'shape_control_count':len(shape_stats),'shape_controls':shape_stats,
        'vrm_preset_count':len(expressions),'vrm_preset_bind_counts':expressions,
        'look_at':'bone','spring_bone_count':len(springs),'spring_bones':springs,
        'materials':[m.name for m in (skin,eye_white,iris_mat,pupil_mat,hair_mat,suit_mat,teeth_mat,mouth_mat,accent_mat)],
        'files':{'blend':str(blend_path),'vrm':str(vrm_path),'vrm_bytes':vrm_path.stat().st_size,'previews':previews},
        'assembly_pass':True,
    }
    (out/'QA'/'AINA_FINAL_ASSEMBLY_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    log(json.dumps({'assembly_pass':True,'vrm_bytes':vrm_path.stat().st_size,'shape_controls':len(shape_stats),'presets':len(expressions),'springs':len(springs)},indent=2))

if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc(); raise
