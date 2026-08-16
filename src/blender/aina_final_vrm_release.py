#!/usr/bin/env python3
"""Release wrapper for AINA final VRM assembly.

Blender 4.5 treats MPFB2 v2 as an extension package and its services require an
installed extension context. Production CI must not depend on UI installation
state, so this wrapper replaces only MPFB body/rig creation with a deterministic
Blender-native humanoid mesh + armature. The identity-locked v15.5 face, 52
shape controls, VRM expressions, LookAt, spring bones, materials and exporter
remain the canonical implementation in aina_final_vrm_assembly.py.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
import bpy
from mathutils import Vector

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import aina_final_vrm_assembly as core


def enable_vrm_only(root:Path):
    vrm_src=root/'vendor'/'VRM-Addon-for-Blender'/'src'
    if str(vrm_src) not in sys.path: sys.path.insert(0,str(vrm_src))
    import io_scene_vrm
    try: io_scene_vrm.register()
    except Exception as e:
        if 'already registered' not in str(e).lower(): core.log(f'VRM register note: {e}')
    return None,None,None


def _basis(axis):
    a=np.asarray(axis,float); a/=max(np.linalg.norm(a),1e-12)
    ref=np.array([0.,0.,1.]) if abs(a[2])<.90 else np.array([0.,1.,0.])
    u=np.cross(a,ref);u/=max(np.linalg.norm(u),1e-12);v=np.cross(a,u);v/=max(np.linalg.norm(v),1e-12)
    return a,u,v


def _build_body_mesh(height):
    # Coordinates are authored for a 1.72 m avatar and uniformly scaled if requested.
    S=float(height/1.72); verts=[]; faces=[]; influences=[]
    def add_tube(p0,p1,r0,r1,bone,segments=18,rings=5,oval=(1.0,.86)):
        p0=np.asarray(p0,float)*S;p1=np.asarray(p1,float)*S;r0*=S;r1*=S
        axis,u,v=_basis(p1-p0);start=len(verts)
        for j in range(rings):
            t=j/(rings-1);c=p0*(1-t)+p1*t;r=r0*(1-t)+r1*t
            for i in range(segments):
                a=2*math.pi*i/segments; q=c+u*(math.cos(a)*r*oval[0])+v*(math.sin(a)*r*oval[1]);verts.append(q.tolist());influences.append(bone)
        for j in range(rings-1):
            for i in range(segments):
                ni=(i+1)%segments;a=start+j*segments+i;b=start+j*segments+ni;c=start+(j+1)*segments+ni;d=start+(j+1)*segments+i;faces.extend([(a,b,c),(a,c,d)])
        # end caps
        for j,center in ((0,p0),(rings-1,p1)):
            ci=len(verts);verts.append(center.tolist());influences.append(bone);base=start+j*segments
            for i in range(segments):
                ni=(i+1)%segments;faces.append((ci,base+ni,base+i) if j==0 else (ci,base+i,base+ni))
    def add_ellipsoid(center,radii,bone,seg=24,rings=12):
        c=np.asarray(center,float)*S;r=np.asarray(radii,float)*S;start=len(verts)
        for j in range(rings+1):
            th=math.pi*j/rings
            for i in range(seg):
                ph=2*math.pi*i/seg;q=c+np.array([r[0]*math.sin(th)*math.cos(ph),r[1]*math.sin(th)*math.sin(ph),r[2]*math.cos(th)]);verts.append(q.tolist());influences.append(bone)
        for j in range(rings):
            for i in range(seg):
                ni=(i+1)%seg;a=start+j*seg+i;b=start+j*seg+ni;c1=start+(j+1)*seg+ni;d=start+(j+1)*seg+i;faces.extend([(a,b,c1),(a,c1,d)])
    # Feminine athletic suit silhouette, neck stops at the head graft boundary.
    add_ellipsoid((0,0,.91),(.165,.105,.145),'pelvis',28,12)
    add_tube((0,0,.96),(0,0,1.18),.145,.170,'spine_01',24,7,(1.0,.72))
    add_tube((0,0,1.17),(0,0,1.38),.170,.185,'spine_02',28,7,(1.0,.68))
    add_tube((0,0,1.36),(0,0,1.46),.170,.105,'spine_03',24,5,(1.0,.72))
    add_tube((0,0,1.43),(0,0,1.51),.058,.052,'neck_01',20,4,(1.0,.92))
    for sg,suf in ((1,'l'),(-1,'r')):
        add_ellipsoid((sg*.185,0,1.385),(.075,.082,.082),f'clavicle_{suf}',20,9)
        add_tube((sg*.18,0,1.38),(sg*.47,0,1.35),.061,.048,f'upperarm_{suf}',18,6,(1.0,.94))
        add_ellipsoid((sg*.47,0,1.35),(.053,.050,.053),f'lowerarm_{suf}',18,8)
        add_tube((sg*.48,0,1.35),(sg*.70,0,1.34),.047,.035,f'lowerarm_{suf}',18,5,(1.0,.92))
        add_ellipsoid((sg*.745,-.002,1.34),(.070,.040,.025),f'hand_{suf}',20,8)
        add_tube((sg*.095,0,.88),(sg*.105,0,.49),.090,.065,f'thigh_{suf}',20,7,(1.0,.88))
        add_ellipsoid((sg*.105,0,.49),(.068,.064,.070),f'calf_{suf}',18,8)
        add_tube((sg*.105,0,.48),(sg*.105,0,.125),.062,.043,f'calf_{suf}',20,7,(1.0,.88))
        add_ellipsoid((sg*.105,-.060,.075),(.070,.135,.045),f'foot_{suf}',22,9)
    return np.asarray(verts,float),np.asarray(faces,np.int32),influences


def _new_bone(arm,name,h,t,parent=None):
    b=arm.edit_bones.new(name);b.head=Vector(h);b.tail=Vector(t)
    if parent and arm.edit_bones.get(parent):b.parent=arm.edit_bones[parent]
    return b


def _finger_chain(arm,side,suf,hand_tip,z,spread):
    sg=1 if suf=='l' else -1
    root=np.array([hand_tip,0,z],float);dx=.025*sg
    defs=[('thumb',-.028),('index',-.013),('middle',0),('ring',.013),('pinky',.026)]
    names=[]
    for label,yo in defs:
        base=root+np.array([0,yo,0]);parent=f'hand_{suf}'
        if label=='thumb': parts=['01','02','03']
        else: parts=['01','02','03']
        for k,part in enumerate(parts):
            nm=f'{label}_{part}_{suf}';h=base+np.array([dx*k,0,0]);t=base+np.array([dx*(k+1),0,0]);_new_bone(arm,nm,h,t,parent);parent=nm;names.append(nm)
    return names


def _create_rig(height):
    S=float(height/1.72); arm=bpy.data.armatures.new('AINA_Humanoid_Armature');rig=bpy.data.objects.new('AINA_Humanoid_Rig',arm);bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
    def B(n,h,t,p=None):return _new_bone(arm,n,tuple(np.asarray(h)*S),tuple(np.asarray(t)*S),p)
    B('pelvis',(0,0,.82),(0,0,.96));B('spine_01',(0,0,.96),(0,0,1.12),'pelvis');B('spine_02',(0,0,1.12),(0,0,1.29),'spine_01');B('spine_03',(0,0,1.29),(0,0,1.43),'spine_02');B('neck_01',(0,0,1.43),(0,0,1.52),'spine_03');B('head',(0,0,1.52),(0,0,1.68),'neck_01')
    for sg,suf in ((1,'l'),(-1,'r')):
        B(f'clavicle_{suf}',(sg*.04,0,1.39),(sg*.18,0,1.39),'spine_03');B(f'upperarm_{suf}',(sg*.18,0,1.39),(sg*.47,0,1.35),f'clavicle_{suf}');B(f'lowerarm_{suf}',(sg*.47,0,1.35),(sg*.70,0,1.34),f'upperarm_{suf}');B(f'hand_{suf}',(sg*.70,0,1.34),(sg*.79,0,1.34),f'lowerarm_{suf}')
        B(f'thigh_{suf}',(sg*.085,0,.89),(sg*.105,0,.49),'pelvis');B(f'calf_{suf}',(sg*.105,0,.49),(sg*.105,0,.125),f'thigh_{suf}');B(f'foot_{suf}',(sg*.105,0,.125),(sg*.105,-.13,.065),f'calf_{suf}');B(f'ball_{suf}',(sg*.105,-.13,.065),(sg*.105,-.205,.055),f'foot_{suf}')
        _finger_chain(arm,sg,suf,.79*S,1.34*S,.02)
    bpy.ops.object.mode_set(mode='OBJECT');bpy.context.view_layer.update();return rig


def create_native_body(_hs,_ts,_hp,height):
    rig=_create_rig(height);v,f,inf=_build_body_mesh(height);body=core.mesh_object('AINA_Body_Base',v,f)
    groups={}
    for i,bone in enumerate(inf):groups.setdefault(bone,[]).append(i)
    for bone,ids in groups.items():
        vg=body.vertex_groups.new(name=bone);vg.add(ids,1.0,'REPLACE')
    mod=body.modifiers.new('AINA_Armature','ARMATURE');mod.object=rig;body.parent=rig
    for p in body.data.polygons:p.use_smooth=True
    core.log(f'Blender-native production body: {len(v)} vertices / {len(f)} triangles / {len(groups)} weighted groups')
    return body,rig

core.enable_addons=enable_vrm_only
core.create_body=create_native_body

if __name__=='__main__':
    core.main()
