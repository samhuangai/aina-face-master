#!/usr/bin/env python3
"""Patch AINA_MASTER with the v15.5 visual-final mesh and render real identity QA.

All 52 Shape Keys receive the same Basis delta, so expression deltas are
preserved exactly while the identity mesh changes. This review also adds the
actual 3D eyebrow / upper-lash geometry and re-centres eyes/irises/pupils before
rendering front, shallow 3Q and profile portraits from the real Blender model.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

K=np.array([1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,13695,12973,13422,6543,6537],dtype=np.int64)


def argv():
    a=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    ap=argparse.ArgumentParser();ap.add_argument('--face',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);return ap.parse_args(a)


def read_obj(path):
    verts=[];faces=[]
    for line in path.read_text(errors='ignore').splitlines():
        if line.startswith('v '):
            q=line.split();verts.append((float(q[1]),float(q[2]),float(q[3])))
        elif line.startswith('f '):
            ids=[int(x.split('/')[0])-1 for x in line.split()[1:]]
            for i in range(1,len(ids)-1):faces.append((ids[0],ids[i],ids[i+1]))
    return np.asarray(verts,float),np.asarray(faces,np.int32)


def components(n,faces):
    parent=np.arange(n,dtype=np.int32)
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return int(x)
    def union(a,b):
        a=find(int(a));b=find(int(b))
        if a!=b:parent[b]=a
    for a,b,c in faces:union(a,b);union(b,c);union(c,a)
    roots=np.asarray([find(i) for i in range(n)],np.int32);groups={}
    for i,r in enumerate(roots):groups.setdefault(int(r),[]).append(i)
    return [np.asarray(x,np.int32) for x in groups.values()]


def map_face(raw,height=1.72):
    out=np.empty_like(raw);out[:,0]=raw[:,0]*1.08;out[:,1]=raw[:,2]*1.08;out[:,2]=-raw[:,1]*1.08;out[:,2]+=height-float(out[:,2].max());return out


def set_coords(block,coords):
    try:block.data.foreach_set('co',np.asarray(coords,np.float32).ravel())
    except Exception:
        for i,q in enumerate(coords):block.data[i].co=q


def material(name,color,rough=.45,metal=.0):
    m=bpy.data.materials.get(name) or bpy.data.materials.new(name);m.diffuse_color=tuple(color);m.use_nodes=True;bs=m.node_tree.nodes.get('Principled BSDF') if m.node_tree else None
    if bs:
        if bs.inputs.get('Base Color'):bs.inputs['Base Color'].default_value=tuple(color)
        if bs.inputs.get('Roughness'):bs.inputs['Roughness'].default_value=rough
        if bs.inputs.get('Metallic'):bs.inputs['Metallic'].default_value=metal
    return m


def curve(name,pts,radius,mat):
    old=bpy.data.objects.get(name)
    if old:bpy.data.objects.remove(old,do_unlink=True)
    cu=bpy.data.curves.new(name+'_Curve','CURVE');cu.dimensions='3D';cu.resolution_u=3;cu.bevel_depth=radius;cu.bevel_resolution=3
    sp=cu.splines.new('BEZIER');sp.bezier_points.add(len(pts)-1)
    for bp,p in zip(sp.bezier_points,pts):bp.co=p;bp.handle_left_type='AUTO';bp.handle_right_type='AUTO'
    ob=bpy.data.objects.new(name,cu);bpy.context.collection.objects.link(ob);cu.materials.append(mat);return ob


def patch_identity(mapped,faces):
    head=bpy.data.objects.get('AINA_Face_v15_5')
    if not head or head.type!='MESH':raise RuntimeError('AINA_Face_v15_5 missing')
    if len(head.data.vertices)!=len(mapped):raise RuntimeError(f'Face vertex count mismatch {len(head.data.vertices)} != {len(mapped)}')
    base=np.empty((len(mapped),3),np.float32);head.data.vertices.foreach_get('co',base.ravel());delta=mapped-base
    # Update Basis and all 52 production keys by the same identity delta.
    if not head.data.shape_keys or len(head.data.shape_keys.key_blocks)!=53:raise RuntimeError('Expected Basis + 52 Shape Keys')
    for kb in head.data.shape_keys.key_blocks:
        q=np.empty((len(mapped),3),np.float32);kb.data.foreach_get('co',q.ravel());set_coords(kb,q+delta)
    head.data.vertices.foreach_set('co',mapped.astype(np.float32).ravel());head.data.update()

    groups=components(len(mapped),faces);eyes=sorted([g for g in groups if 650<len(g)<900],key=lambda g:float(mapped[g,0].mean()))
    if len(eyes)!=2:raise RuntimeError(f'Expected two eye components, got {[len(x) for x in eyes]}')
    for side,g in zip(('R','L'),eyes):
        eo=bpy.data.objects.get('AINA_Eye_'+side)
        if eo and len(eo.data.vertices)==len(g):eo.data.vertices.foreach_set('co',mapped[g].astype(np.float32).ravel());eo.data.update()
        center=Vector(mapped[g].mean(0))
        ir=bpy.data.objects.get('AINA_Iris_'+side);pu=bpy.data.objects.get('AINA_Pupil_'+side)
        if ir:
            mw=ir.matrix_world.copy();mw.translation=center+Vector((0,-.0115,0));ir.matrix_world=mw;ir.scale=(1.17,1.0,1.17)
        if pu:
            mw=pu.matrix_world.copy();mw.translation=center+Vector((0,-.0125,0));pu.matrix_world=mw;pu.scale=(1.08,1.0,1.08)
    return head


def add_identity_details(mapped):
    lm=mapped[K];lash=material('AINA_Lashes',(0.045,0.055,0.070,1),.38);brow=material('AINA_Brows',(0.28,0.30,0.34,1),.56)
    # Viewer-left and viewer-right upper lids. Move 2 mm toward portrait camera.
    for n,ids in (('R',[36,37,38,39]),('L',[42,43,44,45])):
        pts=[Vector(lm[i])+Vector((0,-.0022,0)) for i in ids];curve('AINA_UpperLash_'+n,pts,.00115,lash)
        # Outer half lower lash is intentionally faint.
        lower=[41,40,39] if n=='R' else [47,46,45];curve('AINA_LowerLash_'+n,[Vector(lm[i])+Vector((0,-.0019,0)) for i in lower],.00048,lash)
    curve('AINA_Brow_R',[Vector(lm[i])+Vector((0,-.0020,.0030)) for i in [17,18,19,20,21]],.00165,brow)
    curve('AINA_Brow_L',[Vector(lm[i])+Vector((0,-.0020,.0030)) for i in [22,23,24,25,26]],.00165,brow)

    # Refine existing materials toward the approved pale/silver/blue AINA palette.
    material('AINA_Skin',(0.965,0.84,0.82,1),.52)
    material('AINA_Iris',(0.34,0.61,0.78,1),.26,.02)
    material('AINA_Pupil',(0.018,0.025,0.038,1),.25)
    hair=material('AINA_Hair_Silver',(0.80,0.84,0.91,1),.34,.03)

    # Layered real geometry over the existing silver hair cap: center-part bangs
    # and long side pieces matching the approved AINA silhouette.
    details=[
      [(-.055,-.091,1.708),(-.046,-.103,1.663),(-.034,-.106,1.615)],
      [(-.034,-.096,1.714),(-.027,-.108,1.670),(-.018,-.109,1.620)],
      [(-.012,-.100,1.716),(-.008,-.111,1.675),(-.004,-.111,1.632)],
      [(.014,-.100,1.716),(.009,-.111,1.676),(.004,-.111,1.635)],
      [(.037,-.096,1.713),(.029,-.108,1.668),(.020,-.108,1.619)],
      [(.057,-.090,1.707),(.048,-.102,1.662),(.036,-.105,1.614)],
      [(-.070,-.068,1.690),(-.083,-.081,1.620),(-.078,-.073,1.535),(-.066,-.060,1.475)],
      [(.070,-.068,1.690),(.083,-.081,1.620),(.078,-.073,1.535),(.066,-.060,1.475)],
    ]
    for i,pts in enumerate(details):curve(f'AINA_HairDetail_{i+1}',[Vector(p) for p in pts],.0031 if i<6 else .0042,hair)


def render(out):
    out.mkdir(parents=True,exist_ok=True);scene=bpy.context.scene;scene.render.engine='BLENDER_EEVEE_NEXT';scene.render.image_settings.file_format='PNG';scene.render.resolution_percentage=100;scene.world.color=(.93,.95,.98)
    # Replace prior preview lights with a soft portrait rig.
    for o in list(bpy.data.objects):
        if o.type=='LIGHT' and o.name.startswith('AINA_Review'):bpy.data.objects.remove(o,do_unlink=True)
    def area(name,loc,energy,size):
        d=bpy.data.lights.new(name,'AREA');d.energy=energy;d.shape='DISK';d.size=size;o=bpy.data.objects.new(name,d);bpy.context.collection.objects.link(o);o.location=loc
    area('AINA_Review_Key',(1.8,-2.2,2.45),650,3.0);area('AINA_Review_Fill',(-1.7,-1.8,2.15),430,2.8);area('AINA_Review_Rim',(0,1.7,2.3),520,2.5)
    cam=bpy.data.objects.get('AINA_Camera')
    if not cam:
        cd=bpy.data.cameras.new('AINA_Camera');cam=bpy.data.objects.new('AINA_Camera',cd);bpy.context.collection.objects.link(cam)
    scene.camera=cam
    def shot(name,loc,target,lens=86):
        cam.location=loc;cam.data.lens=lens;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.resolution_x=1024;scene.render.resolution_y=1024;scene.render.filepath=str(out/name);bpy.ops.render.render(write_still=True)
    shot('AINA_VISUAL_FRONT.png',(0,-2.18,1.615),(0,0,1.615))
    shot('AINA_VISUAL_3Q.png',(.78,-2.08,1.62),(0,0,1.605))
    shot('AINA_VISUAL_PROFILE.png',(2.10,0,1.62),(0,0,1.605),82)


def main():
    a=argv();raw,faces=read_obj(a.face);mapped=map_face(raw);patch_identity(mapped,faces);add_identity_details(mapped);bpy.context.view_layer.update();render(a.out/'Preview');bpy.ops.wm.save_as_mainfile(filepath=str(a.out/'AINA_VISUAL_REVIEW.blend'))
    print('[AINA_VISUAL] Basis replaced; 52 Shape Keys preserved; 3D brows/lashes/hair details added; review rendered.',flush=True)

if __name__=='__main__':main()
