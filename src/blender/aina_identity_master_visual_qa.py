#!/usr/bin/env python3
"""Final visual candidate gate for the real AINA Identity Master.

The script first builds the already-validated 52-control expression scene on the
Identity Master OBJ, then adds production inspection geometry only: refined eye
materials, a layered silver updo made from real mesh ribbons, a neck/bust and the
pearl collar. It renders the actual Blender character under neutral and selected
expressions. No replacement effect art and no VRM export are produced here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import aina_surface_expression_qa_v3 as expression_v3

K68=np.array([
    1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,
    3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,
    5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,
    2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,
    13695,12973,13422,6543,6537,
],dtype=np.int64)


def parse_args():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    p=argparse.ArgumentParser();p.add_argument('--face',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    return p.parse_args(argv)


def make_material(name,color,roughness=.42,metallic=0.0,transmission=0.0):
    mat=bpy.data.materials.get(name) or bpy.data.materials.new(name);mat.diffuse_color=tuple(color);mat.use_nodes=True
    bsdf=mat.node_tree.nodes.get('Principled BSDF') if mat.node_tree else None
    if bsdf:
        if bsdf.inputs.get('Base Color'):bsdf.inputs['Base Color'].default_value=tuple(color)
        if bsdf.inputs.get('Roughness'):bsdf.inputs['Roughness'].default_value=roughness
        if bsdf.inputs.get('Metallic'):bsdf.inputs['Metallic'].default_value=metallic
        if bsdf.inputs.get('Transmission Weight'):bsdf.inputs['Transmission Weight'].default_value=transmission
        if bsdf.inputs.get('Subsurface Weight') and 'Skin' in name:bsdf.inputs['Subsurface Weight'].default_value=.10
        if bsdf.inputs.get('Coat Weight') and ('Hair' in name or 'Eye' in name):bsdf.inputs['Coat Weight'].default_value=.18
        if bsdf.inputs.get('Coat Roughness') and ('Hair' in name or 'Eye' in name):bsdf.inputs['Coat Roughness'].default_value=.16
    return mat


def assign(obj,mat):
    if not hasattr(obj.data,'materials'):return
    obj.data.materials.clear();obj.data.materials.append(mat)
    if obj.type=='MESH':
        for p in obj.data.polygons:p.material_index=0;p.use_smooth=True


def mesh_object(name,verts,faces,mat):
    me=bpy.data.meshes.new(name+'_Mesh');me.from_pydata([tuple(v) for v in verts],[],[tuple(int(x) for x in f) for f in faces]);me.update()
    ob=bpy.data.objects.new(name,me);bpy.context.collection.objects.link(ob);assign(ob,mat);return ob


def ribbon(name,points,widths,mat,center=(0,.02,1.63),thickness=.0008):
    pts=np.asarray(points,float);widths=np.asarray(widths,float);center=np.asarray(center,float)
    left=[];right=[]
    for i,p in enumerate(pts):
        if i==0:t=pts[1]-pts[0]
        elif i==len(pts)-1:t=pts[-1]-pts[-2]
        else:t=pts[i+1]-pts[i-1]
        t/=max(np.linalg.norm(t),1e-9)
        radial=p-center;radial/=max(np.linalg.norm(radial),1e-9)
        side=np.cross(t,radial)
        if np.linalg.norm(side)<1e-7:side=np.array([1.,0.,0.])
        side/=max(np.linalg.norm(side),1e-9)
        left.append(p-side*widths[i]*.5);right.append(p+side*widths[i]*.5)
    verts=[]
    for offset in (-thickness*.5,thickness*.5):
        for a,b in zip(left,right):
            normal=np.asarray(a)-center;normal/=max(np.linalg.norm(normal),1e-9)
            verts.extend([a+normal*offset,b+normal*offset])
    n=len(pts);faces=[]
    for layer in (0,1):
        base=layer*2*n
        for i in range(n-1):
            a=base+2*i;b=a+1;c=a+3;d=a+2
            faces.extend([(a,b,c),(a,c,d)] if layer==0 else [(a,c,b),(a,d,c)])
    for i in range(n-1):
        a=2*i;b=2*(i+1);A=2*n+a;B=2*n+b
        faces.extend([(a,b,B),(a,B,A),(a+1,2*n+a+1,2*n+b+1),(a+1,2*n+b+1,b+1)])
    return mesh_object(name,np.asarray(verts,float),np.asarray(faces,np.int32),mat)


def scalp_cap(mat):
    verts=[];faces=[];nphi=96;nt=28;center=np.array([0,.025,1.625]);rx,ry,rz=.103,.091,.118
    for i in range(nphi):
        phi=2*math.pi*i/nphi;side=abs(math.cos(phi));front=math.sin(phi)<0
        tmax=(1.05+.26*side) if front else 2.00
        for j in range(nt):
            th=tmax*j/(nt-1)
            p=center+np.array([rx*math.sin(th)*math.cos(phi),ry*math.sin(th)*math.sin(phi),rz*math.cos(th)])
            verts.append(p)
    for i in range(nphi):
        ni=(i+1)%nphi
        for j in range(nt-1):
            a=i*nt+j;b=ni*nt+j;c=ni*nt+j+1;d=i*nt+j+1;faces.extend([(a,b,c),(a,c,d)])
    return mesh_object('AINA_Identity_HairScalp',np.asarray(verts),np.asarray(faces,np.int32),mat)


def create_updo(mat):
    objects=[scalp_cap(mat)]
    center=np.array([0,.025,1.625])
    # Broad swept locks from a narrow center part. These are tapered real mesh
    # ribbons rather than the old pipe-like curves.
    for side,sg in [('L',-1.),('R',1.)]:
        for i in range(10):
            root=np.array([sg*(.003+.0019*i),-.060+.0006*i,1.724-.0012*i])
            mid=np.array([sg*(.022+.0055*i),-.075+.0018*i,1.695-.0053*i])
            end=np.array([sg*(.050+.0041*i),-.050+.0025*i,1.645-.0066*i])
            objects.append(ribbon(f'AINA_Identity_Crown_{side}_{i+1}',[root,mid,end],[.010,.012,.007],mat,center))
    # Ear-framing locks and a few delicate flyaways.
    for side,sg in [('L',1.),('R',-1.)]:
        for i in range(4):
            p0=np.array([sg*(.067+.004*i),-.046+.004*i,1.674-.005*i])
            p1=np.array([sg*(.080+.003*i),-.057+.003*i,1.612-.010*i])
            p2=np.array([sg*(.074+.002*i),-.052+.002*i,1.535-.014*i])
            objects.append(ribbon(f'AINA_Identity_SideLock_{side}_{i+1}',[p0,p1,p2],[.010,.009,.0035],mat,center))
    # Back sweep into the bun.
    bun_center=np.array([0,.105,1.695])
    for i,x in enumerate(np.linspace(-.078,.078,13)):
        p0=np.array([x*.70,.050+abs(x)*.20,1.708-abs(x)*.18])
        p1=np.array([x,.088,1.640-abs(x)*.12])
        p2=bun_center+np.array([x*.35,-.010,-.020-abs(x)*.10])
        objects.append(ribbon(f'AINA_Identity_BackSweep_{i+1}',[p0,p1,p2],[.009,.011,.006],mat,center))
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64,ring_count=32,radius=1,location=bun_center)
    bun=bpy.context.object;bun.name='AINA_Identity_Bun';bun.scale=(.050,.043,.052);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);assign(bun,mat);objects.append(bun)
    # Layered bun wrap ribbons.
    for i,phase in enumerate(np.linspace(0,2*math.pi,7,endpoint=False)):
        pts=[]
        for t in np.linspace(0,1,9):
            a=phase+1.45*math.pi*t
            pts.append(bun_center+np.array([.050*math.cos(a),.043*math.sin(a),.022*math.sin(2*a)]))
        objects.append(ribbon(f'AINA_Identity_BunWrap_{i+1}',pts,np.linspace(.007,.003,len(pts)),mat,bun_center,.00065))
    return objects


def create_bust(skin,suit):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48,ring_count=24,radius=1,location=(0,.020,1.455))
    neck=bpy.context.object;neck.name='AINA_Identity_Neck';neck.scale=(.050,.044,.105);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);assign(neck,skin)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64,ring_count=32,radius=1,location=(0,.050,1.335))
    bust=bpy.context.object;bust.name='AINA_Identity_Bust';bust.scale=(.235,.105,.165);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);assign(bust,suit)
    n=64;verts=[];faces=[]
    for z,r in [(1.485,.060),(1.535,.058)]:
        for i in range(n):
            a=2*math.pi*i/n;verts.append((r*math.cos(a),.052*math.sin(a),z))
    for i in range(n):
        j=(i+1)%n;faces.extend([(i,j,n+j),(i,n+j,n+i)])
    collar=mesh_object('AINA_Identity_PearlCollar',np.asarray(verts,float),np.asarray(faces,np.int32),suit)
    return [neck,bust,collar]


def main():
    a=parse_args();a.out.mkdir(parents=True,exist_ok=True)
    # Build the proven v3 52-control real-mesh scene first.
    expression_v3.main()
    candidates=[o for o in bpy.data.objects if o.type=='MESH' and o.data.shape_keys and len(o.data.shape_keys.key_blocks)>=53]
    if not candidates:raise RuntimeError('52-control Identity Master head was not found after expression build')
    head=max(candidates,key=lambda o:len(o.data.vertices))
    if len(head.data.shape_keys.key_blocks)-1!=52:raise RuntimeError('Identity Master visual scene lost the 52 controls')

    skin=make_material('AINA_Visual_Skin',(.79,.64,.61,1),.43)
    hair=make_material('AINA_Visual_HairSilver',(.69,.74,.84,1),.25,.05)
    suit=make_material('AINA_Visual_PearlSuit',(.73,.79,.89,1),.26,.13)
    eye_white=make_material('AINA_Visual_EyeWhite',(.955,.975,.995,1),.19)
    iris=make_material('AINA_Visual_Iris',(.13,.38,.57,1),.16,.01)
    pupil=make_material('AINA_Visual_Pupil',(.004,.007,.015,1),.14)
    dark=make_material('AINA_Visual_BrowLash',(.035,.030,.050,1),.28)
    assign(head,skin)
    # Reuse and improve the actual eye/brow/lash geometry created by v3.
    for obj in bpy.data.objects:
        name=obj.name.lower()
        if obj==head:continue
        if 'eyewhite' in name or ('eye' in name and 'iris' not in name and 'pupil' not in name):assign(obj,eye_white)
        elif 'iris' in name:assign(obj,iris)
        elif 'pupil' in name:assign(obj,pupil)
        elif 'brow' in name or 'lash' in name:assign(obj,dark)
    # Render subdivision is real Blender geometry derived from the same shape-key
    # mesh and removes the remaining low-poly surface response.
    sub=head.modifiers.get('AINA_Identity_RenderSubdivision') or head.modifiers.new('AINA_Identity_RenderSubdivision','SUBSURF')
    sub.levels=1;sub.render_levels=2;sub.subdivision_type='CATMULL_CLARK'

    hair_objects=create_updo(hair);bust_objects=create_bust(skin,suit)
    scene=bpy.context.scene
    for obj in list(scene.objects):
        if obj.type in {'LIGHT','CAMERA'}:bpy.data.objects.remove(obj,do_unlink=True)
    scene.render.engine='BLENDER_EEVEE_NEXT';scene.render.image_settings.file_format='PNG';scene.render.resolution_x=800;scene.render.resolution_y=800;scene.render.resolution_percentage=100
    scene.world.color=(.055,.065,.090)
    try:scene.view_settings.look='AgX - Medium High Contrast';scene.view_settings.exposure=.35
    except Exception:pass
    def area(name,loc,energy,size,target=(0,0,1.60)):
        d=bpy.data.lights.new(name,'AREA');d.energy=energy;d.shape='DISK';d.size=size;o=bpy.data.objects.new(name,d);bpy.context.collection.objects.link(o);o.location=loc;o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler()
    area('AINA_Key',(1.25,-1.7,2.25),720,2.7);area('AINA_Fill',(-1.45,-1.5,1.90),350,2.8);area('AINA_Rim',(0,1.5,2.25),470,2.4);area('AINA_FaceSoft',(0,-2.0,1.62),110,2.8)
    cd=bpy.data.cameras.new('AINA_Identity_Visual_Camera');cam=bpy.data.objects.new('AINA_Identity_Visual_Camera',cd);bpy.context.collection.objects.link(cam);scene.camera=cam;cam.data.lens=82
    preview=a.out/'VisualPreview';qa=a.out/'VisualQA';preview.mkdir(exist_ok=True);qa.mkdir(exist_ok=True)
    def clear_keys():
        for obj in scene.objects:
            if obj.type=='MESH' and obj.data.shape_keys:
                for key in obj.data.shape_keys.key_blocks:key.value=0.0
    def apply(values):
        for obj in scene.objects:
            if obj.type!='MESH' or not obj.data.shape_keys:continue
            for key,value in values.items():
                if key in obj.data.shape_keys.key_blocks:obj.data.shape_keys.key_blocks[key].value=float(value)
    def render(name,yaw,values):
        clear_keys();apply(values);angle=math.radians(yaw);distance=1.02;target=Vector((0,0,1.605));cam.location=(distance*math.sin(angle),-distance*math.cos(angle),1.615);cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(preview/name);bpy.ops.render.render(write_still=True)
    cases=[
      ('AINA_VISUAL_NEUTRAL_FRONT.png',0,{}),('AINA_VISUAL_NEUTRAL_20.png',20,{}),('AINA_VISUAL_NEUTRAL_45.png',45,{}),
      ('AINA_VISUAL_LEFT_PROFILE.png',90,{}),('AINA_VISUAL_RIGHT_PROFILE.png',-90,{}),
      ('AINA_VISUAL_HAPPY_FRONT.png',0,{'mouthSmileLeft':.72,'mouthSmileRight':.72,'cheekSquintLeft':.22,'cheekSquintRight':.22}),
      ('AINA_VISUAL_SAD_FRONT.png',0,{'browInnerUp':.58,'mouthFrownLeft':.48,'mouthFrownRight':.48}),
      ('AINA_VISUAL_ANGRY_FRONT.png',0,{'browDownLeft':.62,'browDownRight':.62,'mouthFrownLeft':.32,'mouthFrownRight':.32}),
      ('AINA_VISUAL_SURPRISED_FRONT.png',0,{'browInnerUp':.48,'eyeWideLeft':.62,'eyeWideRight':.62,'jawOpen':.42}),
      ('AINA_VISUAL_BLINK_FRONT.png',0,{'eyeBlinkLeft':1.,'eyeBlinkRight':1.}),
    ]
    for name,yaw,values in cases:render(name,yaw,values)
    clear_keys();blend=a.out/'AINA_IDENTITY_MASTER_VISUAL_QA.blend';bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    report={
      'product':'AINA Identity Master Final Visual Candidate','real_mesh':True,'replacement_effect_art_generated':False,
      'shape_control_count':52,'visual_render_count':len(cases),'rendered_cases':[x[0] for x in cases],
      'hair_geometry_objects':len(hair_objects),'bust_geometry_objects':len(bust_objects),'topology_changed':False,
      'visual_identity_lock':False,'next_gate':'Manual comparison against approved AINA front/3Q/side, then set visual_identity_lock=true only if accepted',
      'files':{'blend':str(blend),'blend_bytes':blend.stat().st_size,'preview_dir':str(preview)},
    }
    (qa/'AINA_IDENTITY_MASTER_VISUAL_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
