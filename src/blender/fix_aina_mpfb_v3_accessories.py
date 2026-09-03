#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector

def parse_args():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    p=argparse.ArgumentParser();p.add_argument('--blend',type=Path,required=True);p.add_argument('--out',type=Path,required=True);return p.parse_args(argv)
def mat(name,color,metal=0.0,rough=.45,emission=None,alpha=1.0):
    m=bpy.data.materials.get(name) or bpy.data.materials.new(name);m.use_nodes=True
    bs=m.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(*color,alpha);bs.inputs['Metallic'].default_value=metal;bs.inputs['Roughness'].default_value=rough
    if emission:
        bs.inputs['Emission Color'].default_value=(*emission,1);bs.inputs['Emission Strength'].default_value=2.4
    if alpha<1:
        bs.inputs['Alpha'].default_value=alpha;m.surface_render_method='DITHERED'
    return m
def select_only(o):
    bpy.ops.object.select_all(action='DESELECT');o.select_set(True);bpy.context.view_layer.objects.active=o
def weight_object(o,arm,bone):
    select_only(o)
    if o.type=='CURVE':bpy.ops.object.convert(target='MESH');o=bpy.context.object
    bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
    for vg in list(o.vertex_groups):o.vertex_groups.remove(vg)
    vg=o.vertex_groups.new(name=bone);vg.add(list(range(len(o.data.vertices))),1.0,'REPLACE')
    for md in list(o.modifiers):
        if md.type=='ARMATURE':o.modifiers.remove(md)
    md=o.modifiers.new('AINA_Armature','ARMATURE');md.object=arm;md.use_vertex_groups=True
    return o
def sphere(name,loc,scale,material,arm,bone='head',segments=48,rings=24):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments,ring_count=rings,location=loc);o=bpy.context.object;o.name=name;o.scale=scale;select_only(o);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);o.data.materials.append(material)
    for p in o.data.polygons:p.use_smooth=True
    return weight_object(o,arm,bone)
def curve(name,pts,bevel,material,arm,bone='head',cyclic=False):
    c=bpy.data.curves.new(name,'CURVE');c.dimensions='3D';c.bevel_depth=bevel;c.bevel_resolution=3;c.resolution_u=12
    s=c.splines.new('BEZIER');s.bezier_points.add(len(pts)-1)
    for b,p in zip(s.bezier_points,pts):b.co=p;b.handle_left_type='AUTO';b.handle_right_type='AUTO'
    s.use_cyclic_u=cyclic;o=bpy.data.objects.new(name,c);bpy.context.collection.objects.link(o);o.data.materials.append(material)
    return weight_object(o,arm,bone)
def cap(name,centre,radii,material,arm):
    rings,segments=22,64;cx,cy,cz=centre;rx,ry,rz=radii;verts=[];faces=[]
    for i in range(rings+1):
        phi=2.03*i/rings;sp,cp=math.sin(phi),math.cos(phi)
        for j in range(segments):
            th=2*math.pi*j/segments;x=cx+rx*sp*math.cos(th);y=cy+ry*sp*math.sin(th);z=cz+rz*cp;verts.append((x,y,z))
    for i in range(rings):
        for j in range(segments):
            a=i*segments+j;b=i*segments+(j+1)%segments;c=(i+1)*segments+(j+1)%segments;d=(i+1)*segments+j
            q=np.mean(np.asarray([verts[a],verts[b],verts[c],verts[d]]),axis=0)
            if (q[1]>.005) or (q[2]>1.605) or (abs(q[0])>.185 and q[2]>1.50):faces.append((a,b,c,d))
    me=bpy.data.meshes.new(name);me.from_pydata(verts,[],faces);me.update();o=bpy.data.objects.new(name,me);bpy.context.collection.objects.link(o);me.materials.append(material)
    for p in me.polygons:p.use_smooth=True
    return weight_object(o,arm,'head')
def look(o,target):o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler()
def render_views(out):
    for o in list(bpy.data.objects):
        if o.type in {'CAMERA','LIGHT'}:bpy.data.objects.remove(o,do_unlink=True)
    if bpy.context.scene.world is None:bpy.context.scene.world=bpy.data.worlds.new('AINA_World')
    w=bpy.context.scene.world;w.use_nodes=True;bg=w.node_tree.nodes.get('Background');bg.inputs['Color'].default_value=(.025,.032,.048,1);bg.inputs['Strength'].default_value=.16
    target=Vector((0,-.105,1.505))
    for name,loc,en,size in [('Key',(-.70,-.95,2.00),230,1.0),('Fill',(.80,-.60,1.68),105,.9),('Rim',(0,.72,2.02),190,.8)]:
        bpy.ops.object.light_add(type='AREA',location=loc);l=bpy.context.object;l.name='LGT_'+name;l.data.energy=en;l.data.shape='DISK';l.data.size=size;look(l,target)
    bpy.ops.object.camera_add();cam=bpy.context.object;cam.name='CAM_AINA';cam.data.lens=84;cam.data.sensor_width=36;bpy.context.scene.camera=cam
    sc=bpy.context.scene;sc.render.engine='BLENDER_EEVEE_NEXT';sc.render.resolution_x=1024;sc.render.resolution_y=1024;sc.render.resolution_percentage=100;sc.render.image_settings.file_format='PNG';sc.view_settings.look='AgX - Medium High Contrast'
    views={'FRONT':(0,-1.25,1.505),'THREE_QUARTER':(.63,-1.05,1.505),'PROFILE':(1.15,-.08,1.505)}
    for name,loc in views.items():cam.location=loc;look(cam,target);sc.render.filepath=str(out/'Preview'/f'AINA_MPFB_V3_{name}.png');bpy.ops.render.render(write_still=True)
    cam.data.lens=62;cam.location=(0,-3.85,.92);look(cam,(0,-.02,.86));sc.render.filepath=str(out/'Preview'/'AINA_MPFB_V3_FULLBODY.png');bpy.ops.render.render(write_still=True)
def shape_count(body):
    if not body.data.shape_keys:return 0,0
    keys=body.data.shape_keys.key_blocks;basis=np.empty(len(keys[0].data)*3,float);keys[0].data.foreach_get('co',basis);basis=basis.reshape(-1,3);nz=0
    for k in keys[1:]:
        a=np.empty(len(k.data)*3,float);k.data.foreach_get('co',a);a=a.reshape(-1,3);nz+=float(np.linalg.norm(a-basis,axis=1).max())>1e-7
    return len(keys)-1,nz
def main():
    a=parse_args();out=a.out.resolve();(out/'Preview').mkdir(parents=True,exist_ok=True);(out/'QA').mkdir(exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(a.blend.resolve()))
    body=bpy.data.objects.get('AINA_MPFB_Body') or max((o for o in bpy.data.objects if o.type=='MESH'),key=lambda o:len(o.data.vertices));arm=bpy.data.objects.get('AINA_MPFB_Humanoid') or next(o for o in bpy.data.objects if o.type=='ARMATURE')
    for o in list(bpy.data.objects):
        if o.name.startswith('AINA_') and o not in {body,arm}:bpy.data.objects.remove(o,do_unlink=True)
    materials={
      'skin':mat('AINA_Skin',(.55,.31,.26),rough=.50),'suit':mat('AINA_Suit_White',(.68,.76,.90),metal=.07,rough=.42),'navy':mat('AINA_Suit_Navy',(.025,.050,.115),metal=.18,rough=.38),
      'white':mat('AINA_EyeWhite',(.80,.86,.94),rough=.24),'iris':mat('AINA_Iris',(.025,.17,.25),rough=.18),'pupil':mat('AINA_Pupil',(.003,.008,.014),rough=.15),'cornea':mat('AINA_Cornea',(.65,.88,1.0),rough=.04,alpha=.18),
      'lash':mat('AINA_Lash',(.025,.018,.035),rough=.48),'brow':mat('AINA_Brow',(.20,.17,.24),rough=.55),'lip':mat('AINA_Lips',(.38,.065,.10),rough=.40),'hair':mat('AINA_SilverHair',(.44,.48,.62),metal=.035,rough=.38),'metal':mat('AINA_HairMetal',(.16,.27,.52),metal=.78,rough=.20),'core':mat('AINA_Core',(.012,.28,.78),metal=.10,rough=.14,emission=(.04,.48,1.0))}
    for i,m in enumerate(list(body.data.materials)):
        if m and 'Skin' in m.name:body.data.materials[i]=materials['skin']
        elif m and 'Navy' in m.name:body.data.materials[i]=materials['navy']
        elif m and 'Suit' in m.name:body.data.materials[i]=materials['suit']
    created=[];ec=[(-.090,-.148,1.526),(.090,-.148,1.526)]
    for side,(x,y,z) in zip(('L','R'),ec):
        created += [sphere(f'AINA_Eye_{side}',(x,y,z),(.044,.022,.020),materials['white'],arm),sphere(f'AINA_Iris_{side}',(x,-.171,z),(.0155,.0028,.0155),materials['iris'],arm),sphere(f'AINA_Pupil_{side}',(x,-.174,z),(.0058,.0022,.0058),materials['pupil'],arm),sphere(f'AINA_Cornea_{side}',(x,-.176,z),(.018,.0018,.018),materials['cornea'],arm)]
        sx=-1 if side=='L' else 1
        created.append(curve(f'AINA_Lash_{side}',[(x+sx*.039,-.178,z+.001),(x+sx*.019,-.179,z+.015),(x,-.180,z+.018),(x-sx*.020,-.179,z+.015),(x-sx*.039,-.178,z+.003)],.0019,materials['lash'],arm))
        created.append(curve(f'AINA_Brow_{side}',[(x+sx*.047,-.163,z+.055),(x+sx*.023,-.165,z+.064),(x,-.166,z+.067),(x-sx*.039,-.164,z+.057)],.0028,materials['brow'],arm))
    created += [sphere('AINA_UpperLip',(0,-.174,1.408),(.048,.0034,.0048),materials['lip'],arm),sphere('AINA_LowerLip',(0,-.175,1.398),(.052,.0038,.0062),materials['lip'],arm)]
    created.append(cap('AINA_Hair_Cap',(0,.036,1.572),(.222,.162,.232),materials['hair'],arm))
    created.append(sphere('AINA_Hair_Bun',(0,.105,1.710),(.105,.083,.100),materials['hair'],arm,segments=56,rings=28))
    for k in range(9):
        ang=2*math.pi*k/9;created.append(sphere(f'AINA_Hair_Braid_{k:02d}',(.060*math.cos(ang),.108+.018*math.sin(ang),1.710+.052*math.sin(ang)),(.022,.018,.021),materials['hair'],arm,segments=22,rings=12))
    for sx in (-1,1):
        for j in range(5):
            created.append(curve(f'AINA_HairSweep_{sx}_{j}',[(sx*(.018+.024*j),-.075,1.632),(sx*(.030+.022*j),-.090,1.590),(sx*(.047+.018*j),-.075,1.545),(sx*(.066+.012*j),-.052,1.495)],.0038-j*.00025,materials['hair'],arm))
        created.append(curve(f'AINA_SideWisp_{sx}',[(sx*.190,-.055,1.565),(sx*.218,-.088,1.475),(sx*.195,-.100,1.390)],.0030,materials['hair'],arm))
    created.append(curve('AINA_HeadBand',[(-.195,-.018,1.615),(-.105,-.115,1.653),(0,-.140,1.670),(.105,-.115,1.653),(.195,-.018,1.615)],.0022,materials['metal'],arm))
    chest=next((b.name for b in arm.data.bones if 'chest' in b.name.lower()),next((b.name for b in arm.data.bones if 'spine' in b.name.lower()),'spine'))
    bpy.ops.mesh.primitive_cone_add(vertices=64,radius1=.142,radius2=.122,depth=.142,location=(0,-.010,1.225));collar=bpy.context.object;collar.name='AINA_High_Collar';collar.data.materials.append(materials['suit']);created.append(weight_object(collar,arm,chest))
    core=sphere('AINA_Chest_Core',(0,-.166,1.120),(.036,.015,.048),materials['core'],arm,chest,32,16);created.append(core)
    render_views(out);bpy.ops.wm.save_as_mainfile(filepath=str(out/'AINA_MPFB_IDENTITY_MASTER_V3.blend'))
    bpy.ops.object.select_all(action='SELECT');bpy.ops.export_scene.gltf(filepath=str(out/'AINA_MPFB_IDENTITY_MASTER_V3.glb'),export_format='GLB',export_skins=True,export_morph=True,export_animations=True,export_apply=False)
    try:bpy.ops.export_scene.fbx(filepath=str(out/'AINA_MPFB_IDENTITY_MASTER_V3.fbx'),use_selection=False,add_leaf_bones=False,bake_anim=False)
    except Exception as e:(out/'QA'/'FBX_WARNING.txt').write_text(str(e))
    kc,nz=shape_count(body);report={'product':'AINA MPFB Identity Master V3','source_artifact_id':9732976672,'body_vertices':len(body.data.vertices),'rig_bones':len(arm.data.bones),'shape_keys':kc,'shape_keys_nonzero':nz,'generated_accessories':len(created),'attachment_method':'identity-transform armature modifier; one-bone full weights','identity_lock':False,'vrm_exported':False}
    (out/'QA'/'AINA_MPFB_V3_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
    assert len(body.data.vertices)==19158 and len(arm.data.bones)>=53 and kc>=52 and nz>=52 and len(created)>=30
if __name__=='__main__':main()
