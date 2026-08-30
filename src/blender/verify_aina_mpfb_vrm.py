#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector

def parse_args():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    p=argparse.ArgumentParser();p.add_argument('--glb',type=Path,required=True);p.add_argument('--out',type=Path,required=True);return p.parse_args(argv)
def look(obj,target):obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()
def wverts(o):
    mw=o.matrix_world;return np.asarray([mw@v.co for v in o.data.vertices],float)
def head_bounds(body):
    p=wverts(body);sel=p[(p[:,2]>1.245)&(np.abs(p[:,0])<.36)]
    if len(sel)<100:sel=p[np.argsort(p[:,2])[-min(4000,len(p)):]]
    return sel.min(0),sel.max(0)
def arr_key(k):
    a=np.empty(len(k.data)*3,np.float64);k.data.foreach_get('co',a);return a.reshape(-1,3)
def shape_stats(body):
    if not body.data.shape_keys:return {},0
    kb=body.data.shape_keys.key_blocks;basis=arr_key(kb.get('Basis') or kb[0]);stats={}
    for k in kb[1:]:
        d=np.linalg.norm(arr_key(k)-basis,axis=1);stats[k.name]=float(d.max())
    return stats,sum(v>1e-7 for v in stats.values())
def reset_keys(body):
    if body.data.shape_keys:
        for k in body.data.shape_keys.key_blocks:k.value=0.0
def set_keys(body,values):
    reset_keys(body)
    if not body.data.shape_keys:return
    for name,val in values.items():
        k=body.data.shape_keys.key_blocks.get(name)
        if k:k.value=val
def setup_render(body):
    for o in list(bpy.data.objects):
        if o.type in {'LIGHT','CAMERA'}:bpy.data.objects.remove(o,do_unlink=True)
    if bpy.context.scene.world is None:bpy.context.scene.world=bpy.data.worlds.new('AINA_Verify_World')
    w=bpy.context.scene.world;w.use_nodes=True;bg=w.node_tree.nodes.get('Background');bg.inputs['Color'].default_value=(.028,.035,.052,1);bg.inputs['Strength'].default_value=.18
    mn,mx=head_bounds(body);target=Vector((0,-.080,float((mn[2]+mx[2])*.5)))
    for name,loc,en,size in [('Key',(-.75,-.85,2.05),260,1.1),('Fill',(.85,-.55,1.70),120,.9),('Rim',(0,.75,2.00),220,.8)]:
        bpy.ops.object.light_add(type='AREA',location=loc);l=bpy.context.object;l.name='LGT_'+name;l.data.energy=en;l.data.shape='DISK';l.data.size=size;look(l,target)
    bpy.ops.object.camera_add();cam=bpy.context.object;cam.name='CAM_AINA_VERIFY';cam.data.lens=82;cam.data.sensor_width=36;bpy.context.scene.camera=cam
    sc=bpy.context.scene;sc.render.engine='BLENDER_EEVEE_NEXT';sc.render.resolution_x=1024;sc.render.resolution_y=1024;sc.render.resolution_percentage=100;sc.render.image_settings.file_format='PNG';sc.render.film_transparent=False;sc.view_settings.look='AgX - Medium High Contrast'
    return sc,cam,target
def render(sc,cam,target,path,loc):
    cam.location=loc;look(cam,target);sc.render.filepath=str(path);bpy.ops.render.render(write_still=True)
def main():
    a=parse_args();out=a.out.resolve();(out/'Preview').mkdir(parents=True,exist_ok=True);(out/'QA').mkdir(exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(a.glb.resolve()))
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH'];arms=[o for o in bpy.context.scene.objects if o.type=='ARMATURE'];body=max(meshes,key=lambda o:len(o.data.vertices));arm=max(arms,key=lambda o:len(o.data.bones)) if arms else None
    stats,nonzero=shape_stats(body);sc,cam,target=setup_render(body);reset_keys(body)
    render(sc,cam,target,out/'Preview'/'AINA_VRM_NEUTRAL_FRONT.png',(0,-1.28,target.z));render(sc,cam,target,out/'Preview'/'AINA_VRM_NEUTRAL_3Q.png',(.66,-1.08,target.z));render(sc,cam,target,out/'Preview'/'AINA_VRM_NEUTRAL_PROFILE.png',(1.18,-.08,target.z))
    poses={'HAPPY':{'mouthSmileLeft':.8,'mouthSmileRight':.8,'cheekSquintLeft':.3,'cheekSquintRight':.3},'BLINK':{'eyeBlinkLeft':1.0,'eyeBlinkRight':1.0},'AA':{'jawOpen':.75,'mouthFunnel':.25},'SURPRISED':{'browInnerUp':.8,'eyeWideLeft':.7,'eyeWideRight':.7,'jawOpen':.65}}
    for name,vals in poses.items():set_keys(body,vals);render(sc,cam,target,out/'Preview'/f'AINA_VRM_{name}.png',(0,-1.28,target.z))
    reset_keys(body);bpy.ops.wm.save_as_mainfile(filepath=str(out/'AINA_VRM_CLEAN_REIMPORT.blend'))
    required=['pelvis','spine_01','spine_02','spine_03','neck_01','head','clavicle_l','upperarm_l','lowerarm_l','hand_l','clavicle_r','upperarm_r','lowerarm_r','hand_r','thigh_l','calf_l','foot_l','thigh_r','calf_r','foot_r'];bone_names={b.name for b in arm.data.bones} if arm else set();missing=[n for n in required if n not in bone_names]
    arkit=['browDownLeft','browDownRight','browInnerUp','browOuterUpLeft','browOuterUpRight','cheekPuff','cheekSquintLeft','cheekSquintRight','eyeBlinkLeft','eyeBlinkRight','eyeLookDownLeft','eyeLookDownRight','eyeLookInLeft','eyeLookInRight','eyeLookOutLeft','eyeLookOutRight','eyeLookUpLeft','eyeLookUpRight','eyeSquintLeft','eyeSquintRight','eyeWideLeft','eyeWideRight','jawForward','jawLeft','jawOpen','jawRight','mouthClose','mouthDimpleLeft','mouthDimpleRight','mouthFrownLeft','mouthFrownRight','mouthFunnel','mouthLeft','mouthLowerDownLeft','mouthLowerDownRight','mouthPressLeft','mouthPressRight','mouthPucker','mouthRight','mouthRollLower','mouthRollUpper','mouthShrugLower','mouthShrugUpper','mouthSmileLeft','mouthSmileRight','mouthStretchLeft','mouthStretchRight','mouthUpperUpLeft','mouthUpperUpRight','noseSneerLeft','noseSneerRight','tongueOut']
    report={'real_clean_import':True,'mesh_objects':len(meshes),'armatures':len(arms),'body_object':body.name,'body_vertices':len(body.data.vertices),'body_triangles':sum(len(p.vertices)-2 for p in body.data.polygons),'bones':len(bone_names),'required_bones_missing':missing,'shape_keys':len(stats),'shape_keys_nonzero':nonzero,'arkit_present':sum(n in stats for n in arkit),'materials':len(bpy.data.materials),'accessory_objects':sum(o.name.startswith('AINA_') and o!=body for o in meshes),'shape_key_max_delta_m':max(stats.values()) if stats else 0.0}
    (out/'QA'/'AINA_VRM_CLEAN_REIMPORT_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
    assert arm and not missing and report['arkit_present']==52 and nonzero>=52 and report['accessory_objects']>=12
if __name__=='__main__':main()
