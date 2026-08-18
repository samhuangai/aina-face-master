#!/usr/bin/env python3
"""Reimport AINA.vrm into a clean Blender scene and hard-gate final delivery."""
from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path
import bpy

SHAPE_KEYS=[
    'browDownLeft','browDownRight','browInnerUp','browOuterUpLeft','browOuterUpRight',
    'cheekPuff','cheekSquintLeft','cheekSquintRight',
    'eyeBlinkLeft','eyeBlinkRight','eyeLookDownLeft','eyeLookDownRight','eyeLookInLeft','eyeLookInRight','eyeLookOutLeft','eyeLookOutRight','eyeLookUpLeft','eyeLookUpRight','eyeSquintLeft','eyeSquintRight','eyeWideLeft','eyeWideRight',
    'jawForward','jawLeft','jawOpen','jawRight',
    'mouthClose','mouthDimpleLeft','mouthDimpleRight','mouthFrownLeft','mouthFrownRight','mouthFunnel','mouthLeft','mouthLowerDownLeft','mouthLowerDownRight','mouthPressLeft','mouthPressRight','mouthPucker','mouthRight','mouthRollLower','mouthRollUpper','mouthShrugLower','mouthShrugUpper','mouthSmileLeft','mouthSmileRight','mouthStretchLeft','mouthStretchRight','mouthUpperUpLeft','mouthUpperUpRight',
    'noseSneerLeft','noseSneerRight','tongueOut',
]
EXPECTED_PRESET_BINDS={
    'happy':4,'angry':4,'sad':3,'relaxed':4,'surprised':4,'neutral':0,
    'aa':2,'ih':2,'ou':2,'ee':4,'oh':2,'blink':2,
    'blink_left':1,'blink_right':1,'look_up':2,'look_down':2,'look_left':2,'look_right':2,
}
PRESETS=list(EXPECTED_PRESET_BINDS)
REQUIRED_HUMANOID=['hips','spine','chest','neck','head','left_upper_arm','right_upper_arm','left_lower_arm','right_lower_arm','left_hand','right_hand','left_upper_leg','right_upper_leg','left_lower_leg','right_lower_leg','left_foot','right_foot']

def args():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    ap=argparse.ArgumentParser(); ap.add_argument('--vrm',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); return ap.parse_args(argv)

def register_vrm(root:Path):
    src=root/'vendor'/'VRM-Addon-for-Blender'/'src'
    if str(src) not in sys.path: sys.path.insert(0,str(src))
    import io_scene_vrm
    try: io_scene_vrm.register()
    except Exception as e:
        print('[AINA_REIMPORT] register note:',e,flush=True)

def clean():
    if bpy.context.object and bpy.context.object.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)

def main():
    a=args(); a.out.mkdir(parents=True,exist_ok=True); (a.out/'QA').mkdir(exist_ok=True)
    if not a.vrm.exists() or a.vrm.stat().st_size<100_000: raise RuntimeError('VRM missing or implausibly small')
    register_vrm(Path.cwd()); clean()
    result=bpy.ops.import_scene.vrm(filepath=str(a.vrm.resolve()))
    if result!={'FINISHED'}: raise RuntimeError(f'VRM reimport failed: {result}')
    bpy.context.view_layer.update()
    from io_scene_vrm.editor.extension import get_armature_extension
    armatures=[o for o in bpy.context.scene.objects if o.type=='ARMATURE']
    if not armatures: raise RuntimeError('No armature after VRM reimport')
    rig=None
    for o in armatures:
        try:
            if get_armature_extension(o.data).spec_version=='1.0': rig=o; break
        except Exception: pass
    rig=rig or armatures[0]
    ext=get_armature_extension(rig.data)
    if ext.spec_version!='1.0': raise RuntimeError(f'Reimported avatar is not VRM 1.0: {ext.spec_version}')

    shape_meshes=[]; all_keys=set()
    for o in bpy.context.scene.objects:
        if o.type=='MESH' and o.data.shape_keys:
            names=[k.name for k in o.data.shape_keys.key_blocks]
            if len(names)>1: shape_meshes.append({'object':o.name,'keys':names})
            all_keys.update(names)
    missing_shapes=[x for x in SHAPE_KEYS if x not in all_keys]
    found_shapes=[x for x in SHAPE_KEYS if x in all_keys]

    hb=ext.vrm1.humanoid.human_bones; humanoid={}; missing_humanoid=[]
    for attr in REQUIRED_HUMANOID:
        bone=''
        if hasattr(hb,attr): bone=getattr(hb,attr).node.bone_name
        humanoid[attr]=bone
        if not bone or bone not in rig.data.bones: missing_humanoid.append(attr)
    for eye in ('left_eye','right_eye'):
        bone=getattr(hb,eye).node.bone_name if hasattr(hb,eye) else ''
        humanoid[eye]=bone
        if not bone or bone not in rig.data.bones: missing_humanoid.append(eye)

    preset=ext.vrm1.expressions.preset; preset_counts={}; preset_missing=[]
    for name in PRESETS:
        expr=getattr(preset,name,None)
        if expr is None:
            preset_missing.append(name); preset_counts[name]=-1
        else:
            preset_counts[name]=len(expr.morph_target_binds)
    preset_mismatches={
        name:{'expected':EXPECTED_PRESET_BINDS[name],'actual':preset_counts.get(name,-1)}
        for name in PRESETS
        if preset_counts.get(name,-1)!=EXPECTED_PRESET_BINDS[name]
    }
    expression_bound=sum(max(0,n) for n in preset_counts.values())

    look_type=str(ext.vrm1.look_at.type)
    spring_count=len(ext.spring_bone1.springs); spring_joints=sum(len(s.joints) for s in ext.spring_bone1.springs)
    spring_names=[{'name':getattr(s,'vrm_name',''),'joints':[j.node.bone_name for j in s.joints]} for s in ext.spring_bone1.springs]
    checks={
        'vrm_1_0':ext.spec_version=='1.0',
        'shape_controls_52':len(found_shapes)==52 and len(missing_shapes)==0,
        'humanoid_required':len(missing_humanoid)==0,
        'preset_18':len(preset_missing)==0 and len(preset_counts)==18,
        'preset_bindings_exact':len(preset_mismatches)==0 and expression_bound==sum(EXPECTED_PRESET_BINDS.values()),
        'look_at':bool(look_type),
        'spring_bones':spring_count>=3 and spring_joints>=6,
        'file_size':a.vrm.stat().st_size>=100_000,
    }
    passed=all(checks.values())
    qa={
        'product':'AINA Final VRM Reimport QA','pass':passed,'source_vrm':str(a.vrm),'vrm_bytes':a.vrm.stat().st_size,
        'armature':rig.name,'spec_version':ext.spec_version,'checks':checks,
        'shape_controls_found':len(found_shapes),'shape_controls_expected':52,
        'missing_shape_controls':missing_shapes,'shape_meshes':shape_meshes,
        'humanoid':humanoid,'missing_humanoid':missing_humanoid,
        'preset_bind_counts':preset_counts,'expected_preset_bind_counts':EXPECTED_PRESET_BINDS,
        'preset_binding_mismatches':preset_mismatches,'missing_presets':preset_missing,
        'presets_verified':18-len(preset_mismatches)-len(preset_missing),
        'presets_expected':18,'total_preset_morph_binds':expression_bound,
        'expected_total_preset_morph_binds':sum(EXPECTED_PRESET_BINDS.values()),
        'look_at_type':look_type,'spring_bone_count':spring_count,'spring_joint_count':spring_joints,'spring_bones':spring_names,
    }
    p=a.out/'QA'/'AINA_VRM_REIMPORT_QA.json'; p.write_text(json.dumps(qa,indent=2),encoding='utf-8')
    bpy.ops.wm.save_as_mainfile(filepath=str(a.out/'AINA_REIMPORT_CHECK.blend'))
    print(json.dumps(qa,indent=2),flush=True)
    if not passed: raise RuntimeError('Final AINA VRM reimport gate failed')

if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc(); raise
