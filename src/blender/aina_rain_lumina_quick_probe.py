#!/usr/bin/env python3
"""Fast, correctly framed four-view probe of the CC-BY Rain GLB conversion."""
from __future__ import annotations
import sys
from pathlib import Path
import bpy
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import aina_rain_official_rig_probe as probe


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def exact_head_bone(armatures):
    exact=('DEF-Head','MSTR-Head_Upper','LTC-Head_Main','DEF-Head_Top')
    for wanted in exact:
        for armature in armatures:
            bone=armature.data.bones.get(wanted)
            if bone:
                world=armature.matrix_world@bone.head_local
                return (exact.index(wanted),-float(bone.length),armature,bone,np.array(world[:],dtype=np.float64))
    return None


def corrected_head_target(meshes,armatures):
    lo,hi,points=probe.bounds(meshes)
    center=(lo+hi)*.5
    names=('geo-rain-head','geo-rain-eyes','geo-rain-eye_cornea','geo-rain-eye_dots','geo-rain-eyelashes','geo-rain-eyebrows')
    selected=[obj for obj in meshes if obj.name.lower() in names]
    if not selected:
        selected=[obj for obj in meshes if any(token in obj.name.lower() for token in ('head','eye','eyelash','eyebrow')) and 'hair' not in obj.name.lower()]
    hp=np.concatenate([probe.world_array(obj) for obj in selected if len(obj.data.vertices)],axis=0)
    head_lo,head_hi=hp.min(0),hp.max(0)
    head_size=head_hi-head_lo
    head_center=(head_lo+head_hi)*.5
    eye_objs=[obj for obj in selected if any(token in obj.name.lower() for token in ('eyes','eye_cornea','eye_dots'))]
    if eye_objs:
        eye_center=np.mean([probe.world_array(obj).mean(0) for obj in eye_objs],axis=0)
        target=np.array([eye_center[0],head_center[1],eye_center[2]-.105*float(head_size[2])])
    else:
        target=np.array([head_center[0],head_center[1],head_lo[2]+.50*float(head_size[2])])
    distance=max(float(head_size[2])*3.05,float(head_size[0])*3.25,1.00)
    hb=exact_head_bone(armatures)
    info={'armature':hb[2].name if hb else None,'bone':hb[3].name if hb else None}
    return lo,hi,center,target,head_lo,head_hi,head_size,distance,info


def quick_locations(center,target,distance):
    return {
        'Y_POS':np.array([target[0],center[1]+distance,target[2]]),
        'Y_NEG':np.array([target[0],center[1]-distance,target[2]]),
        'Y_POS_X_POS_3Q':np.array([target[0]+.42*distance,center[1]+.91*distance,target[2]]),
        'Y_NEG_X_POS_3Q':np.array([target[0]+.42*distance,center[1]-.91*distance,target[2]]),
    }


_original_setup=probe.setup_render

def quick_setup(scene,center,target,size,distance):
    camera=_original_setup(scene,center,target,size,distance)
    scene.render.resolution_x=640
    scene.render.resolution_y=640
    camera.data.lens=70
    return camera


probe.find_head_bone=exact_head_bone
probe.head_target_and_scale=corrected_head_target
probe.view_locations=quick_locations
probe.setup_render=quick_setup


def main():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    model=None
    for i,value in enumerate(argv):
        if value=='--source' and i+1<len(argv): model=Path(argv[i+1])
    if model is None: raise RuntimeError('--source is required')
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(model))
    bpy.context.view_layer.update()
    probe.main()


if __name__=='__main__': main()
