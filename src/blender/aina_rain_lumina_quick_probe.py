#!/usr/bin/env python3
"""Fast four-view probe of the CC-BY Rain GLB conversion."""
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


def quick_locations(center,target,distance):
    return {
        'Y_POS':np.array([target[0],center[1]+distance,target[2]]),
        'Y_NEG':np.array([target[0],center[1]-distance,target[2]]),
        'Y_POS_X_POS_3Q':np.array([target[0]+.43*distance,center[1]+.90*distance,target[2]]),
        'Y_NEG_X_POS_3Q':np.array([target[0]+.43*distance,center[1]-.90*distance,target[2]]),
    }


_original_setup=probe.setup_render

def quick_setup(scene,center,target,size,distance):
    camera=_original_setup(scene,center,target,size,distance)
    scene.render.resolution_x=512
    scene.render.resolution_y=512
    camera.data.lens=88
    return camera


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
