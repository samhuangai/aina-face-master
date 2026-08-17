#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from aina_mpfb_addon_runtime import ensure_mpfb_addon


def main():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    out=Path(argv[0] if argv else 'mpfb_probe');out.mkdir(parents=True,exist_ok=True)
    root=Path.cwd();ensure_mpfb_addon(root)
    from mpfb.services import HumanService,TargetService
    from mpfb.entities.objectproperties import HumanObjectProperties

    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    body=HumanService.create_human(mask_helpers=False,detailed_helpers=False,extra_vertex_groups=True,feet_on_ground=True,scale=.1)
    body.name='AINA_Continuous_Body'
    # feminine, lean, neutral adult proportions; face will be replaced by AINA.
    settings={'gender':1.0,'age':.48,'muscle':.30,'weight':.36,'height':.58,'proportions':.56}
    applied={}
    for k,v in settings.items():
        try:
            HumanObjectProperties.set_value(k,v,entity_reference=body);applied[k]=v
        except Exception as e:
            print(f'[AINA_BODY] macro {k} note: {e}',flush=True)
    try:TargetService.reapply_macro_details(body)
    except Exception as e:print('[AINA_BODY] macro reapply note:',e,flush=True)
    bpy.context.view_layer.update()
    rig=HumanService.add_builtin_rig(body,'game_engine');rig.name='AINA_Humanoid_Rig';bpy.context.view_layer.update()

    dims=[float(x) for x in body.dimensions];verts=len(body.data.vertices);faces=len(body.data.polygons);bones=len(rig.data.bones)
    if verts<1000 or faces<1000 or bones<20:raise RuntimeError(f'Implausible MPFB output v={verts} f={faces} bones={bones}')

    # Simple proof render.
    mat=bpy.data.materials.new('AINA_BodyProbe_Mat');mat.diffuse_color=(.78,.79,.82,1);body.data.materials.clear();body.data.materials.append(mat)
    scene=bpy.context.scene;scene.render.engine='BLENDER_EEVEE_NEXT';scene.render.resolution_x=800;scene.render.resolution_y=1200;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.world.color=(.95,.96,.98)
    cd=bpy.data.cameras.new('Camera');cam=bpy.data.objects.new('Camera',cd);bpy.context.collection.objects.link(cam);scene.camera=cam;cam.location=(0,-4.7,1.05);target=(0,0,1.0);from mathutils import Vector;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.lens=62
    ld=bpy.data.lights.new('Key','AREA');ld.energy=800;ld.size=4.0;lo=bpy.data.objects.new('Key',ld);bpy.context.collection.objects.link(lo);lo.location=(2,-3,3)
    scene.render.filepath=str(out/'AINA_MPFB_BODY_FRONT.png');bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'AINA_MPFB_BODY.blend'))
    report={'pass':True,'continuous_mesh':True,'vertices':verts,'polygons':faces,'bones':bones,'dimensions':dims,'macros':applied,'rig':'game_engine'}
    (out/'AINA_MPFB_BODY_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
