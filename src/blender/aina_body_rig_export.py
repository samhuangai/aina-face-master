#!/usr/bin/env python3
import bpy,json
from pathlib import Path
from mpfb.services import HumanService
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'output_character_base'; OUT.mkdir(exist_ok=True)
body=bpy.data.objects.get('AINA_Body_Base')
if body is None: raise RuntimeError('AINA_Body_Base missing')
rig=HumanService.add_builtin_rig(body,'game_engine'); rig.name='AINA_Rig_Base'
def zb(o):
 p=[o.matrix_world@v.co for v in o.data.vertices]; return min(x.z for x in p),max(x.z for x in p)
a,b=zb(body); s=1.72/(b-a)
for o in (body,rig): o.scale=(s,s,s)
a,b=zb(body)
for o in (body,rig): o.location.z-=a
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'AINA_CHARACTER_BASE_v1.blend'))
for o in bpy.context.scene.objects:o.select_set(o in {body,rig})
bpy.context.view_layer.objects.active=body
bpy.ops.export_scene.gltf(filepath=str(OUT/'AINA_CHARACTER_BASE_v1.glb'),export_format='GLB',use_selection=True)
a,b=zb(body); rep={'version':'AINA Character Base v1','status':'body_rig_base','body_vertices':len(body.data.vertices),'body_polygons':len(body.data.polygons),'height_m':b-a,'rig_bones':len(rig.data.bones),'next':'graft AINA v15 head'}
(OUT/'AINA_CHARACTER_BASE_v1_REPORT.json').write_text(json.dumps(rep,indent=2));print(rep)
