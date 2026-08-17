#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import bpy
import numpy as np

def main():
 argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [];out=Path(argv[0] if argv else 'rig_probe');out.mkdir(parents=True,exist_ok=True)
 body=bpy.data.objects.get('AINA_Clean_Female_Base');rig=bpy.data.objects.get('AINA_Humanoid_Rig')
 if not body or not rig:raise RuntimeError('MPFB body/rig missing')
 names=[b.name for b in rig.data.bones]
 groups={g.name:g.index for g in body.vertex_groups}
 selected={}
 for name in ('head','lips','ears','neck_01'):
  if name not in groups:continue
  gi=groups[name];ids=[]
  for v in body.data.vertices:
   for e in v.groups:
    if e.group==gi and e.weight>=.25:ids.append(v.index);break
  selected[name]=np.asarray(ids,np.int32)
 raw=np.empty((len(body.data.vertices),3),np.float64);body.data.vertices.foreach_get('co',raw.ravel())
 # Disable only helper mask to preserve original vertex order while keeping the
 # armature modifier active. The evaluated mesh is then index-compatible.
 mask_states=[]
 for m in body.modifiers:
  if m.type=='MASK':mask_states.append((m,m.show_viewport,m.show_render));m.show_viewport=False;m.show_render=False
 deps=bpy.context.evaluated_depsgraph_get();ev=body.evaluated_get(deps);me=ev.to_mesh();eva=np.empty((len(me.vertices),3),np.float64);me.vertices.foreach_get('co',eva.ravel());ev.to_mesh_clear()
 for m,sv,sr in mask_states:m.show_viewport=sv;m.show_render=sr
 if len(eva)!=len(raw):raise RuntimeError(f'Evaluated count changed {len(raw)}->{len(eva)}')
 delta=eva-raw
 rep={'pass':True,'bone_count':len(names),'bone_names':names,'vertex_count':len(raw),'armature_eval_delta_max_m':float(np.linalg.norm(delta,axis=1).max()),'armature_eval_delta_mean_m':float(np.linalg.norm(delta,axis=1).mean()),'groups':{}}
 for name,ids in selected.items():
  rep['groups'][name]={'count':int(len(ids)),'raw_min':raw[ids].min(0).tolist(),'raw_max':raw[ids].max(0).tolist(),'eval_min':eva[ids].min(0).tolist(),'eval_max':eva[ids].max(0).tolist(),'mean_delta':delta[ids].mean(0).tolist(),'max_delta_m':float(np.linalg.norm(delta[ids],axis=1).max())}
 (out/'AINA_MPFB_RIG_SPACE_QA.json').write_text(json.dumps(rep,indent=2),encoding='utf-8');print(json.dumps(rep,indent=2),flush=True)
if __name__=='__main__':main()
