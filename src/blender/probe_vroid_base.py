#!/usr/bin/env python3
import bpy, sys, json
from pathlib import Path
from mathutils import Vector

def args():
    a=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    d={}
    for i in range(0,len(a),2): d[a[i].lstrip('-')]=a[i+1]
    return d

def look_at(obj, target):
    obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()

def world_bounds(objs):
    pts=[]
    for o in objs:
        if o.type!='MESH': continue
        mw=o.matrix_world
        pts.extend([mw @ Vector(c) for c in o.bound_box])
    mn=Vector((min(p.x for p in pts),min(p.y for p in pts),min(p.z for p in pts)))
    mx=Vector((max(p.x for p in pts),max(p.y for p in pts),max(p.z for p in pts)))
    return mn,mx

def main():
    p=args(); inp=Path(p['input']).resolve(); out=Path(p['out']).resolve(); out.mkdir(parents=True,exist_ok=True); (out/'Preview').mkdir(exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(inp))
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    for o in meshes:
        for poly in o.data.polygons: poly.use_smooth=True
    face=[o for o in meshes if 'face' in o.name.lower()]
    if not face: face=meshes
    mn,mx=world_bounds(face); center=(mn+mx)*0.5; size=mx-mn
    world=bpy.data.worlds.new('AINA_WORLD'); bpy.context.scene.world=world; world.use_nodes=True
    bg=world.node_tree.nodes.get('Background'); bg.inputs['Color'].default_value=(0.035,0.045,0.065,1); bg.inputs['Strength'].default_value=0.45
    for name,loc,energy,sizeL in [('KEY',(-1.1,-1.6,2.3),900,1.4),('FILL',(1.1,-1.0,1.7),450,1.2),('RIM',(0,1.2,2.2),700,1.0)]:
        bpy.ops.object.light_add(type='AREA',location=loc); l=bpy.context.object; l.name='LGT_'+name; l.data.energy=energy; l.data.shape='DISK'; l.data.size=sizeL; look_at(l,center)
    bpy.ops.object.camera_add(); cam=bpy.context.object; cam.name='CAM_AINA_PROBE'; cam.data.lens=72; cam.data.sensor_width=36; bpy.context.scene.camera=cam
    sc=bpy.context.scene; sc.render.engine='BLENDER_EEVEE_NEXT'; sc.render.resolution_x=900; sc.render.resolution_y=900; sc.render.resolution_percentage=100; sc.render.image_settings.file_format='PNG'; sc.render.film_transparent=False
    sc.view_settings.look='AgX - Medium High Contrast'
    dist=max(size.x,size.z)*3.1
    views={'YNEG':Vector((center.x,center.y-dist,center.z)),'YPOS':Vector((center.x,center.y+dist,center.z)),'XNEG':Vector((center.x-dist,center.y,center.z)),'XPOS':Vector((center.x+dist,center.y,center.z))}
    for name,loc in views.items():
        cam.location=loc; look_at(cam,center); sc.render.filepath=str(out/'Preview'/f'AINA_PROBE_{name}.png'); bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'AINA_VROID_PROBE.blend'))
    bpy.ops.export_scene.gltf(filepath=str(out/'AINA_VROID_PROBE.glb'), export_format='GLB', export_skins=True, export_morph=True, export_animations=True)
    report={'mesh_objects':[o.name for o in meshes], 'face_objects':[o.name for o in face], 'head_bounds_min':list(mn), 'head_bounds_max':list(mx), 'head_center':list(center), 'head_size':list(size), 'shape_keys':{o.name:(len(o.data.shape_keys.key_blocks) if o.data.shape_keys else 0) for o in meshes}, 'armatures':[o.name for o in bpy.context.scene.objects if o.type=='ARMATURE']}
    (out/'PROBE_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
