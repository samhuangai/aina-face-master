#!/usr/bin/env python3
import bpy, sys, json, math
from pathlib import Path
from mathutils import Vector

def args():
    a=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    return {a[i].lstrip('-'):a[i+1] for i in range(0,len(a),2)}

def look(obj,target):
    obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()

def world_vertices(obj):
    mw=obj.matrix_world
    return [mw@v.co for v in obj.data.vertices]

def main():
    p=args(); src=Path(p['blend']).resolve(); out=Path(p['out']).resolve(); (out/'Preview').mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(src))
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    base=next((o for o in meshes if o.name.lower()=='base'), max(meshes,key=lambda o:len(o.data.vertices)))
    pts=world_vertices(base)
    zmin=min(v.z for v in pts); zmax=max(v.z for v in pts); cut=zmin+.70*(zmax-zmin)
    hp=[v for v in pts if v.z>cut]
    mn=Vector((min(v.x for v in hp),min(v.y for v in hp),min(v.z for v in hp)))
    mx=Vector((max(v.x for v in hp),max(v.y for v in hp),max(v.z for v in hp)))
    center=(mn+mx)*.5; size=mx-mn
    if bpy.context.scene.world is None: bpy.context.scene.world=bpy.data.worlds.new('AINA_World')
    w=bpy.context.scene.world; w.use_nodes=True; bg=w.node_tree.nodes.get('Background'); bg.inputs['Color'].default_value=(.025,.032,.048,1); bg.inputs['Strength'].default_value=.33
    for o in [o for o in bpy.context.scene.objects if o.type in {'LIGHT','CAMERA'}]: bpy.data.objects.remove(o,do_unlink=True)
    for name,offset,energy,sizeL in [('Key',(-1.4,-1.8,1.1),1100,1.4),('Fill',(1.2,-1.1,.3),500,1.2),('Rim',(0,1.4,1.2),900,1.0)]:
        bpy.ops.object.light_add(type='AREA',location=center+Vector(offset)*max(size.x,size.z)); l=bpy.context.object;l.name='LGT_'+name;l.data.energy=energy;l.data.shape='DISK';l.data.size=sizeL;look(l,center)
    bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.lens=78;cam.data.sensor_width=36;bpy.context.scene.camera=cam
    sc=bpy.context.scene;sc.render.engine='BLENDER_EEVEE_NEXT';sc.render.resolution_x=900;sc.render.resolution_y=900;sc.render.resolution_percentage=100;sc.render.image_settings.file_format='PNG';sc.render.film_transparent=False
    sc.view_settings.look='AgX - Medium High Contrast'
    dist=max(size.x,size.z)*3.2
    views={'YNEG_FRONT':center+Vector((0,-dist,0)),'YPOS_FRONT':center+Vector((0,dist,0)),'YNEG_3Q':center+Vector((dist*.62,-dist*.86,0)),'YPOS_3Q':center+Vector((dist*.62,dist*.86,0)),'PROFILE':center+Vector((dist,0,0))}
    for n,loc in views.items():cam.location=loc;look(cam,center);sc.render.filepath=str(out/'Preview'/f'AINA_MPFB_CLOSE_{n}.png');bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'AINA_MPFB_CLOSEUP_REVIEW.blend'))
    report={'base':base.name,'mesh_objects':[o.name for o in meshes],'head_min':list(mn),'head_max':list(mx),'head_center':list(center),'head_size':list(size),'armatures':[o.name for o in bpy.context.scene.objects if o.type=='ARMATURE']}
    (out/'MPFB_CLOSEUP_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
