#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    raw=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    return {raw[i].lstrip('-'):raw[i+1] for i in range(0,len(raw),2)}


def look_at(obj,target):
    obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()


def bounds(objects):
    points=[obj.matrix_world@Vector(corner) for obj in objects if obj.type=='MESH' for corner in obj.bound_box]
    if not points: raise RuntimeError('No mesh bounds for quick AINA render')
    return Vector((min(p.x for p in points),min(p.y for p in points),min(p.z for p in points))),Vector((max(p.x for p in points),max(p.y for p in points),max(p.z for p in points)))


def main():
    p=parse_args();source=Path(p['input']).resolve();output=Path(p['output']).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(source))
    for obj in list(bpy.data.objects):
        if obj.type=='MESH' and obj.name.lower().startswith(('icosphere','sphere','cube')) and max(obj.dimensions)>.42:bpy.data.objects.remove(obj,do_unlink=True)
    meshes=[obj for obj in bpy.context.scene.objects if obj.type=='MESH']
    head=[obj for obj in meshes if any(token in obj.name.lower() for token in ('face','hair','updo'))] or meshes
    mn,mx=bounds(head);target=(mn+mx)*.5;target.z=mn.z+(mx.z-mn.z)*.48;size=max(mx.x-mn.x,mx.z-mn.z);distance=max(.58,size*2.50)
    world=bpy.data.worlds.new('AINA_QUICK_WORLD');world.use_nodes=True;bg=world.node_tree.nodes.get('Background');bg.inputs['Color'].default_value=(.012,.018,.032,1);bg.inputs['Strength'].default_value=.2;bpy.context.scene.world=world
    for loc,energy,size_l in [((-.45,.52,target.z+.32),430,.68),((.48,.42,target.z+.08),190,.75),((0,-.45,target.z+.28),300,.55)]:
        bpy.ops.object.light_add(type='AREA',location=loc);light=bpy.context.object;light.data.energy=energy;light.data.shape='DISK';light.data.size=size_l;look_at(light,target)
    bpy.ops.object.camera_add();camera=bpy.context.object;camera.data.lens=86;camera.data.sensor_width=36;camera.location=Vector((target.x,target.y+distance,target.z));look_at(camera,target);bpy.context.scene.camera=camera
    scene=bpy.context.scene;scene.render.engine='BLENDER_EEVEE_NEXT';scene.render.resolution_x=512;scene.render.resolution_y=512;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_depth='8';scene.render.filepath=str(output);scene.render.film_transparent=False
    try:scene.view_settings.look='AgX - Medium High Contrast';scene.view_settings.exposure=-.65
    except Exception:pass
    bpy.ops.render.render(write_still=True)
    if not output.is_file() or output.stat().st_size==0:raise RuntimeError('Quick AINA front render was not produced')
    print({'output':str(output),'bytes':output.stat().st_size,'head_min':list(mn),'head_max':list(mx)})


if __name__=='__main__':main()
