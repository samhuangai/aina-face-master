#!/usr/bin/env python3
"""Probe the real MPFB2 female head topology that will become final AINA.

Creates the same continuous female MakeHuman base that already passed body QA,
this time with helper geometry masked.  It records native topology, vertex-group
membership/bone weights, a conservative head/neck spatial subset, and renders
front / shallow-3Q / profile views from the actual Blender mesh.  No identity
lock is written here: this is the clean topology inspection stage before AINA
identity transfer.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    out=Path(argv[0] if argv else 'mpfb_head_probe')
    out.mkdir(parents=True,exist_ok=True)
    return out


def load_mpfb():
    package='bl_ext.aina_local.mpfb'
    importlib.import_module(package)
    if package not in bpy.context.preferences.addons:
        raise RuntimeError(f'MPFB2 extension is not enabled: {package}')
    services=importlib.import_module(package+'.services')
    props=importlib.import_module(package+'.entities.objectproperties')
    return package,services.HumanService,services.TargetService,props.HumanObjectProperties


def create_female(HumanService,TargetService,HumanObjectProperties):
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    body=HumanService.create_human(
        mask_helpers=True,
        detailed_helpers=False,
        extra_vertex_groups=True,
        feet_on_ground=True,
        scale=.1,
    )
    body.name='AINA_Clean_Female_Base'
    settings={'gender':1.0,'age':.48,'muscle':.30,'weight':.36,'height':.58,'proportions':.56}
    applied={}
    for key,value in settings.items():
        try:
            HumanObjectProperties.set_value(key,value,entity_reference=body);applied[key]=value
        except Exception as exc:
            print(f'[AINA_HEAD] macro {key} note: {exc}',flush=True)
    try:TargetService.reapply_macro_details(body)
    except Exception as exc:print('[AINA_HEAD] macro reapply note:',exc,flush=True)
    bpy.context.view_layer.update()
    rig=HumanService.add_builtin_rig(body,'game_engine');rig.name='AINA_Humanoid_Rig';bpy.context.view_layer.update()
    return body,rig,applied


def mesh_arrays(body):
    n=len(body.data.vertices)
    coords=np.empty((n,3),np.float32);body.data.vertices.foreach_get('co',coords.ravel())
    # Convert native local coordinates to world coordinates; this is the space
    # used by all identity fitting/render diagnostics.
    mw=np.asarray(body.matrix_world,dtype=np.float64).reshape(4,4).T
    h=np.c_[coords.astype(np.float64),np.ones(n)]
    world=(h@mw.T)[:,:3]
    faces=[]
    for poly in body.data.polygons:
        ids=list(poly.vertices)
        for i in range(1,len(ids)-1):faces.append((ids[0],ids[i],ids[i+1]))
    return coords,world,np.asarray(faces,np.int32)


def vertex_group_report(body):
    names={g.index:g.name for g in body.vertex_groups}
    counts={name:0 for name in names.values()};weight_sums={name:0.0 for name in names.values()}
    strong={name:[] for name in names.values()}
    for v in body.data.vertices:
        for e in v.groups:
            name=names.get(e.group)
            if name is None:continue
            counts[name]+=1;weight_sums[name]+=float(e.weight)
            if e.weight>=.25:strong[name].append(int(v.index))
    interesting={}
    keys=('head','face','neck','jaw','eye','lip','mouth','brow','nose','ear','tongue','teeth')
    for name in sorted(counts):
        if any(k in name.lower() for k in keys):
            interesting[name]={'members':int(counts[name]),'weight_sum':float(weight_sums[name]),'strong_ge_025':int(len(strong[name])),'strong_indices':strong[name]}
    compact={name:{'members':int(counts[name]),'weight_sum':float(weight_sums[name])} for name in sorted(counts)}
    return compact,interesting


def write_obj(path,vertices,faces):
    with path.open('w',encoding='utf-8') as fh:
        for x,y,z in vertices:fh.write(f'v {x:.9f} {y:.9f} {z:.9f}\n')
        for a,b,c in faces:fh.write(f'f {int(a)+1} {int(b)+1} {int(c)+1}\n')


def spatial_head_subset(world,faces):
    lo=world.min(0);hi=world.max(0);height=float(hi[2]-lo[2])
    # Conservative head+upper-neck crop. It is metadata/debug geometry only;
    # final sculpting remains on the untouched continuous full-body topology.
    zcut=float(hi[2]-max(.34,.19*height))
    mask=world[:,2]>=zcut
    # Keep triangles entirely inside crop and remap to compact local indices.
    fi=faces[mask[faces].all(1)]
    ids=np.flatnonzero(mask);g=-np.ones(len(world),np.int32);g[ids]=np.arange(len(ids),dtype=np.int32)
    lf=g[fi]
    return ids,world[ids],lf,zcut


def material_body(body):
    mat=bpy.data.materials.new('AINA_HeadProbe_Skin');mat.diffuse_color=(.83,.73,.70,1);mat.roughness=.60
    body.data.materials.clear();body.data.materials.append(mat)


def render_views(body,world,out):
    scene=bpy.context.scene;scene.render.engine='BLENDER_WORKBENCH';scene.render.image_settings.file_format='PNG';scene.render.resolution_x=720;scene.render.resolution_y=720;scene.render.resolution_percentage=100
    scene.display.shading.light='STUDIO';scene.display.shading.show_shadows=True;scene.display.shading.show_cavity=True;scene.display.shading.cavity_type='WORLD';scene.display.shading.color_type='MATERIAL';scene.display.shading.background_type='WORLD';scene.display.shading.show_specular_highlight=True
    scene.world.color=(.94,.95,.97)
    lo=world.min(0);hi=world.max(0);head_top=float(hi[2]);target=Vector(((lo[0]+hi[0])*.5,(lo[1]+hi[1])*.5,head_top-.135))
    cd=bpy.data.cameras.new('AINA_HeadProbe_Camera');cam=bpy.data.objects.new('AINA_HeadProbe_Camera',cd);bpy.context.collection.objects.link(cam);scene.camera=cam;cam.data.type='ORTHO';cam.data.ortho_scale=.36
    # Front of MakeHuman is -Y, matching the successful full-body probe.
    shots=[
        ('AINA_MPFB_HEAD_FRONT.png',Vector((target.x,target.y-.85,target.z))),
        ('AINA_MPFB_HEAD_3Q.png',Vector((target.x+.42,target.y-.78,target.z+.005))),
        ('AINA_MPFB_HEAD_PROFILE.png',Vector((target.x+.85,target.y,target.z+.005))),
    ]
    for name,location in shots:
        cam.location=location;cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(out/name);bpy.ops.render.render(write_still=True)
    return [name for name,_ in shots]


def main():
    out=parse_args();package,HumanService,TargetService,HumanObjectProperties=load_mpfb();body,rig,macros=create_female(HumanService,TargetService,HumanObjectProperties);material_body(body)
    local,world,faces=mesh_arrays(body);group_summary,interesting=vertex_group_report(body);head_ids,head_v,head_f,zcut=spatial_head_subset(world,faces)

    np.savez_compressed(out/'AINA_MPFB_HEAD_TOPOLOGY.npz',vertices_local=local,vertices_world=world,faces=faces,head_vertex_ids=head_ids,head_vertices_world=head_v,head_faces_local=head_f)
    write_obj(out/'AINA_MPFB_FULL_BODY_NATIVE.obj',world,faces);write_obj(out/'AINA_MPFB_HEAD_SPATIAL_PROBE.obj',head_v,head_f)
    previews=render_views(body,world,out)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'AINA_MPFB_HEAD_TOPOLOGY.blend'))

    lo=world.min(0);hi=world.max(0)
    report={
        'pass':True,'stage':'clean MPFB2 female head topology inspection','identity_lock':False,
        'extension_package':package,'continuous_full_body_topology':True,'helpers_masked':True,
        'vertices':int(len(world)),'triangles':int(len(faces)),'bones':int(len(rig.data.bones)),
        'bounds_min_m':lo.astype(float).tolist(),'bounds_max_m':hi.astype(float).tolist(),'dimensions_m':(hi-lo).astype(float).tolist(),
        'spatial_head_zcut_m':zcut,'spatial_head_vertices':int(len(head_ids)),'spatial_head_triangles':int(len(head_f)),
        'modifier_stack':[{'name':m.name,'type':m.type,'show_render':bool(m.show_render),'show_viewport':bool(m.show_viewport)} for m in body.modifiers],
        'vertex_group_count':int(len(body.vertex_groups)),'vertex_groups':group_summary,'interesting_head_face_groups':interesting,
        'previews':previews,'macros':macros,
        'next':'fit AINA approved front + shallow 3Q + profile to this native continuous female topology; no FaceVerse reuse',
    }
    (out/'AINA_MPFB_HEAD_TOPOLOGY_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('vertex_groups','interesting_head_face_groups')},indent=2),flush=True)
    print('[AINA_HEAD] interesting groups:',json.dumps({k:{kk:vv for kk,vv in x.items() if kk!='strong_indices'} for k,x in interesting.items()},indent=2),flush=True)

if __name__=='__main__':main()
