#!/usr/bin/env python3
"""Blender visual gate for the real AINA Identity Master head.

Loads the reconstructed OBJ, preserves its actual topology, uses the real head and
separated eye components, adds only inspection geometry for iris/pupil/brows and
lashes, and renders neutral beauty plus neutral clay at front, 20°, 45° and both
profiles. No replacement effect art, body, animation or VRM packaging is created
at this stage.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

K68 = np.array([
    1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,
    3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,
    5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,
    2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,
    13695,12973,13422,6543,6537,
], dtype=np.int64)


def parse_args():
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--face", type=Path, required=True)
    p.add_argument("--reconstruction-report", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--height", type=float, default=1.72)
    return p.parse_args(argv)


def read_obj(path: Path):
    vertices = []
    faces = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            q = line.split(); vertices.append((float(q[1]),float(q[2]),float(q[3])))
        elif line.startswith("f "):
            ids = [int(x.split("/")[0])-1 for x in line.split()[1:]]
            for i in range(1,len(ids)-1): faces.append((ids[0],ids[i],ids[i+1]))
    return np.asarray(vertices,np.float64), np.asarray(faces,np.int64)


def components(n: int, faces: np.ndarray):
    parent = np.arange(n,dtype=np.int32)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = int(parent[x])
        return x
    def union(a,b):
        ra,rb=find(int(a)),find(int(b))
        if ra != rb: parent[rb]=ra
    for a,b,c in faces:
        union(a,b);union(b,c);union(c,a)
    labels=np.asarray([find(i) for i in range(n)],np.int32)
    groups={}
    for i,r in enumerate(labels): groups.setdefault(int(r),[]).append(i)
    return labels,{k:np.asarray(v,np.int64) for k,v in groups.items()}


def map_face(raw: np.ndarray, height: float):
    out=np.empty_like(raw);s=1.08
    out[:,0]=raw[:,0]*s;out[:,1]=raw[:,2]*s;out[:,2]=-raw[:,1]*s
    out[:,2]+=height-float(out[:,2].max())
    return out


def material(name,color,roughness=.45,metallic=0.0):
    mat=bpy.data.materials.new(name);mat.diffuse_color=tuple(color);mat.use_nodes=True
    bsdf=mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
    if bsdf:
        bsdf.inputs["Base Color"].default_value=tuple(color)
        bsdf.inputs["Roughness"].default_value=roughness
        bsdf.inputs["Metallic"].default_value=metallic
    return mat


def mesh_object(name,vertices,faces,mat):
    mesh=bpy.data.meshes.new(name+"_Mesh")
    mesh.from_pydata([tuple(v) for v in vertices],[],[tuple(int(x) for x in f) for f in faces]);mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
    if mat: mesh.materials.append(mat)
    for p in mesh.polygons: p.use_smooth=True
    return obj


def component_object(name,world,faces,ids,mat):
    mask=np.zeros(len(world),bool);mask[ids]=True
    selected=faces[mask[faces].all(axis=1)]
    remap=np.full(len(world),-1,np.int64);remap[ids]=np.arange(len(ids))
    return mesh_object(name,world[ids],remap[selected],mat)


def disc(name,center,radius,y,mat,vertical_scale=1.04,segments=72):
    center=np.asarray(center,float);verts=[(center[0],y,center[2])]
    for i in range(segments):
        a=2*math.pi*i/segments
        verts.append((center[0]+radius*math.cos(a),y,center[2]+radius*vertical_scale*math.sin(a)))
    faces=[(0,1+i,1+((i+1)%segments)) for i in range(segments)]
    return mesh_object(name,np.asarray(verts,float),np.asarray(faces,np.int32),mat)


def curve(name,points,radius,mat):
    data=bpy.data.curves.new(name+"_Curve","CURVE");data.dimensions="3D";data.resolution_u=5
    data.bevel_depth=radius;data.bevel_resolution=4
    spline=data.splines.new("BEZIER");spline.bezier_points.add(len(points)-1)
    for bp,p in zip(spline.bezier_points,points):
        bp.co=tuple(p);bp.handle_left_type="AUTO";bp.handle_right_type="AUTO"
    obj=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(obj);data.materials.append(mat)
    return obj


def build_contact_sheet(paths:list[Path],out:Path,cols=5):
    # Blender's image API is used to avoid an external Pillow dependency here.
    images=[bpy.data.images.load(str(p),check_existing=False) for p in paths]
    if not images:return
    width=max(im.size[0] for im in images);height=max(im.size[1] for im in images)
    rows=(len(images)+cols-1)//cols
    canvas=bpy.data.images.new(out.stem,width=width*cols,height=height*rows,alpha=False,float_buffer=False)
    pixels=np.ones((height*rows,width*cols,4),np.float32)
    for index,image in enumerate(images):
        arr=np.asarray(image.pixels[:],np.float32).reshape(image.size[1],image.size[0],4)
        row=index//cols;col=index%cols
        y0=(rows-1-row)*height;x0=col*width
        pixels[y0:y0+image.size[1],x0:x0+image.size[0]]=arr
    canvas.pixels.foreach_set(pixels.ravel());canvas.filepath_raw=str(out);canvas.file_format="PNG";canvas.save()


def main():
    a=parse_args();a.out.mkdir(parents=True,exist_ok=True);preview=a.out/"Preview";qa=a.out/"QA"
    preview.mkdir(exist_ok=True);qa.mkdir(exist_ok=True)
    reconstruction=json.loads(a.reconstruction_report.read_text(encoding="utf-8"))
    if reconstruction.get("topology_changed") is not False:
        raise RuntimeError("Identity Master source topology is not preserved")

    bpy.ops.object.select_all(action="SELECT");bpy.ops.object.delete(use_global=False)
    raw,faces=read_obj(a.face);world=map_face(raw,a.height);labels,groups=components(len(raw),faces)
    head_root=max(groups,key=lambda r:len(groups[r]));head_ids=groups[head_root]
    if not np.all(labels[K68]==head_root):raise RuntimeError("Identity anchors left the primary head component")
    eye_groups=sorted([ids for root,ids in groups.items() if root!=head_root and 650<len(ids)<900],key=lambda ids:float(world[ids,0].mean()))
    if len(eye_groups)!=2:raise RuntimeError(f"Expected two separated eye components, got {len(eye_groups)}")

    skin=material("AINA_Identity_Skin",(.79,.64,.61,1),.46)
    eye_white=material("AINA_Identity_EyeWhite",(.96,.975,.99,1),.22)
    iris=material("AINA_Identity_Iris",(.18,.42,.60,1),.18,.02)
    pupil=material("AINA_Identity_Pupil",(.006,.010,.018,1),.16)
    dark=material("AINA_Identity_LashBrow",(.055,.045,.065,1),.32)
    clay=material("AINA_Identity_Clay",(.66,.68,.72,1),.58)

    head=component_object("AINA_IDENTITY_MASTER_HEAD",world,faces,head_ids,skin)
    eyes=[]
    for i,ids in enumerate(eye_groups):
        side="R" if float(world[ids,0].mean())<0 else "L"
        eyes.append(component_object("AINA_IDENTITY_EYE_"+side,world,faces,ids,eye_white))

    lm=world[K68]
    detail_objects=[]
    for side,indices,group in [("R",np.arange(36,42),eye_groups[0]),("L",np.arange(42,48),eye_groups[1])]:
        center=lm[indices].mean(0);front_y=float(np.percentile(world[group,1],2))-0.0007
        detail_objects.append(disc("AINA_IDENTITY_IRIS_"+side,center,.0054,front_y,iris))
        detail_objects.append(disc("AINA_IDENTITY_PUPIL_"+side,center,.0021,front_y-0.0003,pupil,1.0))
        upper=lm[indices[[0,1,2,3]]]
        detail_objects.append(curve("AINA_IDENTITY_LASH_"+side,[(float(p[0]),front_y-0.00055,float(p[2]+0.0005)) for p in upper],.00052,dark))
    for side,indices in [("R",np.arange(17,22)),("L",np.arange(22,27))]:
        pts=lm[indices]
        detail_objects.append(curve("AINA_IDENTITY_BROW_"+side,[(float(p[0]),float(p[1]-0.0015),float(p[2]+0.0010)) for p in pts],.00082,dark))

    scene=bpy.context.scene;scene.render.engine="BLENDER_EEVEE_NEXT";scene.render.image_settings.file_format="PNG"
    scene.render.film_transparent=False;scene.render.resolution_x=700;scene.render.resolution_y=700;scene.render.resolution_percentage=100
    scene.render.image_settings.color_mode="RGBA";scene.world.color=(.94,.95,.97)
    try:
        scene.view_settings.look="AgX - Medium High Contrast";scene.view_settings.exposure=.15
    except Exception:pass
    scene.render.engine="BLENDER_EEVEE_NEXT"
    scene.render.image_settings.file_format="PNG"

    def area(name,loc,energy,size,target):
        data=bpy.data.lights.new(name,"AREA");data.energy=energy;data.shape="DISK";data.size=size
        obj=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(obj);obj.location=loc
        obj.rotation_euler=(Vector(target)-obj.location).to_track_quat("-Z","Y").to_euler()
    center_z=float((lm[27,2]+lm[8,2])*.5);target=(0,0,center_z)
    area("Key",(1.25,-1.65,2.15),570,2.6,target);area("Fill",(-1.35,-1.4,1.85),290,2.5,target);area("Rim",(0,1.45,2.15),360,2.3,target)
    camera_data=bpy.data.cameras.new("AINA_IDENTITY_CAMERA");camera=bpy.data.objects.new("AINA_IDENTITY_CAMERA",camera_data)
    bpy.context.collection.objects.link(camera);scene.camera=camera;camera.data.type="ORTHO"
    head_height=float(world[head_ids,2].max()-world[head_ids,2].min());camera.data.ortho_scale=max(.30,min(.37,head_height*1.12))

    views=[("FRONT",0),("Q3_20",20),("Q3_45",45),("LEFT_PROFILE",90),("RIGHT_PROFILE",-90)]
    beauty_paths=[];clay_paths=[]
    distance=.80
    def render_view(prefix,label,yaw):
        angle=math.radians(yaw);camera.location=(distance*math.sin(angle),-distance*math.cos(angle),center_z)
        camera.rotation_euler=(Vector(target)-camera.location).to_track_quat("-Z","Y").to_euler()
        path=preview/f"AINA_IDENTITY_{prefix}_{label}.png";scene.render.filepath=str(path);bpy.ops.render.render(write_still=True);return path

    for label,yaw in views:beauty_paths.append(render_view("BEAUTY",label,yaw))
    # Clay pass is the actual same mesh, not a separately generated image/model.
    head.data.materials.clear();head.data.materials.append(clay)
    for obj in eyes:
        obj.data.materials.clear();obj.data.materials.append(clay)
    for obj in detail_objects:obj.hide_render=True
    for label,yaw in views:clay_paths.append(render_view("CLAY",label,yaw))
    for obj in detail_objects:obj.hide_render=False

    blend=a.out/"AINA_IDENTITY_MASTER_QA.blend";bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    build_contact_sheet(beauty_paths,qa/"AINA_IDENTITY_BEAUTY_5VIEW.png")
    build_contact_sheet(clay_paths,qa/"AINA_IDENTITY_CLAY_5VIEW.png")

    result={
        "product":"AINA Identity Master Real-Mesh Visual QA",
        "source":str(a.face),
        "real_mesh":True,
        "replacement_effect_art_generated":False,
        "topology_changed":False,
        "vertices":int(len(raw)),
        "faces":int(len(faces)),
        "head_vertices":int(len(head_ids)),
        "eye_components":int(len(eye_groups)),
        "views":[label for label,_ in views],
        "beauty_renders":[str(p) for p in beauty_paths],
        "clay_renders":[str(p) for p in clay_paths],
        "visual_identity_lock":False,
        "manual_gate":"Compare front/20/45/profile against the already-approved AINA references before locking",
        "files":{"blend":str(blend),"blend_bytes":blend.stat().st_size},
    }
    (qa/"AINA_IDENTITY_MASTER_QA.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
