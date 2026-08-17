#!/usr/bin/env python3
"""Apply MPFB AINA candidate coordinates to the real continuous body and render QA."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);return ap.parse_args(argv)


def mat(name,color,rough=.45,metal=0.0):
    old=bpy.data.materials.get(name);m=old or bpy.data.materials.new(name);m.diffuse_color=tuple(color);m.roughness=rough;m.metallic=metal;return m


def sphere(name,loc,scale,material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=16,radius=1.0,location=loc);o=bpy.context.object;o.name=name;o.scale=scale;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);o.data.materials.append(material);return o


def curve(name,pts,radius,material):
    old=bpy.data.objects.get(name)
    if old:bpy.data.objects.remove(old,do_unlink=True)
    cu=bpy.data.curves.new(name+'_Curve','CURVE');cu.dimensions='3D';cu.bevel_depth=radius;cu.bevel_resolution=3;cu.resolution_u=3
    sp=cu.splines.new('BEZIER');sp.bezier_points.add(len(pts)-1)
    for bp,p in zip(sp.bezier_points,pts):bp.co=p;bp.handle_left_type='AUTO';bp.handle_right_type='AUTO'
    ob=bpy.data.objects.new(name,cu);bpy.context.collection.objects.link(ob);cu.materials.append(material);return ob


def apply_candidate(body,path):
    d=np.load(path);v=np.asarray(d['vertices'],np.float32)
    if len(v)!=len(body.data.vertices):raise RuntimeError(f'Candidate/base vertex mismatch {len(v)} != {len(body.data.vertices)}')
    body.data.vertices.foreach_set('co',v.ravel());body.data.update();bpy.context.view_layer.update();return v


def add_face_details():
    skin=mat('AINA_Skin',(0.96,.82,.80,1),.55);white=mat('AINA_EyeWhite',(.96,.98,1,1),.30);iris=mat('AINA_Iris',(.28,.56,.76,1),.24,.02);pupil=mat('AINA_Pupil',(.015,.022,.035,1),.25);dark=mat('AINA_BrowLash',(.055,.060,.072,1),.42);hair=mat('AINA_HairSilver',(.79,.83,.90,1),.34,.03)
    body=bpy.data.objects.get('AINA_Clean_Female_Base');body.data.materials.clear();body.data.materials.append(skin)
    # Eye centers match the semantic sculpt. Front is -Y.
    for sg,label in [(-1,'R'),(1,'L')]:
        c=Vector((sg*.0355,-.1375,1.565));sphere('AINA_Eye_'+label,c,(.0132,.0126,.0124),white);sphere('AINA_Iris_'+label,c+Vector((0,-.0120,0)),(.0078,.0012,.0078),iris);sphere('AINA_Pupil_'+label,c+Vector((0,-.0130,0)),(.0034,.0008,.0034),pupil)
        # soft upper lash with a modest raised outer tail
        x0=sg*.019;x1=sg*.0355;x2=sg*.052
        zouter=1.568 if sg>0 else 1.568
        curve('AINA_UpperLash_'+label,[Vector((x0,-.1510,1.564)),Vector((x1,-.1515,1.570)),Vector((x2,-.1490,zouter))],.0009,dark)
        curve('AINA_Brow_'+label,[Vector((sg*.018,-.1420,1.591)),Vector((sg*.035,-.1450,1.596)),Vector((sg*.054,-.1400,1.592))],.00135,dark)
    # Center-part silver hair; strands are deliberately sparse so the facial
    # geometry remains easy to inspect while preserving AINA's identity frame.
    hair_lines=[
      [(-.082,-.050,1.642),(-.090,-.070,1.585),(-.082,-.060,1.515),(-.070,-.045,1.455)],
      [(.082,-.050,1.642),(.090,-.070,1.585),(.082,-.060,1.515),(.070,-.045,1.455)],
      [(-.060,-.075,1.650),(-.047,-.103,1.615),(-.032,-.114,1.575)],
      [(-.034,-.085,1.657),(-.022,-.112,1.622),(-.012,-.119,1.585)],
      [(.034,-.085,1.657),(.022,-.112,1.622),(.012,-.119,1.585)],
      [(.060,-.075,1.650),(.047,-.103,1.615),(.032,-.114,1.575)],
    ]
    for i,pts in enumerate(hair_lines):curve(f'AINA_Hair_{i+1}',[Vector(p) for p in pts],.0038 if i<2 else .0028,hair)


def render(out):
    out.mkdir(parents=True,exist_ok=True);scene=bpy.context.scene;scene.render.engine='BLENDER_WORKBENCH';scene.render.image_settings.file_format='PNG';scene.render.resolution_x=760;scene.render.resolution_y=760;scene.render.resolution_percentage=100;scene.world.color=(.94,.95,.97)
    scene.display.shading.light='STUDIO';scene.display.shading.show_shadows=True;scene.display.shading.show_cavity=True;scene.display.shading.cavity_type='WORLD';scene.display.shading.color_type='MATERIAL';scene.display.shading.background_type='WORLD';scene.display.shading.show_specular_highlight=True
    # Hide rig in viewport/render overlays; mesh remains continuous and rigged.
    rig=bpy.data.objects.get('AINA_Humanoid_Rig')
    if rig:rig.hide_set(True);rig.hide_render=True
    cd=bpy.data.cameras.get('AINA_IdentityReview_Camera') or bpy.data.cameras.new('AINA_IdentityReview_Camera');cam=bpy.data.objects.get('AINA_IdentityReview_Camera') or bpy.data.objects.new('AINA_IdentityReview_Camera',cd)
    if cam.name not in bpy.context.collection.objects:bpy.context.collection.objects.link(cam)
    scene.camera=cam;cam.data.type='ORTHO';cam.data.ortho_scale=.315;target=Vector((0,-.055,1.555))
    shots=[('AINA_MPFB_AINA_FRONT.png',Vector((0,-.82,1.555))),('AINA_MPFB_AINA_3Q.png',Vector((.37,-.73,1.56))),('AINA_MPFB_AINA_PROFILE.png',Vector((.82,-.055,1.56)))]
    for name,loc in shots:
        cam.location=loc;cam.rotation_euler=(target-loc).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(out/name);bpy.ops.render.render(write_still=True)
    return [x[0] for x in shots]


def main():
    a=parse_args();a.out.mkdir(parents=True,exist_ok=True);body=bpy.data.objects.get('AINA_Clean_Female_Base')
    if not body:raise RuntimeError('AINA_Clean_Female_Base missing from topology probe blend')
    apply_candidate(body,a.candidate);add_face_details();previews=render(a.out/'Preview');bpy.ops.wm.save_as_mainfile(filepath=str(a.out/'AINA_MPFB_AINA_IDENTITY_REVIEW.blend'))
    print('[AINA_MPFB_IDENTITY] real continuous-body review rendered:',previews,flush=True)

if __name__=='__main__':main()
