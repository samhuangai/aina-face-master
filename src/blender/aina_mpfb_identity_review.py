#!/usr/bin/env python3
"""Apply the MPFB AINA candidate to the real continuous body and render naked-head QA.

No eyes/hair/brows are added here. The previous decorative test used raw mesh
coordinates while the displayed MPFB body was armature-evaluated, which made the
accessories appear below the chin. Identity must first pass as geometry alone.
"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector

def parse_args():
 argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [];ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);return ap.parse_args(argv)

def apply_candidate(body,path):
 d=np.load(path);v=np.asarray(d['vertices'],np.float32)
 if len(v)!=len(body.data.vertices):raise RuntimeError(f'Candidate/base vertex mismatch {len(v)} != {len(body.data.vertices)}')
 body.data.vertices.foreach_set('co',v.ravel());body.data.update();bpy.context.view_layer.update();return v

def render(out):
 out.mkdir(parents=True,exist_ok=True);scene=bpy.context.scene;scene.render.engine='BLENDER_WORKBENCH';scene.render.image_settings.file_format='PNG';scene.render.resolution_x=820;scene.render.resolution_y=820;scene.render.resolution_percentage=100;scene.world.color=(.94,.95,.97)
 scene.display.shading.light='STUDIO';scene.display.shading.show_shadows=True;scene.display.shading.show_cavity=True;scene.display.shading.cavity_type='WORLD';scene.display.shading.color_type='MATERIAL';scene.display.shading.background_type='WORLD';scene.display.shading.show_specular_highlight=True
 body=bpy.data.objects.get('AINA_Clean_Female_Base');mat=bpy.data.materials.get('AINA_NakedHead_Clay') or bpy.data.materials.new('AINA_NakedHead_Clay');mat.diffuse_color=(.72,.73,.75,1);mat.roughness=.60;body.data.materials.clear();body.data.materials.append(mat)
 rig=bpy.data.objects.get('AINA_Humanoid_Rig')
 if rig:rig.hide_set(True);rig.hide_render=True
 cd=bpy.data.cameras.get('AINA_IdentityReview_Camera') or bpy.data.cameras.new('AINA_IdentityReview_Camera');cam=bpy.data.objects.get('AINA_IdentityReview_Camera') or bpy.data.objects.new('AINA_IdentityReview_Camera',cd)
 if cam.name not in bpy.context.collection.objects:bpy.context.collection.objects.link(cam)
 scene.camera=cam;cam.data.type='ORTHO';cam.data.ortho_scale=.285;target=Vector((0,-.060,1.555))
 shots=[('AINA_MPFB_AINA_FRONT.png',Vector((0,-.82,1.555))),('AINA_MPFB_AINA_3Q.png',Vector((.37,-.73,1.56))),('AINA_MPFB_AINA_PROFILE.png',Vector((.82,-.060,1.56)))]
 for name,loc in shots:
  cam.location=loc;cam.rotation_euler=(target-loc).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(out/name);bpy.ops.render.render(write_still=True)
 return [x[0] for x in shots]

def main():
 a=parse_args();a.out.mkdir(parents=True,exist_ok=True);body=bpy.data.objects.get('AINA_Clean_Female_Base')
 if not body:raise RuntimeError('AINA_Clean_Female_Base missing from topology probe blend')
 apply_candidate(body,a.candidate);previews=render(a.out/'Preview');bpy.ops.wm.save_as_mainfile(filepath=str(a.out/'AINA_MPFB_AINA_IDENTITY_REVIEW.blend'));print('[AINA_MPFB_IDENTITY] naked-head review rendered:',previews,flush=True)
if __name__=='__main__':main()
