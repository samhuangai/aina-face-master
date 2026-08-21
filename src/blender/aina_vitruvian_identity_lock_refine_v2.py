#!/usr/bin/env python3
"""Bounded visual refinement of the real AINA CC0 FACS identity candidate.

Keeps the 17,161-vertex topology and all 26 expression/viseme deltas unchanged.
The added neutral displacement only addresses failures visible in the actual
candidate renders: long lower third, narrow eye apertures, off-centre irises,
weak nose/lips, broad jaw/neck, and high straight brows. No effect art or VRM.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aina_vitruvian_identity_lock_candidate as base
import aina_vitruvian_identity_sculpt_v2 as v2

ORIGINAL_RESIDUAL = base.residual_sculpt
ORIGINAL_EYES = base.refine_eye_anatomy
ORIGINAL_BROWS = base.create_clean_brows
ORIGINAL_RENDER = base.render_setup


def semantics(skin, basis):
    eye = base.tight_semantic(v2.max_shape_delta(
        skin, basis, ("eyes_closed", "eyes_opened", "eyes_squint")), 0.54)
    brow = base.tight_semantic(v2.max_shape_delta(
        skin, basis, ("eyebrows_",)), 0.52)
    lip = base.tight_semantic(v2.max_shape_delta(
        skin, basis, ("lips_up", "smile_lips", "kiss", "aa_", "ow_")), 0.66)
    mouth = base.tight_semantic(v2.max_shape_delta(
        skin, basis, ("jaw_lower", "mouth_large", "lips_up", "happy", "sad",
                      "kiss", "aa_", "ow_", "p_b_m", "f_v", "ey_eh_uh")), 0.58)
    return eye, brow, lip, mouth


def refined_residual(skin, centres):
    report = ORIGINAL_RESIDUAL(skin, centres)
    keys = skin.data.shape_keys.key_blocks
    basis = v2.key_array(keys.get("Basis") or keys[0])
    p = v2.to_world(skin, basis)
    lo, hi = p.min(0), p.max(0)
    center = (lo + hi) * 0.5
    height, width = float(hi[2]-lo[2]), float(hi[0]-lo[0])
    eyes = sorted(centres, key=lambda q: q[0])
    if len(eyes) >= 2:
        eyes = [eyes[0], eyes[-1]]
        avg = np.mean(eyes, 0)
        face_x, eye_z = float(avg[0]), float(avg[2])
        fs = -1.0 if avg[1] < center[1] else 1.0
    else:
        face_x, eye_z, fs = float(center[0]), float(lo[2]+0.60*height), -1.0
        eyes = [np.array([face_x-0.18*width, center[1]-0.055, eye_z]),
                np.array([face_x+0.18*width, center[1]-0.055, eye_z])]

    front = fs * (p[:,1] - center[1])
    low, high = float(np.quantile(front, .38)), float(np.quantile(front, .94))
    face = v2.smoothstep01((front-low)/max(high-low, 1e-9))
    face *= v2.smoothstep01((0.145-np.abs(p[:,0]-face_x))/0.070)
    eye_sem, brow_sem, lip_sem, mouth_sem = semantics(skin, basis)
    fallback = np.array([face_x, center[1]+fs*.29*height, eye_z-.066])
    mouth = v2.weighted_center(
        p, np.maximum(lip_sem, .35*mouth_sem)*face*
        v2.smoothstep01((eye_z-.006-p[:,2])/.032), fallback)
    mouth_z = float(mouth[2])
    lower_geo = ((p[:,2] < mouth_z) & (p[:,2] > lo[2]+.08*height) &
                 (np.abs(p[:,0]-face_x) < .080) & (face > .34))
    chin_z = float(np.quantile(p[lower_geo,2], .045)) if np.any(lower_geo) else mouth_z-.040
    d = np.zeros_like(p)

    # Shorter lower third and central midface.
    t = np.clip((mouth_z+.007-p[:,2])/max(mouth_z+.007-(chin_z-.015),1e-6),0,1)
    w = face*v2.smoothstep01(t)
    d[:,2] += (mouth_z+(p[:,2]-mouth_z)*.88-p[:,2])*w
    mid = np.clip((eye_z-.002-p[:,2])/max(eye_z-mouth_z,1e-6),0,1)
    mid *= np.clip((p[:,2]-(mouth_z-.006))/max(eye_z-mouth_z,1e-6),0,1)
    d[:,2] += (eye_z+(p[:,2]-eye_z)*.965-p[:,2])*face*v2.smoothstep01(mid)

    # Soft V jaw, smaller chin, narrow neck and crown.
    lower = np.clip((mouth_z+.014-p[:,2])/max(mouth_z+.014-(chin_z-.018),1e-6),0,1)
    lw = face*v2.smoothstep01(lower)
    taper = 1-.115*np.power(lower,1.18)
    d[:,0] += ((p[:,0]-face_x)*taper-(p[:,0]-face_x))*lw
    chin = np.array([face_x, mouth[1]-fs*.003, chin_z+.010])
    cw = v2.add_scaled_region(d,p,chin,(.050,.057,.048),(.80,.96,.86),face,.02,1.10)
    d[:,2] += .0030*cw
    neck = v2.smoothstep01(((chin_z+.002)-p[:,2])/.050)
    neck *= v2.smoothstep01((.095-np.abs(p[:,0]-face_x))/.043)
    d[:,0] += -.145*(p[:,0]-face_x)*neck
    upper = v2.smoothstep01((p[:,2]-(eye_z+.034))/.078)
    upper *= v2.smoothstep01((.130-np.abs(p[:,0]-face_x))/.055)
    d[:,0] += -.050*(p[:,0]-face_x)*upper
    d[:,2] += -.0030*upper*v2.smoothstep01((p[:,2]-(eye_z+.088))/.045)

    # Natural almond eyes.
    for eye in eyes:
        side = -1.0 if eye[0] < face_x else 1.0
        spatial = v2.ellipsoid_weight(p,eye,(.052,.055,.042),0,1.18)
        ew = spatial*np.maximum(eye_sem,.07*brow_sem)
        d[:,0] += (p[:,0]-eye[0])*.055*ew
        d[:,2] += (p[:,2]-eye[2])*.165*ew + .0007*ew
        d[:,1] += fs*.0008*spatial*eye_sem
        outer = eye+np.array([side*.024,0,.001])
        ow = v2.ellipsoid_weight(p,outer,(.026,.032,.022),0,1.08)*eye_sem
        d[:,0] += side*.0010*ow
        d[:,2] += .00125*ow
        under = eye+np.array([0,-fs*.005,-.018])
        uw = v2.add_shift(d,p,under,(.052,.055,.037),(0,fs*.0018,.0007),face,0,1.13)
        d[:,0] += -side*.00035*uw
    d[:,1] += -fs*.0014*brow_sem*face
    d[:,2] += -.0008*brow_sem*face

    # Slim readable nose.
    nose_mask = ((np.abs(p[:,0]-face_x)<.028) & (p[:,2]>mouth_z+.010) &
                 (p[:,2]<eye_z-.003) & (face>.32))
    tip = p[np.where(nose_mask)[0][np.argmax(front[nose_mask])]].copy() if np.any(nose_mask) \
          else np.array([face_x,center[1]+fs*.34*height,mouth_z+.039])
    bridge = np.array([face_x,tip[1],mouth_z+.62*(eye_z-mouth_z)])
    bw = v2.add_scaled_region(d,p,bridge,(.029,.046,.060),(.87,.98,.94),face,0,1.12)
    d[:,1] += fs*.0018*bw
    tw = v2.ellipsoid_weight(p,tip,(.023,.032,.027),.02,1.08)*face
    d[:,1] += fs*.0024*tw
    d[:,2] += .0016*tw
    d[:,0] += -(p[:,0]-face_x)*.065*tw
    nb = np.array([face_x,tip[1],mouth_z+.020])
    nw = v2.add_scaled_region(d,p,nb,(.034,.040,.031),(.84,.98,.92),face,0,1.10)
    d[:,2] += .0010*nw

    # Fuller lips and high apple cheeks.
    lip = v2.ellipsoid_weight(p,mouth,(.052,.050,.031),0,1.14)
    lip *= np.maximum(lip_sem,.45*mouth_sem)
    d[:,0] += (p[:,0]-mouth[0])*.055*lip
    d[:,2] += (p[:,2]-mouth[2])*.235*lip + .0023*lip
    d[:,1] += fs*.0016*lip
    corners = np.clip(np.abs(p[:,0]-mouth[0])/.043,0,1)
    d[:,2] += .00095*np.power(corners,1.7)*lip
    for eye in eyes:
        side = -1.0 if eye[0] < face_x else 1.0
        cheek = np.array([eye[0]+side*.001,tip[1]-fs*.012,mouth_z+.56*(eye_z-mouth_z)])
        ch = v2.add_shift(d,p,cheek,(.056,.056,.052),(0,fs*.0025,.0011),face,0,1.13)
        d[:,0] += -side*.0007*ch

    # Smaller ears.
    ear = v2.smoothstep01((np.abs(p[:,0]-face_x)-width*.33)/(width*.13))
    ear *= v2.smoothstep01((p[:,2]-(mouth_z-.030))/.034)
    ear *= v2.smoothstep01(((eye_z+.030)-p[:,2])/.034)
    ear *= v2.smoothstep01((high-front)/max(high-low,1e-9))
    d[:,0] += -(p[:,0]-face_x)*.070*ear
    d[:,1] += fs*.0012*ear

    length = np.linalg.norm(d,axis=1)
    d *= np.minimum(1,.0075/np.maximum(length,1e-9))[:,None]
    base.apply_same_delta_to_all_keys(skin,d)
    report["refinement_v2"] = {
        "max_additional_displacement_m": float(np.linalg.norm(d,axis=1).max()),
        "rms_additional_displacement_m": float(np.sqrt(np.mean(np.sum(d*d,axis=1)))),
        "mouth_center_world": mouth.tolist(), "chin_z_m": chin_z,
        "nose_tip_world": tip.tolist(),
    }
    return report


def refined_eyes(scene, centres, face_x):
    report = ORIGINAL_EYES(scene,centres,face_x)
    eyeballs = v2.eyeball_objects(scene)
    if not eyeballs:
        return report
    current = sorted([v2.world_vertices(o).mean(0) for o in eyeballs], key=lambda q:q[0])
    meshes = [o for o in scene.objects if o.type == "MESH"]
    allp = np.concatenate([v2.world_vertices(o) for o in meshes],0)
    sc = (allp.min(0)+allp.max(0))*.5
    fs = -1.0 if np.mean(current,0)[1] < sc[1] else 1.0
    maxd, iris_count = 0.0, 0
    for obj in v2.eye_related_objects(scene):
        world = v2.to_world(obj,v2.mesh_local_array(obj)); old = world.copy()
        c = current[0] if world[:,0].mean() < face_x else current[-1]
        offset = np.array([0,fs*.0008,-.0024])
        result = world+offset
        slots = {i for i,m in enumerate(obj.data.materials) if m and "iris" in m.name.lower()}
        ids=set()
        for poly in obj.data.polygons:
            if poly.material_index in slots: ids.update(poly.vertices)
        iris_count += len(ids)
        cc=c+offset
        for i in ids:
            r=result[i]-cc; r[0]*=1.16; r[2]*=1.16; result[i]=cc+r
        v2.set_mesh_local_array(obj,v2.to_local(obj,result))
        if len(result): maxd=max(maxd,float(np.linalg.norm(result-old,axis=1).max()))
    report["refinement_v2"]={"max_displacement_m":maxd,"iris_vertex_count":iris_count,
                              "vertical_offset_m":-.0024,"forward_offset_m":fs*.0008}
    return report


def refined_brows(centres, fs, material):
    names = ORIGINAL_BROWS(centres,fs,material)
    # Bring the generated brows closer to the eye line and add true upper-lash curves.
    for obj in bpy.data.objects:
        if obj.name.startswith("AINA_Candidate_Brow_"):
            obj.location.z -= .0045
    lash_mat = base.make_material("AINA_Refined_Lash",(.006,.006,.010,1),.32)
    for side_name,eye in (("L",sorted(centres,key=lambda q:q[0])[0]),
                          ("R",sorted(centres,key=lambda q:q[0])[-1])):
        sign=-1.0 if eye[0]<0 else 1.0
        curve=bpy.data.curves.new(f"AINA_Refined_Lash_{side_name}_Curve","CURVE")
        curve.dimensions="3D"; curve.resolution_u=5; curve.bevel_depth=.00034; curve.bevel_resolution=2
        spline=curve.splines.new("BEZIER"); offsets=np.linspace(-.026,.026,11)
        spline.bezier_points.add(len(offsets)-1)
        for point,x in zip(spline.bezier_points,offsets):
            n=x/.026; point.co=(float(eye[0]+x),float(eye[1]+fs*.0285),
                               float(eye[2]+.0035+.0041*(1-n*n)+.001*max(0,sign*n)))
            point.handle_left_type="AUTO"; point.handle_right_type="AUTO"
        obj=bpy.data.objects.new(f"AINA_Refined_Lash_{side_name}",curve)
        bpy.context.collection.objects.link(obj); curve.materials.append(lash_mat); names.append(obj.name)
    return names


def refined_render(scene,meshes,centres):
    camera,target,locations=ORIGINAL_RENDER(scene,meshes,centres)
    camera.data.lens=102; scene.render.resolution_x=800; scene.render.resolution_y=800
    scene.world.color=(.026,.030,.043)
    try: scene.view_settings.exposure=-.35
    except Exception: pass
    return camera,target,locations


base.residual_sculpt=refined_residual
base.refine_eye_anatomy=refined_eyes
base.create_clean_brows=refined_brows
base.render_setup=refined_render


def main():
    base.main()
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    out=None
    for i,t in enumerate(argv):
        if t=="--out" and i+1<len(argv): out=Path(argv[i+1]); break
    if not out: return
    rp=out/"QA"/"AINA_IDENTITY_LOCK_CANDIDATE_REPORT.json"
    report=json.loads(rp.read_text())
    report.update({"product":"AINA Real Identity Lock Refinement v2",
                   "identity_lock":False,"visual_identity_lock":False,
                   "candidate":True,"vrm_exported":False,
                   "next_gate":"Inspect actual neutral and expression renders before identity lock."})
    rp.write_text(json.dumps(report,indent=2))
    for src,dst in (("AINA_IDENTITY_LOCK_CANDIDATE.blend","AINA_IDENTITY_LOCK_REFINED_V2.blend"),
                    ("AINA_IDENTITY_LOCK_CANDIDATE.glb","AINA_IDENTITY_LOCK_REFINED_V2.glb")):
        if (out/src).exists(): shutil.copy2(out/src,out/dst)


if __name__=="__main__": main()
