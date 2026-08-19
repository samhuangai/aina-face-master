#!/usr/bin/env python3
"""AINA real-3D visual identity production assembly.

Consumes the existing v15.5 real FaceVerse topology. No replacement reference is
created and no new face version is invented. The script edits the actual mesh,
rebuilds eye placement and hair, keeps the 52 controls, and renders actual
Blender expression QA for visual identity locking.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aina_final_vrm_assembly as base

K = base.K


def _weights(coords, c, r, inner=0.0, outer=1.0):
    c=np.asarray(c,float); r=np.asarray(r,float)
    q=np.sqrt(np.sum(((coords-c)/r)**2,axis=1))
    w=np.zeros(len(coords),float); w[q<=inner]=1.0
    m=(q>inner)&(q<outer)
    if np.any(m):
        t=(q[m]-inner)/(outer-inner+1e-12)
        w[m]=0.5*(1.0+np.cos(np.pi*t))
    return w


def _shift(coords,c,r,d,inner=0.0,outer=1.0):
    coords += _weights(coords,c,r,inner,outer)[:,None]*np.asarray(d,float)


def _scale(coords,c,r,s,inner=0.0,outer=1.0):
    w=_weights(coords,c,r,inner,outer)[:,None]; c=np.asarray(c,float)
    target=c+(coords-c)*np.asarray(s,float)
    coords += w*(target-coords)


def polish_real_face(mapped, head_ids, eye_groups):
    out=mapped.copy(); h=np.asarray(head_ids,np.int64)

    # Youthful AINA silhouette: compact V-shaped lower third.
    z=out[h,2]
    t=np.clip((1.605-z)/0.10,0,1)
    out[h,0] *= (1.0-0.18*(t**1.45))
    m=z<1.58
    w=np.clip((1.58-z[m])/0.08,0,1)
    out[h[m],2] += 0.009*(w**1.15)

    # Slightly narrower temples/upper cranium; hair supplies the outer contour.
    z=out[h,2]; wt=np.clip((z-1.61)/0.09,0,1)
    out[h,0] *= (1.0-0.045*wt)

    lm=out[K].copy()
    # Larger almond eye apertures, lifted outer tails, softer orbit/brow plane.
    for rr in (range(36,42), range(42,48)):
        c=lm[list(rr)].mean(0)
        _scale(out[h],c,(.040,.024,.020),(1.16,1.0,1.12),.10,1.18)
        side=-1.0 if c[0]<0 else 1.0
        _shift(out[h],(c[0]+side*.016,c[1],c[2]),(.018,.020,.016),(0,.0005,.0015),0,1.10)
        _shift(out[h],(c[0],c[1]+.010,c[2]+.015),(.040,.040,.030),(0,.0020,.0015),0,1.05)

    lm=out[K].copy()
    for rr in (range(17,22),range(22,27)):
        c=lm[list(rr)].mean(0)
        _shift(out[h],c,(.042,.032,.026),(0,.0025,.0035),0,1.18)

    # Shorter, narrower, less projected nose.
    lm=out[K].copy(); bridge=lm[27:31].mean(0); nbase=lm[31:36].mean(0)
    _scale(out[h],nbase,(.026,.028,.032),(.76,1.0,.88),0,1.15)
    _shift(out[h],nbase,(.028,.030,.030),(0,.0042,.0052),0,1.15)
    _shift(out[h],bridge,(.023,.030,.045),(0,.0023,.0023),0,1.10)
    _scale(out[h],lm[27],(.024,.028,.028),(.86,1.0,.96),0,1.05)

    # Soft apple cheeks without making the jaw wide.
    for c in ((-.032,-.002,1.577),(.032,-.002,1.577)):
        _shift(out[h],c,(.044,.042,.040),(0,-.0028,.0018),0,1.12)

    # Lighter, smaller lips and compact mouth placement.
    lm=out[K].copy(); mouth=lm[48:60].mean(0)
    _scale(out[h],mouth,(.043,.032,.026),(.82,1.0,.60),0,1.20)
    _shift(out[h],mouth,(.045,.032,.030),(0,.0018,.0032),0,1.10)
    for corner in (lm[48],lm[54]):
        _shift(out[h],corner,(.018,.020,.014),(0,.0005,.0010),0,1.05)

    # Smaller rounded chin, less lower-face projection.
    lm=out[K].copy(); chin=lm[8]
    _scale(out[h],chin,(.050,.050,.050),(.78,.92,.86),0,1.15)
    _shift(out[h],chin,(.052,.050,.048),(0,.0022,.0040),0,1.05)

    # Smaller tucked ears.
    lm=out[K].copy()
    for ii in (0,16):
        c=lm[ii]
        _scale(out[h],c,(.032,.040,.052),(.70,.80,.72),0,1.15)
        _shift(out[h],c,(.034,.042,.055),((.006 if c[0]<0 else -.006),.003,0),0,1.05)

    # Put eyeballs behind the lids instead of protruding through them.
    lm=out[K].copy()
    for ids in eye_groups:
        ids=np.asarray(ids,np.int64); c=out[ids].mean(0)
        target=lm[36:42].mean(0) if c[0]<0 else lm[42:48].mean(0)
        out[ids] += np.array([target[0]-c[0], .0050, target[2]-c[2]])
        c2=out[ids].mean(0)
        out[ids] = c2+(out[ids]-c2)*np.array([1.08,1.02,1.04])

    return out


def polished_create_face_objects(face_path:Path,height,skin,eye_mat,teeth_mat,mouth_mat):
    raw,faces=base.read_obj(face_path); mapped=base.map_face_vertices(raw,height)
    roots,groups=base.component_data(len(raw),faces)
    head_root=max(groups,key=lambda r:len(groups[r]))
    eye_roots=[r for r,g in groups.items() if 650<len(g)<900]
    eye_roots=sorted(eye_roots,key=lambda r:float(mapped[groups[r],0].mean()))
    if len(eye_roots)!=2:
        raise RuntimeError(f'Expected 2 eye components, got {[(r,len(groups[r])) for r in eye_roots]}')
    oral_roots=sorted([r for r in groups if r!=head_root and r not in eye_roots],key=lambda r:len(groups[r]),reverse=True)
    mapped=polish_real_face(mapped,groups[head_root],[groups[r] for r in eye_roots])

    keep_mask=np.array([roots[int(f[0])] not in set(eye_roots) for f in faces],dtype=bool)
    head_faces=faces[keep_mask]
    head=base.mesh_object('AINA_Face_v15_5',mapped,head_faces)
    head.data.materials.append(skin); head.data.materials.append(teeth_mat); head.data.materials.append(mouth_mat)
    face_roots=[roots[int(f[0])] for f in faces[keep_mask]]; oral_big=set(oral_roots[:2])
    for poly,r in zip(head.data.polygons,face_roots):
        poly.material_index=0 if r==head_root else (1 if r in oral_big else 2); poly.use_smooth=True

    eyes=[]
    for r in eye_roots:
        ids=np.asarray(groups[r],dtype=np.int32); remap={int(g):i for i,g in enumerate(ids)}; sf=[]
        for f in faces[roots[faces[:,0]]==r]: sf.append(tuple(remap[int(x)] for x in f))
        eo=base.mesh_object('AINA_Eye_R' if mapped[ids,0].mean()<0 else 'AINA_Eye_L',mapped[ids],np.asarray(sf,np.int32))
        base.assign_single_material(eo,eye_mat)
        for p in eo.data.polygons: p.use_smooth=True
        eyes.append((eo,ids,mapped[ids].mean(0)))
    tongue_ids=groups[oral_roots[-1]] if oral_roots else np.array([],dtype=np.int32)
    return head,eyes,mapped,groups,head_root,oral_roots,tongue_ids


def _hair_curve(name,pts,radius,material,parent,rig):
    ob=base.create_curve(name,pts,radius,material,parent,rig); ob.data.resolution_u=4; ob.data.bevel_resolution=4; return ob


def polished_create_hair(rig,hair_mat,hair_chains):
    verts=[]; faces=[]; nphi=72; nt=24; center=np.array([0,.022,1.628]); rx,ry,rz=.103,.091,.121
    for i in range(nphi):
        phi=2*math.pi*i/nphi; front=math.sin(phi)<0; frontness=max(0.0,-math.sin(phi)); tmax=(1.22-.18*frontness) if front else 2.02
        for k in range(nt):
            th=tmax*k/(nt-1); p=center+np.array([rx*math.sin(th)*math.cos(phi),ry*math.sin(th)*math.sin(phi),rz*math.cos(th)]); verts.append(p.tolist())
    for i in range(nphi):
        ni=(i+1)%nphi
        for k in range(nt-1):
            a=i*nt+k;b=ni*nt+k;c=ni*nt+k+1;d=i*nt+k+1;faces.extend([(a,b,c),(a,c,d)])
    cap=base.mesh_object('AINA_Hair_Cap',np.asarray(verts,float),np.asarray(faces,np.int32)); base.assign_single_material(cap,hair_mat); base.bone_parent_preserve(cap,rig,'head')
    for p in cap.data.polygons:p.use_smooth=True
    bun=base.create_uv_sphere('AINA_Hair_Bun',(0,.095,1.697),(.046,.040,.048),hair_mat,'head',rig)
    for p in bun.data.polygons:p.use_smooth=True

    part=(-.010,-.055,1.724)
    targets=[(-.070,-.070,1.620),(-.061,-.079,1.606),(-.052,-.085,1.595),(-.043,-.090,1.607),(-.033,-.093,1.620),(-.023,-.096,1.628),(-.012,-.098,1.632),(.004,-.098,1.632),(.018,-.097,1.625),(.030,-.094,1.615),(.042,-.090,1.604),(.054,-.084,1.594),(.064,-.077,1.606),(.072,-.069,1.620)]
    for i,t in enumerate(targets):
        p0=np.array(part,float)+np.array([(i-6.5)*.0011,0,0]);p2=np.array(t,float);p1=(p0+p2)/2+np.array([0,-.013,.008])
        _hair_curve(f'AINA_Fringe_{i+1}',[p0.tolist(),p1.tolist(),p2.tolist()],.0022 if i not in (0,13) else .0026,hair_mat,'head',rig)

    for side,s in [('L',1.0),('R',-1.0)]:
        chains=[[(s*.066,-.056,1.662),(s*.078,-.062,1.615),(s*.070,-.058,1.555)],[(s*.073,-.049,1.650),(s*.084,-.053,1.595),(s*.076,-.050,1.535)],[(s*.079,-.038,1.642),(s*.087,-.041,1.585),(s*.079,-.039,1.525)]]
        bones=hair_chains['HairL' if side=='L' else 'HairR']
        for i,pts in enumerate(chains): _hair_curve(f'AINA_SideLock_{side}_{i+1}',pts,.0027,hair_mat,bones[min(i,1)],rig)
    for i,x in enumerate(np.linspace(-.060,.060,7)):
        pts=[(float(x),.078,1.670),(float(x)*1.08,.090,1.610),(float(x)*.92,.080,1.535)]; bone=hair_chains['HairBack'][0 if i in (2,3,4) else 1]
        _hair_curve(f'AINA_BackLock_{i+1}',pts,.0031,hair_mat,bone,rig)


ORIGINAL_MAKE_MATERIAL=base.make_material

def polished_make_material(name,color,metallic=0.0,roughness=.48,emission=None):
    overrides={'AINA_Skin':((.66,.44,.40,1),0.0,.46,None),'AINA_EyeWhite':((.88,.92,.98,1),0.0,.28,None),'AINA_Iris':((.12,.42,.68,1),.02,.20,None),'AINA_Pupil':((.008,.014,.025,1),.0,.22,None),'AINA_Hair_Silver':((.46,.52,.64,1),.12,.28,None),'AINA_Suit_Pearl':((.56,.64,.76,1),.16,.30,None),'AINA_Teeth':((.90,.88,.84,1),0,.32,None),'AINA_MouthInner':((.28,.055,.075,1),0,.48,None)}
    if name in overrides: color,metallic,roughness,emission=overrides[name]
    return ORIGINAL_MAKE_MATERIAL(name,color,metallic,roughness,emission)


def _clear(head):
    if head.data.shape_keys:
        for kb in head.data.shape_keys.key_blocks: kb.value=0.0


def polished_setup_render(out:Path):
    scene=bpy.context.scene; scene.render.engine='BLENDER_EEVEE_NEXT'; scene.render.image_settings.file_format='PNG'; scene.render.film_transparent=False; scene.world.color=(.055,.065,.085)
    for o in scene.objects:
        if o.type=='MESH':
            for p in o.data.polygons:p.use_smooth=True
    for o in list(scene.objects):
        if o.type in {'LIGHT','CAMERA'}: bpy.data.objects.remove(o,do_unlink=True)
    def area(name,loc,energy,size):
        d=bpy.data.lights.new(name,'AREA');d.energy=energy;d.shape='DISK';d.size=size;o=bpy.data.objects.new(name,d);bpy.context.collection.objects.link(o);o.location=loc;o.rotation_euler=(Vector((0,0,1.58))-o.location).to_track_quat('-Z','Y').to_euler()
    area('AINA_Key',(1.7,-2.4,2.5),420,3.2);area('AINA_Fill',(-1.8,-1.8,2.0),210,2.8);area('AINA_Rim',(0,1.9,2.4),300,2.6)
    cd=bpy.data.cameras.new('AINA_Camera');cam=bpy.data.objects.new('AINA_Camera',cd);bpy.context.collection.objects.link(cam);scene.camera=cam
    head=bpy.data.objects.get('AINA_Face_v15_5');previews=out/'Preview';previews.mkdir(parents=True,exist_ok=True)
    if not head:raise RuntimeError('AINA face missing')
    def render(name,loc,target,vals,res=(768,768)):
        _clear(head)
        for k,v in vals.items():
            if head.data.shape_keys and k in head.data.shape_keys.key_blocks:head.data.shape_keys.key_blocks[k].value=float(v)
        cam.location=loc;cam.data.lens=78;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.resolution_x=res[0];scene.render.resolution_y=res[1];scene.render.resolution_percentage=100;scene.render.filepath=str(previews/name);bpy.ops.render.render(write_still=True)
    cases={'AINA_REAL_NEUTRAL_FRONT.png':{},'AINA_REAL_HAPPY_FRONT.png':{'mouthSmileLeft':.82,'mouthSmileRight':.82,'cheekSquintLeft':.30,'cheekSquintRight':.30},'AINA_REAL_SURPRISED_FRONT.png':{'browInnerUp':.55,'eyeWideLeft':.86,'eyeWideRight':.86,'jawOpen':.58},'AINA_REAL_BLINK_FRONT.png':{'eyeBlinkLeft':1.0,'eyeBlinkRight':1.0},'AINA_REAL_AA_FRONT.png':{'jawOpen':.72,'mouthFunnel':.20}}
    for name,vals in cases.items():render(name,(0,-1.22,1.615),(0,0,1.615),vals)
    render('AINA_REAL_NEUTRAL_3Q.png',(.43,-1.13,1.62),(0,0,1.605),{})
    render('AINA_REAL_FULL_BODY_FRONT.png',(0,-4.7,1.05),(0,0,.98),{},(1024,1536));_clear(head)
    return [str(p) for p in sorted(previews.glob('AINA_REAL_*.png'))]


base.create_face_objects=polished_create_face_objects
base.create_hair=polished_create_hair
base.make_material=polished_make_material
base.setup_render=polished_setup_render


def main():
    base.main()
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [];out=None
    for i,a in enumerate(argv):
        if a=='--out' and i+1<len(argv):out=Path(argv[i+1]).resolve();break
    if out:
        qa={'product':'AINA Real 3D Visual Identity Lock Candidate','real_mesh_edited':True,'new_reference_generated':False,'face_version_incremented':False,'visual_identity_lock':False,'gate':'actual Blender neutral front + shallow 3Q + real expression renders must visually pass before final VRM packaging'}
        (out/'QA'/'AINA_VISUAL_IDENTITY_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')

if __name__=='__main__':main()
