#!/usr/bin/env python3
import bpy, sys, math, json
from pathlib import Path
from mathutils import Vector

def args():
    a=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    d={}
    for i in range(0,len(a),2): d[a[i].lstrip('-')]=a[i+1]
    return d

def look_at(obj,target): obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()

def weight_to_bone(obj,arm,bone):
    vg=obj.vertex_groups.new(name=bone); vg.add(list(range(len(obj.data.vertices))),1.0,'REPLACE')
    mod=obj.modifiers.new('AINA_Armature','ARMATURE'); mod.object=arm; obj.parent=arm

def make_curve(name,points,radii,material,bevel,arm,bone='J_Bip_C_Head'):
    cu=bpy.data.curves.new(name,'CURVE'); cu.dimensions='3D'; cu.resolution_u=4; cu.bevel_depth=bevel; cu.bevel_resolution=3
    sp=cu.splines.new('NURBS'); sp.points.add(len(points)-1)
    for i,(p,r) in enumerate(zip(points,radii)):
        sp.points[i].co=(*p,1); sp.points[i].radius=r
    sp.order_u=min(4,len(points)); sp.use_endpoint_u=True
    obj=bpy.data.objects.new(name,cu); bpy.context.collection.objects.link(obj); obj.data.materials.append(material)
    bpy.context.view_layer.objects.active=obj; obj.select_set(True); bpy.ops.object.convert(target='MESH'); obj=bpy.context.active_object
    weight_to_bone(obj,arm,bone); return obj

def remove_prefixes(prefixes):
    for o in list(bpy.data.objects):
        if any(o.name.startswith(p) for p in prefixes):
            bpy.data.objects.remove(o,do_unlink=True)

def principled(mat):
    if not mat or not mat.use_nodes: return None
    return next((n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)

def tune_materials():
    for m in bpy.data.materials:
        n=m.name.lower(); bs=principled(m)
        if not bs: continue
        if 'face_00_skin' in n:
            for x in m.node_tree.nodes:
                if x.type=='MIX_RGB':
                    x.blend_type='MULTIPLY'; x.inputs[0].default_value=0.72; x.inputs[2].default_value=(0.64,0.43,0.40,1)
            bs.inputs['Roughness'].default_value=0.48
            if 'Subsurface Weight' in bs.inputs: bs.inputs['Subsurface Weight'].default_value=0.035
        elif 'body_00_skin' in n:
            bs.inputs['Base Color'].default_value=(0.40,0.21,0.19,1); bs.inputs['Metallic'].default_value=0.0; bs.inputs['Roughness'].default_value=0.52
        elif 'facemouth' in n:
            for x in m.node_tree.nodes:
                if x.type=='MIX_RGB':
                    x.blend_type='MULTIPLY'; x.inputs[0].default_value=0.55; x.inputs[2].default_value=(0.66,0.18,0.24,1)
            bs.inputs['Roughness'].default_value=0.30
        elif 'hairback' in n:
            ramps=[x for x in m.node_tree.nodes if x.type=='VALTORGB']
            if ramps:
                r=ramps[0].color_ramp; r.elements[0].color=(0.045,0.055,0.09,1); r.elements[1].color=(0.48,0.56,0.72,1)
            bs.inputs['Metallic'].default_value=0.08; bs.inputs['Roughness'].default_value=0.42
        elif 'eyeiris' in n:
            ramps=[x for x in m.node_tree.nodes if x.type=='VALTORGB']
            if ramps:
                r=ramps[0].color_ramp; r.elements[0].color=(0.005,0.028,0.045,1); r.elements[1].color=(0.07,0.38,0.55,1)
            bs.inputs['Roughness'].default_value=0.22
        elif 'eyehighlight' in n:
            if 'Emission Strength' in bs.inputs: bs.inputs['Emission Strength'].default_value=0.18
        elif 'eyewhite' in n:
            bs.inputs['Roughness'].default_value=0.26
        elif 'facebrow' in n or 'faceeyeline' in n:
            bs.inputs['Roughness'].default_value=0.48
        elif m.name=='AINA_SilverHair':
            bs.inputs['Base Color'].default_value=(0.42,0.48,0.62,1); bs.inputs['Metallic'].default_value=0.12; bs.inputs['Roughness'].default_value=0.38
        elif m.name=='AINA_SilverShadow':
            bs.inputs['Base Color'].default_value=(0.16,0.20,0.32,1); bs.inputs['Metallic'].default_value=0.10; bs.inputs['Roughness'].default_value=0.42
        elif m.name=='AINA_HairMetal':
            bs.inputs['Base Color'].default_value=(0.22,0.30,0.46,1); bs.inputs['Metallic'].default_value=0.75; bs.inputs['Roughness'].default_value=0.23
        elif m.name=='AINA_White':
            bs.inputs['Base Color'].default_value=(0.42,0.52,0.70,1); bs.inputs['Metallic'].default_value=0.22; bs.inputs['Roughness'].default_value=0.34
        elif m.name=='AINA_Core':
            bs.inputs['Base Color'].default_value=(0.025,0.18,0.42,1)
            if 'Emission Strength' in bs.inputs: bs.inputs['Emission Strength'].default_value=1.2
        elif m.name=='AINA_Lip':
            bs.inputs['Base Color'].default_value=(0.45,0.055,0.10,1); bs.inputs['Roughness'].default_value=0.28

def add_refined_hair(arm):
    silver=bpy.data.materials['AINA_SilverHair']; shadow=bpy.data.materials['AINA_SilverShadow']; metal=bpy.data.materials['AINA_HairMetal']; lash=bpy.data.materials['AINA_Lash']
    # Compact bun: keep the existing editable lobes but make them smaller and less spherical.
    for name,scale,loc in [
        ('AINA_Bun_Center',(0.72,0.72,0.72),(0,-0.067,1.607)),
        ('AINA_Bun_L',(0.68,0.68,0.68),(-0.030,-0.063,1.605)),
        ('AINA_Bun_R',(0.68,0.68,0.68),(0.030,-0.063,1.605))]:
        o=bpy.data.objects.get(name)
        if o: o.scale=scale; o.location=loc
    # Fine layered fringe, deliberately avoiding thick vertical rods.
    ends=[(-0.074,1.476),(-0.054,1.500),(-0.034,1.486),(-0.013,1.512),(0.013,1.512),(0.034,1.486),(0.054,1.500),(0.074,1.476)]
    for i,(x,ze) in enumerate(ends):
        sx=x*0.22
        points=[(sx,-0.018,1.585),(x*0.55,0.045,1.558),(x*0.88,0.086,ze+0.030),(x,0.099,ze)]
        make_curve(f'AINA_V2_Fringe_{i}',points,[0.95,0.72,0.35,0.06],silver,0.00135,arm)
    for side in (-1,1):
        make_curve(f'AINA_V2_SideLock_{side}',[(side*0.070,0.005,1.553),(side*0.090,0.068,1.480),(side*0.087,0.086,1.390),(side*0.078,0.077,1.355)],[0.85,0.56,0.25,0.035],shadow,0.00135,arm)
        for j in range(4):
            x=side*(0.023+0.014*j)
            make_curve(f'AINA_V2_Sweep_{side}_{j}',[(x,0.000,1.495+0.012*j),(side*(0.056+0.006*j),-0.018,1.555),(side*(0.030+0.004*j),-0.058,1.605)],[0.72,0.56,0.10],silver,0.00115,arm)
    # Crown arc instead of a full torus around the forehead.
    make_curve('AINA_V2_CrownBand',[(-0.094,-0.010,1.545),(-0.070,0.018,1.588),(0,0.028,1.615),(0.070,0.018,1.588),(0.094,-0.010,1.545)],[0.65,0.75,0.8,0.75,0.65],metal,0.00115,arm)
    for side in (-1,1):
        for j in range(3):
            x=side*(0.071+0.005*j); z=1.535+0.014*j
            make_curve(f'AINA_V2_Pin_{side}_{j}',[(x,0.033,z),(x+side*0.008,0.023,z+0.018)],[0.55,0.20],metal,0.0009,arm)
        cx=side*0.039
        make_curve(f'AINA_V2_UpperLash_{side}',[(cx-side*0.027,0.098,1.431),(cx,0.102,1.442),(cx+side*0.028,0.098,1.434)],[0.15,0.55,0.08],lash,0.00082,arm)

def tune_existing_objects():
    for n in ('AINA_Collar_L','AINA_Collar_R'):
        o=bpy.data.objects.get(n)
        if o:
            o.scale=(0.70,0.72,0.74); o.location.z-=0.010
    core=bpy.data.objects.get('AINA_CoreCrystal')
    if core: core.scale=(0.82,0.82,0.82)

def setup_scene():
    sc=bpy.context.scene; sc.render.engine='BLENDER_EEVEE_NEXT'; sc.render.resolution_x=820; sc.render.resolution_y=820; sc.render.resolution_percentage=100; sc.render.image_settings.file_format='PNG'; sc.render.film_transparent=False
    sc.view_settings.look='AgX - Medium High Contrast'; sc.view_settings.exposure=-1.35
    if sc.world is None: sc.world=bpy.data.worlds.new('AINA_World_v2')
    sc.world.use_nodes=True; bg=sc.world.node_tree.nodes.get('Background'); bg.inputs['Color'].default_value=(0.018,0.025,0.045,1); bg.inputs['Strength'].default_value=0.08
    for o in list(bpy.data.objects):
        if o.type=='LIGHT': bpy.data.objects.remove(o,do_unlink=True)
    target=Vector((0,0.030,1.475))
    for name,loc,energy,size in [('Key',(-0.62,0.66,1.95),115,0.85),('Fill',(0.55,0.55,1.65),42,0.80),('Rim',(0,-0.48,1.92),80,0.65)]:
        bpy.ops.object.light_add(type='AREA',location=loc); l=bpy.context.object; l.name='LGT_V2_'+name; l.data.energy=energy; l.data.shape='DISK'; l.data.size=size; look_at(l,target)
    cam=bpy.data.objects.get('CAM_AINA')
    if not cam:
        bpy.ops.object.camera_add(); cam=bpy.context.object
    cam.name='CAM_AINA_V2'; cam.data.lens=76; cam.data.sensor_width=36; sc.camera=cam
    return sc,cam,target

def reset(face):
    if face.data.shape_keys:
        for k in face.data.shape_keys.key_blocks: k.value=0

def render(sc,cam,target,path,loc):
    cam.location=loc; look_at(cam,target); sc.render.filepath=str(path); bpy.ops.render.render(write_still=True)

def main():
    a=args(); out=Path(a['out']).resolve(); out.mkdir(parents=True,exist_ok=True); (out/'Preview').mkdir(exist_ok=True)
    face=next(o for o in bpy.context.scene.objects if o.type=='MESH' and o.name.startswith('Face'))
    arm=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
    remove_prefixes(('AINA_Fringe_','AINA_SideLock_','AINA_Sweep_','AINA_UpperLash_','AINA_LashTip_','AINA_UpperLip','AINA_LowerLip','AINA_Pin_','AINA_Headband'))
    tune_materials(); add_refined_hair(arm); tune_existing_objects(); sc,cam,target=setup_scene(); reset(face)
    render(sc,cam,target,out/'Preview'/'AINA_V2_FRONT.png',Vector((0,0.82,1.49)))
    render(sc,cam,target,out/'Preview'/'AINA_V2_THREE_QUARTER.png',Vector((0.43,0.69,1.49)))
    render(sc,cam,target,out/'Preview'/'AINA_V2_PROFILE.png',Vector((0.82,0.02,1.49)))
    if face.data.shape_keys and 'Fcl_ALL_Joy' in face.data.shape_keys.key_blocks:
        face.data.shape_keys.key_blocks['Fcl_ALL_Joy'].value=0.80
    render(sc,cam,target,out/'Preview'/'AINA_V2_HAPPY.png',Vector((0,0.82,1.49))); reset(face)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'AINA_MASTER_VROID_v2.blend'))
    bpy.ops.export_scene.gltf(filepath=str(out/'AINA_EXPORT_VROID_v2.glb'),export_format='GLB',export_skins=True,export_morph=True,export_animations=True,export_apply=False)
    report={'product':'AINA VRoid Identity v2','source':'AINA_MASTER_VROID_v1.blend','lighting_fixed':True,'hair_rebuilt':True,'lip_overlay_removed':True,'rig_preserved':True,'shape_keys':len(face.data.shape_keys.key_blocks) if face.data.shape_keys else 0,'identity_lock':False,'visual_candidate':True}
    (out/'BUILD_REPORT_V2.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
