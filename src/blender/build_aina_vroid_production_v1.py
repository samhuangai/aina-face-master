#!/usr/bin/env python3
import bpy, sys, math, json
from pathlib import Path
from mathutils import Vector

TARGET_NAMES = [
'Fcl_ALL_Neutral','Fcl_ALL_Angry','Fcl_ALL_Fun','Fcl_ALL_Joy','Fcl_ALL_Sorrow','Fcl_ALL_Surprised',
'Fcl_BRW_Angry','Fcl_BRW_Fun','Fcl_BRW_Joy','Fcl_BRW_Sorrow','Fcl_BRW_Surprised',
'Fcl_EYE_Natural','Fcl_EYE_Angry','Fcl_EYE_Close','Fcl_EYE_Close_R','Fcl_EYE_Close_L','Fcl_EYE_Fun','Fcl_EYE_Joy','Fcl_EYE_Joy_R','Fcl_EYE_Joy_L','Fcl_EYE_Sorrow','Fcl_EYE_Surprised','Fcl_EYE_Spread','Fcl_EYE_Iris_Hide','Fcl_EYE_Highlight_Hide',
'Fcl_MTH_Close','Fcl_MTH_Up','Fcl_MTH_Down','Fcl_MTH_Angry','Fcl_MTH_Small','Fcl_MTH_Large','Fcl_MTH_Neutral','Fcl_MTH_Fun','Fcl_MTH_Joy','Fcl_MTH_Sorrow','Fcl_MTH_Surprised','Fcl_MTH_SkinFung','Fcl_MTH_SkinFung_R','Fcl_MTH_SkinFung_L','Fcl_MTH_A','Fcl_MTH_I','Fcl_MTH_U','Fcl_MTH_E','Fcl_MTH_O',
'Fcl_HA_Hide','Fcl_HA_Fung1','Fcl_HA_Fung1_Low','Fcl_HA_Fung1_Up','Fcl_HA_Fung2','Fcl_HA_Fung2_Low','Fcl_HA_Fung2_Up','Fcl_HA_Fung3','Fcl_HA_Fung3_Up','Fcl_HA_Fung3_Low','Fcl_HA_Short','Fcl_HA_Short_Up','Fcl_HA_Short_Low']

def parse_args():
    a=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    d={}
    for i in range(0,len(a),2): d[a[i].lstrip('-')]=a[i+1]
    return d

def clamp(v,a=0.0,b=1.0): return max(a,min(b,v))
def gauss(x,z,cx,cz,rx,rz): return math.exp(-0.5*(((x-cx)/rx)**2+((z-cz)/rz)**2))

def identity_transform(co):
    x,y,z=co.x,co.y,co.z
    t=clamp((1.430-z)/(1.430-1.337))
    x*=1.0-0.105*(t**1.20)
    upper=clamp((z-1.455)/(1.571-1.455))
    x*=1.0-0.035*upper
    lower=clamp((1.390-z)/(1.390-1.337))
    z+=0.0075*(lower**1.30)
    if abs(x)>0.078 and 1.37<z<1.50:
        x*=0.965
    for cx in (-0.044,0.044):
        w=gauss(x,z,cx,1.405,0.035,0.032)
        y+=0.0038*w
        x+=(-1 if cx>0 else 1)*0.0011*w
    if 1.400<z<1.470 and 0.014<abs(x)<0.082 and y>0.020:
        side=1.0 if x>0 else -1.0
        cx=side*0.039; cz=1.431
        x=cx+(x-cx)*0.92
        z=cz+(z-cz)*0.82+0.002
        outer=clamp((abs(x)-0.043)/0.026)
        z+=0.0032*outer
    wb=gauss(x,z,0.0,1.435,0.015,0.030)
    wt=gauss(x,z,0.0,1.402,0.018,0.018)
    wa=gauss(abs(x),z,0.014,1.393,0.012,0.012)
    if y>-0.010:
        y+=0.0045*wb+0.0120*wt+0.0040*wa
        x*=1.0-0.11*wb-0.07*wt
    wp=gauss(x,z,0.0,1.383,0.014,0.012)
    wl=gauss(x,z,0.0,1.374,0.031,0.011)
    if y>0.0:
        y+=0.0024*wp+0.0042*wl
        z+=0.0010*wl
    return Vector((x,y,z))

def rename_shape_keys(face):
    sk=face.data.shape_keys
    if not sk: return
    keys=sk.key_blocks
    if len(keys)-1==len(TARGET_NAMES):
        for kb,name in zip(keys[1:],TARGET_NAMES): kb.name=name
    keys[0].name='Basis'

def deform_all_shape_keys(face):
    if face.data.shape_keys:
        for kb in face.data.shape_keys.key_blocks:
            for p in kb.data: p.co=identity_transform(p.co)
    else:
        for p in face.data.vertices: p.co=identity_transform(p.co)

def find_image(mat):
    if not mat or not mat.use_nodes: return None
    for n in mat.node_tree.nodes:
        if n.type=='TEX_IMAGE' and n.image: return n.image
    return None

def set_blend_method(mat):
    try: mat.surface_render_method='DITHERED'
    except Exception: pass

def build_textured_material(mat, mode):
    img=find_image(mat)
    mat.use_nodes=True; nt=mat.node_tree; nt.nodes.clear()
    out=nt.nodes.new('ShaderNodeOutputMaterial'); bsdf=nt.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value=0.55
    nt.links.new(bsdf.outputs['BSDF'],out.inputs['Surface'])
    if mode=='body':
        bsdf.inputs['Base Color'].default_value=(0.82,0.86,0.94,1); bsdf.inputs['Metallic'].default_value=0.18; bsdf.inputs['Roughness'].default_value=0.32
        return
    if img:
        tex=nt.nodes.new('ShaderNodeTexImage'); tex.image=img; tex.interpolation='Linear'
        if mode=='hair':
            bw=nt.nodes.new('ShaderNodeRGBToBW'); ramp=nt.nodes.new('ShaderNodeValToRGB')
            ramp.color_ramp.elements[0].position=0.08; ramp.color_ramp.elements[0].color=(0.18,0.20,0.28,1)
            ramp.color_ramp.elements[1].position=0.90; ramp.color_ramp.elements[1].color=(0.93,0.95,1.0,1)
            nt.links.new(tex.outputs['Color'],bw.inputs['Color']); nt.links.new(bw.outputs['Val'],ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'],bsdf.inputs['Base Color'])
            nt.links.new(tex.outputs['Alpha'],bsdf.inputs['Alpha']); bsdf.inputs['Metallic'].default_value=0.12; bsdf.inputs['Roughness'].default_value=0.36; set_blend_method(mat)
        elif mode=='iris':
            bw=nt.nodes.new('ShaderNodeRGBToBW'); ramp=nt.nodes.new('ShaderNodeValToRGB')
            ramp.color_ramp.elements[0].color=(0.025,0.10,0.14,1); ramp.color_ramp.elements[1].color=(0.38,0.82,0.92,1)
            nt.links.new(tex.outputs['Color'],bw.inputs['Color']); nt.links.new(bw.outputs['Val'],ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'],bsdf.inputs['Base Color']); nt.links.new(tex.outputs['Alpha'],bsdf.inputs['Alpha']); bsdf.inputs['Metallic'].default_value=0.05; bsdf.inputs['Roughness'].default_value=0.20; set_blend_method(mat)
        elif mode=='skin':
            mix=nt.nodes.new('ShaderNodeMixRGB'); mix.blend_type='MIX'; mix.inputs[0].default_value=0.32; mix.inputs[2].default_value=(0.96,0.76,0.73,1)
            nt.links.new(tex.outputs['Color'],mix.inputs[1]); nt.links.new(mix.outputs['Color'],bsdf.inputs['Base Color']); nt.links.new(tex.outputs['Alpha'],bsdf.inputs['Alpha']); bsdf.inputs['Roughness'].default_value=0.48
            if 'Subsurface Weight' in bsdf.inputs: bsdf.inputs['Subsurface Weight'].default_value=0.055
            set_blend_method(mat)
        elif mode=='mouth':
            mix=nt.nodes.new('ShaderNodeMixRGB'); mix.blend_type='MIX'; mix.inputs[0].default_value=0.30; mix.inputs[2].default_value=(0.86,0.24,0.34,1)
            nt.links.new(tex.outputs['Color'],mix.inputs[1]); nt.links.new(mix.outputs['Color'],bsdf.inputs['Base Color']); nt.links.new(tex.outputs['Alpha'],bsdf.inputs['Alpha']); bsdf.inputs['Roughness'].default_value=0.28; set_blend_method(mat)
        elif mode=='eye_white':
            nt.links.new(tex.outputs['Color'],bsdf.inputs['Base Color']); nt.links.new(tex.outputs['Alpha'],bsdf.inputs['Alpha']); bsdf.inputs['Base Color'].default_value=(0.92,0.95,1.0,1); bsdf.inputs['Roughness'].default_value=0.22; set_blend_method(mat)
        elif mode=='highlight':
            nt.links.new(tex.outputs['Color'],bsdf.inputs['Base Color']); nt.links.new(tex.outputs['Alpha'],bsdf.inputs['Alpha']); bsdf.inputs['Emission Color'].default_value=(0.65,0.9,1.0,1); bsdf.inputs['Emission Strength'].default_value=0.6; set_blend_method(mat)
        elif mode in ('brow','eyeline'):
            tint=(0.19,0.17,0.22,1) if mode=='eyeline' else (0.32,0.28,0.32,1)
            mix=nt.nodes.new('ShaderNodeMixRGB'); mix.blend_type='MULTIPLY'; mix.inputs[2].default_value=tint
            nt.links.new(tex.outputs['Color'],mix.inputs[1]); nt.links.new(mix.outputs['Color'],bsdf.inputs['Base Color']); nt.links.new(tex.outputs['Alpha'],bsdf.inputs['Alpha']); bsdf.inputs['Roughness'].default_value=0.50; set_blend_method(mat)
        else:
            nt.links.new(tex.outputs['Color'],bsdf.inputs['Base Color']); nt.links.new(tex.outputs['Alpha'],bsdf.inputs['Alpha']); set_blend_method(mat)

def material_by_name(name,color,metallic=0.0,rough=0.45,emission=None):
    m=bpy.data.materials.get(name) or bpy.data.materials.new(name); m.use_nodes=True
    bs=m.node_tree.nodes.get('Principled BSDF'); bs.inputs['Base Color'].default_value=color; bs.inputs['Metallic'].default_value=metallic; bs.inputs['Roughness'].default_value=rough
    if emission:
        bs.inputs['Emission Color'].default_value=emission[0]; bs.inputs['Emission Strength'].default_value=emission[1]
    return m

def weight_to_bone(obj,arm,bone):
    if obj.type!='MESH': return
    vg=obj.vertex_groups.new(name=bone); vg.add(list(range(len(obj.data.vertices))),1.0,'REPLACE')
    mod=obj.modifiers.new('AINA_Armature','ARMATURE'); mod.object=arm; obj.parent=arm

def make_curve(name,points,radii,material,bevel=0.0025,bone=None,arm=None):
    cu=bpy.data.curves.new(name,'CURVE'); cu.dimensions='3D'; cu.resolution_u=3; cu.bevel_depth=bevel; cu.bevel_resolution=3; cu.resolution_u=4
    sp=cu.splines.new('NURBS'); sp.points.add(len(points)-1)
    for i,(p,r) in enumerate(zip(points,radii)):
        sp.points[i].co=(*p,1); sp.points[i].radius=r
    sp.order_u=min(4,len(points)); sp.use_endpoint_u=True
    obj=bpy.data.objects.new(name,cu); bpy.context.collection.objects.link(obj); obj.data.materials.append(material)
    bpy.context.view_layer.objects.active=obj; obj.select_set(True); bpy.ops.object.convert(target='MESH'); obj=bpy.context.active_object
    if arm and bone: weight_to_bone(obj,arm,bone)
    return obj

def add_sphere(name,loc,scale,mat,arm,bone):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=40, ring_count=24, location=loc); o=bpy.context.object; o.name=name; o.scale=scale; bpy.ops.object.transform_apply(location=False,rotation=False,scale=True); o.data.materials.append(mat)
    for p in o.data.polygons: p.use_smooth=True
    weight_to_bone(o,arm,bone); return o

def add_box(name,loc,scale,rot,mat,arm,bone):
    bpy.ops.mesh.primitive_cube_add(location=loc,rotation=rot); o=bpy.context.object; o.name=name; o.scale=scale; bpy.ops.object.transform_apply(location=False,rotation=False,scale=True); o.data.materials.append(mat); weight_to_bone(o,arm,bone); return o

def look_at(obj,target): obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()

def setup_scene(out):
    sc=bpy.context.scene; sc.render.engine='BLENDER_EEVEE_NEXT'; sc.render.resolution_x=1000; sc.render.resolution_y=1000; sc.render.resolution_percentage=100; sc.render.image_settings.file_format='PNG'; sc.render.film_transparent=False; sc.view_settings.look='AgX - Medium High Contrast'
    world=bpy.data.worlds.new('AINA_World'); sc.world=world; world.use_nodes=True; bg=world.node_tree.nodes['Background']; bg.inputs['Color'].default_value=(0.028,0.035,0.055,1); bg.inputs['Strength'].default_value=0.42
    target=Vector((0,0.025,1.455))
    for name,loc,energy,size in [('Key',(-0.65,0.62,2.05),1100,0.75),('Fill',(0.62,0.52,1.68),520,0.65),('Rim',(0,-0.55,1.95),850,0.55)]:
        bpy.ops.object.light_add(type='AREA',location=loc); l=bpy.context.object; l.name='LGT_'+name; l.data.energy=energy; l.data.shape='DISK'; l.data.size=size; look_at(l,target)
    bpy.ops.object.camera_add(); cam=bpy.context.object; cam.name='CAM_AINA'; cam.data.lens=82; cam.data.sensor_width=36; sc.camera=cam
    return sc,cam,target

def reset_expression(face):
    if face.data.shape_keys:
        for k in face.data.shape_keys.key_blocks: k.value=0.0

def set_expression(face,name,value=1.0):
    reset_expression(face)
    if face.data.shape_keys and name in face.data.shape_keys.key_blocks: face.data.shape_keys.key_blocks[name].value=value

def render_view(sc,cam,target,path,loc):
    cam.location=loc; look_at(cam,target); sc.render.filepath=str(path); bpy.ops.render.render(write_still=True)

def main():
    a=parse_args(); inp=Path(a['input']).resolve(); out=Path(a['out']).resolve(); out.mkdir(parents=True,exist_ok=True); (out/'Preview').mkdir(exist_ok=True); (out/'QA').mkdir(exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=str(inp))
    face=next(o for o in bpy.context.scene.objects if o.type=='MESH' and o.name.startswith('Face'))
    body=next(o for o in bpy.context.scene.objects if o.type=='MESH' and o.name.startswith('Body'))
    arm=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
    rename_shape_keys(face); deform_all_shape_keys(face)
    for o in (face,body):
        for p in o.data.polygons: p.use_smooth=True
    for mat in face.data.materials:
        n=mat.name.lower(); mode='skin'
        if 'mouth' in n: mode='mouth'
        elif 'eyeiris' in n: mode='iris'
        elif 'eyehighlight' in n: mode='highlight'
        elif 'eyewhite' in n: mode='eye_white'
        elif 'brow' in n: mode='brow'
        elif 'eyeline' in n: mode='eyeline'
        build_textured_material(mat,mode)
    for mat in body.data.materials:
        mode='hair' if 'hair' in mat.name.lower() else 'body'; build_textured_material(mat,mode)
    silver=material_by_name('AINA_SilverHair',(0.72,0.76,0.88,1),0.12,0.30)
    silver_dark=material_by_name('AINA_SilverShadow',(0.35,0.40,0.55,1),0.18,0.30)
    metal=material_by_name('AINA_HairMetal',(0.42,0.50,0.65,1),0.75,0.18)
    lash=material_by_name('AINA_Lash',(0.055,0.045,0.075,1),0.0,0.34)
    lip=material_by_name('AINA_Lip',(0.75,0.18,0.26,1),0.02,0.22)
    crystal=material_by_name('AINA_Core',(0.06,0.38,0.65,1),0.35,0.16,((0.05,0.45,1.0,1),3.2))
    white=material_by_name('AINA_White',(0.86,0.90,0.98,1),0.18,0.28)
    add_sphere('AINA_Bun_Center',(0,-0.055,1.617),(0.058,0.048,0.050),silver,arm,'J_Bip_C_Head')
    add_sphere('AINA_Bun_L',(-0.036,-0.050,1.614),(0.040,0.042,0.044),silver_dark,arm,'J_Bip_C_Head')
    add_sphere('AINA_Bun_R',(0.036,-0.050,1.614),(0.040,0.042,0.044),silver,arm,'J_Bip_C_Head')
    xs=[-0.070,-0.052,-0.035,-0.018,0.0,0.018,0.035,0.052,0.070]
    lengths=[1.425,1.455,1.478,1.493,1.502,1.493,1.478,1.455,1.425]
    for i,(x,ze) in enumerate(zip(xs,lengths)):
        points=[(x*0.35,-0.010,1.583),(x*0.65,0.060,1.555),(x,0.096,ze)]
        make_curve(f'AINA_Fringe_{i}',points,[0.95,0.72,0.12],silver,0.0032,'J_Bip_C_Head',arm)
    for side in (-1,1):
        make_curve(f'AINA_SideLock_{side}',[(side*0.075,0.020,1.555),(side*0.096,0.084,1.470),(side*0.086,0.090,1.365)],[0.9,0.55,0.08],silver_dark,0.0028,'J_Bip_C_Head',arm)
        for j in range(5):
            x=side*(0.030+0.013*j)
            make_curve(f'AINA_Sweep_{side}_{j}',[(x,0.005,1.485+0.015*j),(side*(0.060+0.006*j),-0.030,1.565),(side*(0.018+0.004*j),-0.050,1.610)],[0.8,0.65,0.18],silver,0.0023,'J_Bip_C_Head',arm)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.108,minor_radius=0.0022,major_segments=72,minor_segments=10,location=(0,0.000,1.548)); band=bpy.context.object; band.name='AINA_Headband'; band.data.materials.append(metal); weight_to_bone(band,arm,'J_Bip_C_Head')
    for side in (-1,1):
        for j in range(3): add_box(f'AINA_Pin_{side}_{j}',(side*(0.077+0.006*j),0.035,1.548+0.018*j),(0.0025,0.010,0.022),(0,side*0.22,side*0.18),metal,arm,'J_Bip_C_Head')
    for side in (-1,1):
        cx=side*0.039
        make_curve(f'AINA_UpperLash_{side}',[(cx-side*0.027,0.099,1.431),(cx,0.103,1.443),(cx+side*0.028,0.099,1.434)],[0.25,0.72,0.15],lash,0.0014,'J_Bip_C_Head',arm)
        for j in range(3):
            x=cx+side*(0.022+0.004*j)
            make_curve(f'AINA_LashTip_{side}_{j}',[(x,0.101,1.434+0.002*j),(x+side*0.004,0.103,1.440+0.004*j)],[0.55,0.05],lash,0.0009,'J_Bip_C_Head',arm)
    make_curve('AINA_UpperLip',[(-0.028,0.102,1.376),(-0.013,0.105,1.381),(0,0.106,1.378),(0.013,0.105,1.381),(0.028,0.102,1.376)],[0.1,0.65,0.8,0.65,0.1],lip,0.0020,'J_Bip_C_Head',arm)
    make_curve('AINA_LowerLip',[(-0.027,0.101,1.373),(-0.013,0.105,1.369),(0,0.106,1.368),(0.013,0.105,1.369),(0.027,0.101,1.373)],[0.1,0.55,0.75,0.55,0.1],lip,0.0022,'J_Bip_C_Head',arm)
    add_box('AINA_Collar_L',(-0.035,0.035,1.285),(0.030,0.020,0.070),(0.0,-0.25,-0.24),white,arm,'J_Bip_C_UpperChest')
    add_box('AINA_Collar_R',(0.035,0.035,1.285),(0.030,0.020,0.070),(0.0,0.25,0.24),white,arm,'J_Bip_C_UpperChest')
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=0.035,location=(0,0.078,1.255)); core=bpy.context.object; core.name='AINA_CoreCrystal'; core.scale=(0.72,0.36,1.02); core.rotation_euler[1]=math.radians(45); bpy.ops.object.transform_apply(location=False,rotation=False,scale=True); core.data.materials.append(crystal); weight_to_bone(core,arm,'J_Bip_C_UpperChest')
    sc,cam,target=setup_scene(out)
    reset_expression(face)
    render_view(sc,cam,target,out/'Preview'/'AINA_FRONT.png',Vector((0,0.72,1.455)))
    render_view(sc,cam,target,out/'Preview'/'AINA_THREE_QUARTER.png',Vector((0.43,0.59,1.455)))
    render_view(sc,cam,target,out/'Preview'/'AINA_PROFILE.png',Vector((0.72,0.02,1.455)))
    for key,label in [('Fcl_ALL_Joy','HAPPY'),('Fcl_EYE_Close','BLINK'),('Fcl_MTH_A','AA'),('Fcl_MTH_O','OH')]:
        set_expression(face,key,0.82 if key=='Fcl_ALL_Joy' else 1.0)
        render_view(sc,cam,target,out/'Preview'/f'AINA_{label}.png',Vector((0,0.72,1.455)))
    reset_expression(face)
    full_target=Vector((0,0.0,0.82)); cam.data.lens=58; render_view(sc,cam,full_target,out/'Preview'/'AINA_FULL_BODY.png',Vector((0,3.05,0.88))); cam.data.lens=82
    blend_path=out/'AINA_MASTER_VROID_v1.blend'; bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    glb_path=out/'AINA_EXPORT_VROID_v1.glb'; bpy.ops.export_scene.gltf(filepath=str(glb_path),export_format='GLB',export_skins=True,export_morph=True,export_animations=True,export_apply=False)
    fbx_path=out/'AINA_EXPORT_VROID_v1.fbx'; bpy.ops.export_scene.fbx(filepath=str(fbx_path),use_selection=False,add_leaf_bones=False,bake_anim=False,path_mode='COPY',embed_textures=True)
    report={'product':'AINA VRoid Production Source v1','base':'madjin/vrm-samples fem_vroid.vrm','editable_blend':True,'rig_preserved':True,'shape_key_count':len(face.data.shape_keys.key_blocks) if face.data.shape_keys else 0,'mesh_objects':len([o for o in bpy.context.scene.objects if o.type=='MESH']),'armature_bones':len(arm.data.bones),'identity_lock':False,'visual_candidate':True,'vrm1_patch_pending':True}
    (out/'BUILD_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
