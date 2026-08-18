#!/usr/bin/env python3
"""Fast Blender QA for the actual AINA v15.5 production head.

No effect/reference image is generated. This loads the locked real OBJ, edits its
actual vertices, builds real 3D eye/brow/lash geometry, and renders the actual
model from front and calibrated 20-degree three-quarter views. It intentionally
skips body, VRM export and expensive production packaging so visual identity can
be judged quickly before committing more downstream work.
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector

K=np.array([1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,13695,12973,13422,6543,6537],dtype=np.int64)

def args():
    av=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    p=argparse.ArgumentParser();p.add_argument('--face',type=Path,required=True);p.add_argument('--out',type=Path,required=True);return p.parse_args(av)

def read_obj(path):
    vs=[];fs=[]
    for line in path.read_text(errors='ignore').splitlines():
        if line.startswith('v '):vs.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            q=[int(x.split('/')[0])-1 for x in line.split()[1:]]
            if len(q)==3:fs.append(q)
            else:
                for i in range(1,len(q)-1):fs.append([q[0],q[i],q[i+1]])
    return np.asarray(vs,float),np.asarray(fs,np.int32)

def components(n,faces):
    a=[[] for _ in range(n)]
    for x,y,z in faces:a[x]+=[y,z];a[y]+=[x,z];a[z]+=[x,y]
    root=np.full(n,-1,np.int32);groups={};r=0
    for i in range(n):
        if root[i]>=0:continue
        st=[i];root[i]=r;g=[]
        while st:
            u=st.pop();g.append(u)
            for v in a[u]:
                if root[v]<0:root[v]=r;st.append(v)
        groups[r]=np.asarray(g,np.int32);r+=1
    return root,groups

def mapped(v):
    o=np.empty_like(v);s=1.08;o[:,0]=v[:,0]*s;o[:,1]=v[:,2]*s;o[:,2]=-v[:,1]*s;o[:,2]+=1.72-o[:,2].max();return o

def weight(p,c,r,inner=0.,outer=1.):
    c=np.asarray(c,float);r=np.asarray(r,float);q=np.sqrt(np.sum(((p-c)/r)**2,axis=1));w=np.zeros(len(p));w[q<=inner]=1
    m=(q>inner)&(q<outer)
    if np.any(m):t=(q[m]-inner)/(outer-inner+1e-12);w[m]=.5*(1+np.cos(np.pi*t))
    return w

def shift(a,ids,c,r,d,inner=0.,outer=1.):
    ids=np.asarray(ids,np.int64);p=a[ids].copy();p+=weight(p,c,r,inner,outer)[:,None]*np.asarray(d,float);a[ids]=p

def scale(a,ids,c,r,s,inner=0.,outer=1.):
    ids=np.asarray(ids,np.int64);p=a[ids].copy();c=np.asarray(c,float);w=weight(p,c,r,inner,outer)[:,None];tar=c+(p-c)*np.asarray(s,float);a[ids]=p+w*(tar-p)

def polish(a,head,eye_groups):
    o=a.copy();h=np.asarray(head,np.int64)
    z=o[h,2];t=np.clip((1.570-z)/.068,0,1);p=o[h].copy();p[:,0]*=(1-.075*(t**1.30));o[h]=p
    lm=o[K].copy();chin=lm[8];scale(o,h,chin,(.040,.042,.037),(.83,.98,.95),0,1.08);shift(o,h,chin,(.042,.042,.040),(0,-.0012,.0012),0,1.02)
    z=o[h,2];t=np.clip((z-1.635)/.080,0,1);p=o[h].copy();p[:,0]*=(1-.025*t);o[h]=p
    lm=o[K].copy()
    for rr in (range(36,42),range(42,48)):
        c=lm[list(rr)].mean(0);shift(o,h,(c[0],c[1]+.008,c[2]+.013),(.040,.038,.028),(0,.0045,.0005),0,1.15)
    lm=o[K].copy()
    for rr in (range(17,22),range(22,27)):
        c=lm[list(rr)].mean(0);shift(o,h,c,(.040,.032,.025),(0,.0045,.0002),0,1.15)
    shift(o,h,lm[27],(.032,.032,.040),(0,.0035,0),0,1.10)
    lm=o[K].copy();nb=lm[31:36].mean(0);tip=lm[30];scale(o,h,nb,(.028,.026,.028),(.82,1,.96),0,1.15);shift(o,h,nb,(.030,.030,.032),(0,.0020,.0005),0,1.10);shift(o,h,tip,(.022,.026,.026),(0,.0010,.0003),0,1.05)
    lm=o[K].copy();m=lm[48:60].mean(0);shift(o,h,m,(.052,.040,.038),(0,.0090,0),0,1.20);scale(o,h,m,(.044,.032,.025),(.96,1,.86),0,1.15);shift(o,h,lm[51],(.025,.030,.025),(0,.0020,0),0,1);shift(o,h,lm[57],(.027,.030,.025),(0,.0020,.0005),0,1)
    lm=o[K].copy()
    for c in ((lm[40]+lm[31]+lm[48])/3,(lm[46]+lm[35]+lm[54])/3):shift(o,h,c,(.042,.040,.040),(0,-.0014,.0005),0,1.10)
    lm=o[K].copy()
    for ii in (0,16):
        c=lm[ii];scale(o,h,c,(.034,.043,.055),(.78,.82,.78),0,1.12);shift(o,h,c,(.034,.043,.055),((.004 if c[0]<0 else -.004),.004,0),0,1.05)
    return o

def material(name,color,rough=.42,metal=0.):
    m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True;b=m.node_tree.nodes.get('Principled BSDF');b.inputs['Base Color'].default_value=(*color,1);b.inputs['Roughness'].default_value=rough;b.inputs['Metallic'].default_value=metal;return m

def mesh(name,verts,faces,mat):
    me=bpy.data.meshes.new(name+'_Mesh');me.from_pydata([tuple(x) for x in verts],[],[tuple(map(int,x)) for x in faces]);me.update();me.materials.append(mat);ob=bpy.data.objects.new(name,me);bpy.context.collection.objects.link(ob)
    for p in me.polygons:p.use_smooth=True
    return ob

def almond(name,c,side,white,iris,pupil):
    c=np.asarray(c,float);rx=.0152;rz=.0055;n=56;b=[]
    for i in range(n+1):
        u=-1+2*i/n;b.append((c[0]+rx*u,-.0120,c[2]+rz*(max(0.,1-u*u)**.58)+(0.00042*u if side=='L' else -0.00042*u)))
    for i in range(n+1):
        u=1-2*i/n;b.append((c[0]+rx*u,-.0119,c[2]-rz*.72*(max(0.,1-u*u)**.74)+(0.00022*u if side=='L' else -0.00022*u)))
    v=[(c[0],-.0124,c[2])]+b;f=[(0,1+i,1+((i+1)%len(b))) for i in range(len(b))];mesh(name,v,f,white)
    def disc(nm,r,y,mat,oval=1.04):
        vv=[(c[0],y,c[2])]+[(c[0]+r*math.cos(2*math.pi*i/64),y-.00015,c[2]+r*oval*math.sin(2*math.pi*i/64)) for i in range(64)];ff=[(0,1+i,1+((i+1)%64)) for i in range(64)];mesh(nm,vv,ff,mat)
    disc('AINA_Iris_'+side,.0046,-.01255,iris);disc('AINA_Pupil_'+side,.0019,-.01275,pupil,1.)
def curve(name,pts,r,mat):
    cu=bpy.data.curves.new(name+'_Curve','CURVE');cu.dimensions='3D';cu.resolution_u=5;cu.bevel_depth=r;cu.bevel_resolution=3;sp=cu.splines.new('BEZIER');sp.bezier_points.add(len(pts)-1)
    for bp,p in zip(sp.bezier_points,pts):bp.co=p;bp.handle_left_type='AUTO';bp.handle_right_type='AUTO'
    ob=bpy.data.objects.new(name,cu);bpy.context.collection.objects.link(ob);cu.materials.append(mat);return ob

def main():
    a=args();a.out.mkdir(parents=True,exist_ok=True);raw,fs=read_obj(a.face);roots,gs=components(len(raw),fs);head_root=max(gs,key=lambda r:len(gs[r]));eyes=sorted([r for r,g in gs.items() if 650<len(g)<900],key=lambda r:float(mapped(raw)[g,0].mean()));v=polish(mapped(raw),gs[head_root],[gs[r] for r in eyes])
    skin=material('AINA_Skin',(.82,.69,.67),.48);white=material('AINA_EyeWhite',(.96,.97,.99),.24);iris=material('AINA_Iris',(.17,.37,.48),.20);pupil=material('AINA_Pupil',(.008,.012,.020),.18);dark=material('AINA_LashBrow',(.10,.09,.12),.34);lip=material('AINA_Lip',(.60,.28,.31),.40)
    # Naked external head only: inner-mouth/eye components are deliberately hidden for identity QA.
    mask=roots[fs[:,0]]==head_root;head=mesh('AINA_REAL_HEAD',v,fs[mask],skin);head.data.materials.append(lip);li=1;lm=v[K];mc=lm[48:60].mean(0)
    for p in head.data.polygons:
        c=np.mean([np.asarray(head.data.vertices[i].co) for i in p.vertices],axis=0);q=((c[0]-mc[0])/.023)**2+((c[2]-mc[2])/.0080)**2
        if q<1 and c[1]<.004:p.material_index=li
    for side,rr in [('R',range(36,42)),('L',range(42,48))]:
        c=lm[list(rr)].mean(0);almond('AINA_Eye_'+side,c,side,white,iris,pupil);rx=.0152;pts=[(c[0]-rx,-.0130,c[2]+(.0008 if side=='R' else .0002)),(c[0]-rx*.52,-.0131,c[2]+.0038),(c[0],-.0132,c[2]+.0049),(c[0]+rx*.52,-.0131,c[2]+.0038),(c[0]+rx,-.0130,c[2]+(.0002 if side=='R' else .0008))];curve('AINA_Lash_'+side,pts,.00055,dark)
    for side,ids in [('R',range(17,22)),('L',range(22,27))]:curve('AINA_Brow_'+side,[(float(p[0]),-.0123,float(p[2]+.0003)) for p in lm[list(ids)]],.00085,dark)
    scene=bpy.context.scene;scene.render.engine='BLENDER_EEVEE_NEXT';scene.render.image_settings.file_format='PNG';scene.render.resolution_x=800;scene.render.resolution_y=800;scene.render.resolution_percentage=100;scene.world.color=(.93,.94,.96)
    def area(name,loc,en,size):
        d=bpy.data.lights.new(name,'AREA');d.energy=en;d.shape='DISK';d.size=size;o=bpy.data.objects.new(name,d);bpy.context.collection.objects.link(o);o.location=loc;o.rotation_euler=(Vector((0,0,1.60))-o.location).to_track_quat('-Z','Y').to_euler()
    area('Key',(1.0,-1.5,2.1),500,2.4);area('Fill',(-1.2,-1.4,1.8),260,2.2);area('Rim',(0,1.2,2.0),260,2.0)
    cd=bpy.data.cameras.new('Camera');cam=bpy.data.objects.new('Camera',cd);bpy.context.collection.objects.link(cam);scene.camera=cam;cam.data.lens=85
    for nm,pos in [('AINA_REAL_HEAD_FRONT.png',(0,-.78,1.61)),('AINA_REAL_HEAD_Q3_20.png',(.275,-.755,1.61))]:
        cam.location=pos;cam.rotation_euler=(Vector((0,0,1.61))-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(a.out/nm);bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(a.out/'AINA_REAL_HEAD_QA.blend'))
if __name__=='__main__':main()
