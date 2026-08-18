#!/usr/bin/env python3
"""Layered real-geometry silver hair for AINA visual identity."""
from __future__ import annotations
import math
import numpy as np


def install(visual):
    base=visual.base
    def strand(name,pts,r,mat,parent=None,rig=None):
        ob=base.create_curve(name,pts,r,mat,parent,rig)
        ob.data.resolution_u=5;ob.data.bevel_resolution=4;return ob

    def create_hair(rig,hair_mat,hair_chains):
        # Smooth scalp mass. Fine strands below carry the visible direction; the
        # cap only prevents gaps and keeps a coherent silver silhouette.
        verts=[];faces=[];nphi=96;nt=28;center=np.array([0,.026,1.629]);rx,ry,rz=.104,.094,.122
        for i in range(nphi):
            phi=2*math.pi*i/nphi
            sy=math.sin(phi);frontness=max(0.0,-sy)
            # High centre part and lower temple hairline.
            side_factor=abs(math.cos(phi))
            if sy<0:
                tmax=1.07 + .30*side_factor
            else:
                tmax=2.02
            for k in range(nt):
                th=tmax*k/(nt-1)
                p=center+np.array([rx*math.sin(th)*math.cos(phi),ry*math.sin(th)*math.sin(phi),rz*math.cos(th)])
                verts.append(p.tolist())
        for i in range(nphi):
            ni=(i+1)%nphi
            for k in range(nt-1):
                a=i*nt+k;b=ni*nt+k;c=ni*nt+k+1;d=i*nt+k+1;faces.extend([(a,b,c),(a,c,d)])
        cap=base.mesh_object('AINA_Hair_Cap',np.asarray(verts,float),np.asarray(faces,np.int32));base.assign_single_material(cap,hair_mat);base.bone_parent_preserve(cap,rig,'head')
        for p in cap.data.polygons:p.use_smooth=True

        bun=base.create_uv_sphere('AINA_Hair_Bun',(0,.101,1.697),(.044,.039,.047),hair_mat,'head',rig)
        for p in bun.data.polygons:p.use_smooth=True

        # Parted bangs: left/right groups sweep away from a narrow central part.
        # Thin radii prevent the old cage/pipe appearance.
        for side,sg in [('L',-1.0),('R',1.0)]:
            for i in range(11):
                root=np.array([sg*(.004+.0020*i),-.057+.0005*i,1.723-.0010*i])
                endx=sg*(.017+.0052*i)
                endz=1.655-.0047*i + (.007 if i>7 else 0)
                end=np.array([endx,-.091+.0010*i,endz])
                mid=(root+end)/2 + np.array([sg*.006,-.011,.009])
                strand(f'AINA_Fringe_{side}_{i+1}',[root.tolist(),mid.tolist(),end.tolist()],.00085+.000035*i,hair_mat,'head',rig)

        # A few translucent-looking wisps are represented as extremely fine real
        # curves, not image cards.
        wisps=[
          [(-.003,-.061,1.719),(-.010,-.083,1.674),(-.014,-.094,1.628)],
          [(.004,-.060,1.720),(.010,-.083,1.672),(.012,-.095,1.626)],
          [(-.012,-.058,1.714),(-.025,-.080,1.663),(-.031,-.090,1.614)],
          [(.013,-.058,1.714),(.026,-.080,1.663),(.032,-.090,1.614)],
        ]
        for i,pts in enumerate(wisps):strand(f'AINA_Wisp_{i+1}',pts,.00062,hair_mat,'head',rig)

        # Ear-framing locks. Upper sections follow first spring bones, lower
        # sections the second, so they remain production-dynamic.
        for side,sg,key in [('L',1.0,'HairL'),('R',-1.0,'HairR')]:
            chains=[
              [(sg*.065,-.050,1.674),(sg*.078,-.060,1.625),(sg*.074,-.061,1.580)],
              [(sg*.072,-.043,1.666),(sg*.084,-.052,1.604),(sg*.078,-.055,1.548)],
              [(sg*.078,-.034,1.654),(sg*.088,-.043,1.590),(sg*.080,-.048,1.530)],
              [(sg*.084,-.025,1.645),(sg*.090,-.035,1.578),(sg*.081,-.041,1.516)],
            ]
            bones=hair_chains[key]
            for i,pts in enumerate(chains):strand(f'AINA_SideLock_{side}_{i+1}',pts,.00105+.00010*i,hair_mat,bones[min(i//2,1)],rig)

        # Back flow and bun connection; denser but still fine enough to read as
        # hair instead of tentacles.
        for i,x in enumerate(np.linspace(-.072,.072,11)):
            pts=[(float(x)*.45,.087,1.699),(float(x),.104,1.635),(float(x)*.94,.093,1.555)]
            bone=hair_chains['HairBack'][0 if 2<=i<=8 else 1]
            strand(f'AINA_BackLock_{i+1}',pts,.00115,hair_mat,bone,rig)

        # Directional crown strands lie on top of the cap and visually break the
        # helmet surface while retaining a clean digital-human silhouette.
        for i,a in enumerate(np.linspace(-1.0,1.0,13)):
            root=(a*.008,-.050,1.744)
            end=(a*.082,-.020+abs(a)*.020,1.665-abs(a)*.018)
            mid=((root[0]+end[0])*.50,-.045,(root[2]+end[2])*.50+.012)
            strand(f'AINA_CrownFlow_{i+1}',[root,mid,end],.00055,hair_mat,'head',rig)

    base.create_hair=create_hair
