#!/usr/bin/env python3
"""Render actual AINA v15.4 mesh with corrected reference camera calibration."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw
from rebuild_aina_v133_landmark_laplacian import render_vtk,compare

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--clay',type=Path,required=True);ap.add_argument('--front',type=Path,required=True);ap.add_argument('--q3',type=Path,required=True);ap.add_argument('--side',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    views=[(-90,'right_facing_profile'),(-45,'deep_left_45'),(-25,'q3_25'),(-20,'q3_20_primary'),(-15,'q3_15'),(0,'front'),(45,'deep_right_45'),(90,'left_facing_profile')]
    paths=[]
    for yaw,label in views:
        p=args.out/f'AINA_VTK_{label}_v15.4.png';render_vtk(args.clay,p,yaw);paths.append((yaw,label,p))
    ims=[Image.open(p).convert('RGB') for _,_,p in paths];w=max(x.width for x in ims);h=max(x.height for x in ims)
    sheet=Image.new('RGB',(len(ims)*w,h+30),'white');d=ImageDraw.Draw(sheet)
    for i,((yaw,label,_),im) in enumerate(zip(paths,ims)):
        sheet.paste(im,(i*w+(w-im.width)//2,30));d.text((i*w+8,7),f'{yaw:+d} {label}',fill='black')
    sheet.save(args.out/'AINA_VTK_CALIBRATED_8VIEW_v15.4.png')
    compare(args.front,args.out/'AINA_VTK_front_v15.4.png',args.out/'AINA_REFERENCE_VS_VTK_FRONT_v15.4.png')
    compare(args.q3,args.out/'AINA_VTK_q3_20_primary_v15.4.png',args.out/'AINA_REFERENCE_3Q_VS_VTK_Q3_20_v15.4.png')
    compare(args.side,args.out/'AINA_VTK_left_facing_profile_v15.4.png',args.out/'AINA_REFERENCE_SIDE_VS_VTK_PROFILE_CORRECT_v15.4.png')
    # A shallow-yaw sweep is kept so likeness judgement is not forced to one guessed angle.
    qims=[]
    for yaw,label in [(-15,'q3_15'),(-20,'q3_20_primary'),(-25,'q3_25')]:
        qims.append(Image.open(args.out/f'AINA_VTK_{label}_v15.4.png').convert('RGB'))
    ref=Image.open(args.q3).convert('RGB');H=max([ref.height]+[x.height for x in qims]);ref=ref.resize((int(ref.width*H/ref.height),H));qm=[x.resize((int(x.width*H/x.height),H)) for x in qims];sw=ref.width+sum(x.width for x in qm);s=Image.new('RGB',(sw,H),'white');x=0;s.paste(ref,(x,0));x+=ref.width
    for im in qm:s.paste(im,(x,0));x+=im.width
    s.save(args.out/'AINA_REFERENCE_3Q_YAW_SWEEP_v15.4.png')
if __name__=='__main__':main()
