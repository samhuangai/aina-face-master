#!/usr/bin/env python3
"""Render v14 AINA clay QA against approved front/3Q/side art."""
import argparse
from pathlib import Path
from PIL import Image
from rebuild_aina_v133_landmark_laplacian import render_vtk,compare

def main():
 a=argparse.ArgumentParser();a.add_argument('--clay',type=Path,required=True);a.add_argument('--front',type=Path,required=True);a.add_argument('--q3',type=Path,required=True);a.add_argument('--side',type=Path,required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args();x.out.mkdir(parents=True,exist_ok=True);views=[]
 for yaw,label in [(-90,'left_profile'),(-45,'left_45'),(0,'front'),(45,'right_45'),(90,'right_profile')]:
  p=x.out/f'AINA_VTK_{label}_v14.0.png';render_vtk(x.clay,p,yaw);views.append(p)
 ims=[Image.open(p).convert('RGB') for p in views];W,H=ims[0].size;sheet=Image.new('RGB',(5*W,H),'white')
 for i,im in enumerate(ims):sheet.paste(im,(i*W,0))
 sheet.save(x.out/'AINA_VTK_5VIEW_v14.0.png');compare(x.front,x.out/'AINA_VTK_front_v14.0.png',x.out/'AINA_REFERENCE_VS_VTK_FRONT_v14.0.png');compare(x.q3,x.out/'AINA_VTK_left_45_v14.0.png',x.out/'AINA_REFERENCE_3Q_VS_VTK_45_v14.0.png');compare(x.side,x.out/'AINA_VTK_left_profile_v14.0.png',x.out/'AINA_REFERENCE_SIDE_VS_VTK_PROFILE_v14.0.png')
if __name__=='__main__':main()
