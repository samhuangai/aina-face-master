#!/usr/bin/env python3
"""Render the actual AINA v15.2 mesh for front/45/profile identity QA."""
import argparse
from pathlib import Path
from PIL import Image
from rebuild_aina_v133_landmark_laplacian import render_vtk, compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clay', type=Path, required=True)
    ap.add_argument('--front', type=Path, required=True)
    ap.add_argument('--q3', type=Path, required=True)
    ap.add_argument('--side', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    views = []
    for yaw, label in [
        (-90, 'left_profile'),
        (-45, 'left_45'),
        (0, 'front'),
        (45, 'right_45'),
        (90, 'right_profile'),
    ]:
        p = args.out / f'AINA_VTK_{label}_v15.2.png'
        render_vtk(args.clay, p, yaw)
        views.append(p)

    images = [Image.open(p).convert('RGB') for p in views]
    w, h = images[0].size
    sheet = Image.new('RGB', (5 * w, h), 'white')
    for i, im in enumerate(images):
        sheet.paste(im, (i * w, 0))
    sheet.save(args.out / 'AINA_VTK_5VIEW_v15.2.png')

    compare(args.front, args.out / 'AINA_VTK_front_v15.2.png', args.out / 'AINA_REFERENCE_VS_VTK_FRONT_v15.2.png')
    compare(args.q3, args.out / 'AINA_VTK_left_45_v15.2.png', args.out / 'AINA_REFERENCE_3Q_VS_VTK_45_v15.2.png')
    compare(args.side, args.out / 'AINA_VTK_left_profile_v15.2.png', args.out / 'AINA_REFERENCE_SIDE_VS_VTK_PROFILE_v15.2.png')


if __name__ == '__main__':
    main()
