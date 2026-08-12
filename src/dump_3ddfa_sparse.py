#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
import cv2, yaml, numpy as np

workspace = Path.cwd().resolve()
repo = (workspace / 'vendor/3DDFA_V2').resolve()
sys.path.insert(0, str(repo))
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA

img_path = (workspace / 'references/AINA_APPROVED_FRONT.jpg').resolve()
img = cv2.imread(str(img_path))
if img is None:
    raise SystemExit('reference image missing')

# 3DDFA configs and weights use paths relative to the project root.
os.chdir(repo)
cfg = yaml.load(Path('configs/mb1_120x120.yml').read_text(), Loader=yaml.SafeLoader)
cfg['gpu_mode'] = False
face_boxes = FaceBoxes()
tddfa = TDDFA(**cfg)
boxes = face_boxes(img)
if not boxes:
    h, w = img.shape[:2]
    boxes = [[0.08*w, 0.05*h, 0.92*w, 0.96*h, 1.0]]
param_lst, roi_box_lst = tddfa(img, boxes)
pts = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0].T
if pts.shape != (68,3):
    raise SystemExit(f'unexpected sparse shape {pts.shape}')

out = {
    'image_size': [int(img.shape[1]), int(img.shape[0])],
    'face_boxes': np.asarray(boxes).tolist(),
    'roi_boxes': np.asarray(roi_box_lst).tolist(),
    'landmarks_xyz': pts.tolist(),
}
out_dir = workspace / 'output_sparse'
out_dir.mkdir(exist_ok=True)
(out_dir/'AINA_3DDFA_SPARSE_68.json').write_text(json.dumps(out, indent=2))
np.savetxt(out_dir/'AINA_3DDFA_SPARSE_68.txt', pts, fmt='%.8f')
print(json.dumps(out, indent=2))
