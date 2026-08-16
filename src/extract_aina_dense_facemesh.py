#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np,mediapipe as mp


def extract(path:Path,refine=True):
    im=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if im is None:raise RuntimeError(f'Cannot read {path}')
    h,w=im.shape[:2]
    fm=mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=refine,min_detection_confidence=.25)
    r=fm.process(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));fm.close()
    if not r.multi_face_landmarks:raise RuntimeError(f'MediaPipe detected no face in {path}')
    lm=r.multi_face_landmarks[0].landmark
    pts=np.array([[q.x*w,q.y*h,q.z*w] for q in lm],float)
    return im,pts

def overlay(im,pts,out):
    d=im.copy()
    # Main 468 surface; refined iris points 468+ are marked slightly larger.
    for i,(x,y,z) in enumerate(pts):
        rr=2 if i>=468 else 1
        cv2.circle(d,(int(round(x)),int(round(y))),rr,(0,0,255) if i>=468 else (0,255,0),-1,cv2.LINE_AA)
    cv2.imwrite(str(out),d)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--front',type=Path,required=True);ap.add_argument('--q3',type=Path,required=True);ap.add_argument('--side',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    report={'version':'AINA Dense FaceMesh Target v1','views':{}}
    for name,path in [('front',a.front),('q3',a.q3)]:
        im,pts=extract(path,True);overlay(im,pts,a.out/f'AINA_{name.upper()}_FACEMESH_OVERLAY.png')
        data={'image_size':[int(im.shape[1]),int(im.shape[0])],'landmark_count':int(len(pts)),'landmarks_xyz_px':pts.tolist()}
        (a.out/f'AINA_{name.upper()}_FACEMESH_478.json').write_text(json.dumps(data,indent=2));report['views'][name]={'count':len(pts),'x_span_px':float(np.ptp(pts[:468,0])),'y_span_px':float(np.ptp(pts[:468,1])),'z_span_px':float(np.ptp(pts[:468,2]))}
    if a.side and a.side.exists():
        try:
            im,pts=extract(a.side,True);overlay(im,pts,a.out/'AINA_SIDE_FACEMESH_OVERLAY.png');(a.out/'AINA_SIDE_FACEMESH_478.json').write_text(json.dumps({'image_size':[int(im.shape[1]),int(im.shape[0])],'landmark_count':int(len(pts)),'landmarks_xyz_px':pts.tolist()},indent=2));report['views']['side']={'count':len(pts),'x_span_px':float(np.ptp(pts[:468,0])),'y_span_px':float(np.ptp(pts[:468,1])),'z_span_px':float(np.ptp(pts[:468,2]))}
        except Exception as e:report['views']['side']={'error':str(e)}
    (a.out/'AINA_DENSE_FACEMESH_REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
