#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def detect(path: Path) -> np.ndarray:
    image=cv2.imread(str(path))
    if image is None: raise ValueError(f'Unable to read {path}')
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=.16) as detector:
        result=detector.process(rgb)
    if not result.multi_face_landmarks: raise RuntimeError(f'No face detected in {path}')
    return np.asarray([(p.x,p.y,p.z) for p in result.multi_face_landmarks[0].landmark[:468]],dtype=np.float64)


def similarity(source: np.ndarray,target: np.ndarray):
    source_mean=source.mean(axis=0);target_mean=target.mean(axis=0)
    x=source-source_mean;y=target-target_mean
    u,s,vt=np.linalg.svd(x.T@y)
    rotation=u@vt
    if np.linalg.det(rotation)<0:
        u[:,-1]*=-1;rotation=u@vt
    scale=float(s.sum()/max((x*x).sum(),1e-12))
    translation=target_mean-source_mean@rotation*scale
    return rotation,scale,translation


def align(reference: np.ndarray,model: np.ndarray):
    anchors=np.asarray([10,152,234,454,33,133,362,263,168,1,61,291,70,300,105,334],dtype=np.int64)
    choices=[]
    for mirrored in (False,True):
        candidate=reference[:,:2].copy()
        if mirrored:candidate[:,0]=1.0-candidate[:,0]
        rotation,scale,translation=similarity(candidate[anchors],model[anchors,:2])
        aligned=candidate@rotation*scale+translation
        rms=float(np.sqrt(np.mean(np.sum((aligned[anchors]-model[anchors,:2])**2,axis=1))))
        choices.append((rms,mirrored,aligned,rotation,scale,translation))
    return min(choices,key=lambda item:item[0])


def main():
    if len(sys.argv)!=4:raise SystemExit('extract_aina_full_landmarks.py REFERENCE MODEL OUTPUT_JSON')
    reference_path=Path(sys.argv[1]);model_path=Path(sys.argv[2]);output=Path(sys.argv[3])
    reference=detect(reference_path);model=detect(model_path)
    anchor_rms,mirrored,target,rotation,scale,translation=align(reference,model)
    displacement=target-model[:,:2]
    norms=np.linalg.norm(displacement,axis=1)
    report={
        'reference_image':str(reference_path),
        'model_image':str(model_path),
        'reference_mirrored':mirrored,
        'similarity_scale':scale,
        'similarity_rotation':rotation.tolist(),
        'similarity_translation':translation.tolist(),
        'anchor_rms':anchor_rms,
        'full_rms':float(np.sqrt(np.mean(np.sum(displacement**2,axis=1)))),
        'median_displacement':float(np.median(norms)),
        'p95_displacement':float(np.quantile(norms,.95)),
        'max_displacement':float(norms.max()),
        'model_points':model[:,:2].tolist(),
        'target_points':target.tolist(),
        'displacement':displacement.tolist(),
    }
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in {'model_points','target_points','displacement'}},indent=2))


if __name__=='__main__':main()
