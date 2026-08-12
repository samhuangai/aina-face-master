#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from PIL import Image
import face_alignment

fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cpu',face_detector='sfd')
inputs={
 'front':Path('references/AINA_APPROVED_FRONT.jpg'),
 'three_quarter':Path('references/AINA_APPROVED_3Q.jpg'),
 'side':Path('references/AINA_APPROVED_SIDE.jpg'),
}
out={}
for name,p in inputs.items():
 im=np.asarray(Image.open(p).convert('RGB'))
 preds=fa.get_landmarks_from_image(im)
 if not preds: raise SystemExit(f'no face {name}')
 ctr=np.array([im.shape[1]*.5,im.shape[0]*.5])
 q=min(preds,key=lambda x:np.linalg.norm(np.asarray(x)[:,:2].mean(0)-ctr))
 pts=np.asarray(q,dtype=float)[:,:2]
 out[name]={'image_size':[int(im.shape[1]),int(im.shape[0])],'landmarks_xy':pts.tolist()}
Path('output_target68').mkdir(exist_ok=True)
Path('output_target68/AINA_ALL_TARGET_68.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
