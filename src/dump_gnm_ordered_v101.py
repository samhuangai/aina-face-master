#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
from gnm.shape import gnm_numpy, gnm_landmarks

identity = np.load('output_v101/AINA_identity_coefficients_v10.1.npy').astype(np.float64).reshape(1,-1)
g = gnm_numpy.GNM.from_local(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
vertices = np.asarray(g(identity=identity))[0].astype(np.float64)
triangles = np.asarray(g.triangles, dtype=np.int64)
skin_tri_idx = np.asarray(g.triangle_indices_for_group('skin'), dtype=np.int64)
skin_faces_global = triangles[skin_tri_idx]
skin_ids = np.unique(skin_faces_global.reshape(-1))
cfg = gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
idx = np.asarray(cfg.indices, dtype=np.int64)
bw = np.asarray(cfg.weights, dtype=np.float64)
lm = (vertices[idx] * bw[...,None]).sum(axis=-2)

out=Path('output_ordered');out.mkdir(exist_ok=True)
np.save(out/'GNM_v10.1_ORDERED_VERTICES.npy', vertices.astype(np.float32))
np.save(out/'GNM_v10.1_TRIANGLES.npy', triangles.astype(np.int32))
np.save(out/'GNM_v10.1_SKIN_IDS.npy', skin_ids.astype(np.int32))
np.save(out/'GNM_v10.1_LANDMARK_INDICES.npy', idx.astype(np.int32))
np.save(out/'GNM_v10.1_LANDMARK_WEIGHTS.npy', bw.astype(np.float32))
np.save(out/'GNM_v10.1_LANDMARKS_68.npy', lm.astype(np.float32))
(out/'GNM_v10.1_META.json').write_text(json.dumps({
 'vertices':int(len(vertices)), 'triangles':int(len(triangles)),
 'skin_ids':int(len(skin_ids)), 'landmarks':int(len(lm))
},indent=2))
print('ordered vertices',vertices.shape,'skin',skin_ids.shape,'lm',lm.shape)
