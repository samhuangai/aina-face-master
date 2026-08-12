#!/usr/bin/env python3
from pathlib import Path
import sys, json
import numpy as np
import torch
import trimesh

VENDOR = Path('vendor/FaceVerse_v4')
sys.path.insert(0, str(VENDOR.resolve()))
from faceversev4 import FaceVerseRecon


def load_coeff(path: Path):
    d = np.load(path, allow_pickle=True).item()
    c = np.asarray(d['coeffs'], dtype=np.float32)
    if c.ndim == 1:
        c = c[None]
    return c


def export_mesh(vertices, faces, out_base: Path):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(out_base.with_suffix('.obj'))
    mesh.export(out_base.with_suffix('.ply'))
    mesh.export(out_base.with_suffix('.glb'))
    return mesh


def main():
    out = Path('output_faceverse_v4')
    out.mkdir(exist_ok=True)
    front = load_coeff(Path('fv_front/AINA_APPROVED_FRONT.npy'))
    q3 = load_coeff(Path('fv_3q/AINA_APPROVED_3Q.npy'))

    # FaceVerse V4: id 156, exp 177, tex 251, light 27, rot 3, trans 3, eyes 4.
    id_front = front[:, :156]
    id_3q = q3[:, :156]
    id_blend = 0.72 * id_front + 0.28 * id_3q

    device = torch.device('cpu')
    fvr = FaceVerseRecon(
        'vendor/FaceVerse_v4/data/faceverse_v4_2.npy',
        'vendor/FaceVerse_v4/data/faceverse_resnet50.pth',
        device,
        load_recon=False,
    )

    zeros_exp = torch.zeros((1, fvr.exp_dims), dtype=torch.float32)
    zeros_eye = torch.zeros((1, 4), dtype=torch.float32)
    faces = fvr.fvd['tri']

    results = {}
    for name, id_np in [('front', id_front), ('three_quarter', id_3q), ('blend', id_blend)]:
        id_t = torch.from_numpy(id_np.astype(np.float32))
        with torch.no_grad():
            vertices = fvr.get_vs(id_t, zeros_exp, zeros_eye)[0].cpu().numpy()
        mesh = export_mesh(vertices, faces, out / f'AINA_FACEVERSE_V4_NEUTRAL_{name}')
        results[name] = {
            'vertices': int(len(mesh.vertices)),
            'triangles': int(len(mesh.faces)),
            'bounds': mesh.bounds.tolist(),
            'id_rms': float(np.sqrt(np.mean(id_np ** 2))),
            'id_max_abs': float(np.max(np.abs(id_np))),
        }

    np.save(out / 'AINA_FACEVERSE_V4_ID_FRONT.npy', id_front)
    np.save(out / 'AINA_FACEVERSE_V4_ID_3Q.npy', id_3q)
    np.save(out / 'AINA_FACEVERSE_V4_ID_BLEND.npy', id_blend)
    report = {
        'model': 'FaceVerse V4',
        'identity_dims': 156,
        'expression': 'neutral, all 177 expression coefficients zero',
        'eye_rotation': 'neutral, zero',
        'blend': '72% approved front + 28% approved three-quarter identity coefficients',
        'results': results,
        'identity_lock': False,
        'note': 'This is a new single-image-predicted full-head identity candidate; compare clay against approved AINA art before locking.'
    }
    (out / 'AINA_FACEVERSE_V4_REPORT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
