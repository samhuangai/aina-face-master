#!/usr/bin/env python3
"""AINA Identity Master Reconstruction — real topology-preserving head rebuild.

This stage does not generate replacement effect art. It consumes the actual
locked AINA v15.5 OBJ, uses the already-approved front landmark target plus the
approved 3/4 and side reference images, reconstructs a coherent multi-view 3D
semantic cage, and transfers it to the real head through differential-coordinate
mesh deformation. Vertex order, faces, connected components and downstream 52
shape-control compatibility are preserved.

The output remains a visual-lock candidate. A numeric fit is never allowed to set
``visual_identity_lock`` by itself; the actual Blender front/20/45/profile renders
must be visually approved first.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

K68 = np.array([
    1309,710,3509,2178,385,932,467,2360,5078,9356,7497,7951,7415,9179,10498,7729,8320,
    3367,3887,1988,3270,1914,8915,10259,8989,10874,10356,2577,5429,6355,5794,4670,6511,
    5658,13396,11656,4559,6220,4818,4275,5529,4339,11261,11804,13112,11545,11325,12452,
    2322,6640,4842,6262,11828,13519,9323,13361,12656,5715,5744,6476,6079,6817,6550,
    13695,12973,13422,6543,6537,
], dtype=np.int64)

# MediaPipe FaceMesh points arranged in conventional sparse-68 order.
MP68 = np.array([
    234,93,132,58,172,136,150,149,152,378,379,365,397,288,361,323,454,
    70,63,105,66,107,336,296,334,293,300,
    168,6,197,195,48,115,4,344,278,
    33,160,158,133,153,144,362,385,387,263,373,380,
    61,40,37,0,267,270,291,321,314,17,84,91,
    78,81,13,311,308,402,14,178,
], dtype=np.int64)

SYMMETRY_PAIRS = [
    *[(i, 16-i) for i in range(8)],
    (17,26),(18,25),(19,24),(20,23),(21,22),
    (31,35),(32,34),
    (36,45),(37,44),(38,43),(39,42),(40,47),(41,46),
    (48,54),(49,53),(50,52),(55,59),(56,58),
    (60,64),(61,63),(65,67),
]


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--front-target", type=Path, required=True)
    p.add_argument("--front-reference", type=Path)
    p.add_argument("--q3-reference", type=Path)
    p.add_argument("--side-reference", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--height", type=float, default=1.72)
    return p.parse_args(argv)


def read_obj(path: Path):
    lines = path.read_text(errors="ignore").splitlines()
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in lines:
        if line.startswith("v "):
            q = line.split()
            verts.append((float(q[1]), float(q[2]), float(q[3])))
        elif line.startswith("f "):
            ids = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
            for i in range(1, len(ids) - 1):
                faces.append((ids[0], ids[i], ids[i + 1]))
    return lines, np.asarray(verts, np.float64), np.asarray(faces, np.int64)


def write_obj(lines: list[str], verts: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output: list[str] = []
    vi = 0
    for line in lines:
        if line.startswith("v "):
            x, y, z = verts[vi]
            output.append(f"v {x:.9f} {y:.9f} {z:.9f}")
            vi += 1
        else:
            output.append(line)
    if vi != len(verts):
        raise RuntimeError(f"OBJ vertex rewrite mismatch: {vi} != {len(verts)}")
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def components(n: int, faces: np.ndarray):
    parent = np.arange(n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    for a, b, c in faces:
        union(a, b); union(b, c); union(c, a)
    labels = np.asarray([find(i) for i in range(n)], np.int32)
    groups: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(int(label), []).append(i)
    return labels, {k: np.asarray(v, np.int64) for k, v in groups.items()}


def map_face(raw: np.ndarray, height: float):
    scale = 1.08
    world = np.empty_like(raw)
    world[:, 0] = raw[:, 0] * scale
    world[:, 1] = raw[:, 2] * scale
    world[:, 2] = -raw[:, 1] * scale
    offset = height - float(world[:, 2].max())
    world[:, 2] += offset
    return world, scale, offset


def inverse_map(world: np.ndarray, scale: float, offset: float):
    raw = np.empty_like(world)
    raw[:, 0] = world[:, 0] / scale
    raw[:, 2] = world[:, 1] / scale
    raw[:, 1] = -(world[:, 2] - offset) / scale
    return raw


def normalize_image_points(points: np.ndarray, size: tuple[int, int]):
    w, h = size
    s = 0.5 * max(w, h)
    q = (np.asarray(points, np.float64) - np.array([0.5*w, 0.5*h])) / max(s, 1e-9)
    q[:, 1] *= -1.0
    return q


def load_front_target(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    points = np.asarray(data["landmarks_xy"], np.float64)
    size = tuple(int(x) for x in data["image_size"])
    if points.shape != (68, 2):
        raise RuntimeError(f"Expected front 68x2 target, got {points.shape}")
    return normalize_image_points(points, size), {"detected": True, "source": str(path), "size": list(size)}


def detect_reference_68(path: Path | None):
    if not path or not path.exists():
        return None, {"detected": False, "reason": "reference_missing"}
    try:
        import cv2
        import mediapipe as mp
    except Exception as exc:
        return None, {"detected": False, "reason": f"mediapipe_import_failed: {exc}"}
    image = cv2.imread(str(path))
    if image is None:
        return None, {"detected": False, "reason": "image_decode_failed"}
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.25,
        ) as mesh:
            result = mesh.process(rgb)
    except Exception as exc:
        return None, {"detected": False, "reason": f"facemesh_failed: {exc}"}
    if not result.multi_face_landmarks:
        return None, {"detected": False, "reason": "no_face"}
    lm = result.multi_face_landmarks[0].landmark
    points = np.asarray([(lm[i].x*w, lm[i].y*h) for i in MP68], np.float64)
    return normalize_image_points(points, (w, h)), {
        "detected": True,
        "source": str(path),
        "size": [w, h],
        "detector": "MediaPipe FaceMesh",
    }


def feature_weights():
    w = np.ones(68, np.float64)
    w[:17] = 1.25
    w[17:27] = 0.70
    w[27:36] = 2.15
    w[36:48] = 2.65
    w[48:68] = 2.35
    return w


def fit_weak_camera(points3: np.ndarray, target2: np.ndarray, weights: np.ndarray):
    x = np.c_[points3, np.ones(len(points3))]
    sw = np.sqrt(weights)[:, None]
    beta = np.linalg.lstsq(x*sw, target2*sw, rcond=None)[0]
    a = beta[:3].T
    b = beta[3]
    n1 = np.linalg.norm(a[0]); n2 = np.linalg.norm(a[1])
    scale = max(1e-8, 0.5*(n1+n2))
    r1 = a[0] / max(n1, 1e-9)
    v2 = a[1] - np.dot(a[1], r1)*r1
    r2 = v2 / max(np.linalg.norm(v2), 1e-9)
    r3 = np.cross(r1, r2); r3 /= max(np.linalg.norm(r3), 1e-9)
    r2 = np.cross(r3, r1)
    rotation = np.stack([r1, r2, r3])
    if np.linalg.det(rotation) < 0:
        rotation[2] *= -1
    return rotation, scale, b


def camera_rmse(points3: np.ndarray, target2: np.ndarray, weights: np.ndarray):
    rotation, scale, translate = fit_weak_camera(points3, target2, weights)
    predicted = scale*(points3 @ rotation.T)[:, :2] + translate
    error = np.linalg.norm(predicted-target2, axis=1)
    rmse = float(np.sqrt(np.sum(weights*error*error)/np.sum(weights)))
    return rmse, rotation, scale, translate, predicted, error


def reconstruct_anchor_cage(base: np.ndarray, views: list[dict]):
    anchors = base.copy()
    fw = feature_weights()
    history: list[dict] = []
    for iteration in range(10):
        cameras = []
        for view in views:
            rmse, rotation, scale, translate, predicted, error = camera_rmse(
                anchors, view["target"], fw
            )
            robust = 1.0 / np.maximum(1.0, error / 0.045)
            cameras.append((view, rotation, scale, translate, robust, rmse))
        deltas = np.zeros_like(anchors)
        for i in range(68):
            h = np.eye(3)*160.0
            g = np.zeros(3)
            for view, rotation, scale, translate, robust, _ in cameras:
                r2 = rotation[:2]
                predicted = scale*(r2 @ anchors[i]) + translate
                residual = view["target"][i] - predicted
                weight = float(view["weight"]*fw[i]*robust[i])
                h += weight*(scale*scale)*(r2.T @ r2)
                g += weight*scale*(r2.T @ residual)
            delta = np.linalg.solve(h, g)
            length = float(np.linalg.norm(delta))
            if length > 0.0045:
                delta *= 0.0045/length
            deltas[i] = delta
        anchors += 0.82*deltas
        history.append({
            "iteration": iteration,
            "max_anchor_step_m": float(np.linalg.norm(deltas, axis=1).max()),
            "view_rmse": {v[0]["name"]: float(v[5]) for v in cameras},
        })
        if np.linalg.norm(deltas, axis=1).max() < 1.2e-4:
            break

    # A digital-human identity is symmetric in object space. Pose and lighting
    # asymmetry belong to cameras/materials, not to the neutral master mesh.
    center_x = float(np.mean(anchors[[27,28,29,30,33,51,57,62,66], 0]))
    for left, right in SYMMETRY_PAIRS:
        d = 0.5*(abs(anchors[left,0]-center_x)+abs(anchors[right,0]-center_x))
        yz = 0.5*(anchors[left,1:]+anchors[right,1:])
        anchors[left,0] = center_x-d
        anchors[right,0] = center_x+d
        anchors[left,1:] = yz
        anchors[right,1:] = yz
    for idx in (27,28,29,30,33,51,57,62,66):
        anchors[idx,0] = center_x

    # Art-direction depth constraints: small visible nose, integrated lips and a
    # compact rounded chin. Front camera is on -Y, therefore more negative Y is
    # farther forward.
    eye_plane_y = float(np.mean(anchors[36:48,1]))
    nose_tip = anchors[30].copy()
    anchors[27:31,1] = np.minimum(anchors[27:31,1], eye_plane_y-np.linspace(0.0045,0.0115,4))
    anchors[30,1] = min(float(anchors[30,1]), eye_plane_y-0.0130)
    anchors[31:36,1] = np.minimum(anchors[31:36,1], eye_plane_y-0.0085)
    mouth_y = float(np.mean(anchors[48:60,1]))
    desired_mouth_y = max(mouth_y, float(anchors[30,1])+0.0050)
    anchors[48:68,1] += desired_mouth_y-mouth_y
    anchors[8,1] = min(float(anchors[8,1]), desired_mouth_y+0.0025)
    return anchors, history


def build_uniform_laplacian(n: int, faces: np.ndarray):
    edges = np.vstack([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]])
    edges = np.vstack([edges, edges[:,::-1]])
    data = np.ones(len(edges), np.float64)
    adjacency = sparse.coo_matrix((data, (edges[:,0], edges[:,1])), shape=(n,n)).tocsr()
    adjacency.data[:] = 1.0
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv = 1.0/np.maximum(degree,1.0)
    return sparse.eye(n, format="csr") - sparse.diags(inv) @ adjacency


def laplacian_deform(base: np.ndarray, faces: np.ndarray, anchor_local: np.ndarray, targets: np.ndarray):
    n = len(base)
    lap = build_uniform_laplacian(n, faces)
    smooth = (lap.T @ lap).tocsr()
    constraint = np.full(n, 1e-5, np.float64)
    target = base.copy()
    fw = feature_weights()
    for local, point, weight in zip(anchor_local, targets, fw):
        constraint[int(local)] += 3500.0*float(weight)
        target[int(local)] = point

    # Keep the rear cranium and low neck stable while allowing the visible face
    # and lower contour to rebuild around the approved semantic cage.
    rear = base[:,1] > np.percentile(base[:,1], 70)
    low = base[:,2] < np.percentile(base[:,2], 7)
    top_rear = (base[:,2] > np.percentile(base[:,2], 88)) & rear
    constraint[rear] += 4.0
    constraint[top_rear] += 9.0
    constraint[low] += 120.0

    system = smooth + sparse.diags(constraint)
    rhs_base = smooth @ base
    out = np.empty_like(base)
    for axis in range(3):
        rhs = rhs_base[:,axis] + constraint*target[:,axis]
        out[:,axis] = spsolve(system.tocsc(), rhs)
    displacement = out-base
    length = np.linalg.norm(displacement, axis=1)
    mask = length > 0.026
    if np.any(mask):
        displacement[mask] *= (0.026/length[mask])[:,None]
        out = base+displacement
    return out


def radial_weights(points: np.ndarray, center, radii, inner=0.0, outer=1.0):
    center = np.asarray(center, np.float64); radii = np.asarray(radii, np.float64)
    q = np.sqrt(np.sum(((points-center)/radii)**2, axis=1))
    w = np.zeros(len(points), np.float64); w[q<=inner] = 1.0
    m = (q>inner)&(q<outer)
    if np.any(m):
        t = (q[m]-inner)/max(outer-inner,1e-12)
        w[m] = 0.5*(1.0+np.cos(np.pi*t))
    return w


def region_shift(points: np.ndarray, center, radii, delta, inner=0.0, outer=1.0):
    points += radial_weights(points, center, radii, inner, outer)[:,None]*np.asarray(delta,np.float64)


def region_scale(points: np.ndarray, center, radii, factors, inner=0.0, outer=1.0):
    c = np.asarray(center,np.float64)
    w = radial_weights(points,c,radii,inner,outer)[:,None]
    desired = c+(points-c)*np.asarray(factors,np.float64)
    points += w*(desired-points)


def art_directed_surface_polish(head: np.ndarray, local_k: np.ndarray):
    out = head.copy(); lm = out[local_k].copy()
    cx = float(np.mean(lm[[27,28,29,30,33,51,57],0]))

    # Narrow temples and taper the lower third without creating a knife-point jaw.
    eye_z = float(np.mean(lm[36:48,2])); mouth_z = float(np.mean(lm[48:60,2])); chin_z = float(lm[8,2])
    front = np.exp(-0.5*((out[:,1]+0.005)/0.078)**4)
    upper = np.clip((out[:,2]-eye_z)/0.105,0,1)
    out[:,0] = cx+(out[:,0]-cx)*(1.0-0.040*upper*front)
    lower = np.clip((mouth_z+0.020-out[:,2])/max(mouth_z+0.020-(chin_z-0.012),1e-6),0,1)
    out[:,0] = cx+(out[:,0]-cx)*(1.0-0.095*(lower**1.25)*front)

    lm = out[local_k].copy()
    # True eyelid/orbit continuity around both eyes.
    for ids, outer_idx, inner_idx in [
        (np.arange(36,42),36,39), (np.arange(42,48),45,42)
    ]:
        c = lm[ids].mean(0)
        region_scale(out,c,(0.045,0.036,0.028),(1.045,1.0,1.10),0.05,1.10)
        region_shift(out,(c[0],c[1]+0.008,c[2]-0.013),(0.042,0.038,0.023),(0,-0.0010,0.00045),0,1.10)
        lm = out[local_k].copy(); outer = lm[outer_idx]; side = -1.0 if outer[0]<cx else 1.0
        region_shift(out,outer,(0.018,0.018,0.014),(side*0.0007,-0.00025,0.0008),0,1.02)
        region_shift(out,lm[inner_idx],(0.016,0.017,0.013),(-side*0.0001,-0.00015,0.0001),0,1.02)

    # Small but clearly modelled nose.
    lm = out[local_k].copy(); bridge=lm[27:31].mean(0); tip=lm[30]; base=lm[31:36].mean(0)
    region_scale(out,bridge,(0.026,0.034,0.047),(0.84,1.0,0.94),0.04,1.13)
    region_shift(out,bridge,(0.027,0.035,0.046),(0,-0.0010,0.0005),0.02,1.10)
    region_scale(out,base,(0.030,0.029,0.030),(0.82,1.0,0.94),0.04,1.12)
    region_shift(out,tip,(0.021,0.024,0.023),(0,-0.0018,0.0010),0.02,1.06)

    # Integrated lips with retained central volume and no floating oval perimeter.
    lm = out[local_k].copy(); mouth=lm[48:60].mean(0)
    region_scale(out,mouth,(0.050,0.038,0.030),(0.99,0.90,0.84),0.08,1.14)
    region_shift(out,mouth,(0.052,0.040,0.032),(0,0.0016,0.0008),0.04,1.12)
    lm = out[local_k].copy()
    region_shift(out,lm[[49,50,51,52,53]].mean(0),(0.034,0.026,0.014),(0,-0.0006,0.00025),0,1.04)
    region_shift(out,lm[[55,56,57,58,59]].mean(0),(0.035,0.027,0.015),(0,-0.0008,-0.0001),0,1.04)

    # Youthful centered apple cheeks and a small rounded chin.
    lm = out[local_k].copy()
    for c in ((lm[40]+lm[31]+lm[48])/3.0,(lm[46]+lm[35]+lm[54])/3.0):
        region_shift(out,c,(0.046,0.043,0.041),(0,-0.0015,0.0007),0.02,1.11)
        region_scale(out,c,(0.047,0.044,0.042),(0.98,1.0,1.015),0,1.09)
    lm = out[local_k].copy(); chin=lm[8]
    region_scale(out,chin,(0.047,0.047,0.043),(0.84,0.96,0.92),0.02,1.10)
    region_shift(out,chin,(0.048,0.047,0.043),(0,0.0010,0.0020),0,1.06)
    return out


def main() -> None:
    a = parse_args()
    lines, raw, faces = read_obj(a.mesh)
    if len(raw) <= int(K68.max()):
        raise RuntimeError("Source mesh does not contain the required semantic vertex order")
    labels, groups = components(len(raw), faces)
    head_root = max(groups, key=lambda root: len(groups[root]))
    head_ids = groups[head_root]
    head_mask = np.zeros(len(raw), bool); head_mask[head_ids] = True
    if not np.all(head_mask[K68]):
        raise RuntimeError("Sparse identity anchors are no longer on the primary head component")
    global_to_local = np.full(len(raw), -1, np.int64); global_to_local[head_ids] = np.arange(len(head_ids))
    local_faces_global = faces[head_mask[faces].all(axis=1)]
    local_faces = global_to_local[local_faces_global]
    local_k = global_to_local[K68]

    world, map_scale, map_offset = map_face(raw, a.height)
    head_base = world[head_ids].copy()
    anchor_base = world[K68].copy()

    front_target, front_meta = load_front_target(a.front_target)
    q3_target, q3_meta = detect_reference_68(a.q3_reference)
    side_target, side_meta = detect_reference_68(a.side_reference)
    views = [{"name":"front","target":front_target,"weight":1.0}]
    if q3_target is not None:
        views.append({"name":"q3","target":q3_target,"weight":0.85})
    if side_target is not None:
        views.append({"name":"side","target":side_target,"weight":0.55})

    anchor_target, history = reconstruct_anchor_cage(anchor_base, views)
    head_deformed = laplacian_deform(head_base, local_faces, local_k, anchor_target)
    head_deformed = art_directed_surface_polish(head_deformed, local_k)

    result_world = world.copy(); result_world[head_ids] = head_deformed
    # Keep separated eyeballs and oral components coherent with their semantic
    # regions while preserving every component's own topology.
    eye_groups = sorted(
        [ids for root,ids in groups.items() if root != head_root and 650 < len(ids) < 900],
        key=lambda ids: float(world[ids,0].mean()),
    )
    old_eye = [anchor_base[36:42].mean(0), anchor_base[42:48].mean(0)]
    new_eye = [result_world[K68[36:42]].mean(0), result_world[K68[42:48]].mean(0)]
    for ids, old, new in zip(eye_groups, old_eye, new_eye):
        result_world[ids] += new-old
    mouth_shift = result_world[K68[48:60]].mean(0)-anchor_base[48:60].mean(0)
    for root, ids in groups.items():
        if root == head_root or any(np.array_equal(ids,e) for e in eye_groups):
            continue
        result_world[ids] += mouth_shift

    result_raw = inverse_map(result_world, map_scale, map_offset)
    write_obj(lines, result_raw, a.out)

    tri0 = head_base[local_faces]; tri1 = head_deformed[local_faces]
    area0 = 0.5*np.linalg.norm(np.cross(tri0[:,1]-tri0[:,0],tri0[:,2]-tri0[:,0]),axis=1)
    area1 = 0.5*np.linalg.norm(np.cross(tri1[:,1]-tri1[:,0],tri1[:,2]-tri1[:,0]),axis=1)
    ratio = area1/np.maximum(area0,1e-12)
    displacement = head_deformed-head_base
    view_reports = {}
    for view in views:
        rmse, rotation, scale, translate, predicted, error = camera_rmse(
            result_world[K68], view["target"], feature_weights()
        )
        view_reports[view["name"]] = {
            "normalized_rmse": rmse,
            "max_landmark_error": float(error.max()),
            "camera_rotation_rows": rotation.tolist(),
            "camera_scale": float(scale),
            "camera_translation": translate.tolist(),
        }

    report = {
        "product": "AINA Identity Master Reconstruction Candidate",
        "source": str(a.mesh),
        "output": str(a.out),
        "real_mesh": True,
        "replacement_effect_art_generated": False,
        "topology_changed": False,
        "vertices": int(len(raw)),
        "faces": int(len(faces)),
        "head_vertices": int(len(head_ids)),
        "head_triangles": int(len(local_faces)),
        "semantic_anchor_count": 68,
        "views_used": [v["name"] for v in views],
        "reference_detection": {"front":front_meta,"q3":q3_meta,"side":side_meta},
        "view_fit": view_reports,
        "iteration_history": history,
        "max_head_displacement_m": float(np.linalg.norm(displacement,axis=1).max()),
        "rms_head_displacement_m": float(np.sqrt(np.mean(np.sum(displacement*displacement,axis=1)))),
        "triangle_area_ratio_p01": float(np.percentile(ratio,1)),
        "triangle_area_ratio_p99": float(np.percentile(ratio,99)),
        "degenerate_head_triangles": int(np.sum(area1 < 1e-11)),
        "visual_identity_lock": False,
        "next_gate": "Actual Blender naked-head beauty and clay front/20/45/profile visual approval",
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
