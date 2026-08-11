#!/usr/bin/env python3
"""Compatibility/fix wrapper for AINA v10.3 semantic sculpt."""
import numpy as np
import semantic_sculpt_v103 as m


def fixed_apply_vertical_midface_warp(p, curr, target, face_center_x, face_rx):
    eye_y = .5 * (curr[36:42, 1].mean() + curr[42:48, 1].mean())
    nose_y = curr[33, 1]
    mouth_y = curr[48:60, 1].mean()
    chin_y = curr[8, 1]
    t_eye_y = .5 * (target[36:42, 1].mean() + target[42:48, 1].mean())
    t_nose_y = target[33, 1]
    t_mouth_y = target[48:60, 1].mean()
    t_chin_y = target[8, 1]
    ys = np.array([eye_y, nose_y, mouth_y, chin_y], dtype=np.float64)
    yt = np.array([t_eye_y, t_nose_y, t_mouth_y, t_chin_y], dtype=np.float64)
    order = np.argsort(ys)
    ys, yt = ys[order], yt[order]
    desired_y = np.interp(p[:, 1], ys, yt)
    inside = (p[:, 1] >= ys[0]) & (p[:, 1] <= ys[-1])
    desired_y[~inside] = p[~inside, 1]
    xfade = np.exp(-0.5 * ((p[:, 0] - face_center_x) / max(face_rx, 1e-6)) ** 4)
    w = inside.astype(np.float64) * xfade * .48
    p[:, 1] += (desired_y - p[:, 1]) * w
    return {
        "eye_y_scale_target": float((t_nose_y - t_eye_y) / max(abs(nose_y - eye_y), 1e-8)),
        "lower_face_y_scale_target": float((t_chin_y - t_nose_y) / max(abs(chin_y - nose_y), 1e-8)),
    }


m.apply_vertical_midface_warp = fixed_apply_vertical_midface_warp

if __name__ == "__main__":
    m.main()
