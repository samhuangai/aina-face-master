#!/usr/bin/env python3
import numpy as np
import cv2
import identity_lock_v104 as m


def fixed_detect(fa, im):
    h, w = im.shape[:2]
    s = max(1.0, 720.0 / max(h, w))
    x = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else im
    ps = fa.get_landmarks_from_image(x)
    if not ps:
        raise RuntimeError('no face')
    ctr = np.array([x.shape[1] / 2, x.shape[0] / 2], dtype=np.float64)
    q = min(ps, key=lambda p: np.linalg.norm(np.asarray(p)[:, :2].mean(0) - ctr))
    return np.asarray(q, dtype=np.float64)[:, :2] / s


m.detect = fixed_detect

if __name__ == '__main__':
    m.main()
