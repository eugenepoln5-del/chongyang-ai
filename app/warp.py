"""Smooth lower-face warp. Pixels outside the feathered mask stay exact."""

from __future__ import annotations

import cv2
import numpy as np


def warp_frame(frame: np.ndarray, src: np.ndarray, dst: np.ndarray, mask_pts: np.ndarray) -> np.ndarray:
    """Return a new image. `frame` is not modified."""
    out = frame.copy()
    if src.size == 0 or dst.size == 0 or mask_pts.size == 0:
        return out
    height, width = frame.shape[:2]
    src = src.astype(np.float32).copy()
    dst = dst.astype(np.float32).copy()
    src[:, 0] = np.clip(src[:, 0], 0, width - 1)
    src[:, 1] = np.clip(src[:, 1], 0, height - 1)
    dst[:, 0] = np.clip(dst[:, 0], 0, width - 1)
    dst[:, 1] = np.clip(dst[:, 1], 0, height - 1)
    if float(np.max(np.abs(src - dst))) < 0.35:
        return out

    both = np.vstack([src, dst, mask_pts.astype(np.float32)])
    pad = 28
    x0 = max(int(np.floor(both[:, 0].min())) - pad, 0)
    y0 = max(int(np.floor(both[:, 1].min())) - pad, 0)
    x1 = min(int(np.ceil(both[:, 0].max())) + pad, width - 1)
    y1 = min(int(np.ceil(both[:, 1].max())) + pad, height - 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return out

    roi_src = src.copy()
    roi_dst = dst.copy()
    roi_src[:, 0] -= x0
    roi_src[:, 1] -= y0
    roi_dst[:, 0] -= x0
    roi_dst[:, 1] -= y0
    rw = x1 - x0 + 1
    rh = y1 - y0 + 1
    anchors_s, anchors_d = _border_anchors(rw, rh, roi_dst)
    ctrl_src = np.vstack([roi_src, anchors_s])
    ctrl_dst = np.vstack([roi_dst, anchors_d])
    ctrl_src, ctrl_dst = _dedupe(ctrl_src, ctrl_dst, min_dist=5.0)

    warped = _idw_remap(frame[y0 : y1 + 1, x0 : x1 + 1], ctrl_src, ctrl_dst)
    alpha = _feather_mask((rh, rw), mask_pts.astype(np.float32) - np.array([x0, y0], dtype=np.float32))
    if alpha is None:
        return out
    base = out[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    blended = warped.astype(np.float32) * alpha + base * (1.0 - alpha)
    out[y0 : y1 + 1, x0 : x1 + 1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _border_anchors(width: int, height: int, existing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    step = 34
    pts = []
    for x in range(2, width - 2, step):
        pts.append((x, 2))
        pts.append((x, height - 3))
    for y in range(2, height - 2, step):
        pts.append((2, y))
        pts.append((width - 3, y))
    if not pts:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    arr = np.array(pts, dtype=np.float32)
    if existing.size:
        keep = []
        for p in arr:
            if np.min(np.linalg.norm(existing - p, axis=1)) > 10.0:
                keep.append(p)
        arr = np.stack(keep).astype(np.float32) if keep else np.zeros((0, 2), np.float32)
    return arr, arr.copy()


def _dedupe(src: np.ndarray, dst: np.ndarray, min_dist: float) -> tuple[np.ndarray, np.ndarray]:
    keep_s = []
    keep_d = []
    for s, d in zip(src, dst):
        if keep_d and np.min(np.linalg.norm(np.stack(keep_d) - d, axis=1)) < min_dist:
            continue
        keep_s.append(s)
        keep_d.append(d)
    return np.stack(keep_s).astype(np.float32), np.stack(keep_d).astype(np.float32)


def _idw_remap(roi: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    rh, rw = roi.shape[:2]
    # A coarse field is resized smoothly, which keeps a 720p mouth interactive.
    step = 6
    xs = np.arange(0, rw, step, dtype=np.float32)
    ys = np.arange(0, rh, step, dtype=np.float32)
    if xs[-1] != rw - 1:
        xs = np.append(xs, rw - 1)
    if ys[-1] != rh - 1:
        ys = np.append(ys, rh - 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    samples = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    disp_c = src - dst
    delta = samples[:, None, :] - dst[None, :, :]
    dist2 = np.sum(delta * delta, axis=2)
    weights = 1.0 / (dist2 * dist2 + 1e-3)
    weights /= np.sum(weights, axis=1, keepdims=True)
    disp = (weights @ disp_c).astype(np.float32)
    disp_img = disp.reshape(grid_y.shape[0], grid_x.shape[1], 2)
    disp_full = cv2.resize(disp_img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
    map_x = xx + disp_full[:, :, 0]
    map_y = yy + disp_full[:, :, 1]
    return cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def _feather_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray | None:
    rh, rw = shape
    if polygon.shape[0] < 3:
        return None
    poly = np.round(polygon).astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, rw - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, rh - 1)
    hull = cv2.convexHull(poly)
    filled = np.zeros((rh, rw), np.uint8)
    cv2.fillConvexPoly(filled, hull, 255)
    kernel = np.ones((13, 13), np.uint8)
    filled = cv2.dilate(filled, kernel, iterations=1)
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    alpha = np.clip(dist / 16.0, 0.0, 1.0).astype(np.float32)
    return alpha[:, :, None]
