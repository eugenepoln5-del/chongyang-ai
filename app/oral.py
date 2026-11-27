"""Paint a simple mouth interior into an opened lip polygon.

Warping a closed mouth only stretches lip skin. The cavity, upper teeth,
and tongue are drawn so the opening reads as a mouth. They are clipped to
the inner lip and use a fixed grain so they do not shimmer.
"""

from __future__ import annotations

import cv2
import numpy as np

_GRAIN = np.random.default_rng(7).normal(0.0, 2.2, size=(96, 96)).astype(np.float32)


def cover_opening(image: np.ndarray, polygon: np.ndarray, lip_bgr: np.ndarray) -> np.ndarray:
    """Hide teeth before a closing warp, using the color of the upper lip."""
    if polygon.shape[0] < 6:
        return image
    height, width = image.shape[:2]
    poly = np.round(polygon).astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, width - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, height - 1)
    if abs(float(cv2.contourArea(poly))) < 24.0:
        return image
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    x, y, bw, bh = cv2.boundingRect(poly)
    if bw < 2 or bh < 2:
        return image
    soft = cv2.GaussianBlur(mask[y : y + bh, x : x + bw], (7, 7), 0).astype(np.float32) / 255.0
    if float(soft.max()) < 0.05:
        return image
    color = np.clip(np.asarray(lip_bgr, dtype=np.float32), 0, 255)
    patch = image[y : y + bh, x : x + bw].astype(np.float32)
    fill = np.empty_like(patch)
    fill[:] = color
    mixed = patch * (1.0 - soft[:, :, None]) + fill * soft[:, :, None]
    out = image.copy()
    out[y : y + bh, x : x + bw] = np.clip(mixed, 0, 255).astype(np.uint8)
    return out


def paint_oral_cavity(
    image: np.ndarray,
    polygon: np.ndarray,
    open_ratio: float,
    mouth_wide: float,
    mouth_round: float,
    skin_bgr: np.ndarray,
) -> np.ndarray:
    """Paint into a copy only when the mouth is actually open."""
    if open_ratio < 0.12 or polygon.shape[0] < 6:
        return image
    height, width = image.shape[:2]
    poly = np.round(polygon).astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, width - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, height - 1)
    area = abs(float(cv2.contourArea(poly)))
    if area < 36.0:
        return image

    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    if int(mask.sum()) < 400:
        return image

    x, y, bw, bh = cv2.boundingRect(poly)
    if bw < 4 or bh < 4:
        return image

    skin = np.clip(np.asarray(skin_bgr, dtype=np.float32), 8.0, 255.0)
    luma = float(np.dot(skin, np.array([0.114, 0.587, 0.299], dtype=np.float32)))
    light = float(np.clip(luma / 150.0, 0.45, 1.2))

    cavity = skin * np.array([0.20, 0.16, 0.26], dtype=np.float32)
    layer = np.empty((bh, bw, 3), dtype=np.float32)
    layer[:] = cavity

    yy = np.linspace(0.0, 1.0, bh, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, bw, dtype=np.float32)[None, :]
    vignette = (0.48 + 0.62 * yy) * np.clip(1.0 - 0.28 * xx * xx, 0.55, 1.0)
    layer *= vignette[:, :, None]

    if open_ratio > 0.22:
        tongue = np.zeros((bh, bw), np.float32)
        tcx, tcy = bw // 2, int(bh * (0.58 + 0.08 * mouth_round))
        ax = max(int(bw * (0.26 + 0.06 * (1.0 - mouth_wide))), 2)
        ay = max(int(bh * 0.30), 2)
        cv2.ellipse(tongue, (tcx, tcy), (ax, ay), 0, 0, 360, 1.0, thickness=-1)
        tongue = cv2.GaussianBlur(tongue, (0, 0), 1.6)
        tongue_color = skin * np.array([0.42, 0.36, 0.58], dtype=np.float32)
        t = tongue[:, :, None]
        layer = layer * (1.0 - t * 0.82) + tongue_color * (t * 0.82)

    if open_ratio > 0.16:
        teeth = _teeth_mask(bw, bh, upper=True, wide=mouth_wide, open_ratio=open_ratio)
        teeth_color = np.array([198, 194, 184], dtype=np.float32) * light
        layer = layer * (1.0 - teeth[:, :, None]) + teeth_color * teeth[:, :, None]
    if open_ratio > 0.46 and mouth_round < 0.75:
        teeth = _teeth_mask(bw, bh, upper=False, wide=mouth_wide, open_ratio=open_ratio)
        teeth_color = np.array([176, 172, 162], dtype=np.float32) * light
        layer = layer * (1.0 - teeth[:, :, None]) + teeth_color * teeth[:, :, None]

    grain = _grain(bh, bw)
    layer += grain[:, :, None]
    layer = np.clip(layer, 0, 255)

    soft = cv2.GaussianBlur(mask[y : y + bh, x : x + bw], (5, 5), 0).astype(np.float32) / 255.0
    patch = image[y : y + bh, x : x + bw].astype(np.float32)
    mixed = patch * (1.0 - soft[:, :, None]) + layer * soft[:, :, None]
    painted = image.copy()
    painted[y : y + bh, x : x + bw] = np.clip(mixed, 0, 255).astype(np.uint8)
    return painted


def sample_skin(frame: np.ndarray, xy: tuple[float, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x = int(round(xy[0]))
    y = int(round(xy[1]))
    x0, x1 = max(0, x - 6), min(width, x + 7)
    y0, y1 = max(0, y - 4), min(height, y + 5)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([130.0, 140.0, 170.0], dtype=np.float32)
    return patch.reshape(-1, 3).mean(axis=0).astype(np.float32)


def _teeth_mask(bw: int, bh: int, upper: bool, wide: float, open_ratio: float) -> np.ndarray:
    mask = np.zeros((bh, bw), np.float32)
    cols = np.arange(bw, dtype=np.float32)
    scallop = (0.5 + 0.5 * np.sin(cols / max(bw, 1) * np.pi * 6.0)) * (0.07 * bh + 1.0)
    if upper:
        tooth_h = bh * (0.18 + 0.16 * open_ratio) * (1.0 - 0.18 * wide)
        edge = tooth_h + scallop
        yy = np.arange(bh, dtype=np.float32)[:, None]
        region = yy < edge[None, :]
        fade = np.clip(1.0 - yy / max(edge.max(), 1.0), 0.0, 1.0)
    else:
        tooth_h = bh * (0.10 + 0.08 * open_ratio)
        edge = bh - tooth_h - scallop * 0.6
        yy = np.arange(bh, dtype=np.float32)[:, None]
        region = yy > edge[None, :]
        fade = np.clip((yy - edge[None, :]) / max(tooth_h, 1.0), 0.0, 1.0)
    alpha = region.astype(np.float32) * (0.55 + 0.45 * fade)
    # Gaps between teeth.
    gaps = 0.82 + 0.18 * np.sin(cols / max(bw, 1) * np.pi * 6.0 + 1.2)
    alpha *= gaps[None, :]
    inset = max(int(bw * 0.08), 1)
    alpha[:, :inset] = 0
    alpha[:, bw - inset :] = 0
    mask[:, :] = alpha
    return cv2.GaussianBlur(mask, (3, 3), 0)


def _grain(bh: int, bw: int) -> np.ndarray:
    tile = np.tile(_GRAIN, (bh // 96 + 2, bw // 96 + 2))
    return tile[:bh, :bw]
