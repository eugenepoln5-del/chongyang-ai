"""A drawn face used for the in-app demo and for tests.

The picture is illustration, not a photograph of a person. Landmark
positions match the drawing, so the same warp used on a camera frame
can be checked without a webcam.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.deform import (
    CHIN,
    FOREHEAD,
    INNER_LIP,
    LEFT_CHEEK,
    LEFT_CORNER,
    LEFT_JAW_ANGLE,
    LEFT_TEMPLE,
    NOSE_BASE,
    NOSE_TIP,
    OUTER_LIP,
    RIGHT_CHEEK,
    RIGHT_CORNER,
    RIGHT_JAW_ANGLE,
    RIGHT_TEMPLE,
)

WIDTH = 960
HEIGHT = 540


def synthetic_points(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    pts = np.zeros((478, 2), dtype=np.float32)
    cx, cy = width * 0.5, height * 0.48
    pts[:] = (cx, cy)

    def put(idx: int, x: float, y: float) -> None:
        pts[idx] = (x, y)

    put(FOREHEAD, cx, cy - 168)
    put(CHIN, cx, cy + 168)
    put(NOSE_TIP, cx, cy + 8)
    put(NOSE_BASE, cx, cy + 36)
    put(4, cx, cy - 20)
    put(5, cx - 16, cy + 6)
    put(6, cx + 16, cy + 6)
    put(LEFT_CHEEK, cx - 132, cy + 24)
    put(RIGHT_CHEEK, cx + 132, cy + 24)
    put(LEFT_TEMPLE, cx - 118, cy - 70)
    put(RIGHT_TEMPLE, cx + 118, cy - 70)
    put(LEFT_JAW_ANGLE, cx - 108, cy + 90)
    put(RIGHT_JAW_ANGLE, cx + 108, cy + 90)
    put(58, cx - 96, cy + 120)
    put(288, cx + 96, cy + 120)

    eye_y = cy - 36
    put(33, cx - 78, eye_y)
    put(133, cx - 38, eye_y)
    put(159, cx - 58, eye_y - 14)
    put(145, cx - 58, eye_y + 12)
    put(263, cx + 78, eye_y)
    put(362, cx + 38, eye_y)
    put(386, cx + 58, eye_y - 14)
    put(374, cx + 58, eye_y + 12)
    put(70, cx - 70, eye_y - 32)
    put(105, cx - 42, eye_y - 36)
    put(300, cx + 70, eye_y - 32)
    put(334, cx + 42, eye_y - 36)

    _place_loop(pts, OUTER_LIP, cx, cy + 78, 48, 8, 9, upper_count=11)
    _place_loop(pts, INNER_LIP, cx, cy + 78, 36, 3.2, 3.4, upper_count=11)
    put(LEFT_CORNER, cx - 48, cy + 78)
    put(RIGHT_CORNER, cx + 48, cy + 78)

    jaw = [
        (152, 0.0, 168),
        (148, -28, 160),
        (377, 28, 160),
        (176, -52, 146),
        (400, 52, 146),
        (149, -74, 128),
        (378, 74, 128),
        (150, -90, 112),
        (379, 90, 112),
    ]
    for idx, dx, dy in jaw:
        put(idx, cx + dx, cy + dy)
    return pts


def synthetic_frame(width: int = WIDTH, height: int = HEIGHT) -> tuple[np.ndarray, np.ndarray]:
    points = synthetic_points(width, height)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (34, 32, 30)
    cx, cy = int(width * 0.5), int(height * 0.48)

    cv2.ellipse(image, (cx, cy + 210), (70, 90), 0, 0, 360, (150, 166, 196), -1)
    cv2.ellipse(image, (cx, cy + 8), (148, 186), 0, 0, 360, (168, 188, 214), -1)
    cv2.ellipse(image, (cx, cy - 150), (150, 120), 0, 200, 340, (42, 40, 38), -1)
    cv2.circle(image, (cx - 148, cy + 10), 28, (160, 178, 204), -1)
    cv2.circle(image, (cx + 148, cy + 10), 28, (160, 178, 204), -1)

    for side in (-1, 1):
        ex = int(cx + side * 58)
        ey = int(cy - 36)
        cv2.ellipse(image, (ex, ey), (24, 16), 0, 0, 360, (244, 244, 242), -1)
        cv2.circle(image, (ex, ey + 1), 8, (70, 84, 120), -1)
        cv2.circle(image, (ex, ey + 1), 4, (30, 28, 26), -1)
        cv2.circle(image, (ex - 2, ey - 2), 2, (230, 230, 230), -1)
        brow_y = ey - 28
        cv2.line(image, (ex - 22, brow_y + 4), (ex + 18, brow_y - 2), (60, 62, 70), 4, cv2.LINE_AA)

    cv2.line(image, (cx, cy - 10), (cx, cy + 28), (140, 150, 170), 3, cv2.LINE_AA)
    cv2.ellipse(image, (cx, cy + 34), (16, 10), 0, 0, 180, (140, 150, 170), 2, cv2.LINE_AA)

    outer = points[list(OUTER_LIP)].astype(np.int32)
    inner = points[list(INNER_LIP)].astype(np.int32)
    cv2.fillPoly(image, [outer], (120, 132, 186))
    cv2.fillPoly(image, [inner], (90, 96, 140))
    cv2.polylines(image, [outer], True, (96, 104, 150), 2, cv2.LINE_AA)

    cv2.ellipse(image, (cx - 70, cy + 30), (22, 12), 0, 0, 360, (150, 160, 190), -1)
    cv2.ellipse(image, (cx + 70, cy + 30), (22, 12), 0, 0, 360, (150, 160, 190), -1)
    return image, points


def demo_frame(t: float, width: int = WIDTH, height: int = HEIGHT) -> tuple[np.ndarray, np.ndarray]:
    base, points = _cached_base(width, height)
    sway = math.sin(t * 0.85) * 26.0
    bob = math.cos(t * 0.55) * 10.0
    matrix = np.array([[1.0, 0.0, sway], [0.0, 1.0, bob]], dtype=np.float32)
    frame = cv2.warpAffine(
        base,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    moved = points.copy()
    moved[:, 0] += sway
    moved[:, 1] += bob
    return frame, moved


_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _cached_base(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    key = (width, height)
    if key not in _CACHE:
        _CACHE[key] = synthetic_frame(width, height)
    return _CACHE[key]


def _place_loop(
    pts: np.ndarray,
    indices: tuple[int, ...],
    cx: float,
    cy: float,
    rx: float,
    upper_ry: float,
    lower_ry: float,
    upper_count: int,
) -> None:
    upper = indices[:upper_count]
    lower = indices[upper_count:]
    for i, idx in enumerate(upper):
        t = i / (len(upper) - 1)
        ang = math.pi * (1.0 - t)
        pts[idx] = (cx + rx * math.cos(ang), cy - upper_ry * math.sin(ang))
    for i, idx in enumerate(lower):
        ang = math.pi * (i + 1) / (len(lower) + 1)
        pts[idx] = (cx + rx * 0.92 * math.cos(ang), cy + lower_ry * math.sin(ang))
