"""Turn a viseme into landmark targets for the lower face.

The face position always comes from the current frame, so turning or
leaning moves the mouth with the head. Only the opening, width, and
jaw are driven by audio. Pause is handled by not calling this at all.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from app.viseme import Viseme

FOREHEAD = 10
CHIN = 152
NOSE_TIP = 1
NOSE_BASE = 2
NOSE_BRIDGE = (4, 5, 6)
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
LEFT_TEMPLE = 127
RIGHT_TEMPLE = 356
LEFT_JAW_ANGLE = 132
RIGHT_JAW_ANGLE = 361
LEFT_CORNER = 61
RIGHT_CORNER = 291
UPPER_INNER = 13
LOWER_INNER = 14
UPPER_OUTER = 0

LEFT_EYE = (33, 133, 159, 145)
RIGHT_EYE = (263, 362, 386, 374)
BROWS = (70, 105, 300, 334)

OUTER_LIP = (
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
)
INNER_LIP = (
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
)
UPPER_LIP = (185, 40, 39, 37, 0, 267, 269, 270, 409, 191, 80, 81, 82, 13, 312, 311, 310, 415)
LOWER_LIP = (375, 321, 405, 314, 17, 84, 181, 91, 146, 324, 318, 402, 317, 14, 87, 178, 88, 95)
CORNERS = (61, 291, 78, 308)

# How far the chin and jawline follow the opening. 1 would detach the jaw.
JAW_WEIGHTS = {
    152: 0.55,
    148: 0.38,
    377: 0.38,
    176: 0.24,
    400: 0.24,
    149: 0.12,
    378: 0.12,
    150: 0.06,
    379: 0.06,
}

FIXED = (
    FOREHEAD,
    NOSE_TIP,
    NOSE_BASE,
    *NOSE_BRIDGE,
    LEFT_CHEEK,
    RIGHT_CHEEK,
    LEFT_TEMPLE,
    RIGHT_TEMPLE,
    LEFT_JAW_ANGLE,
    RIGHT_JAW_ANGLE,
    *LEFT_EYE,
    *RIGHT_EYE,
    *BROWS,
    58,
    288,
)


@dataclass
class DeformPlan:
    src: np.ndarray
    dst: np.ndarray
    inner_src: np.ndarray
    inner_dst: np.ndarray
    mask_pts: np.ndarray
    open_ratio: float
    mouth_wide: float
    mouth_round: float
    skin_xy: tuple[float, float]
    lip_xy: tuple[float, float]
    closing: bool
    paint_cavity: bool
    skipped: bool
    note: str


class DeformState:
    def __init__(self) -> None:
        self.open: float | None = None
        self.scale: float | None = None
        self.t: float | None = None

    def reset(self) -> None:
        self.open = None
        self.scale = None
        self.t = None

    def dt(self) -> float:
        now = time.perf_counter()
        if self.t is None:
            self.t = now
            return 1.0 / 30.0
        delta = min(0.2, max(1e-3, now - self.t))
        self.t = now
        return delta

    def smooth(self, value: float, prev: float | None, dt: float, attack: float, release: float) -> float:
        if prev is None:
            return value
        tau = attack if value > prev else release
        alpha = 1.0 - math.exp(-dt / max(tau, 1e-4))
        return prev + alpha * (value - prev)


def plan_deform(
    points: np.ndarray,
    viseme: Viseme,
    strength: float,
    smooth: float,
    state: DeformState,
) -> DeformPlan:
    pts = np.asarray(points, dtype=np.float32)
    empty = _empty(pts)
    if pts.ndim != 2 or pts.shape[0] < 468 or pts.shape[1] < 2:
        empty.note = "人脸点不足"
        return empty

    axis_x, axis_y = _face_axes(pts)
    if axis_x is None or axis_y is None:
        empty.note = "人脸朝向不稳定"
        return empty

    forehead = pts[FOREHEAD]
    chin = pts[CHIN]
    nose = pts[NOSE_TIP]
    nose_base = pts[NOSE_BASE]
    face_h = abs(float(np.dot(chin - forehead, axis_y)))
    nose_chin = abs(float(np.dot(chin - nose_base, axis_y)))
    if face_h < 12.0 or nose_chin < 8.0:
        empty.note = "人脸太小"
        return empty

    left_c = pts[LEFT_CORNER]
    right_c = pts[RIGHT_CORNER]
    upper = pts[UPPER_INNER]
    lower = pts[LOWER_INNER]
    center = (left_c + right_c) * 0.5
    current_open = float(np.dot(lower - upper, axis_y))
    current_width = abs(float(np.dot(right_c - left_c, axis_x)))
    if current_width < 6.0:
        empty.note = "没有看清嘴巴"
        return empty

    left_edge = pts[LEFT_CHEEK]
    right_edge = pts[RIGHT_CHEEK]
    dl = abs(float(np.dot(nose - left_edge, axis_x)))
    dr = abs(float(np.dot(right_edge - nose, axis_x)))
    profile = abs(dr - dl) / (dr + dl + 1e-6)

    strength = float(np.clip(strength, 0.0, 1.8))
    max_open = max(14.0, 0.30 * nose_chin)
    target_open = 1.6 + viseme.jaw_open * (max_open - 1.6)
    target_open *= 1.0 - 0.20 * viseme.mouth_wide + 0.05 * viseme.mouth_round
    target_open = max(1.2, target_open)
    # A smile that is already open should not be stretched into a fake cavern.
    # A closed mouth is still allowed to open the full amount.
    if current_open > 0.50 * max_open and viseme.jaw_open > 0.15:
        extra = max(0.0, viseme.jaw_open - 0.15) * 0.10 * max_open
        target_open = min(target_open, current_open + extra)
    desired_width = current_width * (1.0 + 0.20 * viseme.mouth_wide - 0.30 * viseme.mouth_round)
    desired_width = float(np.clip(desired_width, 0.72 * current_width, 1.38 * current_width))

    d_open = (target_open - current_open) * strength
    width_scale = 1.0 + (desired_width / current_width - 1.0) * strength

    note = ""
    if profile > 0.30:
        damp = max(0.22, 1.0 - (profile - 0.30) / 0.40)
        d_open *= damp
        width_scale = 1.0 + (width_scale - 1.0) * damp
        if profile > 0.48:
            note = "侧面角度较大，口型已减弱"

    d_open = float(np.clip(d_open, -0.22 * face_h, 0.22 * face_h))
    width_scale = float(np.clip(width_scale, 0.68, 1.42))

    dt = state.dt()
    attack = 0.022 + 0.08 * float(np.clip(smooth, 0.0, 1.0))
    release = 0.045 + 0.13 * float(np.clip(smooth, 0.0, 1.0))
    d_open = state.smooth(d_open, state.open, dt, attack, release)
    width_scale = state.smooth(width_scale, state.scale, dt, attack, release)
    state.open = d_open
    state.scale = width_scale

    if abs(d_open) < 0.45 and abs(width_scale - 1.0) < 0.012:
        empty.note = note
        return empty

    dst_map: dict[int, np.ndarray] = {}
    half = max(current_width * 0.5, 1.0)

    def place(idx: int, share: float, corner: bool = False) -> np.ndarray:
        p = pts[idx]
        rel = p - center
        along0 = float(np.dot(rel, axis_x))
        perp = float(np.dot(rel, axis_y))
        along = along0 * width_scale
        if corner:
            perp += d_open * 0.12
        else:
            cw = math.sqrt(max(0.0, 1.0 - (along0 / half) ** 2))
            perp += d_open * share * (0.18 + 0.82 * cw)
        return center + axis_x * along + axis_y * perp

    for idx in UPPER_LIP:
        dst_map[idx] = place(idx, -0.32)
    for idx in LOWER_LIP:
        dst_map[idx] = place(idx, 0.68)
    for idx in CORNERS:
        dst_map[idx] = place(idx, 0.0, corner=True)
    for idx, weight in JAW_WEIGHTS.items():
        dst_map[idx] = pts[idx] + axis_y * (d_open * weight)

    src_list = []
    dst_list = []
    used = set(dst_map)
    for idx, target in dst_map.items():
        src_list.append(pts[idx])
        dst_list.append(target)
    for idx in FIXED:
        if idx in used or idx >= len(pts):
            continue
        src_list.append(pts[idx])
        dst_list.append(pts[idx].copy())

    src = np.stack(src_list).astype(np.float32)
    dst = np.stack(dst_list).astype(np.float32)
    inner_src = np.stack([pts[i] for i in INNER_LIP]).astype(np.float32)
    inner = np.stack([dst_map[i] for i in INNER_LIP]).astype(np.float32)
    closing = d_open < -4.0 and current_open > 8.0
    paint_cavity = (not closing) and current_open < 0.28 * max_open and (current_open + d_open) > 0.16 * max_open
    mask_ids = list(OUTER_LIP) + list(JAW_WEIGHTS) + [NOSE_BASE, LEFT_CHEEK, RIGHT_CHEEK, LEFT_JAW_ANGLE, RIGHT_JAW_ANGLE]
    mask_pts = []
    for idx in mask_ids:
        mask_pts.append(pts[idx])
        if idx in dst_map:
            mask_pts.append(dst_map[idx])
    open_ratio = float(np.clip((current_open + d_open) / max_open, 0.0, 1.0))
    skin_xy = tuple(((pts[NOSE_BASE] + pts[UPPER_OUTER]) * 0.5).tolist())
    lip_xy = (float(pts[UPPER_OUTER, 0]), float(pts[UPPER_OUTER, 1]))
    return DeformPlan(
        src=src,
        dst=dst,
        inner_src=inner_src,
        inner_dst=inner,
        mask_pts=np.stack(mask_pts).astype(np.float32),
        open_ratio=open_ratio,
        mouth_wide=float(viseme.mouth_wide),
        mouth_round=float(viseme.mouth_round),
        skin_xy=(float(skin_xy[0]), float(skin_xy[1])),
        lip_xy=lip_xy,
        closing=closing,
        paint_cavity=paint_cavity,
        skipped=False,
        note=note,
    )


def _face_axes(pts: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    left = np.mean(pts[list(LEFT_EYE)], axis=0)
    right = np.mean(pts[list(RIGHT_EYE)], axis=0)
    axis_x = right - left
    norm = float(np.linalg.norm(axis_x))
    if norm < 4.0:
        return None, None
    axis_x = axis_x / norm
    axis_y = np.array([-axis_x[1], axis_x[0]], dtype=np.float32)
    if float(axis_y[1]) < 0.0:
        axis_y = -axis_y
    return axis_x.astype(np.float32), axis_y.astype(np.float32)


def _empty(pts: np.ndarray) -> DeformPlan:
    del pts
    zero = np.zeros((0, 2), dtype=np.float32)
    return DeformPlan(
        src=zero,
        dst=zero,
        inner_src=zero,
        inner_dst=zero,
        mask_pts=zero,
        open_ratio=0.0,
        mouth_wide=0.0,
        mouth_round=0.0,
        skin_xy=(0.0, 0.0),
        lip_xy=(0.0, 0.0),
        closing=False,
        paint_cavity=False,
        skipped=True,
        note="",
    )
