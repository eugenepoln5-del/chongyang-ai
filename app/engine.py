from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.deform import DeformState, plan_deform
from app.face_tracker import FaceTracker
from app.oral import cover_opening, paint_oral_cavity, sample_skin
from app.viseme import Viseme
from app.warp import warp_frame


@dataclass
class FrameResult:
    original: np.ndarray
    synthesized: np.ndarray
    face_found: bool
    note: str


class LipEngine:
    def __init__(self, model_path: Path | None = None, tracker_mode: str = "video") -> None:
        self.model_path = model_path
        self.tracker_mode = tracker_mode
        self.tracker: FaceTracker | None = None
        self.state = DeformState()

    def ensure_tracker(self) -> FaceTracker:
        if self.tracker is None:
            if self.model_path is None or not Path(self.model_path).is_file():
                raise FileNotFoundError("找不到人脸模型，无法开启摄像头口型。")
            self.tracker = FaceTracker(self.model_path, mode=self.tracker_mode)
        return self.tracker

    def warmup(self) -> None:
        tracker = self.ensure_tracker()
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        tracker.detect(blank)

    def close(self) -> None:
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None

    def process(
        self,
        frame: np.ndarray,
        viseme: Viseme | None,
        strength: float = 1.0,
        smooth: float = 0.35,
        points: np.ndarray | None = None,
    ) -> FrameResult:
        """`frame` is never modified. A missing viseme means: show the real mouth."""
        detected = points
        if detected is None:
            detected = self.ensure_tracker().detect(frame)
        if viseme is None or strength <= 0.001:
            self.state.reset()
            note = "" if detected is not None else "未检测到人脸，合成画面与原片相同"
            return FrameResult(frame, frame.copy(), detected is not None, note)
        if detected is None:
            self.state.reset()
            return FrameResult(frame, frame.copy(), False, "未检测到人脸，合成画面与原片相同")
        return self.render(frame, detected, viseme, strength, smooth)

    def render(
        self,
        frame: np.ndarray,
        points: np.ndarray,
        viseme: Viseme,
        strength: float = 1.0,
        smooth: float = 0.35,
    ) -> FrameResult:
        plan = plan_deform(points, viseme, strength, smooth, self.state)
        if plan.skipped:
            return FrameResult(frame, frame.copy(), True, plan.note)
        source = frame
        if plan.closing:
            lip = sample_skin(frame, plan.lip_xy)
            source = cover_opening(frame, plan.inner_src, lip)
        warped = warp_frame(source, plan.src, plan.dst, plan.mask_pts)
        if plan.paint_cavity:
            skin = sample_skin(frame, plan.skin_xy)
            warped = paint_oral_cavity(
                warped,
                plan.inner_dst,
                plan.open_ratio,
                plan.mouth_wide,
                plan.mouth_round,
                skin,
            )
        return FrameResult(frame, warped, True, plan.note)
