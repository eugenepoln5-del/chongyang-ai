from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


class FaceTracker:
    """MediaPipe face landmarker. One instance is used from a single thread."""

    def __init__(self, model_path: Path, mode: str = "video") -> None:
        import mediapipe as mp
        from mediapipe.tasks.python.core import base_options as base_options_lib
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib

        self._mp = mp
        running = running_mode_lib.VisionTaskRunningMode.VIDEO if mode == "video" else running_mode_lib.VisionTaskRunningMode.IMAGE
        options = FaceLandmarkerOptions(
            base_options=base_options_lib.BaseOptions(model_asset_path=str(model_path)),
            running_mode=running,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mode = mode
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts = 0

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        height, width = frame_bgr.shape[:2]
        scale = 1.0
        max_side = 640.0
        longest = max(height, width)
        image = frame_bgr
        if longest > max_side:
            scale = max_side / float(longest)
            image = cv2.resize(
                frame_bgr,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._mode == "video":
            timestamp = int((time.monotonic() - self._t0) * 1000) + 1
            if timestamp <= self._last_ts:
                timestamp = self._last_ts + 1
            self._last_ts = timestamp
            result = self._landmarker.detect_for_video(mp_image, timestamp)
        else:
            result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        points = np.empty((len(landmarks), 2), dtype=np.float32)
        for i, point in enumerate(landmarks):
            points[i, 0] = point.x * width
            points[i, 1] = point.y * height
        return points

    def close(self) -> None:
        landmarker = getattr(self, "_landmarker", None)
        if landmarker is not None:
            landmarker.close()
            self._landmarker = None
