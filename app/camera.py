from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from app.synthetic import demo_frame


class CameraError(Exception):
    pass


class CameraSource:
    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.cap = None

    def open(self) -> bool:
        self.close()
        self.cap = _open_capture(self.index)
        if self.cap is None or not self.cap.isOpened():
            self.close()
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        return True

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame.copy()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoSource:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))

    def opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> np.ndarray | None:
        if not self.opened():
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame.copy()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class DemoSource:
    def __init__(self) -> None:
        self.t = 0.0

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        self.t += 1.0 / 30.0
        return demo_frame(self.t)

    def close(self) -> None:
        return None


def camera_permission_hint() -> str:
    if sys.platform == "darwin":
        return (
            "打不开摄像头。请到「系统设置 → 隐私与安全性 → 摄像头」，"
            "允许「终端」使用摄像头，然后关掉崇阳AI口型再打开一次。"
        )
    return "打不开摄像头。请确认摄像头没有被别的软件占用，然后再试一次。"


def wait_for_frame(source: CameraSource, attempts: int = 25) -> np.ndarray | None:
    for _ in range(attempts):
        frame = source.read()
        if frame is not None:
            return frame
        time.sleep(0.08)
    return None


def _open_capture(index: int):
    if sys.platform == "darwin":
        api = getattr(cv2, "CAP_AVFOUNDATION", 0)
        cap = cv2.VideoCapture(index, api)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    cap = cv2.VideoCapture(index)
    return cap
