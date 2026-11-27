from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class RecorderError(Exception):
    pass


def ffmpeg_exe() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class Recorder:
    def __init__(self) -> None:
        self.path: Path | None = None
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self.width = 0
        self.height = 0
        self.fps = 25
        self.frames_written = 0
        self.error = ""

    @property
    def active(self) -> bool:
        return self._running

    def start(self, path: Path, frame: np.ndarray) -> None:
        if self._running:
            raise RecorderError("已经在录制。")
        exe = ffmpeg_exe()
        if not exe:
            raise RecorderError("没有找到 ffmpeg，这次没法录制。画面预览不受影响。")
        height, width = frame.shape[:2]
        width -= width % 2
        height -= height % 2
        if width < 16 or height < 16:
            raise RecorderError("画面太小，没法录制。")
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            exe,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            str(path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.path = path
        self.width = width
        self.height = height
        self.frames_written = 0
        self.error = ""
        self._running = True
        self._latest = frame
        self._thread = threading.Thread(target=self._loop, name="chongyang-record", daemon=True)
        self._thread.start()

    def publish(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest = frame

    def stop(self) -> Path | None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        proc = self._proc
        self._proc = None
        if proc is None:
            return self.path
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        err = b""
        try:
            if proc.stderr is not None:
                err = proc.stderr.read()
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            err = b""
        if proc.returncode not in (0, None):
            tail = err.decode("utf-8", errors="replace")[-400:]
            self.error = tail or "录像写入失败"
            return None
        return self.path

    def mux_audio(self, video: Path, audio: Path, start_s: float, duration_s: float) -> Path:
        exe = ffmpeg_exe()
        if exe is None:
            raise RecorderError("没有 ffmpeg，画面已保存，但没有混入声音。")
        destination = video.with_name(video.stem + "-含声音" + video.suffix)
        cmd = [
            exe,
            "-y",
            "-i",
            str(video),
            "-ss",
            f"{max(0.0, start_s):.3f}",
            "-t",
            f"{max(0.1, duration_s):.3f}",
            "-i",
            str(audio),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(destination),
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0 or not destination.is_file():
            raise RecorderError("画面已保存，但混入声音失败。")
        return destination

    def _loop(self) -> None:
        interval = 1.0 / self.fps
        next_tick = time.perf_counter()
        while self._running:
            with self._lock:
                frame = None if self._latest is None else self._latest
            if frame is not None:
                self._write(frame)
            next_tick += interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()

    def _write(self, frame: np.ndarray) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        else:
            frame = frame[: self.height, : self.width]
        try:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            self.frames_written += 1
        except Exception as exc:
            self.error = str(exc)
            self._running = False
