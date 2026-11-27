from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.audio_io import AudioError, write_example_wav
from app.camera import CameraSource, DemoSource, VideoSource, camera_permission_hint, wait_for_frame
from app.engine import LipEngine
from app.paths import data_dir, recordings_dir
from app.player import AudioPlayer
from app.recorder import Recorder, RecorderError


@dataclass
class Snapshot:
    original: np.ndarray
    synthesized: np.ndarray
    note: str
    fps: float
    process_ms: float
    face_found: bool
    source_label: str
    status: str
    audio_name: str
    position_s: float
    duration_s: float
    play_state: str
    recording: bool
    silent_audio: bool
    engine_ready: bool
    engine_error: str


class Session:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.engine = LipEngine(model_path)
        self.audio = AudioPlayer()
        self.recorder = Recorder()
        self.mirror = True
        self.strength = 1.0
        self.smooth = 0.35
        self._source = None
        self._source_kind = ""
        self._source_label = "未开启"
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._snap: Snapshot | None = None
        self._status = "准备好了"
        self._engine_error = ""
        self._engine_ready = False
        self._record_audio_path: Path | None = None
        self._record_audio_start = 0.0
        self._record_audio_clean = False
        self._load_thread = threading.Thread(target=self._load_engine, name="chongyang-model", daemon=True)
        self._load_thread.start()

    @property
    def engine_ready(self) -> bool:
        return self._engine_ready and not self._engine_error

    def _load_engine(self) -> None:
        try:
            self.engine.warmup()
            self._engine_ready = True
            self._status = "人脸模型已就绪"
        except Exception as exc:
            self._engine_error = f"人脸模型加载失败：{exc}"
            self._status = self._engine_error

    def start_camera(self, index: int = 0) -> str | None:
        if self._engine_error:
            return self._engine_error
        if not self._engine_ready:
            return "人脸模型还在准备，请稍等几秒再开摄像头。"
        camera = CameraSource(index)
        if not camera.open():
            return camera_permission_hint()
        frame = wait_for_frame(camera)
        if frame is None:
            camera.close()
            return camera_permission_hint()
        self._swap_source(camera, "camera", "摄像头")
        self._status = "摄像头已开启"
        return None

    def open_video(self, path: Path) -> str | None:
        video = VideoSource(path)
        if not video.opened():
            video.close()
            return "打不开这个视频。请换一个 mp4 或 mov 文件。"
        self._swap_source(video, "video", path.name)
        self._status = "正在播放视频源"
        return None

    def start_demo(self) -> None:
        self._swap_source(DemoSource(), "demo", "示例人物")
        self._status = "正在演示绘制的示例人物"

    def stop_source(self) -> None:
        self._stop_recording_internal()
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2)
        self._close_source()
        self._source_label = "未开启"
        self._status = "画面已停止"
        with self._lock:
            self._snap = None

    def load_audio(self, path: Path) -> str | None:
        try:
            self.audio.load(path)
        except AudioError as exc:
            return str(exc)
        self._status = f"已载入音频：{path.name}"
        return None

    def load_example_audio(self) -> str | None:
        path = data_dir() / "示例音频.wav"
        if not path.is_file():
            write_example_wav(path)
        return self.load_audio(path)

    def play(self) -> str | None:
        if self._source is None:
            return "请先开启摄像头，或点「示例演示」。"
        if not self.audio.loaded:
            return "请先选择一段音频。可以先用「示例音频」试听。"
        message = self.audio.start()
        if message:
            return message
        if self.recorder.active:
            self._record_audio_clean = False
        if self.audio.silent and self.audio.error:
            self._status = self.audio.error
        else:
            self._status = "正在按音频做口型"
        return None

    def pause(self) -> None:
        self.audio.pause()
        if self.recorder.active:
            self._record_audio_clean = False
        self._status = "已暂停，合成画面回到真实嘴型"

    def stop_playback(self) -> None:
        self.audio.stop()
        if self.recorder.active:
            self._record_audio_clean = False
        self._status = "已停止，合成画面回到真实嘴型"

    def seek(self, seconds: float) -> None:
        self.audio.seek(seconds)
        if self.recorder.active:
            self._record_audio_clean = False

    def start_recording(self) -> str | None:
        snap = self.snapshot()
        if snap is None:
            return "还没有画面，无法录制。"
        if self.recorder.active:
            return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = recordings_dir() / f"合成-{stamp}.mp4"
        try:
            self.recorder.start(path, snap.synthesized)
        except RecorderError as exc:
            return str(exc)
        self._record_audio_path = self.audio.path if self.audio.state() == "playing" else None
        self._record_audio_start = self.audio.position() if self._record_audio_path else 0.0
        self._record_audio_clean = self._record_audio_path is not None
        self._status = "正在录制合成画面"
        return None

    def stop_recording(self) -> tuple[Path | None, str]:
        return self._stop_recording_internal()

    def snapshot(self) -> Snapshot | None:
        with self._lock:
            return self._snap

    def shutdown(self) -> None:
        self._stop_recording_internal()
        self.audio.close()
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2)
        self._close_source()
        self.engine.close()

    def _stop_recording_internal(self) -> tuple[Path | None, str]:
        if not self.recorder.active and self.recorder.path is None:
            return None, ""
        was_active = self.recorder.active
        if not was_active:
            return None, ""
        video = self.recorder.stop()
        if video is None or not video.is_file():
            return None, self.recorder.error or "录像保存失败。"
        note = f"已保存：{video}"
        if self._record_audio_clean and self._record_audio_path is not None:
            duration = self.recorder.frames_written / float(self.recorder.fps)
            try:
                mixed = self.recorder.mux_audio(video, self._record_audio_path, self._record_audio_start, duration)
                note = f"已保存：{mixed}"
                video = mixed
            except RecorderError as exc:
                note = f"{video}\n{exc}"
        elif self._record_audio_path is not None:
            note = f"已保存画面：{video}\n录制中途暂停或拖动过进度，为避免声画错位，没有混入声音。"
        self._status = "录像已保存"
        self._record_audio_path = None
        self._record_audio_clean = False
        return video, note

    def _swap_source(self, source, kind: str, label: str) -> None:
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2)
        self._close_source()
        self._source = source
        self._source_kind = kind
        self._source_label = label
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="chongyang-preview", daemon=True)
        self._thread.start()

    def _close_source(self) -> None:
        source = self._source
        self._source = None
        if source is not None and hasattr(source, "close"):
            source.close()

    def _loop(self) -> None:
        fps = 0.0
        while self._running:
            source = self._source
            if source is None:
                break
            started = time.perf_counter()
            item = source.read()
            if item is None:
                time.sleep(0.02)
                continue
            points = None
            if isinstance(item, tuple):
                frame, points = item
            else:
                frame = item
            if frame is None:
                time.sleep(0.02)
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)
                if points is not None:
                    points = points.copy()
                    points[:, 0] = (frame.shape[1] - 1) - points[:, 0]
            viseme = self.audio.current_viseme()
            t0 = time.perf_counter()
            try:
                result = self.engine.process(frame, viseme, self.strength, self.smooth, points=points)
            except Exception as exc:
                result_note = f"这一帧处理失败：{exc}"
                synth = frame.copy()
                face = False
                process_ms = (time.perf_counter() - t0) * 1000.0
                note = result_note
                original = frame
            else:
                original = result.original
                synth = result.synthesized
                face = result.face_found
                note = result.note
                process_ms = (time.perf_counter() - t0) * 1000.0
            if self.audio.take_ended():
                self._status = "播放结束，已回到真实嘴型"
            inst = 1000.0 / process_ms if process_ms > 0.4 else 0.0
            fps = inst if fps == 0.0 else fps * 0.85 + inst * 0.15
            if self.recorder.active:
                self.recorder.publish(synth)
            snap = Snapshot(
                original=original,
                synthesized=synth,
                note=note,
                fps=fps,
                process_ms=process_ms,
                face_found=face,
                source_label=self._source_label,
                status=self._status,
                audio_name=self.audio.name,
                position_s=self.audio.position(),
                duration_s=self.audio.duration,
                play_state=self.audio.state(),
                recording=self.recorder.active,
                silent_audio=self.audio.silent,
                engine_ready=self._engine_ready,
                engine_error=self._engine_error,
            )
            with self._lock:
                self._snap = snap
            spent = time.perf_counter() - started
            delay = (1.0 / 30.0) - spent
            if self._source_kind != "camera" and delay > 0:
                time.sleep(delay)
