from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from app.audio_io import AudioError, decode_audio
from app.viseme import Viseme, VisemeTimeline, analyze_audio


class AudioPlayer:
    """Plays a file and reports the viseme at the sample the speaker is on.

    If the machine has no output device, playback becomes a silent clock so
    the mouth can still be previewed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stereo: np.ndarray | None = None
        self._timeline: VisemeTimeline | None = None
        self.sample_rate = 0
        self.path: Path | None = None
        self.name = ""
        self.silent = False
        self.error = ""
        self._cursor = 0
        self._n = 0
        self._playing = False
        self._paused = False
        self._ended = False
        self._mode = ""
        self._stream = None
        self._clock_origin_perf = 0.0
        self._clock_origin_samples = 0

    @property
    def loaded(self) -> bool:
        return self._stereo is not None

    @property
    def duration(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self._n / float(self.sample_rate)

    def load(self, path: Path) -> None:
        stereo, sample_rate = decode_audio(Path(path))
        if stereo.size == 0:
            raise AudioError("这个音频文件是空的。")
        mono = stereo.mean(axis=1).astype(np.float32)
        timeline = analyze_audio(mono, sample_rate)
        self.stop()
        with self._lock:
            self._stereo = stereo
            self._timeline = timeline
            self.sample_rate = sample_rate
            self._n = int(stereo.shape[0])
            self._cursor = 0
            self.path = Path(path)
            self.name = Path(path).name
            self.silent = False
            self.error = ""
            self._ended = False

    def start(self) -> str | None:
        with self._lock:
            if self._stereo is None:
                return "请先选择一段音频。"
            if self._playing:
                return None
            if self._cursor >= self._n:
                self._cursor = 0
            self._ended = False
            self._paused = False
            self._playing = True
            cursor = self._cursor
        if not self._open_device():
            with self._lock:
                self._mode = "clock"
                self._clock_origin_perf = time.perf_counter()
                self._clock_origin_samples = cursor
                self.silent = True
                self._playing = True
        return None

    def pause(self) -> None:
        with self._lock:
            if not self._playing:
                return
            if self._mode == "clock":
                elapsed = time.perf_counter() - self._clock_origin_perf
                self._cursor = min(self._n, self._clock_origin_samples + int(elapsed * self.sample_rate))
            self._playing = False
            self._paused = True
        self._close_stream()

    def stop(self) -> None:
        with self._lock:
            self._playing = False
            self._paused = False
            self._ended = False
            self._cursor = 0
        self._close_stream()

    def seek(self, seconds: float) -> None:
        with self._lock:
            if self.sample_rate <= 0:
                return
            self._cursor = int(np.clip(seconds, 0.0, self.duration) * self.sample_rate)
            self._cursor = min(self._cursor, self._n)
            self._ended = False
            if self._mode == "clock" and self._playing:
                self._clock_origin_perf = time.perf_counter()
                self._clock_origin_samples = self._cursor

    def position(self) -> float:
        with self._lock:
            return self._position_unlocked()

    def state(self) -> str:
        with self._lock:
            if self._playing:
                return "playing"
            if self._paused:
                return "paused"
            return "stopped"

    def current_viseme(self) -> Viseme | None:
        with self._lock:
            if not self._playing or self._timeline is None or self.sample_rate <= 0:
                return None
            if self._mode == "clock":
                elapsed = time.perf_counter() - self._clock_origin_perf
                cursor = self._clock_origin_samples + int(elapsed * self.sample_rate)
                if cursor >= self._n:
                    self._playing = False
                    self._paused = False
                    self._ended = True
                    self._cursor = 0
                    return None
                self._cursor = cursor
            t = self._cursor / float(self.sample_rate)
            timeline = self._timeline
        return timeline.at(t)

    def take_ended(self) -> bool:
        with self._lock:
            if not self._ended:
                return False
            self._ended = False
            return True

    def close(self) -> None:
        self.stop()

    def _position_unlocked(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        if self._playing and self._mode == "clock":
            elapsed = time.perf_counter() - self._clock_origin_perf
            cursor = self._clock_origin_samples + int(elapsed * self.sample_rate)
            return min(self.duration, cursor / float(self.sample_rate))
        return self._cursor / float(self.sample_rate)

    def _open_device(self) -> bool:
        try:
            import sounddevice as sd
        except Exception:
            self.error = "没有可用的声音输出，口型仍会跟着音频走，但听不到声音。"
            return False
        with self._lock:
            stereo = self._stereo
            sample_rate = self.sample_rate

        def callback(outdata, frames, _time_info, status) -> None:
            del status
            outdata.fill(0)
            with self._lock:
                if not self._playing or stereo is None:
                    return
                start = self._cursor
                end = min(start + frames, self._n)
                if start >= self._n:
                    self._playing = False
                    self._paused = False
                    self._ended = True
                    self._cursor = 0
                    return
                chunk = stereo[start:end]
                outdata[: len(chunk)] = chunk
                self._cursor = end
                if end >= self._n:
                    self._playing = False
                    self._paused = False
                    self._ended = True
                    self._cursor = 0

        try:
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=2,
                dtype="float32",
                blocksize=1024,
                callback=callback,
            )
            stream.start()
        except Exception:
            self.error = "没有可用的声音输出，口型仍会跟着音频走，但听不到声音。"
            return False
        with self._lock:
            self._stream = stream
            self._mode = "device"
            self.silent = False
            self.error = ""
        return True

    def _close_stream(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
            if self._mode == "device":
                self._mode = ""
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
