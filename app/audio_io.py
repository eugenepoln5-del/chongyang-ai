from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np


class AudioError(Exception):
    pass


def decode_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return float32 stereo samples in [-1, 1] and the sample rate."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"找不到音频文件：{path.name}")

    try:
        import soundfile as sf

        data, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
        return _stereo(np.asarray(data, dtype=np.float32)), int(sample_rate)
    except Exception:
        pass

    try:
        import miniaudio

        decoded = miniaudio.decode_file(
            str(path),
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=2,
            sample_rate=44100,
        )
        samples = np.asarray(decoded.samples, dtype=np.float32).reshape(-1, decoded.nchannels)
        return _stereo(samples), int(decoded.sample_rate)
    except Exception:
        pass

    try:
        return _ffmpeg_decode(path)
    except Exception as exc:
        raise AudioError(
            f"读不了这个音频（{path.suffix or '未知格式'}）。请换成 wav 或 mp3。"
        ) from exc


def write_example_wav(path: Path, sample_rate: int = 22050) -> Path:
    """A short synthetic phrase so the mouth can be tried with no recording."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pieces: list[np.ndarray] = [np.zeros(int(sample_rate * 0.35), dtype=np.float32)]
    pattern = [
        ("ah", 0.38),
        ("gap", 0.16),
        ("ee", 0.32),
        ("gap", 0.16),
        ("oh", 0.36),
        ("gap", 0.22),
        ("ah", 0.18),
        ("ee", 0.16),
        ("oh", 0.20),
        ("gap", 0.28),
        ("ah", 0.42),
        ("gap", 0.35),
    ]
    for kind, dur in pattern:
        n = int(sample_rate * dur)
        if kind == "gap":
            pieces.append(np.zeros(n, dtype=np.float32))
            continue
        pieces.append(_vowel(sample_rate, n, kind))
    mono = np.concatenate(pieces)
    mono = np.clip(mono, -1.0, 1.0)
    pcm = (mono * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


def _vowel(sample_rate: int, n: int, kind: str) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sample_rate
    f0 = 150.0
    partials = {
        "ah": [(700, 0.55), (1100, 0.35), (250, 0.2)],
        "ee": [(260, 0.25), (2300, 0.55), (3200, 0.28)],
        "oh": [(280, 0.5), (420, 0.4), (180, 0.35)],
    }[kind]
    sig = np.zeros(n, dtype=np.float32)
    for k in range(1, 8):
        sig += (0.35 / k) * np.sin(2 * np.pi * f0 * k * t)
    for freq, amp in partials:
        sig += amp * np.sin(2 * np.pi * freq * t)
    attack = max(1, int(0.025 * sample_rate))
    release = max(1, int(0.05 * sample_rate))
    env = np.ones(n, dtype=np.float32)
    env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    env[-release:] = np.linspace(1.0, 0.0, release, dtype=np.float32)
    return (sig * env * 0.55).astype(np.float32)


def _stereo(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    elif data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    else:
        data = data[:, :2]
    return np.ascontiguousarray(data, dtype=np.float32)


def _ffmpeg_decode(path: Path) -> tuple[np.ndarray, int]:
    exe = shutil.which("ffmpeg")
    if not exe:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    sample_rate = 44100
    cmd = [
        exe,
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "2",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.PIPE)
    if len(raw) < 8:
        raise AudioError("音频是空的")
    data = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()
    return data, sample_rate
