"""Offline check that does not need a camera or a window."""

from __future__ import annotations

import numpy as np

from app.engine import LipEngine
from app.synthetic import synthetic_frame
from app.version import APP_NAME
from app.viseme import Viseme, analyze_audio


def run() -> int:
    silence = analyze_audio(np.zeros(16000, dtype=np.float32), 16000)
    if float(np.max(silence.frames)) > 0.05:
        print("静音不应该张嘴")
        return 1
    tone = _tone(16000, 1.0, 220.0)
    loud = analyze_audio(np.concatenate([np.zeros(4000), tone, np.zeros(4000)]), 16000)
    if float(np.max(loud.frames[:, 0])) < 0.2:
        print("响的声音应该把嘴张开")
        return 1
    frame, points = synthetic_frame()
    engine = LipEngine(None)
    paused = engine.process(frame, None, points=points)
    if not np.array_equal(paused.synthesized, frame):
        print("暂停时合成画面必须和原片一样")
        return 1
    opened = engine.process(frame, Viseme(0.95, 0.1, 0.05), points=points)
    if np.array_equal(opened.synthesized, frame):
        print("有声音时合成画面应该发生变化")
        return 1
    if not np.array_equal(frame[:15, :15], opened.synthesized[:15, :15]):
        print("画面角落不应被改动")
        return 1
    print(f"{APP_NAME} 自检通过")
    return 0


def _tone(sample_rate: int, seconds: float, freq: float) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    return (0.8 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(run())
