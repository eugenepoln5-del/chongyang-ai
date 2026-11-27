"""Map a waveform to a mouth-shape timeline.

The shapes are language-agnostic: energy opens the jaw, low frequencies
round the lips, and high frequencies spread them. This is not a phoneme
recognizer. It is fast, local, and stable enough to drive a live preview.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HOP_S = 0.010
WIN_S = 0.030


@dataclass(frozen=True)
class Viseme:
    jaw_open: float
    mouth_wide: float
    mouth_round: float

    @staticmethod
    def closed() -> Viseme:
        return Viseme(0.0, 0.0, 0.0)

    def scaled(self, strength: float) -> Viseme:
        s = max(0.0, strength)
        return Viseme(
            _clip01(self.jaw_open * s),
            _clip01(self.mouth_wide * s),
            _clip01(self.mouth_round * s),
        )


@dataclass
class VisemeTimeline:
    hop_s: float
    duration: float
    frames: np.ndarray  # (N, 3) jaw, wide, round

    def at(self, t: float) -> Viseme:
        if self.frames.size == 0 or t < 0.0 or t > self.duration + 0.03:
            return Viseme.closed()
        x = t / self.hop_s
        last = len(self.frames) - 1
        if x >= last:
            row = self.frames[last]
            return Viseme(float(row[0]), float(row[1]), float(row[2]))
        i = int(x)
        frac = x - i
        row = self.frames[i] * (1.0 - frac) + self.frames[i + 1] * frac
        return Viseme(float(row[0]), float(row[1]), float(row[2]))


def analyze_audio(mono: np.ndarray, sample_rate: int) -> VisemeTimeline:
    samples = np.asarray(mono, dtype=np.float32).reshape(-1)
    if samples.size == 0 or sample_rate <= 0:
        return VisemeTimeline(HOP_S, 0.0, np.zeros((0, 3), dtype=np.float32))

    peak = float(np.max(np.abs(samples)))
    if peak < 1e-5:
        duration = samples.size / float(sample_rate)
        n = max(1, int(duration / HOP_S))
        return VisemeTimeline(HOP_S, duration, np.zeros((n, 3), dtype=np.float32))
    samples = samples / peak * 0.9

    win = int(sample_rate * WIN_S)
    hop = max(1, int(sample_rate * HOP_S))
    if win < 16:
        win = 16
    if win % 2 == 0:
        win += 1
    if samples.size < win:
        samples = np.pad(samples, (0, win - samples.size))

    window = np.hanning(win).astype(np.float32)
    n = 1 + (samples.size - win) // hop
    rms = np.zeros(n, dtype=np.float32)
    zcr = np.zeros(n, dtype=np.float32)
    cent = np.zeros(n, dtype=np.float32)
    bands = np.zeros((n, 4), dtype=np.float32)
    flux = np.zeros(n, dtype=np.float32)
    freqs = np.fft.rfftfreq(win, d=1.0 / sample_rate).astype(np.float32)
    band_masks = [
        (freqs >= 80) & (freqs < 400),
        (freqs >= 400) & (freqs < 1200),
        (freqs >= 1200) & (freqs < 2800),
        (freqs >= 2800) & (freqs < 7600),
    ]
    prev_mag = None
    for i in range(n):
        start = i * hop
        frame = samples[start : start + win] * window
        rms[i] = math.sqrt(float(np.dot(frame, frame)) / win + 1e-12)
        signs = np.signbit(frame)
        zcr[i] = float(np.mean(signs[1:] != signs[:-1]))
        mag = np.abs(np.fft.rfft(frame)).astype(np.float32) + 1e-8
        cent[i] = float(np.dot(freqs, mag) / np.sum(mag))
        if prev_mag is not None:
            flux[i] = float(np.mean(np.maximum(0.0, mag - prev_mag)))
        prev_mag = mag
        for b, mask in enumerate(band_masks):
            if np.any(mask):
                bands[i, b] = float(np.mean(mag[mask]))

    db = 20.0 * np.log10(rms + 1e-8)
    floor = float(np.percentile(db, 20))
    speech = (db > -32.0) & (db > floor + 5.0)
    level = np.clip((db - max(floor + 4.0, -34.0)) / 18.0, 0.0, 1.0).astype(np.float32)

    band_sum = np.sum(bands, axis=1, keepdims=True) + 1e-6
    ratio = bands / band_sum
    low, mid, high = ratio[:, 0], ratio[:, 1], ratio[:, 2] + ratio[:, 3]
    cent_n = np.clip((cent - 350.0) / 2800.0, 0.0, 1.0)
    zcr_n = np.clip(zcr / 0.22, 0.0, 1.0)

    open_score = mid + 0.35 * low
    wide_score = high * (0.45 + 0.55 * zcr_n) * (0.35 + 0.65 * cent_n)
    round_score = low * (1.05 - 0.75 * zcr_n) * (1.05 - 0.65 * cent_n)
    scores = np.stack([open_score, wide_score, round_score], axis=1)
    scores = np.clip(scores, 0.0, None)
    scores /= np.sum(scores, axis=1, keepdims=True) + 1e-6

    flux_scale = float(np.percentile(flux, 90)) + 1e-8
    onset = np.clip(flux / flux_scale, 0.0, 1.0).astype(np.float32)

    jaw = level * (0.42 + 0.58 * np.maximum(scores[:, 0], 0.55 * scores[:, 2] + 0.22 * scores[:, 1]))
    jaw = np.clip(jaw + 0.12 * onset, 0.0, 1.0)
    wide = np.clip(scores[:, 1] * (0.35 + 0.9 * level), 0.0, 1.0)
    roundness = np.clip(scores[:, 2] * (0.35 + 0.9 * level), 0.0, 1.0)
    jaw = np.where(speech, jaw, 0.0)
    wide = np.where(speech, wide, 0.0)
    roundness = np.where(speech, roundness, 0.0)
    # Keep one shape dominant so the mouth doesn't smear every cue at once.
    wide = wide * (1.0 - 0.65 * roundness)
    roundness = roundness * (1.0 - 0.45 * wide)

    hop_s = hop / float(sample_rate)
    jaw = _envelope(jaw.astype(np.float32), hop_s, 0.018, 0.070)
    wide = _envelope(wide.astype(np.float32), hop_s, 0.028, 0.090)
    roundness = _envelope(roundness.astype(np.float32), hop_s, 0.028, 0.090)
    frames = np.stack([jaw, wide, roundness], axis=1).astype(np.float32)
    duration = samples.size / float(sample_rate)
    return VisemeTimeline(hop_s, duration, frames)


def _envelope(values: np.ndarray, hop_s: float, attack_s: float, release_s: float) -> np.ndarray:
    out = np.empty_like(values)
    state = float(values[0]) if values.size else 0.0
    out_list = out
    for i, target in enumerate(values):
        tau = attack_s if target > state else release_s
        alpha = 1.0 - math.exp(-hop_s / tau)
        state += alpha * (float(target) - state)
        out_list[i] = state
    return np.clip(out, 0.0, 1.0)


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
