"""Canonical real-time primitives for BioWave applications.

No Qt widgets, disk I/O, sockets, or model training live here.  The classes
are deliberately in-process: EMG windows are small and copying them through a
multiprocessing queue is normally more expensive than the compute itself.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from rf_features import FEATURE_EXTRACTOR_VERSION, extract_window_features_legacy

LOG = logging.getLogger("biowave.realtime")

try:  # Optional: the functional path does not require Numba.
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*_args, **_kwargs):
        return lambda func: func


@njit(fastmath=True, nogil=True, cache=True)
def _time_features_kernel(centered: np.ndarray, output: np.ndarray) -> None:
    """Compiled MAV/RMS/IEMG/variance/WL/ZC/SSC/WAMP kernel.

    FFT features retain NumPy's tested FFT implementation. This kernel is
    opt-in; it never changes the legacy model feature schema.
    """
    n, channels = centered.shape
    for ch in range(channels):
        total = 0.0; squared = 0.0; absolute = 0.0; wl = 0.0
        zc = 0.0; ssc = 0.0; wamp = 0.0
        for i in range(n):
            x = centered[i, ch]
            total += x; squared += x * x; absolute += abs(x)
            if i:
                prev = centered[i - 1, ch]
                delta = abs(x - prev)
                wl += delta
                if prev * x < 0.0 and delta >= 10.0: zc += 1.0
                if delta >= 12.0: wamp += 1.0
            if i > 1:
                a = centered[i - 1, ch] - centered[i - 2, ch]
                b = centered[i - 1, ch] - x
                if a * b > 0.0 and abs(a) + abs(b) >= 8.0: ssc += 1.0
        mean = total / n
        offset = ch * 8
        output[offset] = absolute / n
        output[offset + 1] = (squared / n) ** .5
        output[offset + 2] = absolute
        output[offset + 3] = squared / n - mean * mean
        output[offset + 4] = wl; output[offset + 5] = zc
        output[offset + 6] = ssc; output[offset + 7] = wamp


class SampleRingBuffer:
    """Fixed-size, single-producer ring buffer with explicit gap recovery.

    The caller obtains one chronological copy only when an inference/analysis
    window is actually required; ingest never shifts the complete history.
    """
    def __init__(self, channels: int, capacity: int, dtype=np.float32) -> None:
        if channels <= 0 or capacity <= 0:
            raise ValueError("channels and capacity must be positive")
        self.channels, self.capacity = int(channels), int(capacity)
        self.data = np.empty((self.capacity, self.channels), dtype=dtype)
        self.write_index = 0
        self.sample_count = 0
        self.generation = 0
        self.invalid_until_generation = 0

    def reset(self) -> None:
        self.write_index = self.sample_count = self.generation = self.invalid_until_generation = 0

    def append(self, samples: np.ndarray, *, discontinuity: bool = False) -> int:
        x = np.asarray(samples, dtype=self.data.dtype)
        if x.ndim != 2 or x.shape[1] != self.channels:
            raise ValueError("sample shape does not match ring channels")
        n = len(x)
        if not n: return 0
        if discontinuity:
            # Never synthesize data; require a clean subsequent window.
            self.sample_count = 0
            self.invalid_until_generation = self.generation + self.capacity
        if n >= self.capacity:
            self.data[:, :] = x[-self.capacity:]
            self.write_index = 0
            self.sample_count = self.capacity
        else:
            first = min(n, self.capacity - self.write_index)
            self.data[self.write_index:self.write_index + first] = x[:first]
            remaining = n - first
            if remaining: self.data[:remaining] = x[first:]
            self.write_index = (self.write_index + n) % self.capacity
            self.sample_count = min(self.capacity, self.sample_count + n)
        self.generation += n
        return n

    def has_window(self, size: int) -> bool:
        return 0 < size <= self.capacity and self.sample_count >= size and self.generation >= self.invalid_until_generation

    def latest(self, size: int, out: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.has_window(size):
            raise ValueError("no continuous window available")
        if out is None:
            out = np.empty((size, self.channels), dtype=self.data.dtype)
        if out.shape != (size, self.channels):
            raise ValueError("output window shape mismatch")
        start = (self.write_index - size) % self.capacity
        first = min(size, self.capacity - start)
        out[:first] = self.data[start:start + first]
        if first < size: out[first:] = self.data[:size - first]
        return out


@dataclass
class LatencySnapshot:
    count: int = 0
    min_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0


class StageProfiler:
    """Bounded high-resolution stage timings; no I/O on the hot path."""
    def __init__(self, history: int = 2048) -> None:
        self._samples: dict[str, deque[int]] = {}
        self._history = history

    @staticmethod
    def now_ns() -> int: return time.perf_counter_ns()

    def record_ns(self, stage: str, start_ns: int, end_ns: Optional[int] = None) -> None:
        elapsed = (self.now_ns() if end_ns is None else end_ns) - start_ns
        self._samples.setdefault(stage, deque(maxlen=self._history)).append(max(0, elapsed))

    def snapshot(self) -> dict[str, LatencySnapshot]:
        answer = {}
        for stage, values in self._samples.items():
            a = np.asarray(values, dtype=np.float64) / 1_000_000.0
            answer[stage] = LatencySnapshot(len(a), float(a.min()), float(a.mean()), float(np.percentile(a, 50)),
                                            float(np.percentile(a, 95)), float(np.percentile(a, 99)), float(a.max()))
        return answer


class LatestWindowMailbox:
    """Single-slot handoff. Producer overwrites stale work; consumer sees newest."""
    def __init__(self) -> None:
        self._lock = threading.Lock(); self._event = threading.Event(); self._value: Any = None
        self.dropped = 0

    def submit(self, value: Any) -> None:
        with self._lock:
            if self._value is not None: self.dropped += 1
            self._value = value
        self._event.set()

    def take(self, timeout_s: float = .1) -> Any:
        if not self._event.wait(timeout_s): return None
        with self._lock:
            value, self._value = self._value, None
            if self._value is None: self._event.clear()
            return value


class SklearnBackend:
    """Compatibility backend for existing joblib/scikit-learn artifacts."""
    name = "sklearn"
    def __init__(self, model: Any, class_names: list[str]) -> None:
        self.model, self.class_names = model, [str(x) for x in class_names]
    def infer(self, features: np.ndarray) -> tuple[str, Optional[float], Optional[float], np.ndarray]:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            p = np.asarray(self.model.predict_proba(x)[0], dtype=np.float32)
            i = int(np.argmax(p)); ordered = np.sort(p)
            label = self.class_names[i] if i < len(self.class_names) else str(getattr(self.model, "classes_", [i])[i])
            return label, float(p[i]), float(ordered[-1] - ordered[-2]) if len(p) > 1 else float(p[i]), p
        raw = self.model.predict(x)[0]
        label = self.class_names[int(raw)] if isinstance(raw, (int, np.integer)) and 0 <= int(raw) < len(self.class_names) else str(raw)
        return label, None, None, np.empty(0, dtype=np.float32)
