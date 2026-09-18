"""Reproducible local benchmark; does not report fabricated baseline numbers.

Run from ``code`` with an environment that has project requirements installed:
``python -m benchmarks.benchmark_realtime``.
"""
from __future__ import annotations

import statistics
import time

import numpy as np

from realtime_pipeline import SampleRingBuffer
from rf_features import extract_window_features_legacy, extract_window_features_optimized


def measure(name, fn, iterations=200):
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns(); fn(); samples.append(time.perf_counter_ns() - start)
    ms = np.asarray(samples, dtype=np.float64) / 1e6
    print(f"{name:24} n={len(ms):4} mean={ms.mean():8.3f} ms p95={np.percentile(ms,95):8.3f} ms p99={np.percentile(ms,99):8.3f} ms")


def main():
    rng = np.random.default_rng(42)
    window = rng.normal(0, 30, size=(100, 8)).astype(np.float32)
    ring = SampleRingBuffer(8, 1000)
    batch = rng.normal(0, 30, size=(25, 8)).astype(np.float32)
    ring.append(np.tile(batch, (40, 1)))
    legacy = extract_window_features_legacy(window, 500)
    optimized = extract_window_features_optimized(window, 500)
    np.testing.assert_allclose(optimized, legacy, rtol=2e-5, atol=2e-4)
    measure("legacy features", lambda: extract_window_features_legacy(window, 500))
    measure("optimized features", lambda: extract_window_features_optimized(window, 500))
    measure("ring append", lambda: ring.append(batch))
    measure("ring latest window", lambda: ring.latest(100))
    print("Note: window history (100 samples at 500 Hz = 200 ms) is intentionally not included in processing timings.")


if __name__ == "__main__":
    main()
