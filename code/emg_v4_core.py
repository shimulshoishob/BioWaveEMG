"""Safety-critical, GUI-independent building blocks for the BioWave V4 controller.

This module deliberately has no Qt, serial, UDP socket, or mouse dependency so it
can be replayed and unit tested.  The V3 feature extractor remains the default for
old models; V4 preprocessing must be explicitly declared by model metadata.
"""
from __future__ import annotations

import logging
import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

PREPROCESSING_VERSION = "v2"
LEGACY_PREPROCESSING_VERSION = "legacy-baseline-v1"
FEATURE_EXTRACTOR_VERSION = "rf_features.v1"
LOG = logging.getLogger("biowave.v4")


@dataclass
class SampleBatch:
    """Samples plus transport metadata. Values are never silently discarded."""
    samples: np.ndarray
    packet_sequences: Optional[np.ndarray] = None
    frame_ids: Optional[np.ndarray] = None
    emg_timestamps: Optional[np.ndarray] = None
    imu_ids: Optional[np.ndarray] = None
    imu_timestamps: Optional[np.ndarray] = None
    host_received_monotonic: float = field(default_factory=time.monotonic)
    gap_before: int = 0
    invalid: bool = False


@dataclass
class WirelessStats:
    packets_received: int = 0
    packets_missing: int = 0
    packets_out_of_order: int = 0
    invalid_packets: int = 0
    previous_sequence: Optional[int] = None
    last_arrival: Optional[float] = None
    arrival_intervals: deque = field(default_factory=lambda: deque(maxlen=200))

    def observe(self, sequence: int, arrival: Optional[float] = None) -> int:
        """Record one UDP packet and return the number missing before it.

        UDP sequence numbers are unsigned 32-bit; wrap-around is handled.  A
        duplicate/out-of-order packet is retained in diagnostics but never used
        to make a gap look like valid continuous data.
        """
        arrival = time.monotonic() if arrival is None else arrival
        self.packets_received += 1
        gap = 0
        if self.previous_sequence is not None:
            delta = (int(sequence) - self.previous_sequence) & 0xFFFFFFFF
            if delta == 0 or delta > 0x7FFFFFFF:
                self.packets_out_of_order += 1
            elif delta > 1:
                gap = delta - 1
                self.packets_missing += gap
        if self.last_arrival is not None:
            self.arrival_intervals.append(arrival - self.last_arrival)
        self.previous_sequence = int(sequence)
        self.last_arrival = arrival
        return gap

    @property
    def packet_loss_percent(self) -> float:
        total = self.packets_received + self.packets_missing
        return 100.0 * self.packets_missing / total if total else 0.0

    @property
    def jitter_ms(self) -> float:
        return float(np.std(self.arrival_intervals) * 1000.0) if len(self.arrival_intervals) > 1 else 0.0


@dataclass
class CalibrationProfile:
    rest_baseline: np.ndarray
    rest_std: np.ndarray
    rest_rms: np.ndarray
    flex_rms: np.ndarray
    flex_peak: np.ndarray
    activation_scale: np.ndarray
    quality: list[str]
    valid: bool
    reasons: list[str]
    normalization: str = "activation_rms"


def compute_calibration(rest: np.ndarray, flex: np.ndarray, *, min_samples: int = 250,
                        dead_std: float = 1e-3, rest_noise_limit: float = 80.0,
                        min_activation_ratio: float = 1.25,
                        saturation_limit: Optional[float] = 65534.0) -> CalibrationProfile:
    """Quantify calibration, rejecting unsafe captures rather than guessing."""
    rest = np.asarray(rest, dtype=np.float32)
    flex = np.asarray(flex, dtype=np.float32)
    if rest.ndim != 2 or flex.ndim != 2 or rest.shape[1] != flex.shape[1]:
        raise ValueError("REST and FLEX must be 2-D with identical channel count")
    channels = rest.shape[1]
    if rest.shape[0] < min_samples or flex.shape[0] < min_samples:
        zeros = np.zeros(channels, dtype=np.float32)
        return CalibrationProfile(zeros, zeros, zeros, zeros, zeros, np.ones(channels),
                                  ["INSUFFICIENT"] * channels, False,
                                  ["Insufficient REST or FLEX samples"])
    baseline = np.median(rest, axis=0)
    rest_centered = rest - baseline
    flex_centered = flex - baseline
    rest_std = np.std(rest_centered, axis=0)
    rest_rms = np.sqrt(np.mean(rest_centered ** 2, axis=0))
    flex_rms = np.sqrt(np.mean(flex_centered ** 2, axis=0))
    flex_peak = np.max(np.abs(flex_centered), axis=0)
    # RMS activation is robust against one transient; epsilon prevents divide-by-zero.
    scale = np.maximum(flex_rms, 1e-6)
    quality, reasons = [], []
    finite = np.isfinite(rest).all(axis=0) & np.isfinite(flex).all(axis=0)
    for ch in range(channels):
        if not finite[ch]:
            quality.append("INVALID"); reasons.append(f"CH{ch + 1}: NaN/Inf")
        elif saturation_limit is not None and ((np.abs(rest[:, ch]) >= saturation_limit).any() or (np.abs(flex[:, ch]) >= saturation_limit).any()):
            quality.append("SATURATED"); reasons.append(f"CH{ch + 1}: saturated")
        elif rest_std[ch] <= dead_std and flex_rms[ch] <= dead_std:
            quality.append("DEAD"); reasons.append(f"CH{ch + 1}: no measurable signal")
        elif rest_std[ch] > rest_noise_limit:
            quality.append("NOISY"); reasons.append(f"CH{ch + 1}: unstable REST")
        elif flex_rms[ch] < max(rest_rms[ch] * min_activation_ratio, dead_std * 2):
            quality.append("WEAK"); reasons.append(f"CH{ch + 1}: insufficient FLEX activation")
        else:
            quality.append("GOOD")
    valid = all(q == "GOOD" for q in quality)
    return CalibrationProfile(baseline.astype(np.float32), rest_std.astype(np.float32), rest_rms.astype(np.float32),
                              flex_rms.astype(np.float32), flex_peak.astype(np.float32), scale.astype(np.float32),
                              quality, valid, reasons)


@dataclass
class PreprocessingConfig:
    version: str = PREPROCESSING_VERSION
    sample_rate: float = 500.0
    highpass_hz: float = 20.0
    lowpass_hz: float = 220.0
    notch_hz: float = 50.0
    notch_q: float = 30.0
    normalization: str = "activation_rms"  # "none" retains centered ADC scale.


class RealTimePreprocessor:
    """Causal, stateful band-pass + notch processor.

    Cascaded one-pole high/low pass filters and a normalized RBJ notch are
    stable for the configured 500 Hz stream.  States persist across batches,
    avoiding the non-causal look-ahead and edge artifacts of ``filtfilt``.
    """
    def __init__(self, channels: int, config: PreprocessingConfig, profile: CalibrationProfile):
        self.channels, self.config, self.profile = int(channels), config, profile
        if not 0 < config.highpass_hz < config.lowpass_hz < config.sample_rate / 2:
            raise ValueError("invalid causal band-pass configuration")
        self._hp_x = np.zeros(channels, np.float64); self._hp_y = np.zeros(channels, np.float64)
        self._lp_y = np.zeros(channels, np.float64)
        self._z1 = np.zeros(channels, np.float64); self._z2 = np.zeros(channels, np.float64)
        dt = 1.0 / config.sample_rate
        self._hp_a = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * config.highpass_hz * dt))
        self._lp_a = dt / ((1.0 / (2.0 * math.pi * config.lowpass_hz)) + dt)
        w0 = 2.0 * math.pi * config.notch_hz / config.sample_rate
        alpha = math.sin(w0) / (2.0 * config.notch_q)
        b0, b1, b2, a0, a1, a2 = 1, -2 * math.cos(w0), 1, 1 + alpha, -2 * math.cos(w0), 1 - alpha
        self._b0, self._b1, self._b2 = b0 / a0, b1 / a0, b2 / a0
        self._a1, self._a2 = a1 / a0, a2 / a0

    def process(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.channels or not np.isfinite(x).all():
            raise ValueError("invalid sample batch for preprocessing")
        out = np.empty_like(x)
        baseline = self.profile.rest_baseline
        scale = self.profile.activation_scale
        for i, row in enumerate(x):
            centered = row.astype(np.float64) - baseline
            hp = self._hp_a * (self._hp_y + centered - self._hp_x)
            self._hp_x, self._hp_y = centered, hp
            lp = self._lp_y + self._lp_a * (hp - self._lp_y)
            self._lp_y = lp
            y = self._b0 * lp + self._z1
            self._z1 = self._b1 * lp - self._a1 * y + self._z2
            self._z2 = self._b2 * lp - self._a2 * y
            out[i] = y / scale if self.config.normalization != "none" else y
        return out


@dataclass
class SignalQuality:
    state: str
    channel_states: list[str]
    reason: str = ""


def assess_signal_quality(window: np.ndarray, profile: CalibrationProfile, *, saturation_limit: float = 65534.0,
                          max_noise_multiplier: float = 8.0, min_active_std: float = 1e-4) -> SignalQuality:
    x = np.asarray(window, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(profile.quality) or not np.isfinite(x).all():
        return SignalQuality("SIGNAL_POOR", ["INVALID"] * len(profile.quality), "missing or non-finite samples")
    states = []
    for ch in range(x.shape[1]):
        std, peak = float(np.std(x[:, ch])), float(np.max(np.abs(x[:, ch])))
        if peak >= saturation_limit: states.append("SATURATED")
        elif std <= min_active_std: states.append("DEAD")
        elif profile.rest_std[ch] > 0 and std > max_noise_multiplier * max(profile.flex_rms[ch], profile.rest_std[ch]): states.append("NOISY")
        else: states.append(profile.quality[ch] if profile.quality[ch] != "GOOD" else "GOOD")
    bad = [s for s in states if s != "GOOD"]
    return SignalQuality("GOOD" if not bad else "SIGNAL_POOR", states, ", ".join(bad))


@dataclass
class ModelCompatibility:
    compatible: bool
    warnings: list[str]
    errors: list[str]
    preprocessing_version: str


def expected_feature_count(channels: int) -> int:
    return 15 * channels + channels + channels * (channels - 1) // 2


def validate_model_artifact(artifact: Any, acquisition_rate: Optional[float] = None,
                            acquisition_channels: Optional[int] = None) -> ModelCompatibility:
    warnings, errors = [], []
    if not isinstance(artifact, dict) or "model" not in artifact:
        return ModelCompatibility(False, warnings, ["Joblib artifact must contain a model"], "unknown")
    model = artifact["model"]
    if not hasattr(model, "predict"):
        errors.append("Model has no predict()")
    classes = artifact.get("class_names", artifact.get("classes", getattr(model, "classes_", [])))
    if not list(classes): errors.append("Model classes are missing")
    version = artifact.get("preprocessing_version", LEGACY_PREPROCESSING_VERSION)
    if "preprocessing_version" not in artifact: warnings.append("Model preprocessing metadata is missing; using legacy baseline compatibility mode.")
    if "input_channels" not in artifact: warnings.append("Model input channel metadata is missing; feature count will be used when available.")
    rate = artifact.get("sample_rate")
    if rate is None: warnings.append("Model sample-rate metadata is missing; compatibility cannot be fully verified.")
    elif acquisition_rate is not None and not math.isclose(float(rate), float(acquisition_rate), rel_tol=0, abs_tol=0.01):
        errors.append(f"Model sample rate ({rate} Hz) differs from acquisition ({acquisition_rate} Hz); resampling is not enabled.")
    channels = artifact.get("input_channels")
    if channels is not None and acquisition_channels is not None and int(channels) != int(acquisition_channels):
        errors.append(f"Model expects {channels} EMG channels, device supplies {acquisition_channels}.")
    count = getattr(model, "n_features_in_", None)
    if count is not None and channels is not None and int(count) != expected_feature_count(int(channels)):
        errors.append(f"Model has {count} features; expected {expected_feature_count(int(channels))} for declared channel order.")
    return ModelCompatibility(not errors, warnings, errors, version)


@dataclass
class Decision:
    state: str
    action_allowed: bool
    click_triggered: bool = False


class GestureDecisionEngine:
    """Temporal gate: actions require sustained, confident, quality-approved labels."""
    def __init__(self, *, vote_windows: int = 5, consecutive_required: int = 3,
                 min_confidence: float = .65, min_margin: float = .10, refractory_s: float = .8):
        self.history = deque(maxlen=vote_windows); self.consecutive_required = consecutive_required
        self.min_confidence, self.min_margin, self.refractory_s = min_confidence, min_margin, refractory_s
        self.active: Optional[str] = None; self.candidate: Optional[str] = None; self.candidate_count = 0
        self.last_click = -float("inf")

    def update(self, label: Optional[str], confidence: Optional[float], margin: Optional[float], quality_good: bool,
               now: Optional[float] = None) -> Decision:
        now = time.monotonic() if now is None else now
        valid = quality_good and label not in (None, "", "REST", "UNKNOWN") and confidence is not None and confidence >= self.min_confidence and (margin is None or margin >= self.min_margin)
        if not valid:
            self.history.clear(); self.candidate = None; self.candidate_count = 0; self.active = None
            return Decision("REST" if label == "REST" and quality_good else "UNKNOWN", False)
        self.history.append(label)
        voted = Counter(self.history).most_common(1)[0][0]
        if voted != self.candidate:
            self.candidate, self.candidate_count = voted, 1
        else: self.candidate_count += 1
        if self.candidate_count < self.consecutive_required:
            return Decision("GESTURE_CANDIDATE", False)
        if self.active != voted:
            self.active = voted
            return Decision(f"{voted}_ACTIVE", True)
        return Decision(f"{voted}_HELD", True)

    def trigger_click_once(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        if now - self.last_click < self.refractory_s: return False
        self.last_click = now
        return True


class SessionRecorder:
    """Records raw batches and metadata for deterministic offline replay.

    The recorder intentionally stores raw values, not filtered values, so new
    preprocessing algorithms can be compared against the same acquisition.
    ``.npz`` is used to avoid adding a database or file-format dependency.
    """
    def __init__(self) -> None:
        self._batches: list[SampleBatch] = []

    def append(self, batch: SampleBatch) -> None:
        self._batches.append(batch)

    def save(self, path: str) -> None:
        if not self._batches:
            raise ValueError("no samples recorded")
        samples = np.vstack([b.samples for b in self._batches])
        def concat(name: str, dtype: Any) -> np.ndarray:
            values = [getattr(b, name) for b in self._batches]
            return np.concatenate([v if v is not None else np.full(len(b.samples), -1, dtype=dtype)
                                   for b, v in zip(self._batches, values)])
        np.savez_compressed(path, samples=samples,
                            packet_sequences=concat("packet_sequences", np.int64),
                            frame_ids=concat("frame_ids", np.int64), emg_timestamps=concat("emg_timestamps", np.int64),
                            imu_ids=concat("imu_ids", np.int64), imu_timestamps=concat("imu_timestamps", np.int64))

    @staticmethod
    def replay(path: str, batch_size: int = 25):
        """Yield SampleBatch objects in their original sample order."""
        with np.load(path) as data:
            samples = data["samples"]
            for start in range(0, len(samples), batch_size):
                end = start + batch_size
                yield SampleBatch(samples[start:end], data["packet_sequences"][start:end],
                                  data["frame_ids"][start:end], data["emg_timestamps"][start:end],
                                  data["imu_ids"][start:end], data["imu_timestamps"][start:end])
