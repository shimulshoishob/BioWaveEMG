from functools import lru_cache

import numpy as np

FEATURE_EXTRACTOR_VERSION = "rf_features.v1"
FEATURE_SCHEMA_VERSION = "emg-rf-15+rms-ratio+corr.v1"

DEFAULT_NUM_CHANNELS = 4
FFT_MIN_HZ = 20.0
FFT_MAX_HZ = 220.0
BANDS = [(20.0, 60.0), (60.0, 120.0), (120.0, 220.0)]


def _ensure_window_shape(window):
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("window must be 2D")
    # Prefer (samples, channels). If likely transposed, flip.
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.shape[1] <= 0:
        raise ValueError("window must have at least 1 channel")
    return arr


def _spectral_1d(x, sample_rate):
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    if n < 8:
        return {
            "mean_hz": 0.0,
            "median_hz": 0.0,
            "peak_hz": 0.0,
            "spec_entropy": 0.0,
            "band_power_pct": [0.0, 0.0, 0.0],
        }

    xc = x - np.mean(x)
    win = np.hanning(n).astype(np.float32)
    spec = np.abs(np.fft.rfft(xc * win)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))

    mask = (freqs >= FFT_MIN_HZ) & (freqs <= FFT_MAX_HZ)
    if not np.any(mask):
        return {
            "mean_hz": 0.0,
            "median_hz": 0.0,
            "peak_hz": 0.0,
            "spec_entropy": 0.0,
            "band_power_pct": [0.0, 0.0, 0.0],
        }

    sv = spec[mask]
    fv = freqs[mask]
    total = float(np.sum(sv) + 1e-9)

    peak_hz = float(fv[int(np.argmax(sv))])
    mean_hz = float(np.sum(sv * fv) / total)
    csum = np.cumsum(sv)
    med_hz = float(fv[int(np.argmax(csum >= (0.5 * total)))])

    p = sv / total
    spec_entropy = float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p) + 1e-9))

    band_power = []
    for lo, hi in BANDS:
        bmask = (fv >= lo) & (fv < hi)
        if np.any(bmask):
            band_power.append(float(np.sum(sv[bmask]) / total * 100.0))
        else:
            band_power.append(0.0)

    return {
        "mean_hz": mean_hz,
        "median_hz": med_hz,
        "peak_hz": peak_hz,
        "spec_entropy": spec_entropy,
        "band_power_pct": band_power,
    }


def feature_names(num_channels=DEFAULT_NUM_CHANNELS):
    n_ch = int(max(1, num_channels))
    names = []
    per_ch = [
        "mav",
        "rms",
        "iemg",
        "var",
        "wl",
        "zc",
        "ssc",
        "wamp",
        "mean_hz",
        "median_hz",
        "peak_hz",
        "spec_entropy",
        "bp20_60",
        "bp60_120",
        "bp120_220",
    ]
    for ch in range(1, n_ch + 1):
        for k in per_ch:
            names.append(f"ch{ch}_{k}")

    for ch in range(1, n_ch + 1):
        names.append(f"ch{ch}_rms_ratio")

    for a in range(1, n_ch + 1):
        for b in range(a + 1, n_ch + 1):
            names.append(f"corr_ch{a}_ch{b}")
    return names


def extract_window_features_legacy(window, sample_rate=500):
    """Canonical legacy RF implementation used by deployed joblib artifacts.

    Keep its ordering and numerical operations stable. Optimized implementations
    are deliberately opt-in and tested against this reference.
    """
    arr = _ensure_window_shape(window)
    n_samples = arr.shape[0]
    n_ch = arr.shape[1]
    arr_centered = arr - np.mean(arr, axis=0, keepdims=True)

    zc_thresh = 10.0
    ssc_thresh = 8.0
    wamp_thresh = 12.0

    feats = []
    rms_vals = []
    for ch in range(n_ch):
        x = arr_centered[:, ch]
        abs_x = np.abs(x)
        dx = np.diff(x) if n_samples > 1 else np.array([], dtype=np.float32)

        mav = float(np.mean(abs_x))
        rms = float(np.sqrt(np.mean(np.square(x))))
        iemg = float(np.sum(abs_x))
        var = float(np.var(x))
        wl = float(np.sum(np.abs(dx))) if dx.size else 0.0

        if n_samples > 1:
            zc = int(np.sum(((x[:-1] * x[1:]) < 0) & (np.abs(x[:-1] - x[1:]) >= zc_thresh)))
            wamp = int(np.sum(np.abs(x[1:] - x[:-1]) >= wamp_thresh))
        else:
            zc = 0
            wamp = 0

        if n_samples > 2:
            s1 = x[1:-1] - x[:-2]
            s2 = x[1:-1] - x[2:]
            ssc = int(np.sum(((s1 * s2) > 0) & ((np.abs(s1) + np.abs(s2)) >= ssc_thresh)))
        else:
            ssc = 0

        sp = _spectral_1d(x, sample_rate)
        feats.extend(
            [
                mav,
                rms,
                iemg,
                var,
                wl,
                float(zc),
                float(ssc),
                float(wamp),
                sp["mean_hz"],
                sp["median_hz"],
                sp["peak_hz"],
                sp["spec_entropy"],
                sp["band_power_pct"][0],
                sp["band_power_pct"][1],
                sp["band_power_pct"][2],
            ]
        )
        rms_vals.append(rms)

    rms_vals = np.asarray(rms_vals, dtype=np.float32)
    mean_rms = float(np.mean(rms_vals) + 1e-9)
    feats.extend((rms_vals / mean_rms).tolist())

    # Pairwise channel correlation features.
    std = np.std(arr_centered, axis=0)
    valid = np.isfinite(std) & (std > 1e-8)
    if np.any(valid):
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(arr_centered.T)
    else:
        corr = np.eye(n_ch, dtype=np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(valid):
        corr[~valid, :] = 0.0
        corr[:, ~valid] = 0.0
        np.fill_diagonal(corr, 1.0)
    for a in range(n_ch):
        for b in range(a + 1, n_ch):
            feats.append(float(corr[a, b]))

    return np.asarray(feats, dtype=np.float32)


# Public legacy-compatible entry point. Existing training scripts and joblib
# metadata refer to this exact import path.
extract_window_features = extract_window_features_legacy


@lru_cache(maxsize=32)
def _spectral_constants(n_samples, sample_rate):
    """Immutable FFT constants reused by the optional optimized path."""
    n = int(n_samples)
    rate = float(sample_rate)
    return np.hanning(n).astype(np.float32), np.fft.rfftfreq(n, d=1.0 / rate)


def extract_window_features_optimized(window, sample_rate=500):
    """Low-allocation equivalent of :func:`extract_window_features_legacy`.

    FFTs remain NumPy operations because NumPy's platform FFT backend is both
    mature and more reliable than attempting an incomplete Numba FFT. The
    Hanning window and frequency bins are cached. This function is opt-in until
    feature-equivalence tests have passed in the target runtime.
    """
    arr = _ensure_window_shape(window)
    n_samples, n_ch = arr.shape
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    hanning, freqs = _spectral_constants(n_samples, float(sample_rate))
    out = np.empty(15 * n_ch + n_ch + n_ch * (n_ch - 1) // 2, dtype=np.float32)
    rms_values = np.empty(n_ch, dtype=np.float32)
    write = 0
    for ch in range(n_ch):
        x = centered[:, ch]
        abs_x = np.abs(x)
        delta = np.diff(x) if n_samples > 1 else np.empty(0, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(x))))
        rms_values[ch] = rms
        if n_samples > 1:
            zc = float(np.sum(((x[:-1] * x[1:]) < 0) & (np.abs(x[:-1] - x[1:]) >= 10.0)))
            wamp = float(np.sum(np.abs(x[1:] - x[:-1]) >= 12.0))
        else:
            zc = wamp = 0.0
        if n_samples > 2:
            a, b = x[1:-1] - x[:-2], x[1:-1] - x[2:]
            ssc = float(np.sum(((a * b) > 0) & ((np.abs(a) + np.abs(b)) >= 8.0)))
        else:
            ssc = 0.0
        spectrum = np.abs(np.fft.rfft(x * hanning)) ** 2
        mask = (freqs >= FFT_MIN_HZ) & (freqs <= FFT_MAX_HZ)
        if np.any(mask):
            power, f = spectrum[mask], freqs[mask]
            total = float(np.sum(power) + 1e-9)
            p = power / total
            csum = np.cumsum(power)
            spectral = (
                float(np.sum(power * f) / total), float(f[int(np.argmax(csum >= .5 * total))]),
                float(f[int(np.argmax(power))]),
                float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p) + 1e-9)),
                *[float(np.sum(power[(f >= lo) & (f < hi)]) / total * 100.0) if np.any((f >= lo) & (f < hi)) else 0.0 for lo, hi in BANDS],
            )
        else:
            spectral = (0.0,) * 7
        out[write:write + 15] = (float(np.mean(abs_x)), rms, float(np.sum(abs_x)), float(np.var(x)),
                                  float(np.sum(np.abs(delta))) if delta.size else 0.0, zc, ssc, wamp, *spectral)
        write += 15
    out[write:write + n_ch] = rms_values / (float(np.mean(rms_values)) + 1e-9)
    write += n_ch
    std = np.std(centered, axis=0)
    valid = np.isfinite(std) & (std > 1e-8)
    corr = np.corrcoef(centered.T) if np.any(valid) else np.eye(n_ch, dtype=np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(valid):
        corr[~valid, :] = 0.0; corr[:, ~valid] = 0.0; np.fill_diagonal(corr, 1.0)
    for first in range(n_ch):
        for second in range(first + 1, n_ch):
            out[write] = corr[first, second]; write += 1
    return out


def build_windows_from_sequence(sequence, win_samples, stride_samples):
    arr = _ensure_window_shape(sequence)
    n = arr.shape[0]
    if n < win_samples:
        return []
    out = []
    for s in range(0, n - win_samples + 1, stride_samples):
        out.append(arr[s:s + win_samples, :])
    return out
