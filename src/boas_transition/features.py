#!/usr/bin/env python3
"""Extract label-free, calibrated BOAS physiology with explicit quality flags.

Burst/wave features are morphology proxies, not expert-validated event labels.
Every feature uses at most the current epoch, ending at onset + 30 seconds.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, signal

from .data import load_epochs
from .edf import _channel_groups, _read_channel, read_header

ROOT = Path(__file__).resolve().parents[2]
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30)}


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return half-open true intervals, including intervals at either edge."""
    d = np.diff(np.r_[False, mask, False].astype(int))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def pulse_features(x: np.ndarray, fs: float) -> dict[str, float]:
    names = [
        "hr",
        "sdnn",
        "rmssd",
        "amplitude",
        "valid_fraction",
        "hr_acceleration",
        "interval_outlier_fraction",
        "peak_count",
    ]
    out = {f"ppg_{n}": np.nan for n in names}
    if len(x) < 10 * fs or fs <= 10 or not np.isfinite(x).all() or np.std(x) == 0:
        return out
    y = signal.sosfiltfilt(
        signal.butter(3, [0.5, min(5, 0.45 * fs)], fs=fs, btype="band", output="sos"), x
    )
    # Fix polarity by larger median prominence rather than larger peak count.
    candidates = [
        signal.find_peaks(
            s * y, distance=max(1, int(0.35 * fs)), prominence=max(np.std(y) * 0.4, 1e-10)
        )
        for s in (1, -1)
    ]
    peaks, props = max(candidates, key=lambda z: np.median(z[1]["prominences"]) if len(z[0]) else 0)
    rr = np.diff(peaks) / fs
    if len(rr) < 8:
        return out
    median = np.median(rr)
    valid = (rr >= 0.35) & (rr <= 1.8) & (abs(rr - median) <= 0.25 * median)
    fraction = valid.mean()
    out["ppg_valid_fraction"] = float(fraction)
    out["ppg_interval_outlier_fraction"] = float(1 - fraction)
    out["ppg_peak_count"] = float(len(peaks))
    out["ppg_amplitude"] = float(np.median(props["prominences"]))
    if fraction < 0.8 or valid.sum() < 8:
        return out
    good = rr[valid]
    out["ppg_hr"] = float(60 / np.median(good))
    out["ppg_sdnn"] = float(np.std(good, ddof=1) * 1000)
    # Do not join RR pairs across rejected intervals.
    pair_valid = valid[1:] & valid[:-1]
    if pair_valid.sum() >= 5:
        out["ppg_rmssd"] = float(np.sqrt(np.mean(np.diff(rr)[pair_valid] ** 2)) * 1000)
    times = peaks[1:] / fs
    slope = np.polyfit(times[valid], 60 / good, 1)[0]
    out["ppg_hr_acceleration"] = float(slope)
    return out


def eeg_features(x: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Vectorized PSD features for (epoch, channel, sample) calibrated microvolts."""
    n = x.shape[0]
    bins = signal.detrend(x.reshape(n, 2, 6, int(5 * fs)), axis=-1)
    f, psd = signal.welch(bins, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), axis=-1)
    total_mask = (f >= 0.5) & (f <= 30)
    total = np.trapezoid(psd[..., total_mask], f[total_mask], axis=-1)
    total = np.maximum(total, 1e-20)
    value: dict[str, np.ndarray] = {}
    relative: dict[str, np.ndarray] = {}
    for name, (a, b) in BANDS.items():
        mask = (f >= a) & (f <= b)
        power = np.trapezoid(psd[..., mask], f[mask], axis=-1)
        relative[name] = (power / total).mean(axis=1)
        value["log_" + name] = np.log10(np.maximum(power, 1e-20)).mean(axis=1)
        value["rel_" + name] = relative[name]
    value["log_alpha_theta"] = np.log((relative["alpha"] + 1e-8) / (relative["theta"] + 1e-8))
    value["log_delta_sigma"] = np.log((relative["delta"] + 1e-8) / (relative["sigma"] + 1e-8))
    value["log_slow_fast"] = np.log(
        (relative["delta"] + relative["theta"] + 1e-8)
        / (relative["alpha"] + relative["beta"] + 1e-8)
    )
    value["log_rms"] = np.log10(np.maximum(np.sqrt(np.mean(bins**2, axis=-1)), 1e-8)).mean(axis=1)
    value["log_ptp"] = np.log10(np.maximum(np.ptp(bins, axis=-1), 1e-8)).mean(axis=1)
    p = psd[..., total_mask].copy()
    p /= np.maximum(p.sum(axis=-1, keepdims=True), 1e-20)
    value["spectral_entropy"] = (
        -np.sum(p * np.log(np.maximum(p, 1e-20)), axis=-1) / np.log(p.shape[-1])
    ).mean(axis=1)
    # Lendner range: descriptive slope only, vulnerable to muscle contamination.
    mask = (f >= 30) & (f <= 45)
    lf = np.log10(f[mask])
    lf -= lf.mean()
    ly = np.log10(np.maximum(psd[..., mask], 1e-20))
    value["hf_spectral_slope"] = (np.sum(ly * lf, axis=-1) / np.sum(lf**2)).mean(axis=1)
    hf = (f >= 30) & (f <= 45)
    value["hf_relative_power"] = (np.trapezoid(psd[..., hf], f[hf], axis=-1) / total).mean(axis=1)
    out: dict[str, np.ndarray] = {}
    t = np.arange(6, dtype=float)
    t -= t.mean()
    for name, v in value.items():
        out["eeg_" + name + "_mean"] = v.mean(axis=1)
        out["eeg_" + name + "_slope"] = np.sum(v * t, axis=1) / np.sum(t**2)
        out["eeg_" + name + "_last_first"] = v[:, -1] - v[:, 0]
    centered = x - x.mean(axis=-1, keepdims=True)
    denom = np.sqrt(np.sum(centered[:, 0] ** 2, axis=-1) * np.sum(centered[:, 1] ** 2, axis=-1))
    out["eeg_correlation"] = np.divide(
        np.sum(centered[:, 0] * centered[:, 1], axis=-1), denom, out=np.zeros(n), where=denom > 0
    )
    out["eeg_extreme_fraction"] = np.mean(abs(centered) > 500, axis=(1, 2))
    out["eeg_flat_fraction"] = np.mean(abs(np.diff(x, axis=-1)) < 1e-5, axis=(1, 2))
    # Sigma envelope burst proxy: no hypnogram mask and no future-night statistics.
    sigma = signal.sosfiltfilt(
        signal.butter(3, [12, 16], fs=fs, btype="band", output="sos"), x, axis=-1
    )
    slow = signal.sosfiltfilt(
        signal.butter(3, [0.5, 4], fs=fs, btype="band", output="sos"), x, axis=-1
    )
    envelope = np.sqrt(
        np.maximum(0, ndimage.uniform_filter1d(sigma**2, size=max(1, int(0.2 * fs)), axis=-1))
    )
    thresholds = np.median(envelope, axis=-1, keepdims=True) * 1.5
    rows = []
    for k in range(n):
        count = occupancy = sw_count = sw_amp = 0.0
        for ch in range(2):
            starts, ends = runs(envelope[k, ch] > thresholds[k, ch, 0])
            lengths = (ends - starts) / fs
            keep = (lengths >= 0.5) & (lengths <= 2)
            count += keep.sum() / 2
            occupancy += lengths[keep].sum() / 60
            starts, ends = runs(slow[k, ch] < 0)
            for a, b in zip(starts, ends):
                d = (b - a) / fs
                if 0.25 <= d <= 1 and b + int(0.5 * fs) < x.shape[-1]:
                    neg = np.min(slow[k, ch, a:b])
                    pos = np.max(slow[k, ch, b : b + int(0.5 * fs)])
                    amp = pos - neg
                    if neg <= -40 and amp >= 75:
                        sw_count += 0.5
                        sw_amp += amp / 2
        rows.append((count, occupancy, sw_count, sw_amp / max(sw_count, 1)))
    for j, name in enumerate(
        ["sigma_burst_count", "sigma_burst_fraction", "slow_wave_count", "slow_wave_amplitude"]
    ):
        out["eeg_" + name] = np.array(rows)[:, j]
    return out


def extract_one(path: str, base_path: str, cache: str) -> str:
    p = Path(path)
    night = p.name.split("_")[0]
    target = Path(cache) / (night + "_features.csv.gz")
    if target.exists():
        return str(target)
    header = read_header(p)
    groups = _channel_groups(header)
    eeg = [_read_channel(header, label) for label in groups["eeg"]]
    if len(eeg) != 2 or eeg[0][1] != eeg[1][1]:
        raise ValueError("Expected two EEG channels at the same rate")
    fs = eeg[0][1]
    arrays = []
    for label, (v, _) in zip(groups["eeg"], eeg):
        dim = (
            next(s.physical_dimension for s in header.signals if s.label == label)
            .lower()
            .replace("µ", "u")
        )
        factor = {"uv": 1, "mv": 1000, "v": 1e6}.get(dim)
        if factor is None:
            raise ValueError(f"Unsupported EEG dimension {dim}")
        arrays.append(v * factor)
    eeg = np.vstack(arrays)
    base = load_epochs(Path(base_path))
    events = base.loc[base.participant_id == night, ["participant_id", "onset", "duration"]].copy()
    start = np.rint(events.onset.to_numpy() * fs).astype(int)
    duration = int(round(30 * fs))
    valid = (start >= 0) & (start + duration <= eeg.shape[1]) & (events.duration.to_numpy() == 30)
    events = events.loc[valid].reset_index(drop=True)
    start = start[valid]
    indices = start[:, None] + np.arange(duration)[None, :]
    x = eeg[:, indices].transpose(1, 0, 2)
    features = eeg_features(x, fs)
    output = pd.concat([events[["participant_id", "onset"]], pd.DataFrame(features)], axis=1)
    for group, prefix in [("accel", "acc"), ("gyro", "gyro")]:
        available = groups[group]
        output["imu_" + prefix + "_available"] = bool(available)
        if not available:
            continue
        rates = [_read_channel(header, label) for label in available]
        if len(set(r for _, r in rates)) != 1:
            raise ValueError("Inconsistent IMU rates")
        r = rates[0][1]
        for name in ["rms", "jerk", "peak", "first_last"]:
            output["imu_" + prefix + "_" + name] = np.nan
        for i, onset in enumerate(events.onset):
            pieces = [v[int(round(onset * r)) : int(round((onset + 30) * r))] for v, _ in rates]
            if not pieces or min(map(len, pieces)) < 3:
                continue
            a = np.vstack(pieces)
            a -= np.median(a, axis=1, keepdims=True)
            mag = np.sqrt(np.sum(a * a, axis=0))
            jerk = np.sqrt(np.sum(np.diff(a, axis=1) ** 2, axis=0)) * r
            vals = [
                np.sqrt(np.mean(mag**2)),
                np.sqrt(np.mean(jerk**2)),
                np.quantile(mag, 0.95),
                np.mean(mag[-int(5 * r) :]) - np.mean(mag[: int(5 * r)]),
            ]
            for name, val in zip(["rms", "jerk", "peak", "first_last"], vals):
                output.loc[i, "imu_" + prefix + "_" + name] = val
    output["ppg_available"] = bool(groups["ppg"])
    if groups["ppg"]:
        v, r = _read_channel(header, groups["ppg"][0])
        pulse = []
        long = []
        for onset in events.onset:
            a = int(round(onset * r))
            b = int(round((onset + 30) * r))
            pulse.append(pulse_features(v[a:b], r))
            past = pulse_features(v[max(0, b - int(90 * r)) : b], r)
            long.append({k + "_90s": val for k, val in past.items()})
        output = pd.concat([output, pd.DataFrame(pulse), pd.DataFrame(long)], axis=1)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp.gz")
    output.to_csv(temp, index=False, compression="gzip")
    temp.replace(target)
    return str(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--base", type=Path, default=ROOT)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/features/cache")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    paths = sorted(args.raw_root.rglob("*headband_eeg.edf"))
    if not paths:
        raise FileNotFoundError("No headband EDFs found")
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        jobs = {ex.submit(extract_one, str(p), str(args.base), str(args.cache)): p for p in paths}
        for fut in as_completed(jobs):
            completed.append(fut.result())
            print(f"[{len(completed)}/{len(paths)}] {jobs[fut].name}", flush=True)
    combined = pd.concat([pd.read_csv(p) for p in completed], ignore_index=True)
    combined = combined.sort_values(["participant_id", "onset"]).reset_index(drop=True)
    schema = json.loads((ROOT / "data/metadata/feature_schema.json").read_text())
    if set(combined.columns) != set(schema):
        raise ValueError("Extracted columns do not match the feature schema")
    combined = combined[schema]
    destination = args.cache.parent / "epoch_features.csv.gz"
    temporary = destination.with_name("epoch_features.tmp.csv.gz")
    combined.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(destination)
    print("Complete", combined.shape, flush=True)


if __name__ == "__main__":
    main()
