#!/usr/bin/env python3
"""Descriptive event study with same-night, same-origin-stage stable controls.

Whole-night standardization is used ONLY for effect-size interpretation, never
as classifier input. Offset -2 ends 30 seconds before the labelled boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_epochs

ROOT = Path(__file__).resolve().parents[2]

STAGES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
FEATURES = [
    "eeg_log_alpha_theta_mean",
    "eeg_log_delta_sigma_mean",
    "eeg_rel_delta_mean",
    "eeg_rel_theta_mean",
    "eeg_rel_sigma_mean",
    "eeg_rel_beta_mean",
    "eeg_hf_spectral_slope_mean",
    "eeg_sigma_burst_count",
    "eeg_slow_wave_count",
    "eeg_log_rms_mean",
    "eeg_hf_relative_power_mean",
    "ppg_hr",
    "ppg_hr_acceleration",
    "ppg_rmssd_90s",
    "ppg_interval_outlier_fraction",
    "imu_acc_rms",
    "imu_gyro_rms",
    "eeg_extreme_fraction",
]


def main() -> None:
    base = load_epochs(ROOT)
    features = pd.read_csv(ROOT / "data/features/epoch_features.csv.gz")
    frame = base.merge(features, on=["participant_id", "onset"], validate="one_to_one")
    out = ROOT / "results/retrained"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for night_id, night in frame.groupby("participant_id"):
        night = night.sort_values("onset").set_index("onset")
        stable = night.loc[night.distance_to_human_transition >= 10]
        # Descriptive SD scaling, without transforming classifier inputs.
        sd = night[FEATURES].std().replace(0, np.nan)
        events = night.loc[night.human_transition & night.previous_human_stage.isin(range(5))]
        for t, event in events.iterrows():
            origin = int(event.previous_human_stage)
            dest = int(event.stage_hum)
            controls = stable.loc[stable.stage_hum == origin]
            local = controls.loc[abs(controls.index - t) <= 3600]
            if len(local) >= 3:
                controls = local
            if len(controls) < 3:
                continue
            center = controls[FEATURES].median()
            for offset in [-8, -4, -2, -1]:
                onset = t + 30 * offset
                if onset not in night.index:
                    continue
                record = night.loc[onset]
                # Require an uninterrupted origin-stage run up to the boundary.
                segment = night.loc[(night.index >= onset) & (night.index < t)]
                if len(segment) != -offset or not segment.stage_hum.eq(origin).all():
                    continue
                effects = (record[FEATURES].astype(float) - center) / sd
                for feature, value in effects.items():
                    if np.isfinite(value):
                        rows.append(
                            dict(
                                pid=int(event.pid),
                                participant_id=night_id,
                                boundary_onset=float(t),
                                pair=STAGES[origin] + "→" + STAGES[dest],
                                offset=offset,
                                lead_seconds=-30 * (offset + 1),
                                feature=feature,
                                effect_sd=float(value),
                            )
                        )
    events = pd.DataFrame(rows)
    events.to_csv(out / "physiology_event_effects.csv.gz", index=False, compression="gzip")
    rng = np.random.default_rng(20261013)
    summaries = []
    for pair, subset in [("All", events), *events.groupby("pair")]:
        for (offset, feature), g in subset.groupby(["offset", "feature"]):
            participant = g.groupby("pid").effect_sd.mean().to_numpy()
            if len(participant) < 5:
                continue
            estimates = participant[
                rng.integers(0, len(participant), (2000, len(participant)))
            ].mean(axis=1)
            summaries.append(
                dict(
                    pair=pair,
                    offset=int(offset),
                    lead_seconds=int(g.lead_seconds.iloc[0]),
                    feature=feature,
                    effect_sd=float(participant.mean()),
                    ci_low=float(np.quantile(estimates, 0.025)),
                    ci_high=float(np.quantile(estimates, 0.975)),
                    participants=len(participant),
                    events=len(g),
                )
            )
    pd.DataFrame(summaries).to_csv(out / "physiology_effects.csv", index=False)
    print("Descriptive effects", len(events), "summaries", len(summaries), flush=True)


if __name__ == "__main__":
    main()
