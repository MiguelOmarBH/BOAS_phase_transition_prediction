"""Load BOAS labels and construct temporally explicit model inputs."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

STAGES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def read_archive(path: Path) -> pd.DataFrame:
    """Restore a compact prediction/label archive without pickled objects."""
    with np.load(path, allow_pickle=False) as archive:
        frame = pd.DataFrame({name: archive[name] for name in archive.files})
    if "night" in frame:
        frame["participant_id"] = "sub-" + frame.pop("night").astype(str)
    return frame


def load_epochs(root: Path) -> pd.DataFrame:
    frame = read_archive(root / "data/metadata/epochs.npz")
    frame["duration"] = 30
    return frame.sort_values(["participant_id", "onset"]).reset_index(drop=True)


def assign_folds(epochs: pd.DataFrame) -> pd.DataFrame:
    """Balance five participant groups on the continuous human-valid population."""
    grouped = epochs.groupby("participant_id", sort=False)
    eligible = (
        epochs.stage_hum.isin(STAGES)
        & grouped.stage_hum.shift(1).isin(STAGES)
        & grouped.stage_hum.shift(-1).isin(STAGES)
        & np.isclose(epochs.onset - grouped.onset.shift(1), 30)
        & np.isclose(grouped.onset.shift(-1) - epochs.onset, 30)
    )
    development = epochs.loc[eligible].copy()
    development["fold"] = -1
    for fold, (_, test) in enumerate(GroupKFold(5).split(development, groups=development.pid)):
        development.iloc[test, development.columns.get_loc("fold")] = fold
    return development[["participant_id", "onset", "pid", "fold"]]


def build_frame(root: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    epochs = load_epochs(root)
    feature_path = root / "data/features/epoch_features.csv.gz"
    if not feature_path.exists():
        raise FileNotFoundError(
            "No extracted features found. Run scripts/download_data.py and "
            "scripts/extract_features.py first, or copy the supplied feature cache."
        )
    features = pd.read_csv(feature_path)
    frame = (
        epochs.merge(features, on=["participant_id", "onset"], validate="one_to_one")
        .sort_values(["participant_id", "onset"])
        .reset_index(drop=True)
    )
    sensors = [c for c in features if c.startswith(("eeg_", "ppg_", "imu_"))]
    added: dict[str, pd.Series] = {}
    context = []
    grouped = frame.groupby("participant_id", sort=False)

    for lag in (0, 1, 2, 4):
        stage = grouped.stage_ai_headband.shift(lag)
        contiguous = np.isclose(frame.onset - grouped.onset.shift(lag), lag * 30)
        stage = stage.where(contiguous)
        for label in STAGES:
            name = f"ctx_ai_lag{lag}_{label}"
            added[name] = (stage == label).astype(float)
            context.append(name)

    added["ctx_elapsed_hours"] = frame.onset / 3600
    context.append("ctx_elapsed_hours")
    runs = []
    for _, night in grouped:
        labels = night.stage_ai_headband.to_numpy()
        onsets = night.onset.to_numpy()
        duration = np.ones(len(night))
        for i in range(1, len(night)):
            if (
                labels[i] in STAGES
                and labels[i] == labels[i - 1]
                and np.isclose(onsets[i] - onsets[i - 1], 30)
            ):
                duration[i] = duration[i - 1] + 1
        runs.extend(duration)
    added["ctx_run_epochs"] = pd.Series(runs, index=frame.index)
    context.append("ctx_run_epochs")

    trajectories = [
        c
        for c in sensors
        if (c.startswith("eeg_") and c.endswith("_mean"))
        or c
        in (
            "eeg_sigma_burst_count",
            "eeg_slow_wave_count",
            "ppg_hr",
            "ppg_rmssd",
            "imu_acc_rms",
            "imu_gyro_rms",
        )
    ]
    for name in trajectories:
        for lag in (1, 2, 4, 8):
            previous = (
                grouped[name]
                .shift(lag)
                .where(np.isclose(frame.onset - grouped.onset.shift(lag), lag * 30))
            )
            added[f"{name}_lag{lag}"] = previous
            if lag in (1, 4, 8):
                added[f"{name}_change{lag}"] = frame[name] - previous

    frame = pd.concat([frame, pd.DataFrame(added)], axis=1)
    current_past = sensors + [c for c in added if c.startswith(("eeg_", "ppg_", "imu_"))] + context
    grouped = frame.groupby("participant_id", sort=False)
    following = grouped.stage_ai_headband.shift(-1).where(
        np.isclose(grouped.onset.shift(-1) - frame.onset, 30)
    )
    future_inputs = {f"ctx_ai_next_{label}": (following == label).astype(float) for label in STAGES}
    for name in trajectories:
        future_inputs[f"{name}_next"] = (
            grouped[name].shift(-1).where(np.isclose(grouped.onset.shift(-1) - frame.onset, 30))
        )

    # Measurements end at onset+30; the target boundary starts at onset+60.
    future_target = grouped.human_transition.shift(-2)
    target_valid = (
        np.isclose(grouped.onset.shift(-2) - frame.onset, 60)
        & grouped.stage_hum.shift(-1).isin(STAGES)
        & grouped.stage_hum.shift(-2).isin(STAGES)
    )
    frame["future_boundary_30s"] = future_target.where(target_valid)
    frame = pd.concat([frame, pd.DataFrame(future_inputs)], axis=1)
    offline = current_past + list(future_inputs)

    folds = assign_folds(epochs).drop(columns="pid")
    frame = frame.merge(folds, on=["participant_id", "onset"], validate="one_to_one")
    frame = frame.loc[frame.stage_ai_headband.isin(STAGES)].reset_index(drop=True)
    if frame.groupby("pid").fold.nunique().max() != 1:
        raise ValueError("A participant crosses outer folds")
    if len(frame) != 114611 or frame.duplicated(["participant_id", "onset"]).any():
        raise ValueError("The paired population does not match the frozen BOAS snapshot")
    for name in set(current_past + offline):
        frame[name] = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)

    columns = {
        "current_past": current_past,
        "offline": offline,
        "context_only": context,
        "physiology_only": [c for c in current_past if c not in context] + ["ctx_elapsed_hours"],
    }
    return frame, columns
