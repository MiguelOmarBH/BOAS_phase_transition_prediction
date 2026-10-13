"""Paired stage metrics and uncertainty estimates clustered by participant."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SEED = 20261013


def stage_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    truth = frame.stage_hum.to_numpy(int)
    distance = frame.distance_to_human_transition.to_numpy(float)
    error = prediction != truth
    stable = np.isfinite(distance) & (distance >= 10)
    return {
        "accuracy": float(1 - error.mean()),
        "non_boundary_accuracy": float(1 - error[~frame.human_transition.to_numpy(bool)].mean()),
        "macro_f1": float(f1_score(truth, prediction, labels=range(5), average="macro")),
        "exact_error": float(error[distance == 0].mean()),
        "near_error": float(error[distance <= 1].mean()),
        "stable_error": float(error[stable].mean()),
        "changed_fraction": float((prediction != frame.stage_ai_headband.to_numpy(int)).mean()),
    }


def bootstrap_stage_differences(
    frame: pd.DataFrame, predictions: dict[str, np.ndarray], iterations: int = 2000
) -> pd.DataFrame:
    participants = sorted(frame.pid.unique())
    truth = frame.stage_hum.to_numpy(int)
    distance = frame.distance_to_human_transition.to_numpy(float)
    reference_error = predictions["Bitbrain"] != truth
    masks = {
        "accuracy": np.ones(len(frame), bool),
        "non_boundary_accuracy": ~frame.human_transition.to_numpy(bool),
        "exact_error": distance == 0,
        "near_error": distance <= 1,
        "stable_error": np.isfinite(distance) & (distance >= 10),
    }
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(participants), (iterations, len(participants)))
    rows = []
    for name, prediction in predictions.items():
        if name == "Bitbrain":
            continue
        error = prediction != truth
        for metric, mask in masks.items():
            totals, differences = [], []
            for pid in participants:
                selected = mask & (frame.pid.to_numpy() == pid)
                totals.append(selected.sum())
                differences.append(
                    (error[selected].astype(int) - reference_error[selected].astype(int)).sum()
                )
            totals = np.asarray(totals)
            differences = np.asarray(differences)
            sign = -1 if metric.endswith("accuracy") else 1
            estimates = sign * differences[draws].sum(axis=1) / totals[draws].sum(axis=1)
            rows.append(
                {
                    "model": name,
                    "reference": "Bitbrain",
                    "metric": metric,
                    "difference": float(sign * differences.sum() / totals.sum()),
                    "ci_low": float(np.quantile(estimates, 0.025)),
                    "ci_high": float(np.quantile(estimates, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def directed_errors(frame: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    from .data import STAGES

    rows = []
    for (origin, destination), subset in frame.loc[frame.human_transition].groupby(
        ["previous_human_stage", "stage_hum"]
    ):
        indices = subset.index.to_numpy()
        truth = subset.stage_hum.to_numpy(int)
        row = {
            "transition": STAGES[int(origin)] + "→" + STAGES[int(destination)],
            "epochs": len(subset),
            "participants": subset.pid.nunique(),
        }
        for name, prediction in predictions.items():
            row[name + "_accuracy"] = float((prediction[indices] == truth).mean())
        rows.append(row)
    return pd.DataFrame(rows)
