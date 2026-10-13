"""Selective correction of released Bitbrain labels, with nested grouped validation."""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from .evaluation import SEED, stage_metrics

PARAMETERS = {
    "learning_rate": 0.07,
    "max_iter": 150,
    "max_leaf_nodes": 25,
    "min_samples_leaf": 40,
    "l2_regularization": 3.0,
    "early_stopping": False,
}
THRESHOLDS = np.r_[np.arange(0, 0.81, 0.05), np.inf]


def fit_stage(
    frame: pd.DataFrame,
    train: np.ndarray,
    test: np.ndarray,
    columns: list[str],
    seed: int,
) -> tuple[HistGradientBoostingClassifier, np.ndarray, np.ndarray]:
    labels = frame.iloc[train].stage_hum.to_numpy(int)
    counts = np.bincount(labels, minlength=5)
    weights = np.sqrt(len(labels) / (5 * np.maximum(counts, 1)))[labels]
    weights *= np.where(frame.iloc[train].human_transition.to_numpy(bool), 2.0, 1.0)
    weights /= weights.mean()
    model = HistGradientBoostingClassifier(**PARAMETERS, random_state=seed)
    model.fit(frame.iloc[train][columns], labels, sample_weight=weights)
    raw = model.predict_proba(frame.iloc[test][columns])
    probabilities = np.zeros((len(test), 5))
    probabilities[:, model.classes_.astype(int)] = raw
    reference = frame.iloc[test].stage_ai_headband.to_numpy(int)
    candidate = probabilities.argmax(axis=1)
    advantage = (
        probabilities[np.arange(len(test)), candidate]
        - probabilities[np.arange(len(test)), reference]
    )
    return model, candidate, advantage


def apply_corrections(
    reference: np.ndarray, candidate: np.ndarray, advantage: np.ndarray, threshold: float
) -> np.ndarray:
    output = reference.copy()
    selected = (advantage > threshold) & (candidate != reference)
    output[selected] = candidate[selected]
    return output


def select_threshold(
    frame: pd.DataFrame, candidate: np.ndarray, advantage: np.ndarray
) -> tuple[float, pd.DataFrame]:
    reference = frame.stage_ai_headband.to_numpy(int)
    baseline = stage_metrics(frame, reference)
    rows = []
    for threshold in THRESHOLDS:
        prediction = apply_corrections(reference, candidate, advantage, threshold)
        values = stage_metrics(frame, prediction)
        eligible = (
            values["accuracy"] >= baseline["accuracy"] - 0.002
            and values["stable_error"] <= baseline["stable_error"] + 0.005
        )
        rows.append({"threshold": float(threshold), "eligible": eligible, **values})
    grid = pd.DataFrame(rows)
    # On a tie, keep the larger margin and make fewer changes.
    selected = (
        grid.loc[grid.eligible]
        .sort_values(["exact_error", "threshold"], ascending=[True, False])
        .iloc[0]
    )
    return float(selected.threshold), grid


def nested_staging(frame: pd.DataFrame, columns: list[str], name: str, output: Path) -> np.ndarray:
    output.mkdir(parents=True, exist_ok=True)
    model_directory = output.parent / "models"
    model_directory.mkdir(exist_ok=True)
    prediction = np.full(len(frame), -1)
    selections = []
    for fold in sorted(frame.fold.unique()):
        train = np.flatnonzero(frame.fold.to_numpy() != fold)
        test = np.flatnonzero(frame.fold.to_numpy() == fold)
        candidate = np.full(len(train), -1)
        advantage = np.full(len(train), np.nan)
        inner = frame.iloc[train]
        started = time.monotonic()
        for inner_fold, (inner_train, inner_valid) in enumerate(
            GroupKFold(3).split(inner, groups=inner.pid)
        ):
            if set(inner.iloc[inner_train].pid) & set(inner.iloc[inner_valid].pid):
                raise ValueError("A participant crosses inner folds")
            _, labels, scores = fit_stage(
                frame,
                train[inner_train],
                train[inner_valid],
                columns,
                SEED + fold * 10 + inner_fold,
            )
            candidate[inner_valid] = labels
            advantage[inner_valid] = scores
        if np.isnan(advantage).any():
            raise ValueError("Incomplete inner predictions")
        threshold, grid = select_threshold(inner, candidate, advantage)
        grid.to_csv(output / f"{name}_fold{fold}_inner_grid.csv", index=False)
        model, candidate, advantage = fit_stage(frame, train, test, columns, SEED + 100 + fold)
        prediction[test] = apply_corrections(
            frame.iloc[test].stage_ai_headband.to_numpy(int), candidate, advantage, threshold
        )
        joblib.dump(
            {
                "model": model,
                "feature_columns": columns,
                "threshold": threshold,
                "training_pid": sorted(frame.iloc[train].pid.unique().tolist()),
                "test_pid": sorted(frame.iloc[test].pid.unique().tolist()),
            },
            model_directory / f"{name}_fold{fold}.joblib",
            compress=3,
        )
        selections.append(
            {
                "model": name,
                "fold": int(fold),
                "threshold": threshold,
                **stage_metrics(frame.iloc[test], prediction[test]),
            }
        )
        pd.DataFrame(selections).to_csv(output / f"{name}_nested_selection.csv", index=False)
        print(
            f"{name}: fold {fold + 1}/5, margin={threshold:.2f}, {time.monotonic() - started:.0f}s",
            flush=True,
        )
    if (prediction < 0).any():
        raise ValueError("Incomplete outer predictions")
    return prediction


def forecast_boundaries(
    frame: pd.DataFrame, feature_sets: dict[str, list[str]], output: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = frame.loc[frame.future_boundary_30s.notna()].reset_index(drop=True)
    labels = selected.future_boundary_30s.to_numpy(int)
    predictions = selected[["participant_id", "pid", "onset", "fold", "future_boundary_30s"]].copy()
    rows = []
    variants = {
        "context": feature_sets["context_only"],
        "combined": feature_sets["current_past"],
        "physiology": feature_sets["physiology_only"],
    }
    for name, columns in variants.items():
        probabilities = np.full(len(selected), np.nan)
        for fold in sorted(selected.fold.unique()):
            train = np.flatnonzero(selected.fold.to_numpy() != fold)
            test = np.flatnonzero(selected.fold.to_numpy() == fold)
            # An unweighted loss keeps the observed event prevalence as the base rate.
            model = HistGradientBoostingClassifier(**PARAMETERS, random_state=SEED + fold)
            model.fit(selected.iloc[train][columns], labels[train])
            probabilities[test] = model.predict_proba(selected.iloc[test][columns])[:, 1]
        predictions[name + "_probability"] = probabilities
        rows.append(
            {
                "model": name,
                "n": len(selected),
                "prevalence": float(labels.mean()),
                "auroc": float(roc_auc_score(labels, probabilities)),
                "average_precision": float(average_precision_score(labels, probabilities)),
                "brier": float(brier_score_loss(labels, probabilities)),
                "lead_seconds": 30,
            }
        )
        print(f"Forecast {name}: AUROC={rows[-1]['auroc']:.3f}", flush=True)
    return predictions, pd.DataFrame(rows)
