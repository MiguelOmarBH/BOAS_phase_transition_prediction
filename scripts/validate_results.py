"""Check saved predictions, denominators, folds and published result tables."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from boas_transition.data import assign_folds, load_epochs, read_archive

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    predictions = read_archive(ROOT / "results/staging_predictions.npz")
    assert len(predictions) == 114611 and predictions.pid.nunique() == 100
    assert predictions.participant_id.nunique() == 128
    assert predictions.groupby("pid").fold.nunique().max() == 1
    assert not predictions.duplicated(["participant_id", "onset"]).any()
    epochs = load_epochs(ROOT)
    folds = assign_folds(epochs)
    checked = predictions.merge(
        folds, on=["participant_id", "pid", "onset"], suffixes=("", "_expected")
    )
    assert len(checked) == len(predictions) and checked.fold.eq(checked.fold_expected).all()
    truth = predictions.stage_hum.to_numpy(int)
    distance = predictions.distance_to_human_transition.to_numpy(float)
    stored = pd.read_csv(ROOT / "results/staging_metrics.csv").set_index("model")
    for name in ("Bitbrain", "offline", "current_past", "context_only", "conservative"):
        pred = (
            predictions.stage_ai_headband.to_numpy(int)
            if name == "Bitbrain"
            else predictions["prediction_" + name].to_numpy(int)
        )
        error = pred != truth
        computed = {
            "accuracy": accuracy_score(truth, pred),
            "non_boundary_accuracy": np.mean(~error[~predictions.human_transition.to_numpy(bool)]),
            "macro_f1": f1_score(truth, pred, labels=range(5), average="macro"),
            "exact_error": error[distance == 0].mean(),
            "near_error": error[distance <= 1].mean(),
            "stable_error": error[np.isfinite(distance) & (distance >= 10)].mean(),
            "changed_fraction": (pred != predictions.stage_ai_headband.to_numpy(int)).mean(),
        }
        for metric, value in computed.items():
            assert np.isclose(value, stored.loc[name, metric], atol=1e-12), (name, metric)
    forecasts = read_archive(ROOT / "results/forecast_predictions.npz")
    lookup = epochs.set_index(["participant_id", "onset"])
    targets = pd.MultiIndex.from_arrays([forecasts.participant_id, forecasts.onset + 60])
    future = lookup.loc[targets, "human_transition"].to_numpy(int)
    assert np.array_equal(future, forecasts.future_boundary_30s.to_numpy(int))
    stored = pd.read_csv(ROOT / "results/forecast_metrics.csv").set_index("model")
    for name in ("context", "combined", "physiology"):
        probability = forecasts[name + "_probability"].to_numpy(float)
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
        for metric, value in [
            ("auroc", roc_auc_score(future, probability)),
            ("average_precision", average_precision_score(future, probability)),
            ("brier", brier_score_loss(future, probability)),
        ]:
            assert np.isclose(value, stored.loc[name, metric], atol=1e-12)
    for path in (ROOT / "results").glob("*_inner_grid.csv"):
        grid = pd.read_csv(path)
        selected = (
            grid.loc[grid.eligible]
            .sort_values(["exact_error", "threshold"], ascending=[True, False])
            .iloc[0]
        )
        variant, suffix = path.stem.split("_fold")
        fold = int(suffix.split("_")[0])
        selection = pd.read_csv(ROOT / "results" / f"{variant}_nested_selection.csv").set_index(
            "fold"
        )
        assert np.isclose(selected.threshold, selection.loc[fold, "threshold"])
    print("All result checks passed: paired metrics, participant folds and future target timing.")


if __name__ == "__main__":
    main()
