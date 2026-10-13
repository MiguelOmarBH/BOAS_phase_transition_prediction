"""Recompute the published comparisons from participant-wise held-out predictions."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from boas_transition.data import load_epochs, read_archive
from boas_transition.evaluation import bootstrap_stage_differences, directed_errors, stage_metrics

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    frame = read_archive(ROOT / "results/staging_predictions.npz")
    original = load_epochs(ROOT)
    frame = frame.merge(
        original[["participant_id", "onset", "previous_human_stage"]],
        on=["participant_id", "onset"],
        validate="one_to_one",
    )
    predictions = {"Bitbrain": frame.stage_ai_headband.to_numpy(int)}
    for name in ("offline", "current_past", "context_only", "conservative"):
        predictions[name] = frame["prediction_" + name].to_numpy(int)
    pd.DataFrame(
        [{"model": name, **stage_metrics(frame, values)} for name, values in predictions.items()]
    ).to_csv(ROOT / "results/staging_metrics.csv", index=False)
    bootstrap_stage_differences(frame, predictions).to_csv(
        ROOT / "results/staging_intervals.csv", index=False
    )
    directed_errors(frame, predictions).to_csv(
        ROOT / "results/directed_transitions.csv", index=False
    )
    forecast = read_archive(ROOT / "results/forecast_predictions.npz")
    truth = forecast.future_boundary_30s.to_numpy(int)
    rows = []
    for name in ("context", "combined", "physiology"):
        probability = forecast[name + "_probability"].to_numpy(float)
        rows.append(
            {
                "model": name,
                "n": len(forecast),
                "prevalence": float(truth.mean()),
                "auroc": roc_auc_score(truth, probability),
                "average_precision": average_precision_score(truth, probability),
                "brier": brier_score_loss(truth, probability),
                "lead_seconds": 30,
            }
        )
    pd.DataFrame(rows).to_csv(ROOT / "results/forecast_metrics.csv", index=False)
    print(pd.read_csv(ROOT / "results/staging_metrics.csv").round(5).to_string(index=False))


if __name__ == "__main__":
    main()
