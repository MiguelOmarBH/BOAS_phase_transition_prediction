"""Run the nested participant-wise staging experiment and optional forecasts."""

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd

from boas_transition.data import build_frame
from boas_transition.evaluation import bootstrap_stage_differences, stage_metrics
from boas_transition.modeling import forecast_boundaries, nested_staging

ROOT = Path(__file__).resolve().parents[1]


def save_predictions(frame: pd.DataFrame, path: Path) -> None:
    """Preserve the last complete export if writing is interrupted."""
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv.gz")
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["offline", "current_past", "context_only"],
        default=["offline", "current_past", "context_only"],
    )
    parser.add_argument("--forecast", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/retrained")
    args = parser.parse_args()
    frame, columns = build_frame(ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    output = frame[
        [
            "participant_id",
            "pid",
            "onset",
            "fold",
            "stage_hum",
            "stage_ai_headband",
            "human_transition",
            "distance_to_human_transition",
        ]
    ].copy()
    predictions = {"Bitbrain": frame.stage_ai_headband.to_numpy(int)}
    for name in args.models:
        predictions[name] = nested_staging(frame, columns[name], name, args.output)
        output["prediction_" + name] = predictions[name]
        save_predictions(output, args.output / "staging_predictions.csv.gz")
        pd.DataFrame(
            [
                {"model": name, **stage_metrics(frame, values)}
                for name, values in predictions.items()
            ]
        ).to_csv(args.output / "staging_metrics.csv", index=False)
        bootstrap_stage_differences(frame, predictions).to_csv(
            args.output / "staging_intervals.csv", index=False
        )
    if args.forecast:
        forecasts, metrics = forecast_boundaries(frame, columns, args.output)
        save_predictions(forecasts, args.output / "forecast_predictions.csv.gz")
        metrics.to_csv(args.output / "forecast_metrics.csv", index=False)


if __name__ == "__main__":
    main()
