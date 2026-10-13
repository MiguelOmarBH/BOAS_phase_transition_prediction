from pathlib import Path

import numpy as np

from boas_transition.data import assign_folds, load_epochs, read_archive
from boas_transition.modeling import apply_corrections

ROOT = Path(__file__).resolve().parents[1]


def test_repeated_nights_stay_with_their_participant():
    folds = assign_folds(load_epochs(ROOT))
    assert len(folds) == 119407
    assert folds.pid.nunique() == 100
    assert folds.groupby("pid").fold.nunique().eq(1).all()
    assert folds.fold.nunique() == 5


def test_forecast_has_a_thirty_second_measurement_gap():
    forecasts = read_archive(ROOT / "results/forecast_predictions.npz")
    epochs = load_epochs(ROOT).set_index(["participant_id", "onset"])
    measured_end = forecasts.onset.to_numpy() + 30
    target_start = forecasts.onset.to_numpy() + 60
    assert np.all(target_start - measured_end == 30)
    targets = epochs.loc[list(zip(forecasts.participant_id, target_start)), "human_transition"]
    np.testing.assert_array_equal(
        targets.to_numpy(int), forecasts.future_boundary_30s.to_numpy(int)
    )


def test_original_labels_are_preserved_below_the_margin():
    baseline = np.array([0, 1, 2, 3])
    original = baseline.copy()
    candidate = np.array([1, 2, 3, 4])
    output = apply_corrections(baseline, candidate, np.array([0.1, 0.4, 0.5, 0.7]), 0.5)
    np.testing.assert_array_equal(output, [0, 1, 2, 4])
    np.testing.assert_array_equal(baseline, original)
