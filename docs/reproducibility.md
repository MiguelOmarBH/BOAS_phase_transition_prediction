# Reproducing the analysis

The repository supports two levels of reproduction: checking the published results from included held-out predictions, and rebuilding the complete extraction/training experiment from the pinned public recordings.

## Environment

Python 3.12 was used with NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, scikit-learn 1.8.0 and joblib 1.5.3. The modelling seed is `20261013`. `pyproject.toml` pins the numerical/model dependencies. No GPU or neural-network framework is required.

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[notebook,dev]"
```

On Linux/macOS, activate with `source .venv/bin/activate`. All commands below run from the repository root.

## Check the reported results

```bash
python scripts/validate_results.py
python scripts/evaluate.py
python scripts/make_figures.py
python -m pytest
```

The checks recompute stage and forecast metrics, verify participant grouping, compare the fold allocation with the original-label-derived allocation, and verify that the future target is at `onset + 60 s`. `evaluate.py` rebuilds participant-bootstrap staging intervals and directed transition tables. Scientific figures are generated from those saved arrays and tables.

`results/staging_predictions.npz` and `results/forecast_predictions.npz` contain numeric arrays; `np.load(..., allow_pickle=False)` is sufficient to read them. The `night` array reconstructs recording IDs as `sub-<number>`. Probabilities retain double precision. The exported CSV summaries use the same population for every comparison.

## Explore the notebook

Open `notebooks/BOAS_transition_analysis.ipynb` in VS Code with the Jupyter extension and select `.venv`. The default cells use included results and effects. Optional per-night signal-feature exploration is enabled when `data/features/epoch_features.csv.gz` is present.

The saved cells were executed sequentially in Python/IPython and the rendered figures inspected. A native Jupyter kernel could not be started in the preparation environment because socket creation was restricted; running the notebook through the VS Code kernel remains an environment-specific check.

## Download and extract from source

```bash
python scripts/download_data.py
python scripts/extract_features.py --workers 4
```

The manifest pins 128 wearable EDFs and the relevant label/participant metadata to the NEMAR `on005555/v1.0.0` snapshot. It records source URLs, expected byte sizes and SHA-256 checksums. Downloads are installed atomically only after verification. Correct existing files are reused. No expiring signed URLs are stored in the manifest.

The EDF files total 13.44 GB, so allow additional disk space for partial downloads and derived features. Extraction uses the stored epoch onsets, verified channel calibration and the fixed column schema. PPG and IMU are optional; missing sensors are not inferred from other participants. `--workers` can be reduced for machines with less memory.

For lightweight upstream inspection without EDFs:

```bash
python scripts/download_data.py --metadata-only
```

The compact included labels come from those same public event files. `data/metadata/provenance.json` records the cohort and source snapshot. The actual snapshot uses 100 `pid` values; a catalogue description listing 108 people should not replace the identifiers used for participant grouping.

An existing extracted feature table can be reused at `data/features/epoch_features.csv.gz`. The repository intentionally does not track this 72 MB cache, raw EDFs or fitted model binaries.

## Retrain

```bash
python scripts/train.py --forecast
```

The default experiment trains offline, current/past-input and context-only stage correctors with five outer and three inner folds. Forecasting trains three binary classifiers. A full run takes substantially longer than checking saved results. Use `--models offline` for the main stage comparison only.

New outputs go to `results/retrained/`, leaving the published compact predictions unchanged. Stage-model files are written to `results/models/` and contain the feature order, chosen threshold and training/test participant lists. They are cross-validation models, not a separately externally validated deployment model.

The physiological analysis can be rebuilt after extraction:

```bash
python scripts/analyze_physiology.py
```

It uses complete-night scaling for descriptive effects only. It never changes the raw feature values fed to the classifiers.

## Limits of reproduction

Checksums establish which source files were used, not that Bitbrain's model has been reconstructed. Its released hard labels are inputs and comparators; its weights, probabilities and exact cross-validation splits are outside this project. The fixed cohort and internal validation can reproduce this experiment but cannot replace an external prospective test.
