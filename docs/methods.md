# Methods and code map

The project addresses two related tasks: correcting the released sleep-stage labels at difficult boundaries, and predicting whether a human-labelled boundary will occur after the measurement window. The forecasting output is evaluated separately. It is not used as the gate for the main stage corrector.

## Population and reference

The pinned NEMAR BOAS snapshot contains 128 simultaneous PSG/headband nights, 120,095 aligned epochs and 100 distinct `pid` values. `participant_id` identifies a recording, whereas `pid` identifies the person. All nights with the same `pid` remain together.

A boundary is the first valid human-consensus epoch whose stage differs from the preceding contiguous epoch. Distance is the minimum unsigned number of 30-second epochs to a boundary within a valid segment. Unknown distances are retained as missing and are not treated as stable. Stable periods require a finite distance of at least ten epochs.

The staging cohort requires a valid human stage in the current, preceding and following contiguous epochs. This produces 119,407 human-valid development epochs. The paired comparison additionally requires a valid released headband label for the current epoch, leaving 114,611 epochs, including 9,737 exact boundaries. There are 4,881 `-2` headband labels in the full aligned dataset. Excluded predictions are reported as coverage, not counted as successful classifications.

The five outer folds are balanced by `GroupKFold` on the human-valid development population before restricting to paired labels. Three grouped inner folds are built afresh inside each outer training partition. These details, including the distinction between night and person, are reproduced by `data.assign_folds`.

## Extracted signals

EEG channels are read from their calibrated EDF samples and converted to microvolts using the header units. Sampling rates are read from each channel's metadata. Only complete 30-second epochs are retained.

| Feature family | Computation | Interpretation |
|---|---|---|
| Band powers | Welch spectra in six 5-second windows, 2-second Hann segments with 50% overlap, with bands of delta 0.5–4, theta 4–8, alpha 8–12, sigma 12–16 and beta 16–30 Hz | Absolute log power and relative composition |
| Spectral relationships | Log alpha/theta, delta/sigma and slow/fast ratios | Dynamic relationships between bands |
| Within-epoch dynamics | Mean, slope across the six windows, last-minus-first value | Changes that averages alone can hide |
| Morphology candidates | Sigma-envelope bursts lasting 0.5–2 s above 1.5 times the epoch median, plus negative slow-wave half-cycles with duration/amplitude constraints | Proxies for event-like structure, without validated event labels |
| Amplitude and quality | RMS, peak-to-peak range, inter-channel correlation, spectral entropy, flatness, extreme-amplitude fraction and 30–45 Hz power/slope | Physiological activation together with possible contamination |
| Optical pulse | Bandpass filtering, polarity chosen by median prominence, peaks, intervals 0.35–1.8 s and deviations within 25% of the median | Pulse rate and interval dynamics when quality is sufficient |
| Pulse variability | SDNN and RMSSD when ≥80% of intervals pass checks, with RMSSD restricted to originally consecutive accepted pairs and additional measures from a trailing 90-second window | Pulse variability proxies, not ECG-equivalent clinical HRV |
| Movement | Centred acceleration and angular velocity magnitudes, RMS, 95th percentile, vector jerk and within-epoch changes | Motor activation and a potential source of EEG/PPG artefacts |

The slow-wave proxy requires a negative peak ≤−40 µV and candidate peak-to-peak amplitude ≥75 µV. These thresholds describe a detector candidate, not expert-certified sleep events. The high-frequency slope is a descriptive linear spectral slope, not a separately parameterised aperiodic exponent. No LF/HF ratio is estimated from 30-second pulse windows.

Missing PPG and IMU remain missing, with availability indicators. PPG is present in 112 recordings and six-axis IMU in 94. The complete feature schema is fixed in `data/metadata/feature_schema.json`. Parallel extraction reorders to that schema before writing the cache.

The temporal features add observations 30, 60, 120 and 240 seconds earlier, plus differences at 30, 120 and 240 seconds. Bitbrain context includes one-hot current/prior labels, the duration of its current label run and time elapsed since the start. No whole-night normalisation is used for model inputs.

The offline variant additionally uses the following Bitbrain label and selected signal summaries from the following epoch. The current/past-input variant omits those additional inputs. Because the released Bitbrain labels may themselves use temporal context, this distinction is about our feature availability, not a verified end-to-end real-time system.

## Selective correction

The classifier is `HistGradientBoostingClassifier`, with a learning rate of 0.07, 150 iterations, at most 25 leaves, minimum leaf size 40 and L2 regularisation 3. Early stopping is disabled so that the model does not create an ungrouped internal split.

Within each training partition, class weights are the square root of the inverse class frequency. Boundary epochs receive an additional factor of two, and sample weights are normalised to mean one. Human stages and boundary flags appear here as training targets or weights. They do not enter the feature matrix.

For a candidate label `c` and the released Bitbrain label `b`, the correction score is:

```text
margin = P(stage = c | signals, context) − P(stage = b | signals, context)
```

The candidate is the class with the largest model probability. If its margin exceeds the selected threshold, it replaces the released label. Otherwise, the original label is retained. These weighted-classifier probabilities are used for ranking and margin selection, not claimed as calibrated clinical probabilities.

Inner held-out predictions select among margins from 0 to 0.80 in increments of 0.05, plus a no-correction option. An eligible margin must lose no more than 0.20 percentage points of global accuracy and add no more than 0.50 points of stable-period error on the inner predictions. Among eligible candidates, the smallest boundary error wins. Ties favour the larger threshold.

After selecting the margin, the stage classifier is fitted on the whole outer training partition and applied to the untouched outer test partition. The final reported values pool only these outer held-out predictions. The outer stable-period cost exceeds the inner selection margin, and global noninferiority is not established. Both are retained in the conclusions.

The conservative policy is a secondary post-hoc analysis using the same fitted classifiers. It requires no inner accuracy loss and a stable-error increase ≤0.10 points before selecting its margin. It is not an independent confirmatory experiment.

## Forecasting with an explicit time margin

The input epoch starts at `t` and ends at `t + 30 s`. The target is the human-labelled boundary starting at `t + 60 s`: 30 seconds after the input ends. Targets must remain inside contiguous, human-valid epochs. This yields 114,476 eligible paired forecast observations.

Three unweighted binary classifiers compare Bitbrain context, signals plus context, and signals plus elapsed time without Bitbrain labels. The unweighted loss retains event prevalence as the base rate. AUROC, average precision, Brier score and prevalence are reported. The signals-only variant provides the strongest check of antecedent information independent of released stage labels. The combined variant may carry later information through Bitbrain's upstream algorithm.

## Descriptive physiological effects

Measurements before a boundary are compared with stable controls from the same origin stage and night, preferentially within one hour. The origin stage must remain uninterrupted from the antecedent measurement to the boundary. Each participant receives equal weight in the summary. Whole-night standardisation is used only for these effect sizes and is not supplied to the classifier. Descriptive intervals are not adjusted for all exploratory marker comparisons.

## Uncertainty and interpretation

Paired differences are estimated with 2,000 resamples of complete participants. Every resample carries all nights and epochs from that person. Unknown distances are excluded from the stable subset, not replaced with a stability assumption.

BOAS informed model development and has no independent external holdout for this project. The upstream Bitbrain models were also cross-validated. Without matching their splits, their training could include our outer-test participants when producing labels for other participants. The combined-system evaluation therefore lacks strict upstream independence. The human labels also have 30-second resolution and reflect consensus scoring rather than continuous biological state.

## Code map

| Module | Responsibility |
|---|---|
| `src/boas_transition/edf.py` | Header validation and physical calibration |
| `src/boas_transition/features.py` | EEG, pulse and movement extraction |
| `src/boas_transition/data.py` | Cohort eligibility, temporal inputs and participant folds |
| `src/boas_transition/modeling.py` | Stage correction, nested thresholds and future-boundary classifiers |
| `src/boas_transition/evaluation.py` | Paired metrics and participant-bootstrap intervals |
| `src/boas_transition/physiology.py` | Same-night antecedent/control comparisons |
| `scripts/` | Download, train, evaluate, validate, figures and notebook construction |

The compact prediction archives contain numeric arrays without pickled Python objects. They support checking the reported results without loading models or downloading raw recordings.
