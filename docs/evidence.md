# Evidence behind the physiological hypotheses

This was a directed review of primary sleep-physiology and modelling studies, not a systematic review. The purpose was to identify measurable hypotheses for the two frontal EEG channels and optional pulse/movement signals available in BOAS. Three distinctions guide the interpretation: a marker of an established stage is not necessarily a precursor, an arousal is not necessarily a stage change, and an EEG event is not a consensus-labelled boundary.

## Spectral trajectories

Leguia et al., “Variability and predictability in human sleep” (Brain Communications, 2025). Longitudinal two-channel sub-scalp EEG from eight adults showed that delta/sigma dynamics could predict NREM→REM transitions two to four minutes ahead, with reported AUROC 0.88. The study explicitly compared spectral information with time spent in the preceding stage. This motivated delta/sigma trajectories, multi-minute history and our context-only comparator. The reported 0.88 cannot be transferred to frontal scalp EEG, all five stages or our cohort. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12716275/) and [PubMed](https://pubmed.ncbi.nlm.nih.gov/41426434/)

“Pre-sleep central–occipital alpha–theta dynamics predict NREM2 onset in adult” (Journal of Neural Engineering, 2026). The abstract describes a time-varying alpha/theta relationship and its rate of change during sleep onset, with variable trajectories across fast and slow sleepers. The initial dataset comprised 18 adults, followed by an additional public-data check. This motivated ratio and slope features. The consulted abstract concerns central/occipital sites and first NREM2 onset, so it is a cautious hypothesis for frontal signals and recurrent transitions. [DOI](https://doi.org/10.1088/1741-2552/ae9226) and [PubMed](https://pubmed.ncbi.nlm.nih.gov/42526481/)

## Slow waves and sigma bursts

Siclari et al. (Sleep, 2014). High-density EEG showed dissociated courses of slow-wave number and amplitude during falling asleep, together with evolving spindle density and frequency. This supports representing event morphology alongside band averages. Our implementation extracts simple morphology candidates. It does not recover the spatial processes measured with high-density EEG. [DOI](https://doi.org/10.5665/sleep.4070) and [PubMed](https://pubmed.ncbi.nlm.nih.gov/25197810/)

Lacourse et al. (Journal of Neuroscience Methods, 2019). The A7 spindle detector combines absolute/relative sigma power and the relationship of sigma-filtered activity to the original signal. It provides a methodological reference for event detection. Our sigma-envelope proxy is not A7, and the study's expert-validated detector performance is not a performance claim for this code. [PubMed](https://pubmed.ncbi.nlm.nih.gov/30107208/)

## Autonomic antecedents

Bonnet and Arand (1997). In 12 adults, ECG-derived autonomic changes could precede scored REM, and heart-rate acceleration preceded some EEG arousals. This motivated pulse acceleration and recent variability rather than only a mean pulse rate. The study used ECG and longer HRV windows. BOAS PPG measures are treated as pulse proxies. [PubMed](https://pubmed.ncbi.nlm.nih.gov/9191582/)

“Electroencephalogram Characteristics of Autonomic Arousals During Sleep in Healthy Men.” Tachycardia/bradycardia bouts occurred with and without cortical arousal in a small study. Autonomic activation can therefore provide complementary information, but it should not be equated with a five-stage boundary. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC2289770/)

de Zambotti et al. (Sleep, 2016). A biphasic cardiac response accompanied spontaneous and evoked K-complexes in 40 adolescents. This supports investigating coupling between EEG events and cardiac activity, but does not demonstrate advance prediction of stage changes. No validated K-complex detector or event–pulse coupling model is implemented here. [PubMed](https://pubmed.ncbi.nlm.nih.gov/26856907/)

“Validity of Ultra-Short-Term HRV Analysis Using PPG—A Preliminary Study” (2022). Agreement depends on metric, window duration, quality and sampling. This informed interval checks, preservation of consecutive pairs and an additional trailing 90-second window. No clinical equivalence to ECG HRV is assumed, and LF/HF is not estimated from 30 seconds. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9611389/)

## Arousal, spectral slope and contamination

Lendner et al. (eLife, 2020). The high-frequency EEG spectral slope distinguished arousal states, including wake and REM. This motivated a descriptive 30–45 Hz slope, but the paper establishes state discrimination rather than a universal advance transition predictor. [DOI](https://doi.org/10.7554/eLife.55092)

Kozhemiako et al., “Sources of Variation in the Spectral Slope of the Sleep EEG” (2022). Large-cohort analyses supported state-related slope differences while identifying technical issues involving muscle, ECG and reference choice. This is why high-frequency power, extreme amplitude and movement remain explicit quality-related inputs. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9512622/)

Donoghue et al. (Nature Neuroscience, 2020). Power in a conventional band can conflate oscillatory peaks and aperiodic background. Our linear high-frequency slope is deliberately named a spectral slope. Complete peak/background parameterisation is not implemented. [DOI](https://doi.org/10.1038/s41593-020-00744-x)

## Relation to automated staging

Vallat and Walker (eLife, 2021). Their open-source staging work supports the practical value of interpretable features and context with large, heterogeneous training data. It does not establish transfer of a pretrained PSG model to this two-frontal-channel headband. [DOI](https://doi.org/10.7554/eLife.70092)

Esparza-Iaizzo et al. (Computers in Biology and Medicine, 2026). The Bitbrain-related study examines channel setup, temporal scope and population differences on four other open datasets. It gives context for simpler and real-time sleep scoring. The BOAS release description is the source for the 86.71% wearable and 87.13% PSG figures cited in this repository. [DOI](https://doi.org/10.1016/j.compbiomed.2026.111560) and [BOAS description](https://github.com/OpenNeuroDatasets/ds005555)

## What the sample adds

The descriptive BOAS effects are direction-specific. N2→N3 shows increased slow-wave candidates and delta/sigma. N2→REM shows increased theta and decreased delta. Some movement and pulse-quality measures also precede boundaries. These observations do not resolve whether the predictive information comes from neural dynamics, autonomic activity, movement, contamination or a combination.

The signals-only forecasting model tests the combined information against a future human-labelled target without Bitbrain labels as inputs. It does not validate every individual marker or establish a physiological mechanism. Event-level annotations, better artefact controls and an external cohort would be needed for those stronger claims.
