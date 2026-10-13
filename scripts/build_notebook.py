"""Build the portable exploration notebook with optional execution."""

import argparse
import base64
import io
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def execute_locally(notebook) -> None:
    """Run plain Python cells when a Jupyter kernel cannot bind sockets."""
    import matplotlib.pyplot as plt
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.display import display
    from IPython.utils.capture import capture_output
    from matplotlib.figure import Figure

    shell = InteractiveShell.instance()

    def png(figure):
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight")
        return buffer.getvalue()

    shell.display_formatter.formatters["image/png"].for_type(Figure, png)
    plt.show = lambda *args, **kwargs: display(plt.gcf())
    namespace = {"__name__": "__main__"}
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        with capture_output() as captured:
            exec(compile(cell.source, f"<notebook cell {count}>", "exec"), namespace)
        outputs = []
        for stream, value in (("stdout", captured.stdout), ("stderr", captured.stderr)):
            if value:
                outputs.append(nbformat.v4.new_output("stream", name=stream, text=value))
        for output in captured.outputs:
            data = dict(output.data)
            for kind, value in data.items():
                if isinstance(value, bytes):
                    data[kind] = base64.b64encode(value).decode("ascii")
            outputs.append(
                nbformat.v4.new_output("display_data", data=data, metadata=output.metadata)
            )
        cell.outputs = outputs
        cell.execution_count = count
        plt.close("all")
    notebook.metadata["execution_method"] = (
        "Sequential Python/IPython. Native Jupyter kernel unavailable during preparation"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--local-execution", action="store_true")
    args = parser.parse_args()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.cells = [
        md(
            "# BOAS: sleep-stage boundaries and physiological antecedents\n\n"
            "This notebook follows the project from the released Bitbrain labels to the paired correction experiment and the separate future-boundary task. "
            "The published comparison is offline. The results are internal BOAS development estimates, with a boundary gain and a remaining-epoch cost."
        ),
        md(
            '## Load the included evidence\n\nRun `python -m pip install -e ".[notebook]"` from the repository root. '
            "The default exploration uses compact included predictions. No EDF download is required."
        ),
        code(
            "from pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
            "from IPython.display import display\nfrom boas_transition.data import read_archive\n\n"
            "ROOT = Path.cwd()\nif not (ROOT / 'results').exists():\n    ROOT = ROOT.parent\n"
            "RESULTS = ROOT / 'results'\n"
            "plt.rcParams.update({'font.size': 11, 'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})\n"
            "BASELINE, PROPOSED = '#607F8D', '#B15D37'\n"
            "staging = read_archive(RESULTS / 'staging_predictions.npz')\n"
            "metrics = pd.read_csv(RESULTS / 'staging_metrics.csv').set_index('model')\n"
            "assert len(staging) == 114611\nassert staging.groupby('pid')['fold'].nunique().max() == 1\n"
            "display(metrics.loc[['Bitbrain','offline','current_past','context_only']].round(4))"
        ),
        md(
            "## The boundary problem\n\nA boundary is the first 30-second epoch whose human-consensus stage changes. "
            "The horizontal axis below uses unsigned distance, and 10+ combines epochs at least five minutes away."
        ),
        code(
            "MODEL = 'offline'\n"
            "distance = staging.distance_to_human_transition.to_numpy()\n"
            "bins = np.minimum(distance, 10)\n"
            "truth = staging.stage_hum.to_numpy()\n"
            "fig, ax = plt.subplots(figsize=(9,4), constrained_layout=True)\n"
            "for label, column, color in [('Bitbrain','stage_ai_headband',BASELINE), (MODEL,'prediction_'+MODEL,PROPOSED)]:\n"
            "    values = [100*np.mean(staging[column].to_numpy()[bins==i] == truth[bins==i]) for i in range(11)]\n"
            "    ax.plot(range(11), values, 'o-', color=color, label=label)\n"
            "ax.set_xticks(range(11), [*map(str,range(10)), '10+'])\n"
            "ax.set(xlabel='Distance to human boundary (30-second epochs)', ylabel='Stage accuracy (%)', ylim=(40,100))\n"
            "ax.legend(frameon=False)\nplt.show()"
        ),
        md(
            "## Compare benefit and cost\n\nExact boundaries and the remaining epochs partition the paired population. "
            "The ±30-second and stable subsets are additional, overlapping views. Similar global point estimates do not establish noninferiority."
        ),
        code(
            "baseline = metrics.loc['Bitbrain']\nproposed = metrics.loc[MODEL]\n"
            "comparison = pd.DataFrame({\n"
            "    'Bitbrain': [baseline.accuracy,1-baseline.exact_error,baseline.non_boundary_accuracy,1-baseline.stable_error],\n"
            "    MODEL: [proposed.accuracy,1-proposed.exact_error,proposed.non_boundary_accuracy,1-proposed.stable_error]\n"
            "}, index=['Whole night','Exact boundary','Remaining epochs','Stable periods'])\n"
            "display((comparison*100).round(2))\n"
            "intervals = pd.read_csv(RESULTS/'staging_intervals.csv')\n"
            "display(intervals.loc[intervals.model.eq(MODEL)].round(4))"
        ),
        code(
            "boundary = staging.human_transition.to_numpy(bool)\n"
            "difference = (staging['prediction_'+MODEL].to_numpy()==truth).astype(int) - (staging.stage_ai_headband.to_numpy()==truth).astype(int)\n"
            "accounting = pd.Series({'Boundary epochs': difference[boundary].sum(), 'Remaining epochs': difference[~boundary].sum(), 'Whole night': difference.sum()})\n"
            "fig, ax = plt.subplots(figsize=(8,4), constrained_layout=True)\n"
            "ax.barh(accounting.index, accounting.values, color=[PROPOSED,BASELINE,'#203936'])\n"
            "ax.axvline(0,color='gray',linewidth=1)\n"
            "ax.set_xlabel('Additional correctly classified epochs relative to Bitbrain')\n"
            "ax.set_xlim(accounting.min()-200, accounting.max()+200)\n"
            "for i, value in enumerate(accounting):\n    ax.text(value, i, f' {value:+d}', va='center', ha='left' if value>=0 else 'right')\n"
            "plt.show()"
        ),
        md(
            "## Inspect a particular transition\n\nChange `MINIMUM_EPOCHS` to include sparse cases. "
            "These subgroup results are descriptive. Small counts should not be interpreted as stable estimates."
        ),
        code(
            "MINIMUM_EPOCHS = 200\n"
            "directed = pd.read_csv(RESULTS/'directed_transitions.csv')\n"
            "selected = directed.loc[directed.epochs >= MINIMUM_EPOCHS].copy()\n"
            "selected['accuracy_change_pp'] = 100*(selected[MODEL+'_accuracy']-selected.Bitbrain_accuracy)\n"
            "display(selected[['transition','epochs','participants','Bitbrain_accuracy',MODEL+'_accuracy','accuracy_change_pp']].round(3))"
        ),
        md(
            "## Examine antecedent physiology\n\nChange the transition and feature. Measurements are compared with stable controls of the same origin stage and night. "
            "The plotted time is the end of the measurement window: −30 s means a 30-second gap before the labelled boundary. "
            "Whole-night standardisation is descriptive and is not used by the classifier."
        ),
        code(
            "TRANSITION = 'N2→REM'\nFEATURE = 'eeg_rel_theta_mean'\n"
            "effects = pd.read_csv(RESULTS/'physiology_effects.csv')\n"
            "effect = effects.loc[effects.pair.eq(TRANSITION) & effects.feature.eq(FEATURE)].sort_values('lead_seconds')\n"
            "assert not effect.empty, 'Choose an available transition and feature.'\n"
            "display(effect[['lead_seconds','effect_sd','ci_low','ci_high','participants','events']].round(3))\n"
            "fig, ax = plt.subplots(figsize=(8,4), constrained_layout=True)\n"
            "ax.errorbar(-effect.lead_seconds, effect.effect_sd, yerr=[effect.effect_sd-effect.ci_low, effect.ci_high-effect.effect_sd], fmt='o-', capsize=4, color=PROPOSED)\n"
            "ax.axhline(0,color='gray',linewidth=1)\n"
            "ax.set(title=TRANSITION+'\\n'+FEATURE, xlabel='Measurement window ends before boundary (s)', ylabel='Difference from stable controls (SD)')\n"
            "plt.show()"
        ),
        md(
            "## Predict a future boundary\n\nThese are event-prediction metrics, not stage accuracy. "
            "The signals-only model receives EEG, pulse, movement and elapsed time without Bitbrain stages. "
            "The combined model's upstream Bitbrain context may itself use later information."
        ),
        code(
            "from sklearn.metrics import precision_recall_curve\n"
            "forecast = read_archive(RESULTS/'forecast_predictions.npz')\n"
            "display(pd.read_csv(RESULTS/'forecast_metrics.csv').round(4))\n"
            "fig, ax = plt.subplots(figsize=(8,4), constrained_layout=True)\n"
            "for name, color in [('context','#6C7774'),('physiology',BASELINE),('combined',PROPOSED)]:\n"
            "    precision, recall, _ = precision_recall_curve(forecast.future_boundary_30s, forecast[name+'_probability'])\n"
            "    ax.plot(recall,precision,color=color,label=name)\n"
            "ax.axhline(forecast.future_boundary_30s.mean(),color='gray',linestyle='--',label='Event prevalence')\n"
            "ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1))\n"
            "ax.legend(frameon=False)\nplt.show()"
        ),
        md(
            "## Optional per-night signal features\n\nAfter extraction, select a night and time range. "
            "Missing pulse measures remain missing. This cell is skipped when the feature cache is absent."
        ),
        code(
            "NIGHT = 'sub-100'\nSTART_MINUTE, END_MINUTE = 60, 100\n"
            "feature_path = ROOT/'data/features/epoch_features.csv.gz'\n"
            "if feature_path.exists():\n"
            "    features = pd.read_csv(feature_path)\n"
            "    night = features.loc[features.participant_id.eq(NIGHT) & features.onset.between(START_MINUTE*60,END_MINUTE*60)]\n"
            "    fig, axes = plt.subplots(2,1,figsize=(9,5),sharex=True,constrained_layout=True)\n"
            "    axes[0].plot(night.onset/60,night.eeg_log_delta_sigma_mean,color=PROPOSED,label='log(delta/sigma)')\n"
            "    axes[0].plot(night.onset/60,night.eeg_log_alpha_theta_mean,color=BASELINE,label='log(alpha/theta)')\n"
            "    axes[0].legend()\n"
            "    axes[1].plot(night.onset/60,night.ppg_hr,color=BASELINE)\n"
            "    axes[1].set(xlabel='Minutes since recording start',ylabel='Pulse rate (beats/min)')\n"
            "    plt.show()\n"
            "else:\n    print('Run download_data.py and extract_features.py to enable per-night signal exploration.')"
        ),
        md(
            "## Interpretation\n\nThe primary gain is concentrated at human-labelled boundaries. The global point estimate remains close to Bitbrain, "
            "but stable and remaining-epoch costs and the global interval prevent a claim of validated product superiority. "
            "BOAS is a development dataset. The combined system needs external evaluation with a frozen upstream classifier. "
            "Morphology and autonomic features are predictive candidates rather than validated physiological biomarkers.\n\n"
            "The [methods](../docs/methods.md), [evidence](../docs/evidence.md) and [reproducibility guide](../docs/reproducibility.md) describe the assumptions and code."
        ),
    ]
    if args.local_execution:
        execute_locally(notebook)
    elif args.execute:
        from nbclient import NotebookClient

        NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        notebook.metadata["execution_method"] = "Jupyter kernel"
    nbformat.validate(notebook)
    target = ROOT / "notebooks/BOAS_transition_analysis.ipynb"
    target.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, target)
    print(target)


if __name__ == "__main__":
    main()
