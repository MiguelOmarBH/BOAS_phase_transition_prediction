"""Build the repository figures from the checked held-out predictions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from boas_transition.data import read_archive

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures"
PAPER, INK, MUTED = "#F6F3ED", "#203936", "#6C7774"
BASELINE, PROPOSED = "#607F8D", "#B15D37"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#D5D9D5",
            "svg.fonttype": "none",
            "savefig.facecolor": PAPER,
        }
    )


def heading(fig: plt.Figure, title: str, note: str) -> None:
    fig.text(0.09, 0.96, "BOAS  /  SLEEP TRANSITIONS", fontsize=10, color=MUTED, weight="normal")
    fig.text(0.09, 0.90, title, fontsize=20, weight="normal", color=INK)
    fig.text(0.09, 0.025, note, fontsize=9, color=MUTED, va="bottom")


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        target = FIGURES / f"{name}.{extension}"
        temporary = target.with_name(f"{name}.tmp.{extension}")
        metadata = (
            {"Creator": "Miguel Omar Belhouk Herrero", "Date": None} if extension == "svg" else None
        )
        fig.savefig(temporary, dpi=160, metadata=metadata)
        temporary.replace(target)
    plt.close(fig)


def main() -> None:
    style()
    p = read_archive(ROOT / "results/staging_predictions.npz")
    metrics = pd.read_csv(ROOT / "results/staging_metrics.csv").set_index("model")
    intervals = pd.read_csv(ROOT / "results/staging_intervals.csv")
    truth = p.stage_hum.to_numpy(int)
    distance = p.distance_to_human_transition.to_numpy(float)
    bins = np.minimum(distance, 10)
    rows = []
    for value in range(11):
        mask = bins == value
        rows.append(
            {
                "distance_epochs": value,
                "epochs": int(mask.sum()),
                "Bitbrain_accuracy": float(
                    (p.stage_ai_headband.to_numpy()[mask] == truth[mask]).mean()
                ),
                "proposed_accuracy": float(
                    (p.prediction_offline.to_numpy()[mask] == truth[mask]).mean()
                ),
            }
        )
    distances = pd.DataFrame(rows)
    distances.to_csv(ROOT / "results/accuracy_by_distance.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.22, top=0.78)
    ax.axhline(
        metrics.loc["Bitbrain", "accuracy"] * 100, color=MUTED, linestyle=(0, (3, 3)), linewidth=1.3
    )
    ax.plot(
        distances.distance_epochs,
        distances.Bitbrain_accuracy * 100,
        color=BASELINE,
        marker="o",
        markerfacecolor=PAPER,
        linewidth=2,
        label="Bitbrain",
    )
    ax.plot(
        distances.distance_epochs,
        distances.proposed_accuracy * 100,
        color=PROPOSED,
        marker="o",
        linewidth=2,
        label="Proposed correction",
    )
    ax.set(
        xlim=(-0.2, 10.2),
        ylim=(40, 100),
        ylabel="Stage accuracy (%)",
        xlabel="Distance to the nearest human-labelled boundary (30-second epochs)",
    )
    ax.set_xticks(range(11), [*map(str, range(10)), "10+"])
    ax.grid(axis="y", color="#E2E4DE", linewidth=0.7)
    ax.legend(loc="lower right", frameon=False)
    ax.text(2.2, 88.8, "Bitbrain whole-night accuracy", color=MUTED, fontsize=10)
    heading(
        fig,
        "The difficult part of the night is the boundary",
        "Matched population: 114,611 epochs from 100 people. Distance is unsigned. The 10+ bin means at least five minutes.\n"
        "Each point is a separate distance bin. Epochs with unknown boundary distance are omitted from the curves.",
    )
    save(fig, "boundary_accuracy")

    categories = [
        "Whole night",
        "Exact boundary",
        "Within ±30 seconds",
        "Remaining epochs",
        "Stable periods",
    ]
    bb = metrics.loc["Bitbrain"]
    ours = metrics.loc["offline"]
    baseline = (
        np.array(
            [
                bb.accuracy,
                1 - bb.exact_error,
                1 - bb.near_error,
                bb.non_boundary_accuracy,
                1 - bb.stable_error,
            ]
        )
        * 100
    )
    proposed = (
        np.array(
            [
                ours.accuracy,
                1 - ours.exact_error,
                1 - ours.near_error,
                ours.non_boundary_accuracy,
                1 - ours.stable_error,
            ]
        )
        * 100
    )
    fig, ax = plt.subplots(figsize=(11, 6.4))
    fig.subplots_adjust(left=0.23, right=0.87, bottom=0.19, top=0.78)
    y = np.arange(len(categories))
    for i, (a, b) in enumerate(zip(baseline, proposed)):
        ax.plot([a, b], [i, i], color="#BBC2BE", linewidth=2)
        ax.text(a, i - 0.19, f"{a:.2f}", ha="center", color=BASELINE, fontsize=10)
        ax.text(b, i + 0.25, f"{b:.2f}", ha="center", color=PROPOSED, fontsize=10)
        ax.text(
            102,
            i,
            f"{b - a:+.2f} pp",
            ha="left",
            va="center",
            weight="normal",
            color=INK,
            clip_on=False,
        )
    ax.scatter(
        baseline,
        y,
        s=70,
        facecolors=PAPER,
        edgecolors=BASELINE,
        linewidth=2,
        zorder=3,
        label="Bitbrain",
    )
    ax.scatter(proposed, y, s=65, color=PROPOSED, zorder=4, label="Proposed correction")
    ax.set_yticks(y, categories)
    ax.invert_yaxis()
    ax.set(xlim=(40, 100), ylim=(4.7, -0.7), xlabel="Accuracy (%) on a scale from 40 to 100")
    ax.grid(axis="x", color="#E2E4DE", linewidth=0.7)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.19), ncol=2, frameon=False)
    heading(
        fig,
        "A boundary gain, with similar overall point estimates",
        "The proposed model uses one following epoch. Remaining epochs exclude exact boundaries. Stable means distance ≥ five minutes.\n"
        "Groups overlap and are not additive. Similar point estimates do not demonstrate statistical noninferiority.",
    )
    save(fig, "accuracy_comparison")

    requested = ["accuracy", "exact_error", "near_error", "non_boundary_accuracy", "stable_error"]
    selected = intervals.loc[intervals.model.eq("offline")].set_index("metric").loc[requested]
    estimate = selected.difference.to_numpy() * 100
    low = selected.ci_low.to_numpy() * 100
    high = selected.ci_high.to_numpy() * 100
    for i, metric in enumerate(requested):
        if metric.endswith("error"):
            estimate[i], low[i], high[i] = -estimate[i], -high[i], -low[i]
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(left=0.23, right=0.96, bottom=0.20, top=0.79)
    ax.axvline(0, color=MUTED, linewidth=1.1, linestyle=(0, (3, 3)))
    ax.errorbar(
        estimate,
        y,
        xerr=[estimate - low, high - estimate],
        fmt="o",
        color=PROPOSED,
        markersize=7,
        capsize=4,
        linewidth=2,
    )
    ax.set_yticks(y, categories)
    ax.set(
        ylim=(4.6, -0.6),
        xlim=(-2, 10),
        xlabel="Accuracy difference from Bitbrain (percentage points)",
    )
    ax.grid(axis="x", color="#E2E4DE", linewidth=0.7)
    for i, value in enumerate(estimate):
        ax.text(value, i - 0.22, f"{value:+.2f}", ha="center", fontsize=10, weight="normal")
    heading(
        fig,
        "The gain is localized, and so is the cost",
        "Paired 95% intervals from 2,000 resamples of complete participants. Positive values favour the proposed correction.\n"
        "Whole-night uncertainty spans zero. The higher stable-period error remains visible.",
    )
    save(fig, "paired_differences")

    directed = pd.read_csv(ROOT / "results/directed_transitions.csv")
    directed = directed.loc[directed.epochs >= 200].copy()
    directed["gain"] = directed.offline_accuracy - directed.Bitbrain_accuracy
    directed = directed.sort_values("gain", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.subplots_adjust(left=0.18, right=0.91, bottom=0.17, top=0.82)
    positions = np.arange(len(directed))
    for i, row in enumerate(directed.itertuples()):
        a, b = row.Bitbrain_accuracy * 100, row.offline_accuracy * 100
        ax.plot([a, b], [i, i], color="#BBC2BE", linewidth=2)
        ax.text(
            103,
            i,
            f"n={row.epochs:,}",
            va="center",
            ha="left",
            fontsize=9,
            color=MUTED,
            clip_on=False,
        )
    ax.scatter(
        directed.Bitbrain_accuracy * 100,
        positions,
        s=60,
        facecolors=PAPER,
        edgecolors=BASELINE,
        linewidth=1.8,
        label="Bitbrain",
        zorder=3,
    )
    ax.scatter(
        directed.offline_accuracy * 100,
        positions,
        s=55,
        color=PROPOSED,
        label="Proposed correction",
        zorder=4,
    )
    ax.set_yticks(positions, directed.transition)
    ax.set(
        ylim=(len(directed) - 0.4, -0.6),
        xlim=(0, 100),
        xlabel="Accuracy at the exact human-labelled boundary (%)",
    )
    ax.grid(axis="x", color="#E2E4DE", linewidth=0.7)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.07), ncol=2, frameon=False)
    heading(
        fig,
        "The result depends on the direction of the transition",
        "All transition types with at least 200 paired epochs, ordered by observed accuracy difference.\n"
        "These are descriptive subgroup estimates. Sparse transitions remain in the complete CSV table.",
    )
    save(fig, "directed_transitions")

    effects = pd.read_csv(ROOT / "results/physiology_effects.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.7))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.27, top=0.75, wspace=0.30)
    candidates = [
        ("N2→N3", "eeg_slow_wave_count", "Slow-wave candidates"),
        ("N2→REM", "eeg_rel_theta_mean", "Relative theta power"),
        ("All", "ppg_interval_outlier_fraction", "Pulse interval outliers"),
    ]
    for ax, (pair, feature, title) in zip(axes, candidates):
        effect = effects.loc[effects.pair.eq(pair) & effects.feature.eq(feature)].sort_values(
            "lead_seconds", ascending=False
        )
        ax.errorbar(
            -effect.lead_seconds,
            effect.effect_sd,
            yerr=[effect.effect_sd - effect.ci_low, effect.ci_high - effect.effect_sd],
            fmt="o-",
            color=PROPOSED,
            capsize=3,
            linewidth=1.8,
        )
        ax.axhline(0, color=MUTED, linestyle=(0, (3, 3)), linewidth=1)
        ax.set(title=pair + "\n" + title, xlabel="Window ends before boundary (s)")
        ax.grid(axis="y", color="#E2E4DE", linewidth=0.7)
    axes[0].set_ylabel("Difference from matched stable controls (SD)")
    heading(
        fig,
        "Antecedents include physiology and signal quality",
        "Controls share the origin stage and night. Each person has equal weight. Intervals are descriptive and unadjusted for multiple comparisons.\n"
        "Whole-night scaling is used only for this effect-size analysis. Morphology candidates are not clinically validated events.",
    )
    save(fig, "physiological_antecedents")

    f = read_archive(ROOT / "results/forecast_predictions.npz")
    fm = pd.read_csv(ROOT / "results/forecast_metrics.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.23, top=0.78)
    colors = {"context": MUTED, "combined": PROPOSED, "physiology": BASELINE}
    labels = {
        "context": "Bitbrain context",
        "combined": "Signals + context",
        "physiology": "Signals only",
    }
    for name in ("context", "physiology", "combined"):
        precision, recall, _ = precision_recall_curve(
            f.future_boundary_30s, f[name + "_probability"]
        )
        indices = np.unique(
            np.r_[np.arange(0, len(recall), max(1, len(recall) // 500)), len(recall) - 1]
        )
        ax.plot(
            recall[indices],
            precision[indices],
            color=colors[name],
            linewidth=2,
            label=f"{labels[name]} (AUROC {fm.loc[name, 'auroc']:.3f}, AP {fm.loc[name, 'average_precision']:.3f})",
        )
    ax.axhline(
        fm.prevalence.iloc[0],
        color=MUTED,
        linestyle=(0, (3, 3)),
        linewidth=1,
        label="Event prevalence: 8.55%",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Recall of future human-labelled boundaries",
        ylabel="Precision",
    )
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.grid(color="#E2E4DE", linewidth=0.7)
    heading(
        fig,
        "The signals carry information before the labelled boundary",
        "114,476 held-out epochs. Measurements end 30 seconds before the target boundary. Labels have 30-second resolution.\n"
        "The signals-only model receives no Bitbrain stages. Published Bitbrain context may itself include future information.",
    )
    save(fig, "boundary_forecast")


if __name__ == "__main__":
    main()
