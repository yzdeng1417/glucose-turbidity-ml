"""A compact report rendered exclusively from the current synthetic run."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve


def plot_report(raw, heldout, threshold, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    colors = {0: "#637b92", 1: "#d18145"}
    for ax, channel, title in zip(axes[0], ("glu", "ntu"), ("Glucose-like signal", "Turbidity-like signal")):
        for label in (0, 1):
            traces = raw.loc[raw.growth.eq(label)].head(10)
            values = traces[[f"{channel}_time{i}" for i in range(1, 9)]].to_numpy()
            ax.plot(range(1, 9), values.T, color=colors[label], alpha=0.35, linewidth=1)
            ax.plot([], [], color=colors[label], label=f"Synthetic label {label}")
        ax.set(title=title, xlabel="Toy time index", ylabel="Arbitrary signal units")
        ax.legend(frameon=False, fontsize=8)
    false_positive, true_positive, _ = roc_curve(heldout.growth, heldout.score)
    axes[1, 0].plot(false_positive, true_positive, color="#207f83", linewidth=2)
    axes[1, 0].plot([0, 1], [0, 1], "--", color="#9ba2aa", linewidth=1)
    axes[1, 0].set(title="Synthetic held-out groups: ROC", xlabel="False-positive rate", ylabel="True-positive rate")
    counts = confusion_matrix(heldout.growth, heldout.score >= threshold, labels=[0, 1])
    axes[1, 1].imshow(counts, cmap="Blues")
    for (i, j), value in np.ndenumerate(counts):
        axes[1, 1].text(j, i, str(value), ha="center", va="center", color="white" if value > counts.max()/2 else "#26384a")
    axes[1, 1].set(title="Threshold fixed on development OOF", xticks=[0, 1], yticks=[0, 1], xlabel="Predicted label", ylabel="Synthetic label")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Synthetic workflow demonstration — no experimental performance claims", fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)
