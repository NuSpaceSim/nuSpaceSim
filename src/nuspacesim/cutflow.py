"""Ordered counts of events surviving the active Auger simulation cuts."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


CUTFLOW_STAGES = (
    ("norig", "Original candidates"),
    ("n_involume", "In FoV / volume"),
    ("n_cutEarth", "Cut Earth"),
    ("n_beta", "Emergence-angle range"),
    ("n_exitprob", "Tau exit probability"),
    ("n_energy", "Shower energy"),
    ("n_decayalt", "Decay altitude"),
    ("n_geometry", "Telescope geometry"),
    ("n_startatm", "Atmospheric-start intersection"),
    ("n_grammage", "Available atmospheric grammage"),
    ("n_showerfov", "Shower samples in FoV"),
    ("n_trigger", "Received-energy threshold"),
)


def cutflow_rows(cut_counts):
    """Return (label, count, percent of original, percent of previous) rows.

    Percentages with a zero denominator are undefined and returned as None.
    """
    counts = [int(cut_counts[name]) for name, _ in CUTFLOW_STAGES]
    if any(count < 0 for count in counts):
        raise ValueError("Cut-flow counts must be nonnegative.")
    if any(after > before for before, after in zip(counts, counts[1:])):
        raise ValueError("Cut-flow counts must not increase after a cut.")
    original = counts[0]
    rows = []
    for i, ((_, label), count) in enumerate(zip(CUTFLOW_STAGES, counts)):
        previous = counts[i - 1] if i else original
        rows.append((
            label, count,
            100 * count / original if original else None,
            100 * count / previous if previous else None,
        ))
    return rows


def plot_cutflow(cut_counts, output_path, *, title="Simulation cut flow"):
    """Save a horizontal count chart with an aligned survival-percentage table."""
    rows = cutflow_rows(cut_counts)
    labels, counts = zip(*[(row[0], row[1]) for row in rows])
    fig, (ax, table_ax) = plt.subplots(
        1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1.5, 1]},
        constrained_layout=True,
    )
    ax.barh(np.arange(len(rows)), counts, color="steelblue", height=0.65)
    ax.set_yticks(np.arange(len(rows)), labels)
    # Reserve one top row to align the bars with the table below its header.
    ax.set_ylim(len(rows) - 0.5, -1.5)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, max(1, counts[0]) * 1.4)
    ax.set_xlabel("Events remaining (log scale above 1)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    table_ax.axis("off")

    def percent(value):
        return "—" if value is None else f"{value:.4f}%"

    table = table_ax.table(
        cellText=[[f"{row[1]:,}", percent(row[2]), percent(row[3])] for row in rows],
        colLabels=["n", "% of total", "% of previous"],
        cellLoc="center", bbox=[0, 0, 1, 1], colWidths=[0.34, 0.33, 0.33],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#dbe8f3" if row == 0 else (
            "#f0f4f8" if row % 2 else "white"
        ))
        if row == 0:
            cell.set_text_props(weight="bold")
    fig.suptitle(title)
    output_path = Path(output_path)
    try:
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)
