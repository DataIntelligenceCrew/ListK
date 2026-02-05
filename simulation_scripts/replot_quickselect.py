#!/usr/bin/env python3
"""Generate quickselect publication figure showing oracle calls vs pivot count for various k."""

import csv
import math
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Font setup
FONT_PATH = os.path.expanduser('~/.fonts/LinLibertine_R.otf')
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    FONT_PROP = fm.FontProperties(fname=FONT_PATH, size=9)
    FONT_NAME = FONT_PROP.get_name()
else:
    FONT_NAME = 'serif'
    FONT_PROP = fm.FontProperties(family='serif', size=9)

# Parameters
N, W = 1000, 20
CSV_PATH = os.path.join(SCRIPT_DIR, "T_N1000_W20_algo-quickselect_trials-5000.csv")

# K configurations: k value, color, marker
K_CONFIGS = [
    (1,   '#222255', 'o'),  # dark blue
    (100, '#225555', 's'),  # dark cyan
    (300, '#225522', '^'),  # dark green
    (500, '#666633', 'D'),  # dark yellow
]


def theoretical_cost(N: int, k: int, P: int, W: int) -> float:
    """Theoretical expected oracle calls for quickselect."""
    if P >= W - 1 or P < 1:
        return float('inf')
    kappa = k / N
    denom = P - 1 + kappa ** (P + 1) + (1 - kappa) ** (P + 1)
    if denom == 0:
        return float('inf')
    return (P + 1) / ((W - P) * denom) * N


def load_empirical_data(path: str) -> dict:
    """Load empirical data from CSV, keyed by (k, P)."""
    data = {}
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            k, P = int(row['k']), int(row['P'])
            if k not in data:
                data[k] = {}
            data[k][P] = float(row['T_est'])
    return data


def main():
    empirical = load_empirical_data(CSV_PATH)

    fig, ax = plt.subplots(figsize=(3.43, 2.5725), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Plot each k configuration
    for k, color, marker in K_CONFIGS:
        if k not in empirical:
            continue

        # Empirical scatter
        P_vals = sorted(P for P in empirical[k] if P <= 17)
        ax.scatter(P_vals, [empirical[k][P] for P in P_vals],
                   s=10, color=color, marker=marker, zorder=3)

        # Theoretical curve
        theory_x = np.linspace(1, 17, 200)
        theory_y = [theoretical_cost(N, k, P, W) for P in theory_x]
        valid = [(x, y) for x, y in zip(theory_x, theory_y) if y < 1000]
        if valid:
            ax.plot([x for x, _ in valid], [y for _, y in valid],
                    color=color, linewidth=0.5, zorder=2)

    # Axis setup
    ax.set_xlim(1, 17)
    ax.set_ylim(0, 400)
    ax.set_xticks([1, 5, 9, 13, 17])
    ax.set_yticks([0, 100, 200, 300, 400])
    ax.set_xlabel('Pivot Count (P)', fontsize=9, fontname=FONT_NAME, labelpad=6)
    ax.set_ylabel('Expected Listwise Ranker Calls', fontsize=9, fontname=FONT_NAME)
    ax.tick_params(axis='both', which='major', labelsize=6)

    # Styling
    for spine in ax.spines.values():
        spine.set_color('#666666')

    # Legend: k values + empirical/theory markers
    legend_handles = [
        Line2D([0], [0], marker=m, color=c, markerfacecolor=c,
               markersize=3, linestyle='-', linewidth=0.5, label=f'K = {k}')
        for k, c, m in K_CONFIGS
    ] + [
        Line2D([0], [0], marker='o', color='black', markerfacecolor='black',
               markersize=3, linestyle='None', label='Empirical'),
        Line2D([0], [0], color='black', linewidth=0.5, label='Theory'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='upper left',
              frameon=True, facecolor='white', edgecolor='none',
              ncol=2, columnspacing=1.0, handletextpad=0.5, prop=FONT_PROP)

    ax.set_xticks(range(1, 18), minor=True)
    ax.grid(which='major', color='lightgray', linewidth=0.5, zorder=0)

    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.15, top=0.975)

    # Save
    output_dir = os.path.join(SCRIPT_DIR, 'publication_figures')
    os.makedirs(output_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(output_dir, f'quickselect.{ext}'),
                    dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved to {output_dir}/quickselect.{{png,pdf}}")


if __name__ == '__main__':
    main()
