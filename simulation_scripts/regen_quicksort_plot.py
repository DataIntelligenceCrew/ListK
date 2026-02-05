#!/usr/bin/env python3
"""Generate quicksort publication figure showing oracle calls vs pivot count."""

import csv
import math
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Font setup (Linux Libertine if available, else serif)
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
CSV_PATH = os.path.join(SCRIPT_DIR, "T_N1000_W20_algo-quicksort_trials-5000.csv")


def theoretical_cost(N: int, P: float, W: int) -> float:
    """Theoretical expected oracle calls for quicksort."""
    if P >= W or P < 1:
        return float('inf')
    alpha = 1.0 / ((W - P) * math.log(P + 1))
    return alpha * N * math.log(N) + 0.1 * N


def load_empirical_data(path: str) -> dict:
    """Load empirical data from CSV."""
    data = {}
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            data[int(row['P'])] = float(row['T_est'])
    return data


def main():
    empirical = load_empirical_data(CSV_PATH)

    fig, ax = plt.subplots(figsize=(3.43, 2.5725), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Empirical scatter
    P_vals = sorted(P for P in empirical if P <= 17)
    ax.scatter(P_vals, [empirical[P] for P in P_vals],
               s=10, color='black', marker='o', label='Empirical', zorder=3)

    # Theoretical curve
    theory_x = np.linspace(1, 17, 200)
    theory_y = [theoretical_cost(N, P, W) for P in theory_x]
    ax.plot(theory_x, theory_y, color='black', linewidth=0.5, label='Theory', zorder=2)

    # Axis setup
    ax.set_xlim(1, 17)
    ax.set_ylim(0, 1000)
    ax.set_xticks([1, 5, 9, 13, 17])
    ax.set_yticks([0, 200, 400, 600, 800, 1000])
    ax.set_xlabel('Pivot Count (P)', fontsize=9, fontname=FONT_NAME, labelpad=6)
    ax.set_ylabel('Expected Listwise Ranker Calls', fontsize=9, fontname=FONT_NAME)
    ax.tick_params(axis='both', which='major', labelsize=6)

    # Styling
    for spine in ax.spines.values():
        spine.set_color('#666666')
    ax.legend(fontsize=9, loc='upper left', frameon=True,
              facecolor='white', edgecolor='none', prop=FONT_PROP)
    ax.set_xticks(range(1, 18), minor=True)
    ax.grid(which='major', color='lightgray', linewidth=0.5, zorder=0)

    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.15, top=0.975)

    # Save
    output_dir = os.path.join(SCRIPT_DIR, 'publication_figures')
    os.makedirs(output_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(output_dir, f'quicksort.{ext}'),
                    dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved to {output_dir}/quicksort.{{png,pdf}}")


if __name__ == '__main__':
    main()
