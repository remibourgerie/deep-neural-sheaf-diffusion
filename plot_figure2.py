#!/usr/bin/env python3
"""
Reproduce Figure 2: test accuracy vs depth at a given perturbation level.

Reads results from train_synthetic.py output CSV and produces the depth-scaling
plot from the paper (accuracy vs number of layers, one curve per model variant).

Usage:
    python plot_figure2.py --results results_synthetic.csv
    python plot_figure2.py --results results_synthetic.csv --level 5 --output figure2.png
"""

import argparse
import csv
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Visual encoding (matches paper Figure 2) ─────────────────────────────────

TIER_COLOR = {
    'adj_odd_gate': '#C0392B',   # dark red   — adj+odd+gate
    'adj_odd':      '#E67E22',   # orange     — adj+odd
    'adj_gate':     '#E74C3C',   # light red  — adj+gate
    'odd_gate':     '#8E44AD',   # violet     — odd+gate
    'adj':          '#2471A3',   # blue       — adj only
    'dnsd_plain':   '#1A9E6E',   # green      — no flags
    'nsd':          '#7D3C98',   # purple     — NSD v1
    'gat':          '#1A1A1A',   # near-black
    'baseline':     '#888888',   # grey       — mpnn
}

MODEL_ORDER = [
    # diag before full, matching notebook cell 35 pair order
    ('dnsd_diag', 'adj=True,odd=True,gate=True'),
    ('dnsd_full', 'adj=True,odd=True,gate=True'),
    ('dnsd_diag', 'adj=True,odd=True,gate=False'),
    ('dnsd_full', 'adj=True,odd=True,gate=False'),
    ('dnsd_diag', 'adj=True,odd=False,gate=True'),
    ('dnsd_full', 'adj=True,odd=False,gate=True'),
    ('dnsd_diag', 'adj=False,odd=True,gate=True'),
    ('dnsd_full', 'adj=False,odd=True,gate=True'),
    ('dnsd_diag', 'default'),
    ('dnsd_full', 'default'),
    ('nsd_diag',  'default'),
    ('nsd_full',  'default'),
    ('gat',       'default'),
    ('mpnn',      'default'),
]


def _tier(model, flags):
    has_adj  = 'adj=True'  in flags
    has_odd  = 'odd=True'  in flags
    has_gate = 'gate=True' in flags
    if has_adj and has_odd and has_gate:  return 'adj_odd_gate'
    if has_adj and has_odd:               return 'adj_odd'
    if has_adj and has_gate:              return 'adj_gate'
    if has_odd and has_gate:              return 'odd_gate'
    if has_adj:                           return 'adj'
    if model in ('dnsd_full', 'dnsd_diag'): return 'dnsd_plain'
    if model in ('nsd_full',  'nsd_diag'):  return 'nsd'
    if model == 'gat':                      return 'gat'
    return 'baseline'


def _display_name(model, flags):
    name = model.replace('dnsd_diag', 'DNSD-diag').replace('dnsd_full', 'DNSD-full') \
                .replace('nsd_diag', 'NSD-diag').replace('nsd_full', 'NSD-full') \
                .replace('gat', 'GAT').replace('mpnn', 'MPNN')
    if flags == 'default':
        return name
    # Keep only True flags, map to short labels matching notebook display_name()
    label_map = {'adj=True': 'adj', 'odd=True': 'odd', 'gate=True': 'gate'}
    parts = [label_map[p] for p in flags.split(',') if p in label_map]
    suffix = '-'.join(parts)
    return f"{name}-{suffix}" if suffix else name


def load_results(csv_path: str, level: int):
    """Load results filtered to given level. Returns {(model,flags,nl): [test_acc,...]}"""
    groups = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if int(row['level']) != level:
                continue
            key = (row['model'], row['flags'], int(row['num_layers']))
            try:
                groups[key].append(float(row['test_acc']))
            except ValueError:
                pass
    return groups


def aggregate(groups):
    """Return {(model,flags,nl): (mean, std)}"""
    agg = {}
    for key, accs in groups.items():
        mean = sum(accs) / len(accs)
        std = math.sqrt(sum((a - mean)**2 for a in accs) / len(accs)) if len(accs) > 1 else 0.0
        agg[key] = (mean, std)
    return agg


def plot(agg: dict, level: int, output_path: str = None):
    # Collect all (model, flags) pairs present in data
    present = set((m, f) for m, f, _ in agg.keys())

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.8, 1], wspace=0)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')

    plotted = {}

    for model, flags in [mf for mf in MODEL_ORDER if mf in present]:
        # Collect depths for this model+flags
        depths = sorted([nl for m, f, nl in agg.keys() if m == model and f == flags])
        if not depths:
            continue

        means = [agg[(model, flags, nl)][0] * 100 for nl in depths]
        stds  = [agg[(model, flags, nl)][1] * 100 for nl in depths]

        tier = _tier(model, flags)
        color = TIER_COLOR[tier]
        is_full = 'full' in model
        ls = '--' if is_full else '-'
        lw = 2.5 if tier == 'adj_odd_gate' else 1.8
        mfc = 'white' if is_full else color
        zorder = 4 if tier == 'adj_odd_gate' else 3
        ms = 7 if tier == 'adj_odd_gate' else 5

        line, = ax.plot(depths, means, color=color, ls=ls, lw=lw, zorder=zorder,
                        marker='o', ms=ms, mfc=mfc, mec=color, mew=1.5,
                        label=_display_name(model, flags))
        ax.fill_between(depths,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.10, color=color, zorder=zorder - 1)
        plotted[(model, flags)] = line

    all_depths = sorted(set(nl for _, _, nl in agg.keys()))
    ax.set_xlabel('Depth (number of layers)', fontsize=12)
    ax.set_ylabel('Test accuracy (%)', fontsize=12)
    ax.set_title(f'Community detection G{level} — depth scaling', fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_xticks([int(d) for d in all_depths])
    ax.set_ylim(28, 102)
    ax.grid(True, alpha=0.3)

    leg_items = [(m, f) for m, f in MODEL_ORDER if (m, f) in plotted]
    leg_handles = [plotted[mf] for mf in leg_items]
    leg_labels  = [_display_name(m, f) for m, f in leg_items]

    ax_leg.legend(leg_handles, leg_labels, ncol=1, fontsize=11,
                  loc='center', borderaxespad=0, frameon=True,
                  handlelength=2.2, labelspacing=0.55)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()

    # Print numeric table
    print(f"\n=== Accuracy table at G{level} ===")
    depths_sorted = sorted(set(nl for _, _, nl in agg.keys()))
    header = f"{'Model':<35}" + "".join(f"  L{d:2d}" for d in depths_sorted)
    print(header)
    for model, flags in [mf for mf in MODEL_ORDER if mf in present]:
        row = f"{_display_name(model, flags):<35}"
        for d in depths_sorted:
            key = (model, flags, d)
            if key in agg:
                mean, std = agg[key]
                row += f"  {mean*100:5.1f}"
            else:
                row += f"  {'—':>5}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='CSV file from train_synthetic.py')
    parser.add_argument('--level', type=int, default=5, help='Perturbation level to plot (default 5)')
    parser.add_argument('--output', default=None, help='Save figure to this path (e.g. figure2.png)')
    args = parser.parse_args()

    groups = load_results(args.results, args.level)
    if not groups:
        print(f"No data found for level={args.level} in {args.results}")
        return

    agg = aggregate(groups)
    plot(agg, args.level, args.output)


if __name__ == '__main__':
    main()
