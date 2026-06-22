# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "adjustText"]
# ///
"""Figure generation for the KORE paper.

Reads CSV outputs from ``results/`` and writes paired
``.pdf``/``.png`` files into ``paper/figures/``. The visual standard is
restraint: white facecolors, a single canonical color per method
(``SCHEME_COLORS``), no in-axis ratio annotations, panel labels rendered
as left-aligned italic-descriptor margin labels.
"""

from pathlib import Path
import json
import logging
import os

import numpy as np
import pandas as pd

logging.getLogger('fontTools').setLevel(logging.ERROR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Honor ``KORE_RESULTS_DIR`` / ``KORE_FIGURES_DIR`` so a non-editable
# install on a read-only checkout (Databricks Workspace Files,
# site-packages) can redirect inputs and outputs to a writable Volume or
# DBFS path without monkey-patching.
_RESULTS_DIR_OVERRIDE = os.environ.get('KORE_RESULTS_DIR')
_FIGURES_DIR_OVERRIDE = os.environ.get('KORE_FIGURES_DIR')
RESULTS = (
    Path(_RESULTS_DIR_OVERRIDE).expanduser()
    if _RESULTS_DIR_OVERRIDE
    else PROJECT_ROOT / 'results'
)
OUT = (
    Path(_FIGURES_DIR_OVERRIDE).expanduser()
    if _FIGURES_DIR_OVERRIDE
    else PROJECT_ROOT / 'paper' / 'figures'
)
OUT.mkdir(parents=True, exist_ok=True)

COL = 6.75
HALF = 3.25
PHI = (1 + np.sqrt(5)) / 2

# ---------------------------------------------------------------------
# Canonical palettes. Every method has one color paper-wide; every
# structural family has one color paper-wide; dimensions reuse the same
# four-color cycle wherever they appear.
# ---------------------------------------------------------------------

# A single navy hero (KORE) carried paper-wide, a warm clay for the CV
# gold standard, and a cohesive cool-slate family for the classical
# full-grid criteria so the "field" reads as one group behind the hero.
SCHEME_COLORS = {
    'KORE': '#16486B',   # deep navy -- the hero
    'CV':   '#C0652F',   # warm clay -- the accuracy gold standard
    'GCV':  '#6E859B',   # slate
    'Cp':   '#9AAAB8',   # light slate
    'AIC':  '#C2CAD2',   # pale slate
    'BIC':  '#4F708A',   # steel
}

STRUCT_COLORS = {
    'additive': '#2C7A7B',   # teal
    'pairwise': '#9C4F6C',   # muted rose
}

# Sequential ramp for ORDERED variables (spline degree, input dimension).
# Light to dark navy so "more" reads darker; this replaces the
# categorical-rainbow look of unrelated hues on an ordered axis.
_SEQ_RAMP = mcolors.LinearSegmentedColormap.from_list(
    'kore_seq', ['#AFC8DE', '#5A8FB8', '#16486B', '#0B2A40'])


def seq_colors(n, lo=0.12, hi=1.0):
    """``n`` colors along the navy ramp, light to dark."""
    return [_SEQ_RAMP(t) for t in np.linspace(lo, hi, n)]


DIM_LEVELS = [10, 20, 40, 80]
DIM_COLORS = dict(zip(DIM_LEVELS, seq_colors(len(DIM_LEVELS))))
DIM_MARKERS = {10: 'o', 20: 's', 40: 'D', 80: '^'}

DEGREE_LEVELS = [2, 3, 5]
DEGREE_COLORS = dict(zip(DEGREE_LEVELS, seq_colors(len(DEGREE_LEVELS))))
DEGREE_MARKERS = {2: 's', 3: 'o', 5: 'D'}

# Per-method palette for the real-world benchmark. KORE stays navy
# paper-wide; the spline-family selectors reuse SCHEME_COLORS so they read
# identically across the synthetic and real-world figures; the remaining
# families share a hue each (linear greens, tree ambers, booster reds,
# kernel purples, neighbours teal, neural rose) so the panel reads by
# family at a glance.
METHOD_COLORS = {
    'kore':              SCHEME_COLORS['KORE'],
    'cv_spline':         SCHEME_COLORS['CV'],
    'gcv_spline':        SCHEME_COLORS['GCV'],
    'cp_spline':         SCHEME_COLORS['Cp'],
    'aic_spline':        SCHEME_COLORS['AIC'],
    'bic_spline':        SCHEME_COLORS['BIC'],
    'pygam':             '#7FA8C9',
    'linear':            '#3F7D54',
    'ridge_cv':          '#5B9B72',
    'lasso_cv':          '#86B89A',
    'elasticnet_cv':     '#A9CDB6',
    'random_forest':     '#9C5A2D',
    'extra_trees':       '#BE7B3E',
    'hist_gbm':          '#D9A05B',
    'xgboost':           '#8A3324',
    'lightgbm':          '#B65B47',
    'catboost':          '#6B2018',
    'svr_rbf':           '#7D5BA6',
    'kernel_ridge_rbf':  '#A98BC9',
    'knn':               '#2C7A7B',
    'mlp':               '#B05A8A',
}

METHOD_LABEL = {
    'kore':              'KORE',
    'cv_spline':         'CV+spline',
    'gcv_spline':        'GCV+spline',
    'cp_spline':         'Cp+spline',
    'aic_spline':        'AIC+spline',
    'bic_spline':        'BIC+spline',
    'pygam':             'pyGAM',
    'linear':            'OLS',
    'ridge_cv':          'RidgeCV',
    'lasso_cv':          'LassoCV',
    'elasticnet_cv':     'ElasticNetCV',
    'random_forest':     'RandomForest',
    'extra_trees':       'ExtraTrees',
    'hist_gbm':          'HistGBM',
    'xgboost':           'XGBoost',
    'lightgbm':          'LightGBM',
    'catboost':          'CatBoost',
    'svr_rbf':           'SVR-RBF',
    'kernel_ridge_rbf':  'KernelRidge',
    'knn':               'KNN',
    'mlp':               'MLP',
}

# Aliases retained for backward compatibility with code that still uses
# the short single-letter names below.
C_KORE = SCHEME_COLORS['KORE']
C_CV = SCHEME_COLORS['CV']
C_GCV = SCHEME_COLORS['GCV']
C_ADD = STRUCT_COLORS['additive']
C_PAIR = STRUCT_COLORS['pairwise']

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'grid.color': '#cccccc',
    'grid.alpha': 0.4,
    'grid.linewidth': 0.4,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.3,
    'lines.markersize': 4.3,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def panel_label(ax, letter: str, descriptor: str = '', x: float = -0.16, y: float = 1.06):
    """Left-aligned panel label: bold ``(letter)`` followed by an optional
    italic descriptor in the margin above the axes.

    The descriptor is positioned by measuring the rendered bounding box of
    the bold ``(letter)`` text in display coordinates and adding a fixed
    pixel gap, then inverse-transforming back to axes-fraction
    coordinates. That keeps the visible gap constant across figure
    widths, axes counts, and figure scales, so ``(a)additive targets``
    collisions cannot recur regardless of the surrounding panel size.
    """
    t = ax.text(x, y, f'({letter})', transform=ax.transAxes,
                va='bottom', fontsize=9, fontweight='bold')
    if not descriptor:
        return
    fig = ax.figure
    fig.canvas.draw()
    bbox_disp = t.get_window_extent(renderer=fig.canvas.get_renderer())
    gap_px = 14.0  # constant visible gap between (letter) and descriptor
    inv = ax.transAxes.inverted()
    x_after, _ = inv.transform((bbox_disp.x1 + gap_px, bbox_disp.y0))
    ax.text(x_after, y, descriptor, transform=ax.transAxes,
            va='bottom', fontsize=9, fontstyle='italic')


def save(fig, name: str):
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png')
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure: effective-density collapse
# ---------------------------------------------------------------------

def fig_law_collapse():
    df = read_csv('law_collapse.csv')
    fig, axes = plt.subplots(1, 2, figsize=(COL, COL / (2.2 * PHI) + 0.45),
                             gridspec_kw={'wspace': 0.34})
    beta = 4

    def plot_panel(ax, family: str, exp: float, rho_label: str):
        sub = df[df['family'] == family].copy()
        all_d = sorted(sub['d'].unique())
        for d in all_d:
            cur = sub[sub['d'] == d]
            agg = cur.groupby('density').agg(
                y_mean=('rmse', 'mean'),
                y_std=('rmse', 'std'),
                y_count=('rmse', 'count'),
            ).reset_index()
            agg['y_se'] = agg['y_std'].fillna(0.0) / np.sqrt(agg['y_count'])
            ax.errorbar(agg['density'], agg['y_mean'], yerr=agg['y_se'],
                        fmt=DIM_MARKERS.get(d, 'o') + '-',
                        color=DIM_COLORS.get(d, 'gray'),
                        capsize=2, markersize=4.8, label=rf'$d={d}$')

        rho = np.linspace(sub['density'].min(), sub['density'].max(), 300)
        ref_d = all_d[len(all_d) // 2]
        ref = sub[sub['d'] == ref_d].groupby('density').agg(
            y_mean=('rmse', 'mean')).reset_index().sort_values('density')
        mid = len(ref) // 2
        rho0 = float(ref.iloc[mid]['density'])
        y0 = float(ref.iloc[mid]['y_mean'])
        ax.plot(rho, y0 * (rho / rho0) ** exp, ':', color='black', lw=1.0,
                label=rf'theory $\rho^{{{exp:.2f}}}$')

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(True, which='both', axis='both')
        ax.set_xlabel(rho_label)
        ax.set_ylabel(r'test RMSE at KORE-selected $G^\star$')
        despine(ax)
        ax.legend(frameon=False, fontsize=6.3, handlelength=1.3,
                  labelspacing=0.22, handletextpad=0.35, markerscale=0.7,
                  loc='best')

    plot_panel(axes[0], 'additive',
               -1.0 * beta / (2 * beta + 1), r'$\rho = n/d$')
    plot_panel(axes[1], 'pairwise',
               -1.0 * beta / (2 * beta + 2), r'$\rho = n/s$')

    panel_label(axes[0], 'a', 'additive targets')
    panel_label(axes[1], 'b', 'sparse pairwise targets')
    save(fig, 'fig_law_collapse')


# ---------------------------------------------------------------------
# Figures: frontier against exhaustive search. Three independent claims,
# three single-panel floats.
# ---------------------------------------------------------------------

def _frontier_labels(df):
    return [rf"{'Add' if fam == 'additive' else 'Pair'} $d={int(d)}$"
            for fam, d in zip(df['family'], df['d'])]


def fig_frontier_cost():
    df = read_csv('frontier_summary.csv').sort_values(['family', 'd']).reset_index(drop=True)
    labels = _frontier_labels(df)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(HALF + 1.05, 0.36 * len(df) + 1.10))
    for i in y:
        ax.plot([df.iloc[i]['kore_fits_mean'], df.iloc[i]['cv_fits_mean']],
                [i, i], color='#dddddd', lw=4, solid_capstyle='round', zorder=1)
    ax.scatter(df['cv_fits_mean'], y, s=30, color=SCHEME_COLORS['CV'], marker='s',
               edgecolors='white', lw=0.3, zorder=3, label='3-fold CV')
    ax.scatter(df['gcv_fits_mean'], y, s=22, color=SCHEME_COLORS['GCV'], marker='D',
               edgecolors='white', lw=0.3, zorder=3, label='GCV / Cp / AIC / BIC')
    ax.scatter(df['kore_fits_mean'], y, s=34, color=SCHEME_COLORS['KORE'], marker='o',
               edgecolors='white', lw=0.3, zorder=4, label='KORE')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('model fits for selection')
    ax.legend(frameon=False, fontsize=7, loc='lower right',
              handletextpad=0.3, borderaxespad=0.4)
    ax.invert_yaxis()
    ax.grid(True, axis='x')
    despine(ax)
    fig.tight_layout()
    save(fig, 'fig_frontier_cost')


def fig_frontier_accuracy():
    df = read_csv('frontier_summary.csv').sort_values(['family', 'd']).reset_index(drop=True)
    labels = _frontier_labels(df)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(HALF + 1.05, 0.36 * len(df) + 1.10))
    ratios = df['rmse_ratio_kore_cv']
    ax.barh(y, ratios, height=0.52, color=SCHEME_COLORS['KORE'],
            edgecolor='white', lw=0.3)
    ax.axvline(1.0, lw=0.7, ls='--', color='black', alpha=0.35)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    lo = max(0.92, float(ratios.min()) - 0.02)
    hi = max(1.04, float(ratios.max()) + 0.02)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r'KORE / CV RMSE')
    ax.invert_yaxis()
    ax.grid(True, axis='x')
    despine(ax)
    fig.tight_layout()
    save(fig, 'fig_frontier_accuracy')


def fig_frontier_summary():
    ms = read_csv('frontier_method_summary.csv')

    fig, ax = plt.subplots(figsize=(HALF + 0.55, (HALF + 0.55) / PHI + 0.30))
    order = ['KORE', 'GCV', 'Cp', 'BIC', 'AIC']
    ms_ord = ms.set_index('method').loc[order].reset_index()
    yy = np.arange(len(ms_ord))
    ax.barh(yy, ms_ord['gm_rmse_ratio_vs_cv'], height=0.62,
            color=[SCHEME_COLORS[m] for m in ms_ord['method']],
            edgecolor='white', lw=0.3)
    ax.axvline(1.0, lw=0.7, ls='--', color='black', alpha=0.35)
    ax.set_yticks(yy)
    ax.set_yticklabels(ms_ord['method'], fontsize=7)
    rmin = float(ms_ord['gm_rmse_ratio_vs_cv'].min())
    rmax = float(ms_ord['gm_rmse_ratio_vs_cv'].max())
    ax.set_xlim(min(0.94, rmin - 0.02), max(1.55, rmax + 0.05))
    ax.set_xlabel(r'geometric-mean RMSE / CV')
    ax.invert_yaxis()
    ax.grid(True, axis='x')
    despine(ax)
    fig.tight_layout()
    save(fig, 'fig_frontier_summary')


# ---------------------------------------------------------------------
# Figure: theory-aligned benchmark suite
# ---------------------------------------------------------------------

def fig_benchmarks():
    df = read_csv('benchmark_summary.csv')
    keep = [
        'Nguyen-1', 'Nguyen-4', 'Nguyen-5', 'Nguyen-7', 'Nguyen-9 (2D add)',
        'Nguyen-10 (2D int)', 'Friedman-1 (5D)', 'SparseAdd-20D', 'SparsePair-10D'
    ]
    sub = df[df['equation'].isin(keep)].copy().sort_values('ratio_kore_cv').reset_index(drop=True)
    ms = read_csv('benchmark_method_summary_main.csv')
    y = np.arange(len(sub))

    fig, axes = plt.subplots(
        1, 3, figsize=(COL, 0.32 * len(sub) + 1.1),
        gridspec_kw={'width_ratios': [1.20, 0.80, 0.95], 'wspace': 0.12},
        constrained_layout=True,
    )

    ax = axes[0]
    ax.barh(y, sub['ratio_kore_cv'], height=0.56,
            color=SCHEME_COLORS['KORE'], edgecolor='white', lw=0.3)
    ax.axvline(1.0, lw=0.7, ls='--', color='black', alpha=0.35)
    ax.set_yticks(y)
    ax.set_yticklabels(sub['equation'], fontsize=6.8)
    ax.set_xlabel(r'KORE / CV RMSE')
    ax.set_xlim(0.0, max(1.18, float(sub['ratio_kore_cv'].max()) + 0.05))
    ax.invert_yaxis()
    ax.grid(True, axis='x')
    despine(ax)
    panel_label(ax, 'a', 'smooth low-order benchmarks')

    ax = axes[1]
    ax.barh(y, sub['fit_speedup'], height=0.56,
            color=STRUCT_COLORS['additive'], edgecolor='white', lw=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel('CV fits / KORE fits')
    ax.set_xlim(0, max(13.0, float(sub['fit_speedup'].max()) + 1.0))
    ax.invert_yaxis()
    ax.grid(True, axis='x')
    despine(ax)
    panel_label(ax, 'b', 'fit-count reduction')

    ax = axes[2]
    order = ['KORE', 'GCV', 'Cp', 'BIC', 'AIC']
    ms_ord = ms.set_index('method').loc[order].reset_index()
    yy = np.arange(len(ms_ord))
    ax.barh(yy, ms_ord['gm_rmse_ratio_vs_cv'], height=0.62,
            color=[SCHEME_COLORS[m] for m in ms_ord['method']],
            edgecolor='white', lw=0.3)
    ax.axvline(1.0, lw=0.7, ls='--', color='black', alpha=0.35)
    ax.set_yticks(yy)
    ax.set_yticklabels(ms_ord['method'], fontsize=7)
    rmin = float(ms_ord['gm_rmse_ratio_vs_cv'].min())
    ax.set_xlim(min(0.86, rmin - 0.02), 1.04)
    ax.set_xlabel(r'GM RMSE / CV')
    ax.invert_yaxis()
    ax.grid(True, axis='x')
    despine(ax)
    panel_label(ax, 'c', 'method summary')

    save(fig, 'fig_benchmarks')


# ---------------------------------------------------------------------
# Appendix figure: full benchmark suite
# ---------------------------------------------------------------------

def fig_app_full_benchmarks():
    df = read_csv('benchmark_summary.csv').sort_values('ratio_kore_cv').reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(HALF + 1.65, 0.34 * len(df) + 0.95))
    for i in y:
        vals = [df.iloc[i]['kore_rmse_mean'],
                df.iloc[i]['cv_rmse_mean'],
                df.iloc[i]['gcv_rmse_mean']]
        ax.plot([min(vals), max(vals)], [i, i], lw=0.5, color='#eeeeee', zorder=0)
    ax.scatter(df['cv_rmse_mean'], y, s=22, marker='s',
               color=SCHEME_COLORS['CV'], zorder=3, label='CV')
    ax.scatter(df['kore_rmse_mean'], y, s=28, marker='o',
               color=SCHEME_COLORS['KORE'], zorder=4, label='KORE')
    ax.scatter(df['gcv_rmse_mean'], y, s=16, marker='D',
               color=SCHEME_COLORS['GCV'], zorder=2, label='GCV')
    ax.set_yticks(y)
    ax.set_yticklabels(df['equation'], fontsize=7)
    ax.set_xlabel('mean test RMSE')
    ax.set_xscale('log')
    ax.legend(frameon=False, fontsize=6.8, loc='best',
              handletextpad=0.4)
    ax.invert_yaxis()
    ax.grid(True, axis='x', which='both')
    despine(ax)
    fig.tight_layout()
    save(fig, 'fig_app_full_benchmarks')


# ---------------------------------------------------------------------
# Appendix figure: graph discovery
# ---------------------------------------------------------------------

def fig_app_discovery():
    df = read_csv('discovery.csv')
    best = df.loc[df.groupby(['d', 'n_per_pair', 'seed'])['f1'].idxmax()].reset_index(drop=True)
    agg = best.groupby(['d', 'n_per_pair']).agg(
        f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
        rr_mean=('rmse_ratio', 'mean'), rr_std=('rmse_ratio', 'std')).reset_index()
    agg['f1_se'] = agg['f1_std'].fillna(0.0) / np.sqrt(5)
    agg['rr_se'] = agg['rr_std'].fillna(0.0) / np.sqrt(5)

    fig, axes = plt.subplots(1, 2, figsize=(COL, COL / (2.2 * PHI) + 0.35),
                             gridspec_kw={'wspace': 0.38})

    ax = axes[0]
    for d in sorted(agg['d'].unique()):
        cur = agg[agg['d'] == d].sort_values('n_per_pair')
        ax.errorbar(cur['n_per_pair'], cur['f1_mean'], yerr=cur['f1_se'],
                    fmt=DIM_MARKERS.get(d, 'o') + '-',
                    color=DIM_COLORS.get(d, 'gray'),
                    capsize=2, markersize=5, label=rf'$d={d}$')
    ax.axhline(1.0, lw=0.6, ls=':', color='black', alpha=0.3)
    ax.set_xlabel(r'samples per true pair ($n / s$)')
    ax.set_ylabel(r'F1 score')
    ax.set_ylim(0.55, 1.05)
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(True, axis='both')
    despine(ax)
    panel_label(ax, 'a', 'graph recovery')

    ax = axes[1]
    for d in sorted(agg['d'].unique()):
        cur = agg[agg['d'] == d].sort_values('n_per_pair')
        ax.errorbar(cur['n_per_pair'], cur['rr_mean'], yerr=cur['rr_se'],
                    fmt=DIM_MARKERS.get(d, 'o') + '-',
                    color=DIM_COLORS.get(d, 'gray'),
                    capsize=2, markersize=5, label=rf'$d={d}$')
    ax.axhline(1.0, lw=0.6, ls=':', color='black', alpha=0.3)
    ax.set_xlabel(r'samples per true pair ($n / s$)')
    ax.set_ylabel(r'discovered / oracle RMSE')
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(True, axis='both')
    despine(ax)
    panel_label(ax, 'b', 'discovery cost')

    save(fig, 'fig_app_discovery')


# ---------------------------------------------------------------------
# Appendix figure: robustness
# ---------------------------------------------------------------------

def fig_app_robustness():
    df = read_csv('robustness.csv').copy()
    df['ratio'] = df['kore_rmse'] / df['cv_rmse']

    order = ['correct (control)', '3-way interactions', 'non-smooth', 'misspecified graph']
    short = {
        'correct (control)': 'control',
        '3-way interactions': '3-way',
        'non-smooth': 'non-smooth',
        'misspecified graph': 'wrong graph',
    }
    color_for = {
        'correct (control)': SCHEME_COLORS['KORE'],
        '3-way interactions': STRUCT_COLORS['additive'],
        'non-smooth': SCHEME_COLORS['CV'],
        'misspecified graph': STRUCT_COLORS['pairwise'],
    }

    fig, ax = plt.subplots(figsize=(COL * 0.62, COL / (2.0 * PHI) + 0.18))
    xpos = 0.0
    xticks = []
    xlabels = []
    rng = np.random.default_rng(7)
    for scenario in order:
        cur = df[df['scenario'] == scenario]
        for d in sorted(cur['d'].unique()):
            vals = cur[cur['d'] == d]['ratio'].values
            jit = rng.uniform(-0.10, 0.10, len(vals))
            ax.scatter(np.full_like(vals, xpos, dtype=float) + jit, vals,
                       s=18, color=color_for[scenario], alpha=0.78,
                       edgecolors='white', lw=0.3, zorder=3)
            ax.plot([xpos - 0.18, xpos + 0.18],
                    [np.median(vals), np.median(vals)],
                    color=color_for[scenario], lw=1.6, zorder=4)
            xticks.append(xpos)
            xlabels.append(f"{short[scenario]} $d={d}$")
            xpos += 1.0
        xpos += 0.8
    ax.axhline(1.0, lw=0.7, ls='--', color='black', alpha=0.35)
    ax.set_yscale('log')
    ax.set_yticks([0.8, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 50.0])
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel('KORE / CV RMSE')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=30, ha='right', fontsize=6.5)
    handles = [Line2D([0], [0], marker='o', ls='', color=color_for[s],
                      markersize=4, label=short[s]) for s in order]
    ax.legend(handles=handles, frameon=False, fontsize=6,
              loc='upper left', ncol=2, columnspacing=0.8)
    ax.grid(True, axis='y', which='major')
    despine(ax)
    save(fig, 'fig_app_robustness')


# ---------------------------------------------------------------------
# Appendix figure: scaling with dimension
# ---------------------------------------------------------------------

def fig_app_scaling():
    df = read_csv('scaling.csv')
    agg = df.groupby(['family', 'd']).agg(
        kore_time=('kore_time', 'mean'), cv_time=('cv_time', 'mean'),
        kore_fits=('kore_fits', 'mean'), cv_fits=('cv_fits', 'mean'),
        kore_rmse=('kore_rmse', 'mean'), cv_rmse=('cv_rmse', 'mean')).reset_index()
    agg['time_speedup'] = agg['cv_time'] / agg['kore_time']
    agg['fit_speedup'] = agg['cv_fits'] / agg['kore_fits']
    agg['rmse_ratio'] = agg['kore_rmse'] / agg['cv_rmse']

    fig, axes = plt.subplots(1, 2, figsize=(COL, COL / (2.2 * PHI) + 0.65),
                             gridspec_kw={'wspace': 0.36})
    fig.subplots_adjust(bottom=0.30)

    ax = axes[0]
    for family, marker in [('additive', 'o'), ('pairwise', 's')]:
        color = STRUCT_COLORS[family]
        cur = agg[agg['family'] == family].sort_values('d')
        ax.plot(cur['d'], cur['fit_speedup'], marker=marker,
                color=color, label=f'{family}: fit-count')
        ax.plot(cur['d'], cur['time_speedup'], marker=marker, ls='--',
                color=color, alpha=0.85, label=f'{family}: wall-clock')
    ax.set_xlabel(r'dimension $d$')
    ax.set_ylabel('CV / KORE cost')
    ax.legend(frameon=False, fontsize=5.8, loc='upper center',
              bbox_to_anchor=(0.5, -0.22), ncol=2, columnspacing=0.8,
              handlelength=1.4)
    ax.grid(True, axis='both')
    despine(ax)
    panel_label(ax, 'a', 'cost advantage with $d$')

    ax = axes[1]
    for family, marker in [('additive', 'o'), ('pairwise', 's')]:
        color = STRUCT_COLORS[family]
        cur = agg[agg['family'] == family].sort_values('d')
        ax.plot(cur['d'], cur['rmse_ratio'], marker=marker,
                color=color, label=family)
    ax.axhline(1.0, lw=0.7, ls='--', color='black', alpha=0.35)
    ax.set_xlabel(r'dimension $d$')
    ax.set_ylabel('KORE / CV RMSE')
    ax.legend(frameon=False, fontsize=6.5, loc='best')
    ax.grid(True, axis='both')
    despine(ax)
    panel_label(ax, 'b', 'accuracy parity')

    save(fig, 'fig_app_scaling')


# ---------------------------------------------------------------------
# Appendix figure: degree ablation
# ---------------------------------------------------------------------

def fig_app_degree_ablation():
    df = read_csv('degree_ablation.csv')
    sm = read_csv('degree_ablation_summary.csv')

    fig, ax = plt.subplots(figsize=(HALF + 0.35, (HALF + 0.35) / PHI))
    palette = DEGREE_COLORS
    markers = DEGREE_MARKERS

    for degree in sorted(df['degree'].unique()):
        sub = df[df['degree'] == degree]
        agg = sub.groupby('density').agg(
            g_mean=('G_dagger', 'mean'),
            g_std=('G_dagger', 'std'),
            count=('G_dagger', 'count'),
        ).reset_index()
        agg['g_se'] = agg['g_std'].fillna(0.0) / np.sqrt(agg['count'])
        ax.errorbar(
            agg['density'], agg['g_mean'], yerr=agg['g_se'],
            fmt=markers[degree] + '-', color=palette[degree], capsize=2,
            markersize=4.2, label=rf'$k={degree}$'
        )

        row = sm[sm['degree'] == degree].iloc[0]
        rho = np.linspace(agg['density'].min(), agg['density'].max(), 300)
        mid = len(agg) // 2
        rho0 = float(agg.iloc[mid]['density'])
        y0 = float(agg.iloc[mid]['g_mean'])
        slope = float(row['predicted_g_exponent'])
        ax.plot(rho, y0 * (rho / rho0) ** slope, ':', color=palette[degree], lw=0.9)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\rho = n / d$')
    ax.set_ylabel(r'plug-in resolution $\widehat{G}^\dagger$')
    ax.legend(frameon=False, fontsize=6.5, handlelength=1.3)
    ax.grid(True, axis='both', which='both')
    despine(ax)
    save(fig, 'fig_app_degree_ablation')


# ---------------------------------------------------------------------
# Figures: search-free plug-in consistency (Theorem 2). One claim, one
# figure: bias scale, noise scale, and plug-in vs population optimum
# each get their own single-panel float.
# ---------------------------------------------------------------------

def _consistency_xticks(ax, n_grid):
    ax.set_xticks(n_grid)
    ax.set_xticklabels([f'{int(n)}' for n in n_grid],
                       rotation=30, ha='right', fontsize=6.5)


def fig_consistency_bias():
    sm = read_csv('plugin_consistency_summary.csv').sort_values('n_train')
    n_grid = sm['n_train'].values

    fig, ax = plt.subplots(figsize=(HALF + 0.55, (HALF + 0.55) / PHI + 0.30))
    ax.fill_between(sm['n_train'], sm['A_ratio_q25'], sm['A_ratio_q75'],
                    color=SCHEME_COLORS['KORE'], alpha=0.18, lw=0.0)
    ax.plot(sm['n_train'], sm['A_ratio_median'], '-o',
            color=SCHEME_COLORS['KORE'], markersize=4)
    ax.axhline(1.0, lw=0.7, ls='--', color='black', alpha=0.35)
    ax.set_xscale('log', base=2)
    ax.set_xlabel(r'sample size $n$')
    ax.set_ylabel(r'$\widehat{A}_f / A_f$')
    _consistency_xticks(ax, n_grid)
    ax.grid(True, axis='both', which='both')
    despine(ax)
    fig.tight_layout()
    save(fig, 'fig_consistency_bias')


def fig_consistency_noise():
    sm = read_csv('plugin_consistency_summary.csv').sort_values('n_train')
    n_grid = sm['n_train'].values

    fig, ax = plt.subplots(figsize=(HALF + 0.55, (HALF + 0.55) / PHI + 0.30))
    ax.fill_between(sm['n_train'], sm['tau_ratio_q25'], sm['tau_ratio_q75'],
                    color=SCHEME_COLORS['KORE'], alpha=0.18, lw=0.0)
    ax.plot(sm['n_train'], sm['tau_ratio_median'], '-o',
            color=SCHEME_COLORS['KORE'], markersize=4)
    ax.axhline(1.0, lw=0.7, ls='--', color='black', alpha=0.35)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r'sample size $n$')
    ax.set_ylabel(r'$\widehat{\tau}_f / \tau_f$')
    _consistency_xticks(ax, n_grid)
    ax.grid(True, axis='both', which='both')
    despine(ax)
    fig.tight_layout()
    save(fig, 'fig_consistency_noise')


def fig_consistency_plugin():
    df = read_csv('plugin_consistency.csv')
    sm = read_csv('plugin_consistency_summary.csv').sort_values('n_train')
    n_grid = sm['n_train'].values

    fig, ax = plt.subplots(figsize=(HALF + 0.95, (HALF + 0.95) / PHI + 0.20))
    seed_means = (df.groupby('n_train')
                    .agg(mean=('G_dagger', 'mean'),
                         std=('G_dagger', 'std'))
                    .reset_index().sort_values('n_train'))
    ax.fill_between(seed_means['n_train'],
                    seed_means['mean'] - seed_means['std'],
                    seed_means['mean'] + seed_means['std'],
                    color=SCHEME_COLORS['KORE'], alpha=0.18, lw=0.0,
                    label=r'$\widehat{G}_f^\dagger \pm 1\sigma$')
    ax.plot(seed_means['n_train'], seed_means['mean'], '-o',
            color=SCHEME_COLORS['KORE'], markersize=4,
            label=r'$\widehat{G}_f^\dagger$ mean')
    ax.plot(sm['n_train'], sm['G_bullet'], 'D-', color='#444444',
            markersize=3.5, lw=0.9, label=r'$G_f^{\bullet}$ population')
    ax.set_xscale('log', base=2)
    ax.set_xlabel(r'sample size $n$')
    ax.set_ylabel(r'spline resolution')
    _consistency_xticks(ax, n_grid)
    ax.legend(frameon=False, fontsize=6.8, loc='best',
              handlelength=1.4, labelspacing=0.32)
    ax.grid(True, axis='both', which='both')
    despine(ax)
    fig.tight_layout()
    save(fig, 'fig_consistency_plugin')


# ---------------------------------------------------------------------
# Figure: conceptual illustration (purely illustrative)
# ---------------------------------------------------------------------

def fig_conceptual():
    """Conceptual bias-variance U-curve with KORE vs CV."""
    rng = np.random.default_rng(42)

    G_eval = np.arange(3, 21, dtype=float)
    G_fine = np.linspace(2.0, 21.5, 500)

    def bias_sq(G):
        return 2.1 * np.exp(-0.28 * G)

    def var_term(G):
        return 0.0013 * G ** 2

    def err(G):
        return 0.15 + bias_sq(G) + var_term(G)

    err_fine = err(G_fine)
    G_star_kore = float(G_fine[int(np.argmin(err_fine))])

    cv_means = err(G_eval) + rng.normal(0, 0.010, len(G_eval))
    cv_stds = 0.025 + 0.025 * np.abs(G_eval - G_star_kore) / 12

    fig, ax = plt.subplots(figsize=(HALF + 0.2, (HALF + 0.2) / PHI))

    ax.plot(G_fine, err_fine, color='black', lw=2.0, zorder=2,
            label=r'$\mathrm{Err}(G)$')
    ax.plot(G_fine, 0.15 + bias_sq(G_fine), '--',
            color=STRUCT_COLORS['additive'], lw=0.9, alpha=0.55,
            label=r'bias$^2$')
    ax.plot(G_fine, 0.15 + var_term(G_fine), '--',
            color=STRUCT_COLORS['pairwise'], lw=0.9, alpha=0.55,
            label='variance')

    ax.errorbar(G_eval, cv_means, yerr=cv_stds, fmt='o', color=SCHEME_COLORS['CV'],
                elinewidth=0.6, capsize=1.6, capthick=0.6, markersize=3.5,
                markeredgecolor='white', markeredgewidth=0.3,
                zorder=3, alpha=0.75, label='CV: search all $G$')

    y_star = err(G_star_kore)
    ax.scatter([G_star_kore], [y_star], s=180, marker='*',
               facecolors=SCHEME_COLORS['KORE'],
               edgecolors='white', lw=0.6, zorder=7,
               label=r'KORE: closed-form $G^\star$')

    ax.legend(frameon=False, fontsize=6, loc='upper right',
              handlelength=1.4, labelspacing=0.35, handletextpad=0.4)

    ax.set_xlabel(r'resolution $G$')
    ax.set_ylabel(r'test error')
    ax.set_xlim(2.0, 21.5)
    ax.set_ylim(0.20, 1.20)
    ax.grid(True, axis='both')
    despine(ax)
    fig.tight_layout()
    save(fig, 'fig_conceptual')


# ---------------------------------------------------------------------
# Figures: real-world benchmark (Section 4.4). Read CSVs produced by
# experiments.run_real_data; skip silently when the run has not been
# executed yet, since the synthetic figures must continue to render
# regardless.
# ---------------------------------------------------------------------


def _have_real_data() -> bool:
    return (RESULTS / 'real_data_summary.csv').exists() and (RESULTS / 'real_data_pareto.csv').exists()


# Quiet family palette for the real-data Pareto figure. KORE owns the
# canonical scheme blue; every other method takes the muted hue of its
# model family so the reader groups 21 methods into six visual bins at
# a glance rather than parsing 21 individual colours.
FAMILY_COLORS = {
    'linear':    '#3F7D54',
    'spline':    SCHEME_COLORS['KORE'],
    'tree':      '#BE7B3E',
    'kernel':    '#7D5BA6',
    'neighbors': '#2C7A7B',
    'neural':    '#B05A8A',
    'kore':      SCHEME_COLORS['KORE'],
}


def _pareto_frontier(points):
    """Return the lower-left Pareto frontier of (x, y) points, sorted by x."""
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    frontier = []
    best_y = float('inf')
    for x, y, m in pts:
        if y < best_y:
            frontier.append((x, y, m))
            best_y = y
    return frontier


def fig_real_data_pareto():
    """Headline Pareto frontier on the smooth-low-d subset (post-one-hot
    dimension at most 30), the pre-registered regime in which the
    bias-variance theory of Section 3 is calibrated. Both axes are
    ratios to KORE, log-scaled, so KORE sits at (1, 1). A wide aspect
    ratio stretches the horizontal compute decades and compresses the
    vertical RMSE band so the elbow at KORE is visible at a glance;
    family colouring groups 21 methods into six visual bins; Pareto-
    frontier methods carry full opacity and larger markers while
    dominated methods recede."""
    if not _have_real_data():
        return
    subset_path = RESULTS / 'real_data_subset.csv'
    if not subset_path.exists():
        return
    pareto = pd.read_csv(subset_path)
    if pareto.empty:
        return
    pareto = pareto[(pareto['gm_rmse_ratio'] > 0) & (pareto['total_fit_time_s'] > 0)].copy()
    kore_row = pareto[pareto['method'] == 'kore']
    if kore_row.empty:
        return
    kore_time = float(kore_row['total_fit_time_s'].iloc[0])
    pareto['time_ratio'] = pareto['total_fit_time_s'] / kore_time

    pts = list(zip(pareto['time_ratio'], pareto['gm_rmse_ratio'], pareto['method']))
    frontier_methods = {m for _, _, m in _pareto_frontier(pts)}

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    ax.axvline(1.0, lw=0.6, ls='--', color='#555555', alpha=0.45, zorder=1)
    ax.axhline(1.0, lw=0.6, ls='--', color='#555555', alpha=0.45, zorder=1)

    texts = []
    families_seen = []
    for _, row in pareto.iterrows():
        m = row['method']
        fam = row['family'] if m != 'kore' else 'kore'
        is_kore = (m == 'kore')
        on_frontier = m in frontier_methods
        color = FAMILY_COLORS.get(fam, '#888888')
        if is_kore:
            ax.scatter(
                row['time_ratio'], row['gm_rmse_ratio'],
                s=240, marker='*', color=color,
                edgecolors='#B8860B', lw=1.3, zorder=8,
            )
        else:
            ax.scatter(
                row['time_ratio'], row['gm_rmse_ratio'],
                s=58 if on_frontier else 26,
                marker='o', color=color,
                edgecolors='white', lw=0.5,
                alpha=1.0 if on_frontier else 0.55,
                zorder=6 if on_frontier else 4,
            )
            if fam not in families_seen:
                families_seen.append(fam)
        texts.append(ax.text(
            row['time_ratio'], row['gm_rmse_ratio'],
            METHOD_LABEL.get(m, m),
            fontsize=7.2 if (is_kore or on_frontier) else 6.6,
            fontweight='bold' if is_kore else 'normal',
            color=color if (is_kore or on_frontier) else '#555555',
            alpha=1.0 if (is_kore or on_frontier) else 0.85,
            zorder=9 if is_kore else (7 if on_frontier else 5),
        ))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1.5e-3, 1.3e3)
    ax.set_ylim(0.89, 2.05)
    ax.set_xlabel('total fit time / KORE')
    ax.set_ylabel('geometric-mean RMSE / KORE')
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3])
    ax.set_xticklabels([r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', '1', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$'])
    ax.set_yticks([0.95, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0])
    ax.set_yticklabels(['0.95', '1.0', '1.1', '1.25', '1.5', '1.75', '2.0'])
    ax.minorticks_off()
    ax.grid(True, which='major', axis='both')
    despine(ax)

    family_label = {
        'linear': 'linear', 'spline': 'spline', 'tree': 'tree',
        'kernel': 'kernel', 'neighbors': 'k-NN', 'neural': 'neural',
    }
    legend_handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=FAMILY_COLORS[f], markeredgecolor='white',
               markeredgewidth=0.5, markersize=5.5, label=family_label[f])
        for f in ['linear', 'spline', 'tree', 'kernel', 'neighbors', 'neural']
        if f in families_seen
    ]
    legend_handles.append(Line2D([0], [0], marker='*', color='none',
                                 markerfacecolor=FAMILY_COLORS['kore'],
                                 markeredgecolor='#B8860B', markeredgewidth=1.0,
                                 markersize=10, label='KORE'))
    ax.legend(handles=legend_handles, loc='lower left', frameon=False,
              ncol=1, handletextpad=0.5, labelspacing=0.4, borderpad=0.3)

    try:
        from adjustText import adjust_text
        adjust_text(
            texts, ax=ax,
            expand=(1.18, 1.35),
            arrowprops=dict(arrowstyle='-', color='#999999', lw=0.45, alpha=0.55, shrinkA=4, shrinkB=2),
            only_move={'text': 'xy', 'static': 'xy', 'explode': 'xy', 'pull': 'xy'},
            min_arrow_len=2.0,
        )
    except ImportError:
        pass

    fig.tight_layout()
    save(fig, 'fig_real_data_pareto')


def _select_top_competitors(summary: pd.DataFrame, n: int = 4) -> list:
    """Return the n strongest non-KORE methods by per-dataset gap in
    Compute-Normalized Lift over OLS against KORE. ``cnl_median`` is
    the per-cell ``max(0, max(0, r2) - max(0, r2_ols)) /
    (1 + fit_time_s)`` written by ``_aggregate_real_data_results``;
    higher CNL is better, so a smaller (CNL_kore - CNL_competitor)
    gap means a stronger competitor. The ``linear`` method is excluded
    by construction since it is the OLS baseline (CNL identically
    zero)."""
    if 'cnl_median' not in summary.columns:
        return []
    pivot = summary.pivot_table(index='dataset', columns='method',
                                values='cnl_median')
    if 'kore' not in pivot.columns:
        return []
    ref = pivot['kore']
    rows = []
    for m in pivot.columns:
        if m == 'linear':
            continue
        if m == 'kore':
            continue
        diff = (ref - pivot[m]).dropna()
        if diff.empty:
            continue
        rows.append((m, float(diff.mean())))
    rows.sort(key=lambda r: r[1])
    return [m for m, _ in rows[:n]]


def _fig_real_data_forest(name: str, subset_only: bool):
    if not _have_real_data():
        return
    summary = pd.read_csv(RESULTS / 'real_data_summary.csv')
    if 'cnl_median' not in summary.columns:
        return
    if subset_only:
        subset_path = RESULTS / 'real_data_subset.csv'
        if not subset_path.exists():
            return
        subset_methods = pd.read_csv(subset_path)
        if subset_methods.empty:
            return
        full = pd.read_csv(RESULTS / 'real_data.csv')
        if 'dataset' not in full.columns:
            return
        smooth_lowd = set(full[full.get('d', 0).fillna(0).astype(int) <= 30]['dataset'].unique())
        summary = summary[summary['dataset'].isin(smooth_lowd)]

    if summary.empty:
        return

    competitors = _select_top_competitors(summary, n=4)
    if not competitors:
        return
    pivot = summary.pivot_table(index='dataset', columns='method',
                                values='cnl_median')
    if 'kore' not in pivot.columns:
        return
    # Per-dataset CNL gap versus KORE: positive = KORE has higher
    # Compute-Normalized Lift over OLS on that dataset; negative =
    # competitor has the higher CNL. Floor each CNL at a small epsilon
    # so the log axis stays defined when a competitor adds no
    # detectable lift over OLS.
    eps = 1e-3
    log_ratio = np.log10((pivot['kore'].clip(lower=eps).to_frame().values
                          / pivot[competitors].clip(lower=eps).values))
    ratio = pd.DataFrame(log_ratio, index=pivot.index, columns=competitors)

    cols = [m for m in competitors if m in ratio.columns]
    ratio = ratio[cols].dropna(how='all')
    sort_col = cols[0]
    ratio = ratio.sort_values(sort_col, ascending=True)

    # Clip the view window so the panel stays readable. Out-of-range
    # values are drawn as edge arrows at the boundary; readers can see
    # the dataset name and that the method fell off scale without the
    # whole panel collapsing onto the few outlier rows.
    lo_clip, hi_clip = -1.0, 3.0

    y = np.arange(len(ratio))
    fig, ax = plt.subplots(figsize=(HALF + 2.55, 0.30 * len(ratio) + 1.55))
    n_methods = len(cols)
    for j, m in enumerate(cols):
        vals = ratio[m].to_numpy()
        yj = y + (j - (n_methods - 1) / 2) * 0.14
        color = METHOD_COLORS.get(m, '#888888')
        finite = np.isfinite(vals)
        in_range = finite & (vals >= lo_clip) & (vals <= hi_clip)
        below = finite & (vals < lo_clip)
        above = finite & (vals > hi_clip)
        ax.scatter(vals[in_range], yj[in_range],
                   s=24, marker='o', color=color,
                   edgecolors='white', lw=0.3,
                   label=METHOD_LABEL.get(m, m), zorder=3)
        if below.any():
            ax.scatter(np.full(below.sum(), lo_clip), yj[below],
                       s=30, marker='<', color=color,
                       edgecolors='white', lw=0.3, zorder=3)
        if above.any():
            ax.scatter(np.full(above.sum(), hi_clip), yj[above],
                       s=30, marker='>', color=color,
                       edgecolors='white', lw=0.3, zorder=3)
    ax.axvline(0.0, lw=0.8, ls='--', color=METHOD_COLORS['kore'], alpha=0.55,
               zorder=2)
    ax.text(0.0, -0.6, 'KORE', ha='center', va='bottom',
            fontsize=6.5, color=METHOD_COLORS['kore'], fontweight='bold')
    ax.set_xlim(lo_clip, hi_clip)
    ax.set_xticks([-1, 0, 1, 2, 3])
    ax.set_xticklabels([r'$0.1\times$', r'$1\times$', r'$10\times$',
                        r'$100\times$', r'$1000\times$'])
    ax.set_yticks(y)
    ax.set_yticklabels(ratio.index, fontsize=6.5)
    ax.set_xlabel(r'$\mathrm{CNL}_{\mathrm{KORE}}\,/\,\mathrm{CNL}_{\mathrm{competitor}}$  '
                  r'(Compute-Normalized Lift over OLS, log axis)')
    ax.legend(frameon=False, fontsize=6.5, loc='center left',
              bbox_to_anchor=(1.02, 0.5), handletextpad=0.3,
              borderaxespad=0.0)
    ax.invert_yaxis()
    ax.grid(True, axis='x', which='major', alpha=0.35)
    despine(ax)
    fig.tight_layout()
    save(fig, name)


def fig_real_data_full_cnl():
    _fig_real_data_forest('fig_real_data_full_cnl', subset_only=False)


def fig_real_data_subset_cnl():
    _fig_real_data_forest('fig_real_data_subset_cnl', subset_only=True)


def fig_real_data_significance():
    if not _have_real_data():
        return
    sig_path = RESULTS / 'real_data_significance.csv'
    if not sig_path.exists():
        return
    sig = pd.read_csv(sig_path)
    if sig.empty or 'median_delta' not in sig.columns:
        return
    # The headline significance file carries the per-cell CNL
    # ``median_delta`` and ``direction`` columns at alpha = 1 (see
    # ``_aggregate_real_data_results``). A positive median delta means
    # KORE has the higher Compute-Normalized Lift over OLS on the
    # typical paired cell; a negative median delta means the competitor
    # does.
    sig['kore_wins'] = sig['median_delta'].astype(float) > 0
    sig = sig.sort_values(['kore_wins', 'wilcoxon_p_holm'],
                          ascending=[False, True]).reset_index(drop=True)

    y = np.arange(len(sig))
    neg_log_p = -np.log10(np.clip(sig['wilcoxon_p_holm'].values, 1e-300, 1.0))
    kore_color = METHOD_COLORS['kore']
    loss_color = '#B0535A'
    bar_colors = [kore_color if w else loss_color for w in sig['kore_wins']]

    fig, ax = plt.subplots(figsize=(HALF + 2.45, 0.30 * len(sig) + 1.45))
    ax.barh(y, neg_log_p, height=0.62, color=bar_colors,
            edgecolor='white', lw=0.4, zorder=3)
    ax.axvline(-np.log10(0.05), lw=0.7, ls='--', color='black', alpha=0.45,
               zorder=2)
    ax.text(-np.log10(0.05), -0.7, r'$p_{\mathrm{Holm}}=0.05$',
            ha='center', va='bottom', fontsize=6.5, color='#444444')
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABEL.get(m, m) for m in sig['method']],
                       fontsize=7)
    ax.set_xlabel(r'$-\log_{10}$ Holm-corrected $p$')
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=kore_color, edgecolor='white', lw=0.4),
        plt.Rectangle((0, 0), 1, 1, facecolor=loss_color, edgecolor='white', lw=0.4),
    ]
    ax.legend(handles,
              ['KORE has higher CNL',
               'competitor has higher CNL'],
              frameon=False, fontsize=6.5, loc='lower right',
              handletextpad=0.4, handlelength=1.0)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.35)
    despine(ax)
    fig.tight_layout()
    save(fig, 'fig_real_data_significance')


def fig_real_data_memory():
    """Per-cell peak resident-set size on the real-world benchmark.

    Reads the per-cell ``rss_peak_mb`` column of ``real_data.csv`` and
    plots two panels: (a) the empirical CDF of cell peak RSS aggregated
    by method family, with the soft per-cell cap drawn as a vertical
    reference line; (b) the top-five (method, dataset) RSS offenders as
    a horizontal bar chart. The figure documents that steady-state
    per-worker RSS stays inside the cap for every family, with a small
    long tail of legitimately heavy outliers on the largest datasets.
    """
    if not _have_real_data():
        return
    full = RESULTS / 'real_data.csv'
    if not full.exists():
        return
    df = pd.read_csv(full)
    if 'rss_peak_mb' not in df.columns:
        return
    mem = df.dropna(subset=['rss_peak_mb']).copy()
    mem = mem[mem['rss_peak_mb'] > 0]
    if mem.empty:
        return
    cap_mb = float(os.environ.get('KORE_CELL_RSS_CAP_MB', 8000.0))

    fig = plt.figure(figsize=(COL + 0.4, COL / 2.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.55)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Carve KORE out of the spline family so the closed-form selector is
    # not confounded with the classical full-grid spline criteria, which
    # have very different per-cell RSS profiles.
    is_kore = mem['method'] == 'kore'
    fams_other = ['spline', 'tree', 'kernel', 'neighbors', 'neural', 'linear']
    series = [('kore', mem.loc[is_kore, 'rss_peak_mb'].to_numpy())]
    for fam in fams_other:
        sub = mem.loc[(~is_kore) & (mem['family'] == fam), 'rss_peak_mb'].to_numpy()
        series.append((fam, sub))
    for fam, sub in series:
        if sub.size == 0:
            continue
        sub_sorted = np.sort(sub)
        cdf = np.arange(1, len(sub_sorted) + 1) / len(sub_sorted)
        color = FAMILY_COLORS.get(fam, '#888888')
        ax_a.plot(sub_sorted, cdf, lw=1.4, color=color, label=fam, zorder=3)
    ax_a.axvline(cap_mb, lw=0.7, ls='--', color='black', alpha=0.55, zorder=2)
    ax_a.text(cap_mb * 1.10, 0.55, f'{cap_mb:.0f} MiB cap', ha='left',
              va='center', fontsize=6.6, color='#444444', rotation=90)
    ax_a.set_xscale('log')
    ax_a.set_xlabel(r'per-cell peak RSS  (MiB, log)', fontsize=8.0)
    ax_a.set_ylabel(r'empirical CDF', fontsize=8.0)
    ax_a.set_ylim(0, 1.02)
    ax_a.grid(True, which='both', alpha=0.35)
    ax_a.legend(frameon=False, fontsize=6.4,
                loc='upper center', bbox_to_anchor=(0.5, -0.18),
                ncol=4, handletextpad=0.4, handlelength=1.4,
                columnspacing=1.1)
    despine(ax_a)
    panel_label(ax_a, 'a', 'distribution by family')

    top = (mem.groupby(['method', 'dataset'])['rss_peak_mb']
              .max().sort_values(ascending=False).head(8).reset_index())
    labels = [f"{METHOD_LABEL.get(m, m)} on {ds}"
              for m, ds in zip(top['method'], top['dataset'])]
    colors = [FAMILY_COLORS.get(_method_family(m), '#888888')
              for m in top['method']]
    y = np.arange(len(top))
    ax_b.barh(y, top['rss_peak_mb'].to_numpy(), height=0.62,
              color=colors, edgecolor='white', lw=0.4, zorder=3)
    ax_b.axvline(cap_mb, lw=0.7, ls='--', color='black', alpha=0.55, zorder=2)
    ax_b.set_xscale('log')
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels, fontsize=6.8)
    ax_b.invert_yaxis()
    ax_b.set_xlabel(r'peak RSS  (MiB, log)', fontsize=8.0)
    ax_b.grid(True, axis='x', which='both', alpha=0.35)
    despine(ax_b)
    panel_label(ax_b, 'b', 'top RSS offenders')

    save(fig, 'fig_real_data_memory')


def fig_real_data_kore_vs_knn():
    """Per-dataset CNL scatter of KORE versus the kNN baseline.

    Each marker is a single dataset's median Compute-Normalized Lift
    over OLS at alpha = 1. Points above the y=x diagonal are datasets
    on which KORE extracts more OLS-relative lift per unit compute than
    kNN; points below are the converse.
    """
    if not _have_real_data():
        return
    s = pd.read_csv(RESULTS / 'real_data_summary.csv')
    if 'cnl_median' not in s.columns:
        return
    kore = s[s['method'] == 'kore'].set_index('dataset')['cnl_median']
    knn = s[s['method'] == 'knn'].set_index('dataset')['cnl_median']
    common = kore.index.intersection(knn.index)
    if len(common) == 0:
        return
    x = knn.loc[common].to_numpy()
    y = kore.loc[common].to_numpy()
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    hi = float(max(x.max(), y.max())) * 1.05
    if hi <= 0:
        hi = 1.0
    ax.plot([0, hi], [0, hi], color='#444444', lw=0.8, ls='--', alpha=0.5,
            zorder=2)
    ax.scatter(x, y, s=34, color=SCHEME_COLORS['KORE'],
               edgecolors='#0a3a66', lw=0.5, alpha=0.8, zorder=3)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('CNL: kNN')
    ax.set_ylabel('CNL: KORE')
    ax.grid(True, which='major', alpha=0.4)
    despine(ax)
    fig.tight_layout()
    save(fig, 'fig_real_data_kore_vs_knn')


def fig_real_data_rank_vs_n():
    """KORE Friedman rank versus training-size quartile.

    Bins datasets into four roughly equal-sized groups by ``n``. Per
    quartile, computes the mean per-dataset rank for every method (low
    rank = better, matching the Friedman ``-cnl_median`` convention).
    KORE is plotted in the canonical scheme blue; the remaining methods
    sit as a faint backdrop.
    """
    if not _have_real_data():
        return
    meta_path = RESULTS / 'dataset_metadata.csv'
    if not meta_path.exists():
        return
    s = pd.read_csv(RESULTS / 'real_data_summary.csv')
    meta = pd.read_csv(meta_path)
    if 'cnl_median' not in s.columns or 'n' not in meta.columns:
        return
    pivot = s.pivot_table(index='dataset', columns='method',
                          values='cnl_median')
    cm = [m for m in pivot.columns if pivot[m].notna().all()]
    if not cm or 'kore' not in cm:
        return
    rank_pivot = (-pivot[cm]).rank(axis=1, method='average')
    meta = meta.set_index('dataset')
    n_per = meta['n'].reindex(rank_pivot.index)
    valid = n_per.dropna()
    rank_pivot = rank_pivot.loc[valid.index]
    n_per = valid

    quartiles = pd.qcut(n_per, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                        duplicates='drop')
    cuts = pd.qcut(n_per, q=4, retbins=True, duplicates='drop')[1]
    qlabels = list(quartiles.cat.categories)
    quart_means = rank_pivot.groupby(quartiles, observed=False).mean()
    x = np.arange(len(qlabels))

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for m in cm:
        if m == 'kore':
            continue
        ax.plot(x, quart_means[m].to_numpy(), color='#999999', lw=0.8,
                alpha=0.3, zorder=2)
    ax.plot(x, quart_means['kore'].to_numpy(), color=SCHEME_COLORS['KORE'],
            lw=1.8, marker='o', markersize=6,
            markeredgecolor='white', markeredgewidth=0.6,
            label='KORE', zorder=4)

    backdrop = Line2D([0], [0], color='#999999', lw=0.8, alpha=0.6,
                      label='other 20 methods')
    kore_handle = Line2D([0], [0], color=SCHEME_COLORS['KORE'], lw=1.8,
                         marker='o', markersize=6,
                         markeredgecolor='white', markeredgewidth=0.6,
                         label='KORE')
    ax.legend(handles=[kore_handle, backdrop], frameon=False,
              loc='center left', bbox_to_anchor=(1.02, 0.5),
              handletextpad=0.4, borderaxespad=0.0)

    tick_labels = []
    for i, q in enumerate(qlabels):
        lo = cuts[i]
        hi = cuts[i + 1]
        if i == 0:
            tick_labels.append(f'{q}\n(n<{int(round(hi))})')
        elif i == len(qlabels) - 1:
            tick_labels.append(f'{q}\n(n>={int(round(lo))})')
        else:
            tick_labels.append(f'{q}\n({int(round(lo))}-{int(round(hi))})')
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=7.5)
    ax.set_xlabel('Training-size quartile')
    ax.set_ylabel('Mean Friedman rank (lower is better)')
    ax.grid(True, axis='y', alpha=0.4)
    despine(ax)
    fig.tight_layout()
    save(fig, 'fig_real_data_rank_vs_n')


def fig_real_data_rank_vs_d():
    """KORE per-dataset Friedman rank versus post-one-hot dimension.

    Single panel scatter of KORE's per-dataset rank on ``-cnl_median``
    against the post-one-hot dimension ``d_onehot``. A vertical
    reference at ``d = 30`` marks the smooth-low-d cutoff. Two
    horizontal segments report the median KORE rank in each region.
    """
    if not _have_real_data():
        return
    meta_path = RESULTS / 'dataset_metadata.csv'
    if not meta_path.exists():
        return
    s = pd.read_csv(RESULTS / 'real_data_summary.csv')
    meta = pd.read_csv(meta_path)
    if 'cnl_median' not in s.columns or 'd_onehot' not in meta.columns:
        return
    pivot = s.pivot_table(index='dataset', columns='method',
                          values='cnl_median')
    cm = [m for m in pivot.columns if pivot[m].notna().all()]
    if not cm or 'kore' not in cm:
        return
    rank_pivot = (-pivot[cm]).rank(axis=1, method='average')
    kore_rank = rank_pivot['kore']
    meta = meta.set_index('dataset')
    d_per = meta['d_onehot'].reindex(kore_rank.index)
    df = pd.DataFrame({'d': d_per, 'rank': kore_rank}).dropna()
    if df.empty:
        return
    df = df.sort_values('d')

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.scatter(df['d'].to_numpy(), df['rank'].to_numpy(),
               s=34, color=SCHEME_COLORS['KORE'], alpha=0.7,
               edgecolors='white', lw=0.4, zorder=3)

    low = df[df['d'] <= 30.0]
    high = df[df['d'] > 30.0]
    if not low.empty:
        med_low = float(low['rank'].median())
        ax.hlines(med_low, low['d'].min(), 30.0,
                  color='#003c66', lw=2.0, alpha=0.85, zorder=4)
    if not high.empty:
        med_high = float(high['rank'].median())
        ax.hlines(med_high, 30.0, high['d'].max(),
                  color='#003c66', lw=2.0, alpha=0.85, zorder=4)

    ax.axvline(30.0, color='#444444', lw=0.8, ls='--', alpha=0.5, zorder=2)
    y_max = float(df['rank'].max())
    y_min = float(df['rank'].min())
    y_pad = 0.05 * (y_max - y_min)
    ax.set_ylim(y_min - y_pad, y_max + 2.5 * y_pad)
    ax.text(29.3, y_max + 1.4 * y_pad, 'd = 30 cutoff',
            ha='right', va='center', fontsize=7, color='#444444')
    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='',
               color=SCHEME_COLORS['KORE'], alpha=0.7,
               markersize=6, markeredgecolor='white', markeredgewidth=0.4,
               label='per dataset'),
        Line2D([0], [0], color='#003c66', lw=2.0, alpha=0.85,
               label='median in region'),
    ]
    ax.legend(handles=legend_handles, loc='center right',
              bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)

    ax.set_xlabel('Post-one-hot dimension d')
    ax.set_ylabel('KORE Friedman rank (lower is better)')
    ax.grid(True, alpha=0.4)
    despine(ax)
    fig.tight_layout()
    save(fig, 'fig_real_data_rank_vs_d')


def fig_real_data_joint_significance():
    """Two-panel figure for the Compute-Normalized Lift Wilcoxon test.

    Panel (a): at the headline compute weight alpha = 1, methods are
    sorted by Holm-corrected p-value of the per-cell paired statistic
    delta = CNL_alpha(kore) - CNL_alpha(competitor). Bars are
    -log10(p_holm), colored by direction (KORE-better vs KORE-worse on
    Compute-Normalized Lift over OLS).

    Panel (b): sensitivity sweep over alpha in {0, 0.25, 0.5, 1, 2};
    the line shows the count of methods with significantly lower CNL
    than KORE at p_holm < 0.05, and the count with significantly
    higher CNL than KORE.
    """
    if not _have_real_data():
        return
    path = RESULTS / 'real_data_joint_significance.csv'
    if not path.exists():
        return
    sig = pd.read_csv(path)
    if sig.empty:
        return

    headline = sig[sig['alpha'] == 1.0].copy()
    if headline.empty:
        return
    headline = headline.sort_values('wilcoxon_p_holm').reset_index(drop=True)

    kore_color = METHOD_COLORS['kore']
    loss_color = '#B0535A'
    bar_colors = [
        kore_color if d == 'kore_better' else
        loss_color if d == 'kore_worse' else '#999999'
        for d in headline['direction']
    ]
    neg_log_p = -np.log10(np.clip(headline['wilcoxon_p_holm'].values, 1e-300, 1.0))

    fig = plt.figure(figsize=(COL, 0.30 * len(headline) + 2.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    y = np.arange(len(headline))
    ax_a.barh(y, neg_log_p, height=0.62, color=bar_colors,
              edgecolor='white', lw=0.4, zorder=3)
    ax_a.axvline(-np.log10(0.05), lw=0.7, ls='--', color='black', alpha=0.45,
                 zorder=2)
    ax_a.text(-np.log10(0.05), -1.2, r'$p_{\mathrm{Holm}}=0.05$',
              ha='center', va='bottom', fontsize=6.5, color='#444444')
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([METHOD_LABEL.get(m, m) for m in headline['method']],
                         fontsize=7.4)
    ax_a.set_xlabel(r'$-\log_{10}$ Holm-corrected $p$  ($\alpha=1$)',
                    fontsize=8.0)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=kore_color, edgecolor='white', lw=0.4),
        plt.Rectangle((0, 0), 1, 1, facecolor=loss_color, edgecolor='white', lw=0.4),
    ]
    ax_a.legend(handles,
                ['KORE has higher CNL',
                 'competitor has higher CNL'],
                frameon=False, fontsize=6.4,
                loc='upper center', bbox_to_anchor=(0.5, -0.13),
                ncol=2, handletextpad=0.4, handlelength=1.0)
    ax_a.invert_yaxis()
    ax_a.grid(True, axis='x', alpha=0.35)
    despine(ax_a)
    panel_label(ax_a, 'a', 'Compute-Normalized Lift over OLS')

    alphas = sorted(sig['alpha'].unique())
    n_better = []
    n_worse = []
    for a in alphas:
        s = sig[sig['alpha'] == a]
        n_better.append(int((s['direction'] == 'kore_better').sum()))
        n_worse.append(int((s['direction'] == 'kore_worse').sum()))
    ax_b.plot(alphas, n_better, marker='o', lw=1.4, color=kore_color,
              label='KORE higher CNL', zorder=4)
    ax_b.plot(alphas, n_worse, marker='s', lw=1.4, color=loss_color,
              label='competitor higher CNL', zorder=4)
    ax_b.set_xlabel(r'compute weight $\alpha$', fontsize=8.0)
    ax_b.set_ylabel(r'methods at $p_{\mathrm{Holm}}<0.05$', fontsize=8.0)
    major_ticks = [a for a in alphas if a in (0.0, 1.0, 2.0)]
    minor_ticks = [a for a in alphas if a not in major_ticks]
    ax_b.set_xticks(minor_ticks, minor=True)
    ax_b.set_xticks(major_ticks)
    ax_b.set_xticklabels([f'{a:g}' for a in major_ticks], fontsize=7.4)
    ax_b.set_ylim(bottom=0)
    ax_b.grid(True, axis='y', alpha=0.35)
    ax_b.legend(frameon=False, fontsize=6.8, loc='center right',
                handletextpad=0.4, handlelength=1.6)
    despine(ax_b)
    panel_label(ax_b, 'b', r'sensitivity to $\alpha$')

    save(fig, 'fig_real_data_joint_significance')


def _cd_equivalence_groups(sorted_methods, ranks, cd):
    """Maximal cliques of methods within CD of each other, sorted by
    left endpoint. Returns a list of (i, j) index pairs into
    sorted_methods such that ranks[j] - ranks[i] <= cd."""
    groups = []
    n = len(sorted_methods)
    for i in range(n):
        j = i
        while j + 1 < n and ranks[j + 1] - ranks[i] <= cd:
            j += 1
        if j > i:
            groups.append((i, j))
    out = []
    for g in groups:
        if not any(o[0] <= g[0] and o[1] >= g[1] and o != g for o in groups):
            out.append(g)
    return out


def fig_real_data_cd():
    """Demsar 2006 critical-difference diagram for the Friedman/Nemenyi
    test on the Compute-Normalized Lift over OLS
    ``cnl = max(0, max(0, r2_test) - max(0, r2_ols)) / (1 + fit_time_s)``
    (alpha = 1). The horizontal axis is mean rank on ``-cnl_median``
    (lower mean rank = larger OLS-relative lift per unit compute);
    methods whose mean-rank difference is at most the Nemenyi critical
    difference are connected by horizontal equivalence bars. OLS itself
    has CNL identically zero by construction and ranks at the right
    floor of the diagram as the operational baseline."""
    if not _have_real_data():
        return
    fr_path = RESULTS / 'real_data_friedman.json'
    if not fr_path.exists():
        return
    with open(fr_path) as fh:
        fr = json.load(fh)
    mean_ranks = fr['mean_ranks']
    cd = float(fr['cd'])
    methods = sorted(mean_ranks.keys(), key=lambda m: mean_ranks[m])
    ranks = [mean_ranks[m] for m in methods]
    k = len(methods)

    left_idx = list(range(k // 2))
    right_idx = list(range(k // 2, k))
    n_left = len(left_idx)
    n_right = len(right_idx)
    label_step = 0.55
    label_pad = 0.45
    rank_axis_y = 0.0
    groups = _cd_equivalence_groups(methods, ranks, cd)
    bar_step = 0.18
    # The lowest equivalence bar sits at ``rank_axis_y + bar_floor_pad``;
    # the rank tick text is anchored at ``rank_axis_y + 0.16`` with
    # ``va='bottom'``, so its bbox grows upward from the anchor and
    # collides with the bar at the previous 0.30 offset. 0.55 leaves
    # ~0.25 data-units of clearance between the text top and the
    # bottom bar's stroke.
    bar_floor_pad = 0.55
    bar_top = rank_axis_y + bar_floor_pad + bar_step * (len(groups) - 1)
    cd_scale_y = bar_top + 0.55

    fig, ax = plt.subplots(figsize=(COL, 4.2))
    ax.set_xlim(0.5, k + 0.5)
    y_bottom = -max(n_left, n_right) * label_step - label_pad - 0.6
    ax.set_ylim(y_bottom, cd_scale_y + 0.45)
    ax.invert_xaxis()

    ax.hlines(rank_axis_y, 1, k, color='#444444', lw=0.7, zorder=3)
    for r in range(1, k + 1):
        ax.vlines(r, rank_axis_y - 0.05, rank_axis_y + 0.05,
                  color='#444444', lw=0.6, zorder=3)
        ax.text(r, rank_axis_y + 0.16, f'{r}', ha='center', va='bottom',
                fontsize=6.8, color='#444444')

    ax.plot([1, 1 + cd], [cd_scale_y, cd_scale_y], color='#444444', lw=1.0, zorder=3)
    ax.vlines([1, 1 + cd], cd_scale_y - 0.06, cd_scale_y + 0.06,
              color='#444444', lw=0.9, zorder=3)
    ax.text(1 + cd / 2, cd_scale_y + 0.12, f'CD = {cd:.2f}',
            ha='center', va='bottom', fontsize=7.5, color='#222222')

    for bar_i, (i, j) in enumerate(groups):
        y = bar_top - bar_step * bar_i
        ax.plot([ranks[i], ranks[j]], [y, y],
                color='#222222', lw=2.6, solid_capstyle='round', zorder=5)

    # Horizontal leader segments terminate at the visual edge of the
    # label's bounding box rather than at the text anchor itself, so the
    # colored leader does not cross the parenthetical rank values or the
    # method name. The label text carries an opaque white bbox at high
    # zorder, masking any residual pixel overlap from the leader's
    # finite stroke width passing behind it.
    left_x_anchor = 0.6
    right_x_anchor = k + 0.4
    text_bbox = dict(facecolor='white', edgecolor='none', pad=1.2)
    fig.canvas.draw()
    inv = ax.transData.inverted()

    for plot_i, i in enumerate(left_idx):
        m = methods[i]
        r = ranks[i]
        fam = 'kore' if m == 'kore' else _method_family(m)
        color = FAMILY_COLORS.get(fam, '#444444')
        weight = 'bold' if m == 'kore' else 'normal'
        y = rank_axis_y - label_pad - label_step * plot_i
        ax.plot([r, r], [rank_axis_y, y], color=color, lw=0.7, alpha=0.85, zorder=2)
        t = ax.text(left_x_anchor - 0.10, y,
                    f'{METHOD_LABEL.get(m, m)}  ({r:.2f})',
                    ha='right', va='center', fontsize=7.4,
                    color=color, fontweight=weight, zorder=6, bbox=text_bbox)
        bbox_disp = t.get_window_extent(renderer=fig.canvas.get_renderer())
        x_text_far_data, _ = inv.transform((bbox_disp.x1, bbox_disp.y0))
        leader_end = max(left_x_anchor + 0.05, x_text_far_data + 0.05)
        if r > leader_end:
            ax.plot([r, leader_end], [y, y], color=color, lw=0.7,
                    alpha=0.85, zorder=2)

    for plot_i, i in enumerate(right_idx):
        m = methods[i]
        r = ranks[i]
        fam = 'kore' if m == 'kore' else _method_family(m)
        color = FAMILY_COLORS.get(fam, '#444444')
        weight = 'bold' if m == 'kore' else 'normal'
        y = rank_axis_y - label_pad - label_step * (n_right - 1 - plot_i)
        ax.plot([r, r], [rank_axis_y, y], color=color, lw=0.7, alpha=0.85, zorder=2)
        t = ax.text(right_x_anchor + 0.10, y,
                    f'({r:.2f})  {METHOD_LABEL.get(m, m)}',
                    ha='left', va='center', fontsize=7.4,
                    color=color, fontweight=weight, zorder=6, bbox=text_bbox)
        bbox_disp = t.get_window_extent(renderer=fig.canvas.get_renderer())
        x_text_near_data, _ = inv.transform((bbox_disp.x1, bbox_disp.y0))
        leader_end = min(right_x_anchor - 0.05, x_text_near_data - 0.05)
        if r < leader_end:
            ax.plot([r, leader_end], [y, y], color=color, lw=0.7,
                    alpha=0.85, zorder=2)

    for spine in ('top', 'right', 'left', 'bottom'):
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    save(fig, 'fig_real_data_cd')


def fig_real_data_spline_selectors():
    """Per-dataset paired log-ratio strip plot for the four classical
    spline resolution-selection criteria. Each point is a single
    dataset's Compute-Normalized Lift log-ratio
    ``log10(CNL_kore / CNL_other)``; points to the right of the
    equality line are datasets where KORE has a strictly higher CNL
    over OLS than the competitor. The median ratio per competitor is
    overlaid as a diamond. GCV and C_p coincide on this sweep (shared
    row); pyGAM uses a continuous smoothing parameter and is reported
    in the broader-ML comparison."""
    if not _have_real_data():
        return
    summary_path = RESULTS / 'real_data_summary.csv'
    if not summary_path.exists():
        return
    s = pd.read_csv(summary_path)
    if 'cnl_median' not in s.columns:
        return
    kore = s[s['method'] == 'kore'].set_index('dataset')['cnl_median']
    if kore.empty:
        return

    # GCV and C_p coincide at machine precision on every dataset in this
    # sweep (the C_p sigma^2 pilot is taken from the GCV-preselected
    # candidate, so the two criteria pick the same G), so they share a
    # single row labeled "GCV / Cp" to avoid a duplicate panel.
    competitors = [
        ('cv_spline',   'exhaustive CV'),
        ('gcv_spline',  r'GCV / $C_p$'),
        ('aic_spline',  'AIC'),
        ('bic_spline',  'BIC'),
    ]

    # CNL in [0, 1]; floor at a small epsilon so the log axis stays
    # defined when a competitor adds no detectable lift over OLS.
    # log10(CNL_kore / CNL_other) > 0 means KORE extracts more
    # OLS-relative lift per unit compute on that dataset.
    eps = 1e-3
    rows = []
    for i, (m, label) in enumerate(competitors):
        other = s[s['method'] == m].set_index('dataset')['cnl_median']
        common = kore.index.intersection(other.index)
        if len(common) == 0:
            continue
        for ds in common:
            k = kore.loc[ds]
            o = other.loc[ds]
            if not (np.isfinite(k) and np.isfinite(o)):
                continue
            rows.append({'method': m, 'label': label, 'i': i,
                         'log_ratio': float(np.log10(max(k, eps) / max(o, eps)))})
    if not rows:
        return
    df = pd.DataFrame(rows)

    kore_color = FAMILY_COLORS['kore']
    loss_color = '#B0535A'

    fig, ax = plt.subplots(figsize=(COL, 2.8))
    rng = np.random.default_rng(0)

    tick_labels = []
    for i, (m, label) in enumerate(competitors):
        sub = df[df['method'] == m]
        if sub.empty:
            tick_labels.append(label)
            continue
        n = len(sub)
        wins = int((sub['log_ratio'] > 0).sum())
        med = float(sub['log_ratio'].median())

        jitter = rng.uniform(-0.16, 0.16, size=n)
        colors = np.where(sub['log_ratio'].to_numpy() > 0, kore_color, loss_color)
        ax.scatter(sub['log_ratio'], i + jitter, s=30,
                   c=colors, edgecolors='white', lw=0.4,
                   alpha=0.80, zorder=4)
        ax.scatter(med, i, s=160, marker='D',
                   facecolor='white', edgecolor='#222222', lw=1.2, zorder=6)
        ax.scatter(med, i, s=44, marker='D',
                   color='#222222', zorder=7)
        tick_labels.append(f'{label}\n({wins}/{n})')

    ax.axvline(0.0, color='#444444', lw=0.7, ls='--', alpha=0.6, zorder=2)

    ax.set_yticks(range(len(competitors)))
    ax.set_yticklabels(tick_labels, fontsize=8.0)
    ax.invert_yaxis()
    ax.set_xlim(-1.0, 3.5)
    ax.set_xlabel(r'$\log_{10}\left[\mathrm{CNL}_{\mathrm{KORE}}'
                  r'/\mathrm{CNL}_{\mathrm{competitor}}\right]$',
                  fontsize=8.0)
    ax.grid(True, axis='x', alpha=0.35)
    despine(ax)
    fig.tight_layout()
    save(fig, 'fig_real_data_spline_selectors')


def _method_family(method: str) -> str:
    """Map a method name to its family for FAMILY_COLORS lookup. Mirrors
    the family column produced by experiments.run_real_data."""
    fam_map = {
        'kore': 'kore', 'cv_spline': 'spline', 'gcv_spline': 'spline',
        'cp_spline': 'spline', 'aic_spline': 'spline', 'bic_spline': 'spline',
        'pygam': 'spline',
        'linear': 'linear', 'ridge_cv': 'linear', 'lasso_cv': 'linear',
        'elasticnet_cv': 'linear',
        'random_forest': 'tree', 'extra_trees': 'tree', 'hist_gbm': 'tree',
        'xgboost': 'tree', 'lightgbm': 'tree', 'catboost': 'tree',
        'svr_rbf': 'kernel', 'kernel_ridge_rbf': 'kernel',
        'knn': 'neighbors', 'mlp': 'neural',
    }
    return fam_map.get(method, 'neighbors')


def _real_data_main():
    fig_real_data_pareto()
    fig_real_data_cd()
    fig_real_data_spline_selectors()
    fig_real_data_full_cnl()
    fig_real_data_subset_cnl()
    fig_real_data_significance()
    fig_real_data_joint_significance()
    fig_real_data_memory()
    fig_real_data_kore_vs_knn()
    fig_real_data_rank_vs_n()
    fig_real_data_rank_vs_d()


if __name__ == '__main__':
    fig_conceptual()
    fig_law_collapse()
    fig_frontier_cost()
    fig_frontier_accuracy()
    fig_frontier_summary()
    fig_benchmarks()
    fig_app_full_benchmarks()
    fig_app_discovery()
    fig_app_robustness()
    fig_app_scaling()
    fig_app_degree_ablation()
    fig_consistency_bias()
    fig_consistency_noise()
    fig_consistency_plugin()
    _real_data_main()
