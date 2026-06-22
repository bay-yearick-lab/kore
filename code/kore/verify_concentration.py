"""Concentration sanity check for the leverage-calibrated pilot solve.

For each ``n in (300, 600, 1200, 2400, 4800, 9600)``, fixes a single
deterministic additive target on ``d = 10`` and resamples ``M = 100``
sub-Gaussian noise realisations with sigma = 0.5. Each realisation
fits the two pilots ``(G_a = 1, G_b = floor(0.75 * G_max_eff))``,
calls :func:`kore.lib.pilot_solve`, and records ``(A_hat, tau_hat)``.

The empirical standard deviation of each pilot estimate is plotted
against the Prop-A.2 envelope ``c / sqrt(n)``, with the prefactor
``c`` fitted by least-squares on the log scale. The two-panel figure
and the underlying CSV land under ``KORE_FIGURES_DIR`` and
``KORE_RESULTS_DIR`` respectively.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .lib import _fit_additive_with_press, pilot_solve
from .intrinsic import make_additive_target


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FIG_OVR = os.environ.get("KORE_FIGURES_DIR")
_RES_OVR = os.environ.get("KORE_RESULTS_DIR")
FIG_OUT = (Path(_FIG_OVR).expanduser() if _FIG_OVR
           else PROJECT_ROOT / "paper" / "figures")
RES_OUT = (Path(_RES_OVR).expanduser() if _RES_OVR
           else PROJECT_ROOT / "results")
FIG_OUT.mkdir(parents=True, exist_ok=True)
RES_OUT.mkdir(parents=True, exist_ok=True)


D = 10
DEGREE = 3
BETA = DEGREE + 1
SIGMA = 0.5
M_NOISE = 100
N_GRID = (300, 600, 1200, 2400, 4800, 9600)
RIDGE = 1e-8
SAFETY = 0.45
MASTER_SEED = 2026

# Quiet palette: navy for empirical points, neutral grey for the
# fitted envelope, matching the global figure standard.
KORE_BLUE = "#16486B"
ENVELOPE = "#444444"


def _p_add(G: int, d: int, degree: int) -> int:
    return d * (G + degree) - (d - 1)


def _eff_max_g(n: int, d: int, degree: int, max_G: int = 20) -> int:
    g = max_G
    while g > 1 and _p_add(g, d, degree) >= SAFETY * n:
        g -= 1
    return g


def _one_n(n: int, X: np.ndarray, f_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(MASTER_SEED + n)
    eff = _eff_max_g(n, D, DEGREE)
    G_a = 1
    G_b = max(2, int(np.floor(0.75 * eff)))
    p_a = _p_add(G_a, D, DEGREE)
    p_b = _p_add(G_b, D, DEGREE)

    A_hats = np.empty(M_NOISE)
    tau_hats = np.empty(M_NOISE)
    for m in range(M_NOISE):
        y = f_true + rng.normal(0.0, SIGMA, size=n)
        ma = _fit_additive_with_press(X, y, G_a, degree=DEGREE, ridge=RIDGE)
        mb = _fit_additive_with_press(X, y, G_b, degree=DEGREE, ridge=RIDGE)
        A_h, tau_h = pilot_solve(
            ma["loo"], mb["loo"], G_a, G_b, p_a, p_b, n, BETA, ridge=1e-12,
        )
        A_hats[m] = A_h
        tau_hats[m] = tau_h
    return A_hats, tau_hats


def _fit_envelope(n_arr: np.ndarray, std_arr: np.ndarray) -> tuple[float, np.ndarray]:
    # Least-squares prefactor for predicted_std = c / sqrt(n) on a
    # log-log scale: log std = log c - 0.5 log n, solved by averaging.
    mask = (std_arr > 0) & np.isfinite(std_arr)
    log_c = float(np.mean(np.log(std_arr[mask]) + 0.5 * np.log(n_arr[mask])))
    c = float(np.exp(log_c))
    return c, c / np.sqrt(n_arr)


def main() -> None:
    t0 = time.perf_counter()
    rng_X = np.random.default_rng(MASTER_SEED)
    f = make_additive_target(D, seed=MASTER_SEED)

    n_arr = np.array(N_GRID, dtype=float)
    std_A = np.empty(len(N_GRID))
    std_tau = np.empty(len(N_GRID))
    for i, n in enumerate(N_GRID):
        X = rng_X.uniform(0.0, 1.0, size=(n, D))
        f_true = f(X)
        A_hats, tau_hats = _one_n(n, X, f_true)
        std_A[i] = float(np.std(A_hats, ddof=1))
        std_tau[i] = float(np.std(tau_hats, ddof=1))

    cA, env_A = _fit_envelope(n_arr, std_A)
    cT, env_T = _fit_envelope(n_arr, std_tau)

    pd.DataFrame({
        "n": n_arr.astype(int),
        "std_A_hat": std_A,
        "std_tau_hat": std_tau,
        "predicted_envelope_A": env_A,
        "predicted_envelope_tau": env_T,
    }).to_csv(RES_OUT / "concentration_envelope.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, std, env, title in (
        (axes[0], std_A, env_A, r"$\widehat{A}_f$"),
        (axes[1], std_tau, env_T, r"$\widehat{\tau}_f$"),
    ):
        ax.loglog(n_arr, std, "o", color=KORE_BLUE, markersize=6,
                  markeredgecolor="white", markeredgewidth=0.5,
                  label="empirical")
        ax.loglog(n_arr, env, "--", color=ENVELOPE, lw=1.0,
                  label=r"$c/\sqrt{n}$")
        ax.set_xlabel("training size n")
        ax.set_ylabel(f"empirical std of {title}")
        ax.grid(True, which="both", alpha=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.18, top=0.95,
                        wspace=0.40)
    fig.savefig(FIG_OUT / "fig_concentration_envelope.pdf")
    fig.savefig(FIG_OUT / "fig_concentration_envelope.png", dpi=150)
    plt.close(fig)

    print(f"DONE  ({time.perf_counter() - t0:.2f} s wall)")


if __name__ == "__main__":
    main()
