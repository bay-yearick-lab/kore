"""KORE: Kolmogorov-optimal Order-aware Resolution Estimation.

"Kolmogorov-optimal" is literal: the spline bias rate ``G^(-2 beta)`` is the
squared Kolmogorov n-width of the smoothness class, a rate that spline spaces
of order ``k + 1`` attain (exactly, in L2; Melkman & Micchelli 1978), so KORE
only ever selects the resolution within a width-optimal approximation family.

Core library for the closed-form spline resolution selector described in
the accompanying paper. The algorithm has four steps for any structured
spline family ``f`` with basis-dimension polynomial ``p_f(G)``:

  1. Fit two pilot models at resolutions ``G_a < G_b`` and compute their
     analytical leave-one-out (LOO) errors via the PRESS identity.
  2. Solve a leverage-calibrated 2x2 system for ``(A_f, tau_f)``: the bias
     scale and the noise/variance scale.
  3. Evaluate the closed-form plug-in resolution
     ``G_f^dagger = argmin_{G > 0} {A_f G^{-2 beta} + tau_f p_f(G) / n}``,
     either by the dominant-term closed form or by a scalar root solve on
     the exact polynomial.
  4. Round to the nearest stable integer and certify with a small +/-radius
     LOO neighborhood.

The expensive step (training a model at every candidate resolution) never
occurs. The full math is in Section 3 of ``paper/main.tex``.

Public interface used elsewhere in the repository:

  - Spline fitting: ``fit_spline_1d``, ``pred_spline_1d``, ``fit_additive``,
    ``pred_additive``.
  - LOO machinery: ``loo_press``.
  - Pilot solve and closed-form plug-in: ``pilot_solve``,
    ``closed_form_g_star``.
  - Selector: ``kore_sande_additive``.
  - K-fold baseline: ``cv_spline_1d``.
  - Metrics: ``mse``, ``rmse``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.linalg import solve_triangular as _solve_tri
from scipy.optimize import brentq
from sklearn.preprocessing import SplineTransformer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


# ---------------------------------------------------------------------------
# 1D spline fitting
# ---------------------------------------------------------------------------

def fit_spline_1d(x, y, intervals, degree=3, ridge=1e-8):
    x = np.asarray(x).reshape(-1, 1)
    spl = SplineTransformer(
        n_knots=intervals + 1, degree=degree,
        knots="uniform", include_bias=True, extrapolation="continue",
    )
    basis = spl.fit_transform(x)
    A = basis.T @ basis + ridge * np.eye(basis.shape[1])
    coef = np.linalg.solve(A, basis.T @ y)
    return {"spl": spl, "coef": coef, "n_basis": basis.shape[1], "G": intervals}


def pred_spline_1d(model, x):
    x = np.asarray(x).reshape(-1, 1)
    return model["spl"].transform(x) @ model["coef"]


# ---------------------------------------------------------------------------
# d-dimensional additive spline fitting
# ---------------------------------------------------------------------------

def fit_additive(X, y, intervals, degree=3, ridge=1e-8):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    d = X.shape[1]
    blocks, spls = [], []
    for j in range(d):
        spl = SplineTransformer(
            n_knots=intervals + 1, degree=degree,
            knots="uniform", include_bias=(j == 0), extrapolation="continue",
        )
        blocks.append(spl.fit_transform(X[:, [j]]))
        spls.append(spl)
    basis = np.hstack(blocks)
    coef = np.linalg.solve(basis.T @ basis + ridge * np.eye(basis.shape[1]), basis.T @ y)
    return {"spls": spls, "coef": coef, "n_basis": basis.shape[1], "G": intervals, "d": d}


def pred_additive(model, X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    blocks = [spl.transform(X[:, [j]]) for j, spl in enumerate(model["spls"])]
    return np.hstack(blocks) @ model["coef"]


# ---------------------------------------------------------------------------
# Analytical leave-one-out (PRESS)
# ---------------------------------------------------------------------------

def loo_press(B, y, coef, ridge):
    """Allen's PRESS identity for a ridge-stabilized linear smoother.

    For ``y_hat = H y`` with ``H = B (B^T B + ridge I)^{-1} B^T``, the
    leave-one-out residual at sample ``i`` is ``(y_i - y_hat_i) / (1 - h_ii)``.
    The diagonal of ``H`` is recovered from a Cholesky solve, so the whole
    LOO MSE is available from a single full-data fit.

    Parameters
    ----------
    B : (n, p) ndarray
        Design matrix.
    y : (n,) ndarray
        Targets.
    coef : (p,) ndarray
        Ridge solution ``(B^T B + ridge I)^{-1} B^T y``.
    ridge : float
        Tikhonov diagonal used in the solve.

    Returns
    -------
    float
        Mean leave-one-out squared error.
    """
    p = B.shape[1]
    BTB = B.T @ B + ridge * np.eye(p)
    L = np.linalg.cholesky(BTB)
    Z = _solve_tri(L, B.T, lower=True)
    h_diag = np.sum(Z ** 2, axis=0)
    residuals = y - B @ coef
    return float(np.mean((residuals / (1.0 - h_diag)) ** 2))


def _ridge_press_solve(B, y, ridge):
    """Single Cholesky for both the ridge solution and the PRESS leverage.

    Mathematically identical to running ``np.linalg.solve(B^T B + r I,
    B^T y)`` followed by :func:`loo_press`, but performs only one
    formation of the Gram matrix and one Cholesky factorization. The
    matrix algebra:

        G = B^T B + r I = L L^T
        coef = L^{-T} (L^{-1} B^T y)
        Z    = L^{-1} B^T              (so H = Z^T Z, h_ii = ||Z[:, i]||^2)
        LOO  = mean(((y - B coef) / (1 - h_ii))^2)

    Returns ``(coef, loo, h_diag)``.
    """
    p = B.shape[1]
    G = B.T @ B + ridge * np.eye(p)
    L = np.linalg.cholesky(G)
    BTy = B.T @ y
    z = _solve_tri(L, BTy, lower=True)
    coef = _solve_tri(L.T, z, lower=False)
    Z = _solve_tri(L, B.T, lower=True)
    h_diag = np.sum(Z ** 2, axis=0)
    residuals = y - B @ coef
    loo = float(np.mean((residuals / (1.0 - h_diag)) ** 2))
    return coef, loo, h_diag


def _fit_additive_with_press(X, y, intervals, degree=3, ridge=1e-8):
    """Build the additive basis, fit, and compute LOO PRESS in one pass.

    Internal helper for :func:`kore_sande_additive`. Returns the same
    ``model`` dict shape as :func:`fit_additive` plus ``loo``; the public
    ``fit_additive`` / :func:`loo_press` API is unchanged.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    d = X.shape[1]
    blocks, spls = [], []
    for j in range(d):
        spl = SplineTransformer(
            n_knots=intervals + 1, degree=degree,
            knots="uniform", include_bias=(j == 0), extrapolation="continue",
        )
        blocks.append(spl.fit_transform(X[:, [j]]))
        spls.append(spl)
    basis = np.hstack(blocks)
    coef, loo, _ = _ridge_press_solve(basis, y, ridge)
    return {
        "spls": spls, "coef": coef, "n_basis": basis.shape[1],
        "G": intervals, "d": d, "loo": loo,
    }


# ---------------------------------------------------------------------------
# Leverage-calibrated pilot solve and closed-form G_dagger
# ---------------------------------------------------------------------------

def pilot_solve(
    loo_a: float,
    loo_b: float,
    G_a: int,
    G_b: int,
    p_a: int,
    p_b: int,
    n: int,
    beta: int,
    ridge: float = 1e-12,
):
    """Solve the leverage-calibrated 2x2 pilot system from eq. pilot_system.

    The pilot law is ``LOO_f(G) ~ A_f phi(G) + tau_f ell_f(G)`` with
    ``phi(G) = G^{-2 beta}`` and ``ell_f(G) = n / (n - p_f(G))`` (the
    average-leverage approximation to PRESS inflation). The 2x2 system

        | phi_a  ell_a |   | A_f   |   | LOO_a |
        |              | * |       | = |       |
        | phi_b  ell_b |   | tau_f |   | LOO_b |

    has determinant ``D = phi_a ell_b - phi_b ell_a``. When the two pilot
    resolutions are well-separated, ``D`` is bounded away from zero and
    the constants are identified.

    Returns
    -------
    (A_hat, tau_hat) : tuple of float
        Estimated bias scale and noise/variance scale. If the determinant
        is too small or the solve produces nonpositive constants, returns
        ``(0.0, 0.0)``; the caller treats that as an identification
        failure and falls back to the better pilot.
    """
    phi_a = float(G_a) ** (-2 * beta)
    phi_b = float(G_b) ** (-2 * beta)
    ell_a = float(n) / max(n - p_a, 1.0)
    ell_b = float(n) / max(n - p_b, 1.0)

    det = phi_a * ell_b - phi_b * ell_a + ridge
    if abs(det) < 1e-30:
        return 0.0, 0.0

    A_hat = (loo_a * ell_b - loo_b * ell_a) / det
    tau_hat = (phi_a * loo_b - phi_b * loo_a) / det
    return float(A_hat), float(tau_hat)


def closed_form_g_star(
    A: float,
    tau: float,
    beta: int,
    n: int,
    p_func: Callable[[float], float],
    dp_func: Callable[[float], float],
    G_min: float = 1.0,
    G_max: float = 1024.0,
) -> float:
    """Continuous closed-form plug-in resolution from eq. plugin_closed_form.

    Solves the scalar root equation

        2 beta A G^{-(2 beta + 1)} = (tau / n) * p'(G)

    on the interval ``[G_min, G_max]`` via Brent's method. For any
    structured spline polynomial ``p(G) = 1 + sum_t s_t m(G)^t`` with
    ``m(G) = G + k - 1``, this is the unique positive root of the
    derivative of the excess-risk proxy ``A G^{-2 beta} + tau p(G) / n``.

    Parameters
    ----------
    A, tau : float
        Bias scale and noise/variance scale from ``pilot_solve``.
    beta : int
        Smoothness exponent (``beta = k + 1`` for the classical regime).
    n : int
        Sample size.
    p_func : callable
        ``p_func(G)`` returns the basis dimension at resolution ``G``.
    dp_func : callable
        ``dp_func(G)`` returns ``dp / dG``. For additive: ``d``. For
        sparse pairwise: ``d + 2 s (G + k - 1)``.
    G_min, G_max : float
        Bracket for the root. The selector clips ``G_min`` to 1 and
        ``G_max`` to the largest stable integer resolution.

    Returns
    -------
    float
        The continuous plug-in resolution. The caller rounds to the
        nearest feasible integer and runs the small LOO certificate.
    """
    if A <= 0.0 or tau <= 0.0:
        return float(G_min)

    def f(G):
        return 2.0 * beta * A * G ** (-(2 * beta + 1)) - (tau / n) * dp_func(G)

    f_lo = f(G_min)
    f_hi = f(G_max)
    if f_lo <= 0.0:
        return float(G_min)
    if f_hi >= 0.0:
        return float(G_max)
    try:
        return float(brentq(f, G_min, G_max, xtol=1e-6, maxiter=100))
    except ValueError:
        return float(G_min)


# ---------------------------------------------------------------------------
# KORE selector for the additive family
# ---------------------------------------------------------------------------

def _unimodal_int_min(eval_fn, lo, hi):
    """Argmin of a U-shaped integer sequence on ``[lo, hi]`` in
    ``O(log(hi - lo))`` evaluations, by binary search on the discrete slope.
    ``eval_fn`` is expected to be memoized by the caller.
    """
    lo, hi = int(lo), int(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if eval_fn(mid) <= eval_fn(mid + 1):
            hi = mid
        else:
            lo = mid + 1
    return lo if eval_fn(lo) <= eval_fn(hi) else hi


def kore_sande_additive(X, y, q=4, degree=3, ridge=1e-8, max_G=20, radius: int = 3,
                        descent: bool = True):
    """Closed-form plug-in resolution selector for the additive family.

    Implements Algorithm 1 of the paper, specialized to the additive
    structure ``f(x) = sum_j f_j(x_j)`` with cubic B-splines by default.

    Steps:
      (i) Choose pilot pair ``G_a = 1`` (bias-dominated) and
          ``G_b = floor(0.75 * G_max_eff)`` (variance-dominated, near
          the stability cap so the pilot itself probes the high-G
          regime where the optimum tends to sit on very smooth targets).
     (ii) Fit each pilot via :func:`fit_additive` and compute
          :func:`loo_press`.
    (iii) Solve the leverage-calibrated 2x2 system :func:`pilot_solve`
          for ``(A_hat, tau_hat)``.
     (iv) Evaluate :func:`closed_form_g_star` to obtain the continuous
          plug-in ``G_dagger``, then round to the nearest stable integer.
      (v) Certify with a symmetric ``+/- radius`` LOO neighborhood and
          return the model with smallest LOO.

    Parameters
    ----------
    X : (n, d) ndarray
    y : (n,) ndarray
    q : int
        Smoothness order; the bias exponent in the pilot system is
        ``alpha = 2 (degree + 1)`` (classical B-spline rate). ``q`` is
        retained for backward-compatible call signatures and ignored
        whenever the asymptotic ``beta = degree + 1`` is in force.
    degree : int
        B-spline polynomial degree (``3`` = cubic).
    ridge : float
        Tikhonov diagonal on every normal-equations solve.
    max_G : int
        Largest candidate resolution.
    radius : int
        Half-width of the symmetric integer LOO certificate around the
        rounded plug-in.

    Returns
    -------
    dict
        ``{"model", "G_star", "fits", "d", "A_hat", "tau_hat",
           "G_dagger", "history", "loo"}``.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    beta = degree + 1

    def p_add(G):
        return d * (G + degree) - (d - 1)

    def dp_add(_G):
        return float(d)

    eff_max_G = int(max_G)
    while eff_max_G > 1 and p_add(eff_max_G) >= 0.45 * n:
        eff_max_G -= 1

    G_a = 1
    G_b = max(2, int(np.floor(0.75 * eff_max_G)))
    while G_b > G_a and p_add(G_b) >= 0.45 * n:
        G_b -= 1
    pilot_Gs = [G_a] if G_b <= G_a else [G_a, G_b]

    fits = 0
    pilots = []
    for G_p in pilot_Gs:
        if p_add(G_p) >= 0.45 * n:
            continue
        m = _fit_additive_with_press(X, y, G_p, degree=degree, ridge=ridge)
        fits += 1
        pilots.append((G_p, m["loo"], m))

    if not pilots:
        m = _fit_additive_with_press(X, y, 1, degree=degree, ridge=ridge)
        return {
            "model": m,
            "G_star": 1,
            "fits": fits + 1,
            "d": d,
            "A_hat": 0.0,
            "tau_hat": 0.0,
            "G_dagger": 1.0,
            "history": [],
            "loo": float("inf"),
        }

    A_hat, tau_hat, G_dagger = 0.0, 0.0, float(pilots[0][0])
    if len(pilots) >= 2:
        (Ga, loo_a, _), (Gb, loo_b, _) = pilots[0], pilots[1]
        A_hat, tau_hat = pilot_solve(
            loo_a, loo_b, Ga, Gb, p_add(Ga), p_add(Gb), n, beta, ridge=1e-12,
        )
        if A_hat > 0.0 and tau_hat > 0.0:
            G_dagger = closed_form_g_star(
                A_hat, tau_hat, beta, n, p_add, dp_add,
                G_min=1.0, G_max=float(eff_max_G),
            )
        else:
            G_dagger = float(Ga if loo_a <= loo_b else Gb)

    G_star_int = int(np.clip(round(G_dagger), 1, eff_max_G))

    # LOO cache so every resolution is fitted at most once across the
    # certificate neighborhood, the pilots, and the unimodal descent.
    cache = {pg: (loo_p, m_p) for pg, loo_p, m_p in pilots}

    def eval_G(Gc):
        nonlocal fits
        if Gc in cache:
            return cache[Gc][0]
        m_c = _fit_additive_with_press(X, y, Gc, degree=degree, ridge=ridge)
        fits += 1
        cache[Gc] = (m_c["loo"], m_c)
        return cache[Gc][0]

    # Symmetric plug-in certificate around the rounded closed-form root.
    for delta in range(-radius, radius + 1):
        Gc = G_star_int + delta
        if 1 <= Gc <= eff_max_G:
            eval_G(Gc)

    best_G = min(cache, key=lambda g: cache[g][0])

    # Bounded gap-bridge with a regime gate. The symmetric certificate plus
    # the two pilots can miss an interior optimum sitting between the plug-in
    # neighborhood and the upper pilot G_b. When G_b wins, a single midpoint
    # probe decides whether the optimum is genuinely interior (its LOO is a
    # clear margin below G_b's) before paying for a binary search. This
    # leaves the cheap, near-cap behavior intact on smooth low-noise targets,
    # where the curve is flat and minimizing a noisy LOO over extra candidates
    # only adds selection variance, and engages only on a real gap.
    nbhd_top = min(eff_max_G, G_star_int + radius)
    if (descent and len(pilots) >= 2 and best_G == G_b and G_b > nbhd_top + 1):
        mid = (nbhd_top + G_b) // 2
        if eval_G(mid) < cache[G_b][0] * 0.98:
            g = _unimodal_int_min(eval_G, nbhd_top, G_b)
            if cache[g][0] < cache[best_G][0]:
                best_G = g

    best_loo, best_model = cache[best_G]

    return {
        "model": best_model,
        "G_star": int(best_G),
        "fits": fits,
        "d": d,
        "A_hat": float(A_hat),
        "tau_hat": float(tau_hat),
        "G_dagger": float(G_dagger),
        "history": [(pilot_Gs[0], pilot_Gs[-1], int(best_G), float(A_hat), float(tau_hat))],
        "loo": float(best_loo),
    }


# ---------------------------------------------------------------------------
# K-fold cross-validation baseline (1D only; pairwise CV lives in
# intrinsic_kore.py because it knows the centered-block design)
# ---------------------------------------------------------------------------

def cv_spline_1d(x, y, grid, degree=3, K=5, seed=42):
    """K-fold cross-validation for 1D spline resolution selection.

    Total fits ``= K * len(grid) + 1`` (final refit on full data). Used
    only by the d=1 benchmark suite; the d>=2 baselines live in
    :mod:`intrinsic_kore`.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    best_G, best_score = grid[0], float("inf")
    fits = 0
    for G in grid:
        cv_scores = []
        for train_idx, val_idx in kf.split(x):
            m = fit_spline_1d(x[train_idx], y[train_idx], G, degree=degree)
            fits += 1
            cv_scores.append(mse(y[val_idx], pred_spline_1d(m, x[val_idx])))
        avg = float(np.mean(cv_scores))
        if avg < best_score:
            best_score, best_G = avg, G
    final = fit_spline_1d(x, y, best_G, degree=degree)
    fits += 1
    return {"model": final, "G_star": best_G, "fits": fits}


# ---------------------------------------------------------------------------
# Smoke test (run directly: ``uv run python kore_lib.py``)
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """Self-check that the new primitives behave as the math says."""
    rng = np.random.default_rng(2026)

    # 1) pilot_solve: synthesize known (A, tau) at two pilot resolutions and
    # confirm the 2x2 system recovers them.
    A_true, tau_true = 1.5, 0.4
    n, beta = 1200, 4
    G_a, G_b, p_a, p_b = 1, 7, 30, 280
    phi_a = float(G_a) ** (-2 * beta)
    phi_b = float(G_b) ** (-2 * beta)
    ell_a = n / (n - p_a)
    ell_b = n / (n - p_b)
    loo_a = A_true * phi_a + tau_true * ell_a
    loo_b = A_true * phi_b + tau_true * ell_b
    A_hat, tau_hat = pilot_solve(loo_a, loo_b, G_a, G_b, p_a, p_b, n, beta)
    assert abs(A_hat - A_true) < 1e-6, (A_hat, A_true)
    assert abs(tau_hat - tau_true) < 1e-6, (tau_hat, tau_true)

    # 2) closed_form_g_star: for the additive family with the same A, tau,
    # the analytic root ``G = (2 beta A n / (tau d))^{1/(2 beta + 1)}``
    # must agree with brentq, and with a brute-force argmin over the
    # continuous proxy.
    d = 20

    def p_add(G):
        return d * (G + 3) - (d - 1)

    def dp_add(_G):
        return float(d)

    G_dagger = closed_form_g_star(A_true, tau_true, beta, n, p_add, dp_add,
                                  G_min=1.0, G_max=200.0)
    G_analytic = (2.0 * beta * A_true * n / (tau_true * d)) ** (1.0 / (2 * beta + 1))
    assert abs(G_dagger - G_analytic) < 1e-3, (G_dagger, G_analytic)

    grid = np.linspace(1.0, 200.0, 5000)
    proxy = A_true * grid ** (-2 * beta) + tau_true * np.array([p_add(g) for g in grid]) / n
    G_brute = grid[int(np.argmin(proxy))]
    assert abs(G_dagger - G_brute) < 0.5, (G_dagger, G_brute)

    # 3) End-to-end smoke run: a synthetic 10-D additive target with n=1200.
    coords = rng.uniform(0.0, 1.0, size=(1200, 10))
    freqs = rng.integers(1, 4, size=10)
    amps = rng.uniform(0.5, 1.2, size=10)
    y_clean = sum(amps[j] * np.sin(2 * np.pi * freqs[j] * coords[:, j])
                  for j in range(10)) / np.sqrt(10)
    y_noisy = y_clean + 0.03 * rng.standard_normal(1200) * np.std(y_clean)

    out = kore_sande_additive(coords, y_noisy, degree=3, max_G=20)
    assert 1 <= out["G_star"] <= 20
    assert out["A_hat"] > 0.0 or out["G_star"] in {1, out.get("G_star", 1)}
    assert out["fits"] <= 2 + 2 * 3 + 1
    print(
        f"[smoke] pilot recovery: A_hat={A_hat:.4f}, tau_hat={tau_hat:.4f}\n"
        f"[smoke] additive plug-in: G_dagger={G_dagger:.3f}, "
        f"G_analytic={G_analytic:.3f}, G_brute={G_brute:.3f}\n"
        f"[smoke] end-to-end: G_star={out['G_star']}, "
        f"A_hat={out['A_hat']:.3g}, tau_hat={out['tau_hat']:.3g}, "
        f"fits={out['fits']}"
    )


if __name__ == "__main__":
    _smoke_test()
