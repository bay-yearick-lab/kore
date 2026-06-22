"""Interaction-aware KORE: sparse pairwise spline design and structure-aware
selectors. Mirrors :mod:`kore.lib` but for the pairwise family
``f(x) = mu + sum_j f_j(x_j) + sum_{(i,j) in E} f_{ij}(x_i, x_j)`` whose
basis polynomial is

    p_pair(G) = 1 + d (G + k - 1) + |E| (G + k - 1)^2.

Hosts the pairwise plug-in selector :func:`kore_sande_pairwise`, the
discovery-based wrapper :func:`auto_kore_discover`, and the classical
full-grid baselines (GCV / Cp / AIC / BIC) used in the experiments.
"""

import math
from typing import List, Optional, Sequence, Tuple, Dict, Any

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer

from .lib import (
    mse,
    fit_additive,
    pred_additive,
    loo_press,
    pilot_solve,
    closed_form_g_star,
    kore_sande_additive,
    _ridge_press_solve,
    _unimodal_int_min,
)


Pair = Tuple[int, int]


def _ensure_2d(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _all_pairs(d: int) -> List[Pair]:
    return [(i, j) for i in range(d) for j in range(i + 1, d)]


def _pairwise_effective_max_g(n: int, d: int, n_pairs: int, degree: int, max_G: int, safety: float = 0.45) -> int:
    # Largest G in [1, max_G] with p_pair(G) < safety * n, or 0 if even G=1 violates the cap.
    if max_G < 1:
        return 0
    if pairwise_basis_dim(d, 1, degree=degree, n_pairs=n_pairs) >= safety * n:
        return 0
    lo, hi = 1, max_G
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        p = pairwise_basis_dim(d, mid, degree=degree, n_pairs=n_pairs)
        if p < safety * n:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def pairwise_basis_dim(d: int, G: int, degree: int = 3, n_pairs: Optional[int] = None) -> int:
    if n_pairs is None:
        n_pairs = d * (d - 1) // 2
    m = G + degree - 1  # include_bias=False
    return 1 + d * m + n_pairs * (m * m)


def _fit_centered_blocks(X: np.ndarray, G: int, degree: int = 3):
    X = _ensure_2d(X)
    d = X.shape[1]
    spls = []
    means = []
    blocks = []
    for j in range(d):
        spl = SplineTransformer(
            n_knots=G + 1,
            degree=degree,
            knots="uniform",
            include_bias=False,
            extrapolation="continue",
        )
        B = spl.fit_transform(X[:, [j]])
        mu = B.mean(axis=0, keepdims=True)
        Bc = B - mu
        spls.append(spl)
        means.append(mu)
        blocks.append(Bc)
    return spls, means, blocks


def _transform_centered_blocks(X: np.ndarray, spls, means):
    X = _ensure_2d(X)
    blocks = []
    for j, (spl, mu) in enumerate(zip(spls, means)):
        B = spl.transform(X[:, [j]])
        blocks.append(B - mu)
    return blocks


def _pairwise_design_from_blocks(blocks: Sequence[np.ndarray], pairs: Sequence[Pair]):
    n = blocks[0].shape[0]
    pieces = [np.ones((n, 1), dtype=float)]
    pieces.extend(blocks)
    for i, j in pairs:
        Bi = blocks[i]
        Bj = blocks[j]
        Bij = (Bi[:, :, None] * Bj[:, None, :]).reshape(n, -1)
        pieces.append(Bij)
    return np.hstack(pieces)


def fit_pairwise_sparse(X, y, intervals: int, pairs: Optional[Sequence[Pair]] = None,
                        degree: int = 3, ridge: float = 1e-8):
    X = _ensure_2d(X)
    n, d = X.shape
    if pairs is None:
        pairs = _all_pairs(d)
    pairs = list(pairs)
    spls, means, blocks = _fit_centered_blocks(X, intervals, degree=degree)
    B = _pairwise_design_from_blocks(blocks, pairs)
    p = B.shape[1]
    A = B.T @ B + ridge * np.eye(p)
    coef = np.linalg.solve(A, B.T @ y)
    return {
        "spls": spls,
        "means": means,
        "coef": coef,
        "n_basis": p,
        "G": intervals,
        "pairs": pairs,
        "d": d,
        "degree": degree,
        "ridge": ridge,
        "design": B,
    }


def _fit_pairwise_sparse_with_press(X, y, intervals: int, pairs: Sequence[Pair],
                                    degree: int = 3, ridge: float = 1e-8):
    """Build the pairwise design, fit, and compute LOO PRESS in one pass.

    Internal helper for :func:`kore_sande_pairwise`. Returns the same
    ``model`` dict shape as :func:`fit_pairwise_sparse` plus ``loo``;
    the public ``fit_pairwise_sparse`` API is unchanged.
    """
    X = _ensure_2d(X)
    n, d = X.shape
    pairs = list(pairs)
    spls, means, blocks = _fit_centered_blocks(X, intervals, degree=degree)
    B = _pairwise_design_from_blocks(blocks, pairs)
    coef, loo, _ = _ridge_press_solve(B, y, ridge)
    return {
        "spls": spls,
        "means": means,
        "coef": coef,
        "n_basis": B.shape[1],
        "G": intervals,
        "pairs": pairs,
        "d": d,
        "degree": degree,
        "ridge": ridge,
        "design": B,
        "loo": loo,
    }


def pred_pairwise_sparse(model, X):
    X = _ensure_2d(X)
    blocks = _transform_centered_blocks(X, model["spls"], model["means"])
    B = _pairwise_design_from_blocks(blocks, model["pairs"])
    return B @ model["coef"]


def _loo_pairwise_model(model, y):
    B = model.get("design")
    if B is None:
        raise ValueError("Model missing cached design matrix for LOO.")
    return loo_press(B, y, model["coef"], model["ridge"])


def kore_sande_pairwise(X, y, pairs: Optional[Sequence[Pair]] = None, q: int = 4,
                        degree: int = 3, ridge: float = 1e-8, max_G: int = 8,
                        radius: int = 1, descent: bool = True):
    """Closed-form plug-in resolution selector for the sparse pairwise family.

    Mirrors :func:`kore_lib.kore_sande_additive` with the pairwise basis
    polynomial ``p_pair(G) = 1 + d m(G) + s m(G)^2`` and ``m(G) = G + k - 1``.
    Steps: fit pilots at ``G_a = 1`` and ``G_b = floor(0.75 * G_max_eff)``,
    solve the leverage-calibrated 2x2 system for ``(A_hat, tau_hat)``,
    obtain the continuous ``G_dagger`` from :func:`closed_form_g_star`,
    round to the nearest stable integer, and certify with a symmetric
    ``+/- radius`` LOO neighborhood.
    """
    X = _ensure_2d(X)
    n, d = X.shape
    if pairs is None:
        pairs = _all_pairs(d)
    pairs = list(pairs)
    n_pairs = len(pairs)
    beta = degree + 1
    eff_max_G = _pairwise_effective_max_g(n, d, n_pairs, degree, max_G)

    if eff_max_G == 0:
        # p_pair(1) already violates the 0.45 * n stability rule; skip without allocating.
        return {
            "model": None, "G_star": 0, "fits": 0,
            "structure": "pairwise", "loo": float("inf"),
            "A_hat": 0.0, "tau_hat": 0.0, "G_dagger": 0.0,
            "pairs": pairs, "skipped": True,
            "skip_reason": "p_pair(1) >= 0.45 * n",
            "p_pair_at_1": pairwise_basis_dim(d, 1, degree=degree, n_pairs=n_pairs),
            "n": n,
        }

    def p_pair(G):
        return pairwise_basis_dim(d, int(round(G)) if isinstance(G, (int, np.integer)) else G,
                                  degree=degree, n_pairs=n_pairs)

    def p_pair_continuous(G):
        m = G + degree - 1
        return 1.0 + d * m + n_pairs * m * m

    def dp_pair_continuous(G):
        m = G + degree - 1
        return float(d + 2.0 * n_pairs * m)

    G_a = 1
    G_b = max(2, int(np.floor(0.75 * eff_max_G)))
    while G_b > G_a and pairwise_basis_dim(d, G_b, degree=degree, n_pairs=n_pairs) >= 0.45 * n:
        G_b -= 1
    pilot_Gs = [G_a] if G_b <= G_a else [G_a, G_b]

    fits = 0
    pilots: List[Tuple[int, float, Dict[str, Any]]] = []
    for G_p in pilot_Gs:
        if pairwise_basis_dim(d, G_p, degree=degree, n_pairs=n_pairs) >= 0.45 * n:
            continue
        m = _fit_pairwise_sparse_with_press(
            X, y, G_p, pairs=pairs, degree=degree, ridge=ridge,
        )
        fits += 1
        pilots.append((G_p, m["loo"], m))

    if not pilots:
        m = _fit_pairwise_sparse_with_press(
            X, y, 1, pairs=pairs, degree=degree, ridge=ridge,
        )
        return {
            "model": m, "G_star": 1, "fits": 1,
            "structure": "pairwise", "loo": m["loo"],
            "A_hat": 0.0, "tau_hat": 0.0, "G_dagger": 1.0, "pairs": pairs,
        }

    A_hat, tau_hat = 0.0, 0.0
    G_dagger = float(pilots[0][0])
    if len(pilots) >= 2:
        Ga, loo_a, _ = pilots[0]
        Gb, loo_b, _ = pilots[1]
        p_a = pairwise_basis_dim(d, Ga, degree=degree, n_pairs=n_pairs)
        p_b = pairwise_basis_dim(d, Gb, degree=degree, n_pairs=n_pairs)
        A_hat, tau_hat = pilot_solve(loo_a, loo_b, Ga, Gb, p_a, p_b, n, beta)
        if A_hat > 0.0 and tau_hat > 0.0:
            G_dagger = closed_form_g_star(
                A_hat, tau_hat, beta, n,
                p_pair_continuous, dp_pair_continuous,
                G_min=1.0, G_max=float(eff_max_G),
            )
        else:
            G_dagger = float(Ga if loo_a <= loo_b else Gb)

    G_star_int = int(np.clip(round(G_dagger), 1, eff_max_G))

    cache = {pg: (loo_p, m_p) for pg, loo_p, m_p in pilots}

    def _feasible(Gc):
        return (1 <= Gc <= eff_max_G
                and pairwise_basis_dim(d, Gc, degree=degree, n_pairs=n_pairs) < 0.45 * n)

    def eval_G(Gc):
        nonlocal fits
        if Gc in cache:
            return cache[Gc][0]
        m_c = _fit_pairwise_sparse_with_press(
            X, y, Gc, pairs=pairs, degree=degree, ridge=ridge,
        )
        fits += 1
        cache[Gc] = (m_c["loo"], m_c)
        return cache[Gc][0]

    # Symmetric plug-in certificate around the rounded closed-form root.
    for delta in range(-radius, radius + 1):
        Gc = G_star_int + delta
        if _feasible(Gc):
            eval_G(Gc)

    best_G = min(cache, key=lambda g: cache[g][0])

    # Bounded gap-bridge with a regime gate (see the additive selector for
    # the rationale): when G_b wins, a single midpoint probe decides whether
    # the optimum is genuinely interior before paying for a binary search,
    # leaving the cheap near-cap behavior intact on smooth low-noise targets.
    nbhd_top = min(eff_max_G, G_star_int + radius)
    if (descent and len(pilots) >= 2 and best_G == G_b and G_b > nbhd_top + 1):
        mid = (nbhd_top + G_b) // 2
        if _feasible(mid) and eval_G(mid) < cache[G_b][0] * 0.98:
            g = _unimodal_int_min(eval_G, nbhd_top, G_b)
            if cache[g][0] < cache[best_G][0]:
                best_G = g

    best_loo, best_model = cache[best_G]

    return {
        "model": best_model,
        "G_star": int(best_G),
        "fits": fits,
        "structure": "pairwise",
        "loo": float(best_loo),
        "pairs": pairs,
        "A_hat": float(A_hat),
        "tau_hat": float(tau_hat),
        "G_dagger": float(G_dagger),
    }


def kore_diagnostic(X, y, *, degree: int = 3, ridge: float = 1e-8,
                     max_G: int = 20) -> Dict[str, Any]:
    """Practitioner go/no-go diagnostic for KORE on a tabular regression task.

    Runs the additive pilot pair (the same two fits ``kore_sande_additive``
    would run) and exposes the intermediate quantities so a caller can
    decide whether to trust the closed-form selector before committing
    to a final fit. Returns the post-one-hot dimension, effective
    density ``rho = n / d``, the 2x2 pilot system's condition number,
    the fitted bias and noise/variance scales ``A_hat``, ``tau_hat``,
    the continuous closed-form ``G_dagger``, the basis-fraction
    stability margin ``0.45 - p(G_dagger) / n``, an overall
    ``suitable`` flag, a human-readable ``reason``, and a
    ``recommendation`` chosen by an additive-vs-pairwise LOO comparison
    (the same comparison ``auto_kore_intrinsic`` makes).

    Decision rule (theoretical, not learned):
      * ``suitable = False`` if ``d_onehot > 30`` (outside the
        pre-registered smooth-low-d regime where ``rho >= 50`` is
        plausible at typical CTR23 sample sizes),
      * ``suitable = False`` if the pilot system condition number
        exceeds ``1e6`` (the leverage-calibrated solve is ill-posed),
      * ``suitable = False`` if ``stability_margin < 0.05`` (the
        plug-in resolution sits within 5 percentage points of the
        ``p / n < 0.45`` stability cap),
      * otherwise ``suitable = True`` and the recommendation comes
        from the LOO winner.
    """
    X = _ensure_2d(X)
    y = np.asarray(y).ravel()
    n, d = X.shape
    beta = degree + 1

    def p_add(G):
        return d * (G + degree) - (d - 1)

    eff_max_G = int(max_G)
    while eff_max_G > 1 and p_add(eff_max_G) >= 0.45 * n:
        eff_max_G -= 1
    G_a = 1
    G_b = max(2, int(np.floor(0.75 * eff_max_G)))
    while G_b > G_a and p_add(G_b) >= 0.45 * n:
        G_b -= 1

    A_hat = 0.0
    tau_hat = 0.0
    G_dagger = float(G_a)
    cond = float("inf")
    if G_b > G_a:
        m_a = fit_additive(X, y, G_a, degree=degree, ridge=ridge)
        m_b = fit_additive(X, y, G_b, degree=degree, ridge=ridge)
        blocks_a = [spl.transform(X[:, [j]]) for j, spl in enumerate(m_a["spls"])]
        blocks_b = [spl.transform(X[:, [j]]) for j, spl in enumerate(m_b["spls"])]
        loo_a = loo_press(np.hstack(blocks_a), y, m_a["coef"], ridge)
        loo_b = loo_press(np.hstack(blocks_b), y, m_b["coef"], ridge)
        p_a = p_add(G_a)
        p_b = p_add(G_b)
        phi_a = float(G_a) ** (-2 * beta)
        phi_b = float(G_b) ** (-2 * beta)
        ell_a = float(n) / max(n - p_a, 1.0)
        ell_b = float(n) / max(n - p_b, 1.0)
        M = np.array([[phi_a, ell_a], [phi_b, ell_b]], dtype=float)
        try:
            cond = float(np.linalg.cond(M))
        except np.linalg.LinAlgError:
            cond = float("inf")
        A_hat, tau_hat = pilot_solve(loo_a, loo_b, G_a, G_b, p_a, p_b, n, beta)
        if A_hat > 0.0 and tau_hat > 0.0:
            G_dagger = closed_form_g_star(
                A_hat, tau_hat, beta, n, p_add, lambda _G: float(d),
                G_min=1.0, G_max=float(eff_max_G),
            )

    p_at_dagger = p_add(int(np.clip(round(G_dagger), 1, eff_max_G)))
    stability_margin = 0.45 - p_at_dagger / max(n, 1)
    rho = float(n) / max(float(d), 1.0)

    reasons: List[str] = []
    suitable = True
    if d > 30:
        suitable = False
        reasons.append(f"d_onehot={d} exceeds 30 (outside pre-registered smooth-low-d regime)")
    if cond > 1e6:
        suitable = False
        reasons.append(f"pilot condition number {cond:.2e} > 1e6 (ill-posed pilot solve)")
    if stability_margin < 0.05:
        suitable = False
        reasons.append(f"stability margin {stability_margin:.3f} < 0.05 (plug-in near the p/n < 0.45 cap)")

    recommendation = "out-of-scope: use a tree ensemble or kernel method"
    if suitable:
        try:
            chosen = auto_kore_intrinsic(X, y, degree=degree, ridge=ridge,
                                          max_G_add=max_G)
            recommendation = (
                "use additive" if chosen.get("structure") == "additive"
                else "use pairwise"
            )
        except Exception:
            recommendation = "use additive"

    return {
        "suitable": bool(suitable),
        "reason": "; ".join(reasons) if reasons else "all checks passed",
        "n": int(n),
        "d_onehot": int(d),
        "effective_density": float(rho),
        "pilot_condition_number": float(cond),
        "A_hat": float(A_hat),
        "tau_hat": float(tau_hat),
        "G_dagger": float(G_dagger),
        "stability_margin": float(stability_margin),
        "recommendation": recommendation,
    }


def cv_pairwise_sparse(X, y, grid: Sequence[int], pairs: Optional[Sequence[Pair]] = None,
                       degree: int = 3, ridge: float = 1e-8, K: int = 5, seed: int = 42):
    X = _ensure_2d(X)
    y = np.asarray(y)
    if pairs is None:
        pairs = _all_pairs(X.shape[1])
    pairs = list(pairs)
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    best_G = None
    best_score = float("inf")
    fits = 0
    for G in grid:
        fold_scores = []
        for tr_idx, va_idx in kf.split(X):
            m = fit_pairwise_sparse(X[tr_idx], y[tr_idx], G, pairs=pairs, degree=degree, ridge=ridge)
            fits += 1
            pred = pred_pairwise_sparse(m, X[va_idx])
            fold_scores.append(mse(y[va_idx], pred))
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score = score
            best_G = G
    final = fit_pairwise_sparse(X, y, best_G, pairs=pairs, degree=degree, ridge=ridge)
    fits += 1
    return {"model": final, "G_star": best_G, "fits": fits, "cv_score": best_score, "structure": "pairwise"}




def refine_kore_additive(X, y, q: int = 4, degree: int = 3, ridge: float = 1e-8,
                         max_G: int = 20, radius: int = 3, descent: bool = True):
    """Closed-form plug-in selector for the additive family with a wider
    LOO certificate.

    Thin wrapper around :func:`kore_lib.kore_sande_additive` that fixes the
    certificate radius to a slightly larger value and adds the
    ``structure="additive"`` key the experiment driver expects.
    """
    res = kore_sande_additive(X, y, q=q, degree=degree, ridge=ridge,
                              max_G=max_G, radius=radius, descent=descent)
    res["structure"] = "additive"
    return res


def auto_kore_intrinsic(X, y, pairs: Optional[Sequence[Pair]] = None,
                        q: int = 4, degree: int = 3, ridge: float = 1e-8,
                        max_G_add: int = 20, max_G_pair: int = 8,
                        descent: bool = True):
    X = _ensure_2d(X)
    y = np.asarray(y)
    add = None
    pair = None
    # additive via existing Sande selector
    add = refine_kore_additive(X, y, q=q, degree=degree, ridge=ridge, max_G=max_G_add, radius=3, descent=descent)

    pair = kore_sande_pairwise(X, y, pairs=pairs, q=q, degree=degree, ridge=ridge, max_G=max_G_pair, descent=descent)

    chosen = add if add["loo"] <= pair["loo"] else pair
    chosen = dict(chosen)
    pair_cand = {"G_star": pair["G_star"], "fits": pair["fits"], "loo": pair["loo"]}
    if pair.get("skipped"):
        pair_cand["skipped"] = True
        pair_cand["skip_reason"] = pair.get("skip_reason", "")
    chosen["candidates"] = {
        "additive": {"G_star": add["G_star"], "fits": add["fits"], "loo": add["loo"]},
        "pairwise": pair_cand,
    }
    chosen["fits_total"] = add["fits"] + pair["fits"]
    return chosen


def cv_auto_intrinsic(X, y, grid_add: Sequence[int], grid_pair: Sequence[int],
                      pairs: Optional[Sequence[Pair]] = None,
                      degree: int = 3, ridge: float = 1e-8, K: int = 5, seed: int = 42):
    X = _ensure_2d(X)
    y = np.asarray(y)
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)

    fits = 0
    best = None
    best_structure = None
    best_G = None

    # additive
    for G in grid_add:
        fold_scores = []
        for tr_idx, va_idx in kf.split(X):
            m = fit_additive(X[tr_idx], y[tr_idx], G, degree=degree, ridge=ridge)
            fits += 1
            pred = pred_additive(m, X[va_idx])
            fold_scores.append(mse(y[va_idx], pred))
        score = float(np.mean(fold_scores))
        if best is None or score < best:
            best = score
            best_structure = "additive"
            best_G = G

    # pairwise
    if pairs is None:
        pairs = _all_pairs(X.shape[1])
    pairs = list(pairs)
    for G in grid_pair:
        fold_scores = []
        for tr_idx, va_idx in kf.split(X):
            m = fit_pairwise_sparse(X[tr_idx], y[tr_idx], G, pairs=pairs, degree=degree, ridge=ridge)
            fits += 1
            pred = pred_pairwise_sparse(m, X[va_idx])
            fold_scores.append(mse(y[va_idx], pred))
        score = float(np.mean(fold_scores))
        if best is None or score < best:
            best = score
            best_structure = "pairwise"
            best_G = G

    if best_structure == "additive":
        final = fit_additive(X, y, best_G, degree=degree, ridge=ridge)
    else:
        final = fit_pairwise_sparse(X, y, best_G, pairs=pairs, degree=degree, ridge=ridge)
    fits += 1
    return {"model": final, "structure": best_structure, "G_star": best_G, "fits": fits, "cv_score": best}


# ----------------------------- synthetic targets -----------------------------

def make_additive_target(d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    amps = rng.uniform(0.5, 1.2, size=d)
    freqs = rng.integers(1, 4, size=d)
    phases = rng.uniform(0.0, 2 * np.pi, size=d)
    scales = rng.uniform(0.8, 1.2, size=d)

    def f(X):
        X = _ensure_2d(X)
        val = np.zeros(X.shape[0])
        for j in range(d):
            x = X[:, j]
            val += amps[j] * np.sin(2 * np.pi * freqs[j] * x + phases[j])
            val += 0.35 * scales[j] * np.cos(np.pi * (j + 1) * x / (d + 1))
        return val / np.sqrt(d)

    return f


def make_sparse_pairwise_target(d: int, seed: int = 0, pairs: Optional[Sequence[Pair]] = None):
    rng = np.random.default_rng(seed)
    if pairs is None:
        pairs = [(j, j + 1) for j in range(0, d - 1, 2)]
    pairs = list(pairs)
    pair_w = rng.uniform(0.8, 1.3, size=len(pairs))
    main_w = rng.uniform(0.2, 0.5, size=d)

    def f(X):
        X = _ensure_2d(X)
        val = np.zeros(X.shape[0])
        for idx, (i, j) in enumerate(pairs):
            val += pair_w[idx] * np.sin(np.pi * X[:, i] * X[:, j])
            val += 0.4 * pair_w[idx] * np.cos(np.pi * (X[:, i] + X[:, j]))
        for j in range(d):
            val += main_w[j] * np.sin(2 * np.pi * X[:, j])
        return val / np.sqrt(max(len(pairs), 1) + 0.25 * d)

    return f, pairs


def make_dataset(func, n_train: int, n_test: int, d: int, noise_frac: float = 0.05, seed: int = 0,
                 ranges: Optional[List[Tuple[float, float]]] = None):
    rng = np.random.default_rng(seed)
    if ranges is None:
        X_train = rng.random((n_train, d))
        X_test = rng.random((n_test, d))
    else:
        X_train = np.column_stack([
            rng.uniform(lo, hi, n_train) for lo, hi in ranges
        ])
        X_test = np.column_stack([
            rng.uniform(lo, hi, n_test) for lo, hi in ranges
        ])
    y_clean = func(X_train)
    sigma = noise_frac * np.std(y_clean)
    y_train = y_clean + rng.normal(scale=max(sigma, 1e-10), size=n_train)
    y_test = func(X_test)
    return X_train, y_train, X_test, y_test, sigma


# ----------------------------- graph discovery ------------------------------

def screen_pairs_residual(X, y, degree: int = 3, ridge: float = 1e-8,
                          G_add: int = 5, G_pair: int = 2,
                          frac_threshold: float = 0.02,
                          max_pairs: Optional[int] = None) -> List[Pair]:
    """Discover active pairwise interactions via residual-based screening.

    1. Fit additive model at moderate resolution to capture main effects.
    2. Compute residuals.
    3. For each candidate pair (i,j), fit interaction-only centered tensor
       blocks to the residuals and measure fraction of residual variance
       explained.
    4. Return pairs whose explained fraction exceeds *frac_threshold*,
       sorted by decreasing evidence, capped at *max_pairs*.
    """
    X = _ensure_2d(X)
    n, d = X.shape
    if max_pairs is None:
        max_pairs = d

    add_model = fit_additive(X, y, G_add, degree=degree, ridge=ridge)
    y_hat = pred_additive(add_model, X)
    resid = y - y_hat
    resid_var = float(np.var(resid))
    if resid_var < 1e-15:
        return []

    spls, means, blocks = _fit_centered_blocks(X, G_pair, degree=degree)

    scores: List[Tuple[Pair, float]] = []
    for i in range(d):
        for j in range(i + 1, d):
            Bi, Bj = blocks[i], blocks[j]
            Bij = (Bi[:, :, None] * Bj[:, None, :]).reshape(n, -1)
            p = Bij.shape[1]
            if p >= 0.45 * n:
                continue
            try:
                A = Bij.T @ Bij + ridge * np.eye(p)
                coef = np.linalg.solve(A, Bij.T @ resid)
                fitted = Bij @ coef
                frac = float(np.var(fitted)) / resid_var
                scores.append(((i, j), frac))
            except Exception:
                continue

    scores.sort(key=lambda x: x[1], reverse=True)
    selected: List[Pair] = []
    for pair, frac in scores:
        if frac < frac_threshold:
            break
        selected.append(pair)
        if len(selected) >= max_pairs:
            break
    return selected


def auto_kore_discover(X, y, q: int = 4, degree: int = 3, ridge: float = 1e-8,
                       max_G_add: int = 20, max_G_pair: int = 8,
                       G_screen: int = 5, frac_threshold: float = 0.02,
                       max_pairs: Optional[int] = None):
    """Full Interaction-Aware KORE with automatic graph discovery.

    Screens for pairwise interactions, then runs *auto_kore_intrinsic*
    with the discovered graph.  If no interactions are found, falls back
    to additive-only KORE.
    """
    X = _ensure_2d(X)
    y = np.asarray(y)

    discovered = screen_pairs_residual(
        X, y, degree=degree, ridge=ridge, G_add=G_screen,
        frac_threshold=frac_threshold, max_pairs=max_pairs,
    )

    if not discovered:
        result = refine_kore_additive(X, y, q=q, degree=degree, ridge=ridge,
                                      max_G=max_G_add)
        result["discovered_pairs"] = []
        result["n_discovered"] = 0
        return result

    result = auto_kore_intrinsic(
        X, y, pairs=discovered, q=q, degree=degree, ridge=ridge,
        max_G_add=max_G_add, max_G_pair=max_G_pair,
    )
    result["discovered_pairs"] = discovered
    result["n_discovered"] = len(discovered)
    return result


# ----------------------------- GCV baselines --------------------------------

def gcv_additive(X, y, grid: Sequence[int], degree: int = 3,
                 ridge: float = 1e-8):
    """Select additive resolution by Generalized Cross-Validation.

    GCV(G) = (1/n) RSS / (1 - p/n)^2.
    """
    X = _ensure_2d(X)
    y = np.asarray(y)
    n = len(y)
    best_G, best_gcv = None, float("inf")
    fits = 0
    for G in grid:
        m = fit_additive(X, y, G, degree=degree, ridge=ridge)
        fits += 1
        blocks = [spl.transform(X[:, [j]]) for j, spl in enumerate(m["spls"])]
        B = np.hstack(blocks)
        p = B.shape[1]
        rss = float(np.sum((y - B @ m["coef"]) ** 2))
        denom = max(1.0 - p / n, 0.01) ** 2
        gcv = rss / n / denom
        if gcv < best_gcv:
            best_gcv, best_G = gcv, G
    final = fit_additive(X, y, best_G, degree=degree, ridge=ridge)
    fits += 1
    return {"model": final, "G_star": best_G, "fits": fits,
            "structure": "additive", "gcv": best_gcv}


def _information_criterion_value(name: str, rss: float, n: int, p: int,
                                 sigma2_ref: Optional[float] = None) -> float:
    rss = max(float(rss), 1e-12)
    if name == "gcv":
        return rss / n / max(1.0 - p / n, 0.01) ** 2
    if name == "aic":
        return n * math.log(rss / n) + 2.0 * p
    if name == "bic":
        return n * math.log(rss / n) + math.log(max(n, 2)) * p
    if name == "cp":
        if sigma2_ref is None:
            raise ValueError("Cp requires sigma2_ref.")
        sigma2_ref = max(float(sigma2_ref), 1e-12)
        return rss / sigma2_ref - n + 2.0 * p
    raise ValueError(f"Unknown criterion: {name}")


def _sigma2_reference(candidates: Sequence[Dict[str, Any]], n: int) -> float:
    """Noise variance estimate for Mallows' Cp.

    We use the GCV-preselected candidate as the reference model, i.e. we
    pick the G that minimizes GCV and compute sigma2 = RSS / (n - p) at
    that G. This is a standard trick that avoids the "saturated richest
    candidate" pathology (near-zero RSS collapses sigma2 to zero and makes
    Cp prefer overfit models).
    """
    def _gcv_score(rec):
        denom = max(1.0 - rec["p"] / n, 0.01) ** 2
        return rec["rss"] / max(n, 1) / denom

    pilot = min(candidates, key=_gcv_score)
    dof = max(n - pilot["p"], 1)
    return max(pilot["rss"] / dof, 1e-12)


def _select_from_candidates(candidates: Sequence[Dict[str, Any]], n: int,
                            criterion_names: Sequence[str], fit_final,
                            sigma2_ref: Optional[float] = None):
    if sigma2_ref is None:
        sigma2_ref = _sigma2_reference(candidates, n)
    out = {}
    for name in criterion_names:
        best = min(
            candidates,
            key=lambda rec: _information_criterion_value(
                name, rec["rss"], n, rec["p"], sigma2_ref=sigma2_ref
            ),
        )
        final = fit_final(best["G"])
        out[name] = {
            "model": final,
            "G_star": best["G"],
            "fits": len(candidates) + 1,
            "score": _information_criterion_value(
                name, best["rss"], n, best["p"], sigma2_ref=sigma2_ref
            ),
            "sigma2_ref": sigma2_ref,
        }
    return out


def _collect_additive_candidates(X, y, grid, degree, ridge):
    cands = []
    for G in grid:
        m = fit_additive(X, y, G, degree=degree, ridge=ridge)
        blocks = [spl.transform(X[:, [j]]) for j, spl in enumerate(m["spls"])]
        B = np.hstack(blocks)
        cands.append({
            "G": G,
            "rss": float(np.sum((y - B @ m["coef"]) ** 2)),
            "p": int(B.shape[1]),
        })
    return cands


def criteria_additive(X, y, grid: Sequence[int], degree: int = 3,
                      ridge: float = 1e-8,
                      criterion_names: Sequence[str] = ("gcv", "aic", "bic", "cp"),
                      sigma2_ref: Optional[float] = None):
    """Select additive resolution by classical information criteria."""
    X = _ensure_2d(X)
    y = np.asarray(y)
    n = len(y)
    candidates = _collect_additive_candidates(X, y, grid, degree, ridge)
    selected = _select_from_candidates(
        candidates,
        n,
        criterion_names,
        fit_final=lambda G: fit_additive(X, y, G, degree=degree, ridge=ridge),
        sigma2_ref=sigma2_ref,
    )
    for name, result in selected.items():
        result["structure"] = "additive"
        result[name] = result["score"]
    return selected


def gcv_pairwise_sparse(X, y, grid: Sequence[int],
                        pairs: Optional[Sequence[Pair]] = None,
                        degree: int = 3, ridge: float = 1e-8):
    """Select sparse pairwise resolution by GCV."""
    X = _ensure_2d(X)
    y = np.asarray(y)
    n = len(y)
    if pairs is None:
        pairs = _all_pairs(X.shape[1])
    pairs = list(pairs)
    best_G, best_gcv = None, float("inf")
    fits = 0
    for G in grid:
        pdim = pairwise_basis_dim(X.shape[1], G, degree=degree,
                                  n_pairs=len(pairs))
        if pdim >= 0.45 * n:
            continue
        m = fit_pairwise_sparse(X, y, G, pairs=pairs, degree=degree,
                                ridge=ridge)
        fits += 1
        B = m["design"]
        p = B.shape[1]
        rss = float(np.sum((y - B @ m["coef"]) ** 2))
        denom = max(1.0 - p / n, 0.01) ** 2
        gcv = rss / n / denom
        if gcv < best_gcv:
            best_gcv, best_G = gcv, G
    if best_G is None:
        best_G = grid[0]
    final = fit_pairwise_sparse(X, y, best_G, pairs=pairs, degree=degree,
                                ridge=ridge)
    fits += 1
    return {"model": final, "G_star": best_G, "fits": fits,
            "structure": "pairwise", "gcv": best_gcv}


def _collect_pairwise_candidates(X, y, grid, pairs, degree, ridge, n):
    cands = []
    for G in grid:
        pdim = pairwise_basis_dim(X.shape[1], G, degree=degree,
                                  n_pairs=len(pairs))
        if pdim >= 0.45 * n:
            continue
        m = fit_pairwise_sparse(X, y, G, pairs=pairs, degree=degree,
                                ridge=ridge)
        B = m["design"]
        cands.append({
            "G": G,
            "rss": float(np.sum((y - B @ m["coef"]) ** 2)),
            "p": int(B.shape[1]),
        })
    return cands


def criteria_pairwise_sparse(X, y, grid: Sequence[int],
                             pairs: Optional[Sequence[Pair]] = None,
                             degree: int = 3, ridge: float = 1e-8,
                             criterion_names: Sequence[str] = ("gcv", "aic", "bic", "cp"),
                             sigma2_ref: Optional[float] = None):
    """Select pairwise resolution by classical information criteria."""
    X = _ensure_2d(X)
    y = np.asarray(y)
    n = len(y)
    if pairs is None:
        pairs = _all_pairs(X.shape[1])
    pairs = list(pairs)

    candidates = _collect_pairwise_candidates(X, y, grid, pairs, degree, ridge, n)

    if not candidates:
        fallback = fit_pairwise_sparse(X, y, grid[0], pairs=pairs, degree=degree,
                                       ridge=ridge)
        return {
            name: {
                "model": fallback,
                "G_star": grid[0],
                "fits": 1,
                "structure": "pairwise",
                "score": float("inf"),
                name: float("inf"),
                "sigma2_ref": float("nan"),
            }
            for name in criterion_names
        }

    selected = _select_from_candidates(
        candidates,
        n,
        criterion_names,
        fit_final=lambda G: fit_pairwise_sparse(X, y, G, pairs=pairs,
                                                degree=degree, ridge=ridge),
        sigma2_ref=sigma2_ref,
    )
    for name, result in selected.items():
        result["structure"] = "pairwise"
        result[name] = result["score"]
    return selected


def gcv_auto_intrinsic(X, y, grid_add: Sequence[int], grid_pair: Sequence[int],
                       pairs: Optional[Sequence[Pair]] = None,
                       degree: int = 3, ridge: float = 1e-8):
    """Structure selection + resolution by GCV over both families."""
    add = gcv_additive(X, y, grid_add, degree=degree, ridge=ridge)
    pair = gcv_pairwise_sparse(X, y, grid_pair, pairs=pairs, degree=degree,
                               ridge=ridge)
    if add["gcv"] <= pair["gcv"]:
        chosen = dict(add)
    else:
        chosen = dict(pair)
    chosen["fits_total"] = add["fits"] + pair["fits"]
    return chosen


def criteria_auto_intrinsic(X, y, grid_add: Sequence[int], grid_pair: Sequence[int],
                            pairs: Optional[Sequence[Pair]] = None,
                            degree: int = 3, ridge: float = 1e-8,
                            criterion_names: Sequence[str] = ("gcv", "aic", "bic", "cp")):
    """Structure selection + resolution by classical information criteria.

    Cp scores are only comparable across structure families when they share
    a single noise-variance reference. We therefore collect candidates from
    both families first, compute one common sigma2 from the GCV-preselected
    candidate across the union, and pass that reference into each family.
    """
    X = _ensure_2d(X)
    y_arr = np.asarray(y)
    n = len(y_arr)
    if pairs is None:
        pair_list = _all_pairs(X.shape[1])
    else:
        pair_list = list(pairs)
    add_cands = _collect_additive_candidates(X, y_arr, grid_add, degree, ridge)
    pair_cands = _collect_pairwise_candidates(X, y_arr, grid_pair, pair_list,
                                              degree, ridge, n)
    sigma2_ref = _sigma2_reference(add_cands + pair_cands, n)

    add = criteria_additive(X, y_arr, grid_add, degree=degree, ridge=ridge,
                            criterion_names=criterion_names,
                            sigma2_ref=sigma2_ref)
    pair = criteria_pairwise_sparse(X, y_arr, grid_pair, pairs=pair_list,
                                    degree=degree, ridge=ridge,
                                    criterion_names=criterion_names,
                                    sigma2_ref=sigma2_ref)
    chosen = {}
    for name in criterion_names:
        add_res = add[name]
        pair_res = pair[name]
        winner = dict(add_res) if add_res["score"] <= pair_res["score"] else dict(pair_res)
        winner["fits_total"] = add_res["fits"] + pair_res["fits"]
        chosen[name] = winner
    return chosen


# ----------------------------- robustness targets ---------------------------

def make_threeway_target(d: int, seed: int = 0):
    """Target with genuine 3-way interactions (violates r<=2 assumption)."""
    rng = np.random.default_rng(seed)
    triples = [(j, j + 1, j + 2) for j in range(0, d - 2, 3)]
    triple_w = rng.uniform(0.8, 1.3, size=len(triples))
    main_w = rng.uniform(0.2, 0.5, size=d)

    def f(X):
        X = _ensure_2d(X)
        val = np.zeros(X.shape[0])
        for idx, (i, j, k) in enumerate(triples):
            val += triple_w[idx] * np.sin(np.pi * X[:, i] * X[:, j] * X[:, k])
        for j in range(d):
            val += main_w[j] * np.sin(2 * np.pi * X[:, j])
        return val / np.sqrt(max(len(triples), 1) + 0.25 * d)

    return f, triples


def make_nonsmooth_target(d: int, seed: int = 0):
    """Piecewise-constant target (violates smoothness assumption)."""
    rng = np.random.default_rng(seed)
    thresholds = rng.uniform(0.3, 0.7, size=d)
    weights = rng.uniform(0.5, 1.5, size=d)

    def f(X):
        X = _ensure_2d(X)
        val = np.zeros(X.shape[0])
        for j in range(d):
            val += weights[j] * np.where(X[:, j] > thresholds[j], 1.0, -1.0)
        return val / np.sqrt(d)

    return f


def make_misspecified_pairwise(d: int, seed: int = 0):
    """Sparse pairwise target where the *supplied* graph is wrong.

    True pairs are (0,1),(2,3),... but the supplied graph is shifted:
    (1,2),(3,4),...  -- testing robustness to graph misspecification.
    """
    true_pairs = [(j, j + 1) for j in range(0, d - 1, 2)]
    wrong_pairs = [(j, j + 1) for j in range(1, d - 1, 2)]
    func, _ = make_sparse_pairwise_target(d, seed=seed, pairs=true_pairs)
    return func, true_pairs, wrong_pairs


# ----------------------------- benchmark targets ----------------------------

def make_friedman1(seed: int = 0):
    """Friedman #1 (d=5): 10sin(pi*x1*x2) + 20(x3-0.5)^2 + 10*x4 + 5*x5.

    Has one pairwise interaction (x1,x2) and three additive terms.
    """
    d = 5
    pairs = [(0, 1)]

    def f(X):
        X = _ensure_2d(X)
        return (10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
                + 20.0 * (X[:, 2] - 0.5) ** 2
                + 10.0 * X[:, 3] + 5.0 * X[:, 4])

    return f, d, pairs


def make_friedman2(seed: int = 0):
    """Friedman #2 (d=4): sqrt(x1^2 + (x2*x3 - 1/(x2*x4))^2)."""
    d = 4
    pairs = [(0, 1), (1, 2), (1, 3)]

    def f(X):
        X = _ensure_2d(X)
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        inner = x2 * x3 - 1.0 / (x2 * x4 + 1e-8)
        return np.sqrt(x1 ** 2 + inner ** 2)

    return f, d, pairs


def make_franke():
    """Classic Franke function (d=2), additive in spirit but with interactions."""
    d = 2
    pairs = [(0, 1)]

    def f(X):
        X = _ensure_2d(X)
        x1, x2 = X[:, 0], X[:, 1]
        return (0.75 * np.exp(-((9 * x1 - 2) ** 2 + (9 * x2 - 2) ** 2) / 4)
                + 0.75 * np.exp(-(9 * x1 + 1) ** 2 / 49 - (9 * x2 + 1) / 10)
                + 0.5 * np.exp(-((9 * x1 - 7) ** 2 + (9 * x2 - 3) ** 2) / 4)
                - 0.2 * np.exp(-(9 * x1 - 4) ** 2 - (9 * x2 - 7) ** 2))

    return f, d, pairs


def make_sparse_additive_highdim(d: int = 20, n_active: int = 5, seed: int = 0):
    """High-dimensional additive target with only n_active coordinates active."""
    rng = np.random.default_rng(seed)
    active = rng.choice(d, size=n_active, replace=False)
    amps = rng.uniform(0.5, 1.5, size=n_active)
    freqs = rng.integers(1, 4, size=n_active)

    def f(X):
        X = _ensure_2d(X)
        val = np.zeros(X.shape[0])
        for idx, j in enumerate(active):
            val += amps[idx] * np.sin(2 * np.pi * freqs[idx] * X[:, j])
        return val / np.sqrt(n_active)

    return f, sorted(active.tolist())
