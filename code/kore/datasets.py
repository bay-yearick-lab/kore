"""Real-world dataset suite for the KORE benchmark.

The registry here is the OpenML-CTR23 curated tabular regression suite
(Fischer, Liu, Pfisterer, Bischl 2023, AutoML 2023 Workshop), all 35
datasets, plus the Combined Cycle Power Plant dataset (Tufekci 2014)
which is a long-standing GAM-literature classic that does not appear in
CTR23. CTR23 already contains the other UCI classics (concrete,
airfoil, california_housing, red_wine, white_wine, energy_efficiency,
forest_fires, naval_propulsion_plant), so deduplication leaves a single
unique UCI addition.

Total: 36 unique datasets.

CTR23 inclusion criteria (paraphrased from the suite description) cover
the smooth-tabular regression regime well: 500 to 100000 instances,
fewer than 5000 features after one-hot encoding, i.i.d. observations
(no time dependencies), continuous target, no ethical concerns, no
trivial linear solutions. The pre-registered smooth-low-d subset
applied here further restricts to ``d_onehot <= 30``; CTR23 already
enforces ``n >= 500`` and rules out time-series tasks.

Every fetch is cached to ``data/`` (gitignored). The first call hits
OpenML; subsequent calls deserialize from the parquet cache, which
keeps the experiment driver hermetic once the cache is populated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import io
import logging
import os
import urllib.request
import warnings
import zipfile

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Honor ``KORE_DATA_DIR`` so a non-editable install on a read-only
# checkout (Databricks Workspace Files, site-packages) can redirect the
# parquet cache to a writable Volume or DBFS path without monkey-patching.
_DATA_DIR_OVERRIDE = os.environ.get("KORE_DATA_DIR")
DATA_DIR = (
    Path(_DATA_DIR_OVERRIDE).expanduser()
    if _DATA_DIR_OVERRIDE
    else PROJECT_ROOT / "data"
)
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSpec:
    """One row of the benchmark registry.

    Attributes
    ----------
    name :
        Stable canonical name; used as the cache filename and reported
        in the per-dataset table.
    openml_id :
        OpenML ``data_id``. ``None`` only for sklearn-direct fetchers.
    source :
        Either ``'openml-ctr23'`` for the curated suite or
        ``'uci-classic'`` for the single deduplicated UCI addition.
    citation :
        Bibliographic anchor used in the paper.
    """

    name: str
    openml_id: int | None
    source: str
    citation: str


DATASETS_CTR23: tuple[DatasetSpec, ...] = (
    DatasetSpec("abalone", 44956, "openml-ctr23", "Nash et al. 1994"),
    DatasetSpec("airfoil_self_noise", 44957, "openml-ctr23", "Brooks et al. 1989"),
    DatasetSpec("auction_verification", 44958, "openml-ctr23", "Ordowski 2022"),
    DatasetSpec("concrete_compressive_strength", 44959, "openml-ctr23", "Yeh 1998"),
    DatasetSpec("energy_efficiency", 44960, "openml-ctr23", "Tsanas and Xifara 2012"),
    DatasetSpec("forest_fires", 44962, "openml-ctr23", "Cortez and Morais 2007"),
    DatasetSpec("physiochemical_protein", 44963, "openml-ctr23", "Rana 2013"),
    DatasetSpec("superconductivity", 44964, "openml-ctr23", "Hamidieh 2018"),
    DatasetSpec("geographical_origin_of_music", 44965, "openml-ctr23", "Zhou et al. 2014"),
    DatasetSpec("solar_flare", 44966, "openml-ctr23", "Bradshaw 1989"),
    DatasetSpec("student_performance_por", 44967, "openml-ctr23", "Cortez and Silva 2008"),
    DatasetSpec("naval_propulsion_plant", 44969, "openml-ctr23", "Coraddu et al. 2016"),
    DatasetSpec("QSAR_fish_toxicity", 44970, "openml-ctr23", "Cassotti et al. 2015"),
    DatasetSpec("white_wine", 44971, "openml-ctr23", "Cortez et al. 2009"),
    DatasetSpec("red_wine", 44972, "openml-ctr23", "Cortez et al. 2009"),
    DatasetSpec("grid_stability", 44973, "openml-ctr23", "Arzamasov et al. 2018"),
    DatasetSpec("video_transcoding", 44974, "openml-ctr23", "Deneke et al. 2014"),
    DatasetSpec("wave_energy", 44975, "openml-ctr23", "Neshat et al. 2018"),
    DatasetSpec("sarcos", 44976, "openml-ctr23", "Vijayakumar and Schaal 2000"),
    DatasetSpec("california_housing", 44977, "openml-ctr23", "Pace and Barry 1997"),
    DatasetSpec("cpu_activity", 44978, "openml-ctr23", "DELVE 1996"),
    DatasetSpec("diamonds", 44979, "openml-ctr23", "Wickham 2016"),
    DatasetSpec("kin8nm", 44980, "openml-ctr23", "DELVE 1996"),
    DatasetSpec("pumadyn32nh", 44981, "openml-ctr23", "DELVE 1996"),
    DatasetSpec("miami_housing", 44983, "openml-ctr23", "Bourassa et al. 2021"),
    DatasetSpec("cps88wages", 44984, "openml-ctr23", "Berndt 1991"),
    DatasetSpec("socmob", 44987, "openml-ctr23", "Biblarz and Raftery 1993"),
    DatasetSpec("kings_county", 44989, "openml-ctr23", "Harlfoxem 2016"),
    DatasetSpec("brazilian_houses", 44990, "openml-ctr23", "Aguiar 2019"),
    DatasetSpec("fps_benchmark", 44992, "openml-ctr23", "Davies 2019"),
    DatasetSpec("health_insurance", 44993, "openml-ctr23", "Lantz 2019"),
    DatasetSpec("cars", 44994, "openml-ctr23", "Kuhn 2018"),
    DatasetSpec("fifa", 45012, "openml-ctr23", "Leone 2022"),
    DatasetSpec("Moneyball", 41021, "openml-ctr23", "Lewis 2003"),
    DatasetSpec("space_ga", 45402, "openml-ctr23", "Pace and Barry 1997"),
)

# Combined Cycle Power Plant (Tufekci 2014) is the one UCI classic that
# does not overlap CTR23 (CTR23 carries concrete, airfoil, wine, energy,
# california_housing, naval_propulsion, forest_fires already). It has no
# OpenML hosting, so the loader fetches the original UCI archive
# directly. The ``openml_id`` field is ``None`` for this entry.
DATASETS_UCI_CLASSICS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "combined_cycle_power_plant",
        None,
        "uci-classic",
        "Tufekci 2014",
    ),
)

DATASETS: tuple[DatasetSpec, ...] = DATASETS_CTR23 + DATASETS_UCI_CLASSICS


def _cache_path(spec: DatasetSpec) -> Path:
    return DATA_DIR / f"{spec.name}.parquet"


def _normalize_target_column(frame: pd.DataFrame, target_col: str | None) -> tuple[pd.DataFrame, str]:
    if target_col is not None and target_col in frame.columns:
        return frame, target_col
    last_col = frame.columns[-1]
    return frame, last_col


def _fetch_openml_frame(spec: DatasetSpec) -> pd.DataFrame:
    """Pull a dataset from OpenML by ``data_id`` and return the merged frame.

    The frame contains both features and the target as the last column,
    matching the convention CTR23 uses on OpenML.
    """
    from sklearn.datasets import fetch_openml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bunch = fetch_openml(
            data_id=spec.openml_id,
            as_frame=True,
            parser="auto",
            cache=False,
        )
    frame = bunch.frame.copy()
    if bunch.target is not None and bunch.target.name not in frame.columns:
        frame[bunch.target.name] = bunch.target.values
    if bunch.target is not None:
        target_name = bunch.target.name
        cols = [c for c in frame.columns if c != target_name] + [target_name]
        frame = frame[cols]
    return frame


# UCI Combined Cycle Power Plant archive. The zip ships an Excel
# workbook with the raw data on Sheet 1 and five randomized 5x2 cross
# validation folds on the remaining sheets; only Sheet 1 is consumed.
_UCI_CCPP_URL = (
    "https://archive.ics.uci.edu/static/public/294/"
    "combined+cycle+power+plant.zip"
)


def _fetch_uci_ccpp_frame() -> pd.DataFrame:
    """Download CCPP from UCI and return the raw frame (target = ``PE``)."""
    req = urllib.request.Request(
        _UCI_CCPP_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        with zf.open("CCPP/Folds5x2_pp.xlsx") as fh:
            xlsx_bytes = fh.read()
    frame = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name=0)
    target_name = "PE"
    if target_name not in frame.columns:
        raise RuntimeError(
            f"CCPP frame missing expected target column {target_name!r}; "
            f"got columns {list(frame.columns)}"
        )
    cols = [c for c in frame.columns if c != target_name] + [target_name]
    return frame[cols]


def _fetch_frame(spec: DatasetSpec) -> pd.DataFrame:
    if spec.source == "uci-classic" and spec.name == "combined_cycle_power_plant":
        return _fetch_uci_ccpp_frame()
    if spec.openml_id is None:
        raise RuntimeError(
            f"Dataset {spec.name!r} has no openml_id and no UCI loader."
        )
    return _fetch_openml_frame(spec)


def fetch_one(spec: DatasetSpec, *, force_refresh: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` as numpy arrays after caching to ``data/``.

    Categorical features are one-hot encoded (with ``drop_first=False``
    to preserve interpretability of every level). Missing feature
    values are median-imputed; rows with missing target are dropped.
    """
    cache = _cache_path(spec)
    if cache.exists() and not force_refresh:
        frame = pd.read_parquet(cache)
    else:
        frame = _fetch_frame(spec)
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache, index=False)

    target_col = frame.columns[-1]
    feat_cols = list(frame.columns[:-1])

    work = frame.dropna(subset=[target_col]).copy()
    y = pd.to_numeric(work[target_col], errors="coerce").to_numpy(dtype=np.float64)
    X_frame = work[feat_cols]

    cat_cols = [c for c in feat_cols if X_frame[c].dtype.name == "category" or X_frame[c].dtype == object]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    pieces = []
    if num_cols:
        X_num = X_frame[num_cols].apply(pd.to_numeric, errors="coerce")
        # Drop columns that are entirely missing (or coerced entirely to
        # NaN). Their median is NaN, so the per-row finite-mask filter
        # below would otherwise discard every observation. Several CTR23
        # datasets ship sentinel columns of this form (e.g.
        # ``GpuNumberOfExecutionUnits`` in ``fps_benchmark``).
        keep_num = [c for c in num_cols if not X_num[c].isna().all()]
        dropped = [c for c in num_cols if c not in keep_num]
        if dropped:
            logger.warning(
                "Dataset %s: dropping all-missing numeric columns %s.",
                spec.name, dropped,
            )
        X_num = X_num[keep_num]
        for c in keep_num:
            if X_num[c].isna().any():
                X_num[c] = X_num[c].fillna(X_num[c].median())
        if keep_num:
            pieces.append(X_num)
    if cat_cols:
        X_cat = pd.get_dummies(
            X_frame[cat_cols].astype("category"),
            drop_first=False,
            dummy_na=False,
        ).astype(np.float64)
        pieces.append(X_cat)

    if not pieces:
        raise RuntimeError(f"Dataset {spec.name} has no usable feature columns.")

    X_full = pd.concat(pieces, axis=1).to_numpy(dtype=np.float64)

    finite_rows = np.isfinite(y) & np.all(np.isfinite(X_full), axis=1)
    X_full = X_full[finite_rows]
    y = y[finite_rows]

    return X_full, y


def dataset_schema(spec: DatasetSpec) -> dict:
    """Return the canonical schema record used in the per-dataset table."""
    X, y = fetch_one(spec)
    cache = _cache_path(spec)
    raw = pd.read_parquet(cache)
    target_col = raw.columns[-1]
    feat_cols = list(raw.columns[:-1])
    n_cat = sum(
        1
        for c in feat_cols
        if raw[c].dtype.name == "category" or raw[c].dtype == object
    )
    n_num = len(feat_cols) - n_cat
    return {
        "name": spec.name,
        "source": spec.source,
        "citation": spec.citation,
        "openml_id": spec.openml_id,
        "n": int(X.shape[0]),
        "d_onehot": int(X.shape[1]),
        "d_raw_numeric": int(n_num),
        "d_raw_categorical": int(n_cat),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "y_mean": float(np.mean(y)),
    }


def is_smooth_lowd(X: np.ndarray, y: np.ndarray, *, max_d: int = 30, min_n: int = 500) -> bool:
    """Pre-registered smooth-low-d filter.

    A dataset is in the subset when ``n >= 500`` (already enforced by
    CTR23), the post-one-hot dimension is at most ``max_d = 30``, and
    the target carries at least five distinct values (already enforced
    by CTR23). The latter two are checked here as belt-and-braces.
    """
    n, d = X.shape
    if n < min_n:
        return False
    if d > max_d:
        return False
    if np.unique(y).size < 5:
        return False
    return True


def iter_datasets(
    subset: str = "full",
    *,
    max_n: int | None = None,
    seed: int = 0,
):
    """Yield ``(spec, X, y)`` over the registry.

    Parameters
    ----------
    subset :
        ``'full'`` for all 36 datasets, ``'smooth_lowd'`` for the
        pre-registered subset (``n >= 500``, ``d_onehot <= 30``,
        target with at least five distinct values).
    max_n :
        Optional row cap. When set, datasets with ``n > max_n`` are
        uniformly subsampled with the supplied ``seed`` to keep
        per-cell compute bounded for the experiment driver. The
        subsampling preserves the row order so cached splits stay
        deterministic.
    seed :
        Seed for the subsampler.
    """
    if subset not in {"full", "smooth_lowd"}:
        raise ValueError(f"Unknown subset {subset!r}; use 'full' or 'smooth_lowd'.")

    rng = np.random.default_rng(seed)
    for spec in DATASETS:
        try:
            X, y = fetch_one(spec)
        except Exception as exc:
            logger.warning("Skipping %s: fetch failed (%s).", spec.name, exc)
            continue

        if subset == "smooth_lowd" and not is_smooth_lowd(X, y):
            continue

        if max_n is not None and X.shape[0] > max_n:
            idx = rng.choice(X.shape[0], size=max_n, replace=False)
            idx.sort()
            X, y = X[idx], y[idx]

        yield spec, X, y


def _summary_table() -> pd.DataFrame:
    rows = []
    for spec in DATASETS:
        try:
            row = dataset_schema(spec)
        except Exception as exc:
            row = {
                "name": spec.name,
                "source": spec.source,
                "citation": spec.citation,
                "openml_id": spec.openml_id,
                "n": -1,
                "d_onehot": -1,
                "error": str(exc),
            }
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = _summary_table()
    print(table.to_string(index=False))
    print(f"\nTotal datasets: {len(table)}")
    if "error" not in table.columns:
        ok = table[(table["n"] >= 500) & (table["d_onehot"] <= 30)]
        print(f"Smooth-low-d subset (n>=500, d_onehot<=30): {len(ok)}")
