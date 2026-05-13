#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


INSTALL_COMMAND = (
    "conda activate ppd_pca && "
    "conda install -c conda-forge -y numpy pandas pyarrow scikit-learn matplotlib joblib tqdm"
)
REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
    "tqdm": "tqdm",
}


def _require_dependencies() -> None:
    missing = [
        package_name
        for module_name, package_name in REQUIRED_IMPORTS.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        raise SystemExit(
            "Missing required package(s) in the active environment: "
            f"{', '.join(missing)}\nInstall with:\n  {INSTALL_COMMAND}"
        )


_require_dependencies()

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import sklearn  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from tqdm import tqdm  # noqa: E402


EXPECTED_EMB_DIM = 3584
VALID_SPLITS = ("train", "validation", "test")
VALID_KINDS = ("raw", "l2")
L2_EPS = 1e-12


@dataclass(frozen=True)
class ShardInfo:
    split: str
    shard_id: int
    vector_path: Path
    meta_path: Path
    num_rows: int


@dataclass(frozen=True)
class Selection:
    split: str
    range_tag: str
    shard_start: Optional[int]
    shard_end: Optional[int]
    shards: List[ShardInfo]
    total_rows: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit or apply row-level PCA over Phase 1 user row vectors."
    )
    parser.add_argument("--mode", choices=("fit", "transform"), required=True)
    parser.add_argument(
        "--vector-root",
        type=Path,
        default=Path("/data/roycecho/PPD/pca_data/user_row_pca/vectors"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/roycecho/PPD/pca_data/user_row_pca"),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--shard-start", type=int, default=None)
    parser.add_argument("--shard-end", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--fit-split", choices=VALID_SPLITS, default=None)
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--run-raw", action="store_true")
    parser.add_argument("--run-l2", action="store_true")

    parser.add_argument("--load-pca", type=Path, default=None)
    parser.add_argument("--transform-split", choices=VALID_SPLITS, default=None)
    parser.add_argument("--plot-max-points", type=int, default=20000)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if (args.shard_start is None) != (args.shard_end is None):
        raise ValueError("--shard-start and --shard-end must be provided together.")
    if args.shard_start is not None:
        if args.shard_start < 0:
            raise ValueError(f"--shard-start must be >= 0, got {args.shard_start}")
        if args.shard_end <= args.shard_start:
            raise ValueError(
                "--shard-end must be greater than --shard-start, "
                f"got start={args.shard_start}, end={args.shard_end}"
            )
    if args.plot_max_points <= 0:
        raise ValueError(f"--plot-max-points must be > 0, got {args.plot_max_points}")

    if args.mode == "fit":
        if args.fit_split is None:
            raise ValueError("--mode fit requires --fit-split.")
        if args.n_components <= 0:
            raise ValueError(f"--n-components must be > 0, got {args.n_components}")
        if args.max_rows <= 0:
            raise ValueError(f"--max-rows must be > 0, got {args.max_rows}")
        if not args.run_raw and not args.run_l2:
            raise ValueError("--mode fit requires at least one of --run-raw or --run-l2.")
    else:
        if args.load_pca is None:
            raise ValueError("--mode transform requires --load-pca.")
        if args.transform_split is None:
            raise ValueError("--mode transform requires --transform-split.")


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _vector_path(vector_root: Path, split: str, shard_id: int) -> Path:
    return vector_root / split / f"{split}_shard{shard_id}_mean.npy"


def _meta_path(vector_root: Path, split: str, shard_id: int) -> Path:
    return vector_root / split / f"{split}_shard{shard_id}_meta.parquet"


def _pca_dir(output_root: Path, run_name: str) -> Path:
    return output_root / "pca" / run_name


def _plots_dir(output_root: Path, run_name: str) -> Path:
    return output_root / "plots" / run_name


def _range_tag(split: str, shard_start: Optional[int], shard_end: Optional[int]) -> str:
    if shard_start is None:
        return f"{split}_all"
    return f"{split}_shard{shard_start}_{shard_end}"


def _model_path(pca_dir: Path, kind: str, effective_n_components: int) -> Path:
    return pca_dir / f"row_pca_{kind}_{effective_n_components}.pkl"


def _scores_path(pca_dir: Path, range_tag: str, kind: str) -> Path:
    return pca_dir / f"{range_tag}_row_pc_scores_{kind}.npy"


def _combined_meta_path(pca_dir: Path, range_tag: str) -> Path:
    return pca_dir / f"{range_tag}_row_meta.parquet"


def _stats_path(pca_dir: Path, kind: str) -> Path:
    return pca_dir / f"pc_stats_{kind}.json"


def _explained_variance_csv_path(pca_dir: Path, kind: str) -> Path:
    return pca_dir / f"explained_variance_{kind}.csv"


def _fit_config_path(pca_dir: Path) -> Path:
    return pca_dir / "fit_config.json"


def _transform_config_path(pca_dir: Path, range_tag: str, kind: str) -> Path:
    return pca_dir / f"transform_config_{range_tag}_{kind}.json"


def _plot_paths(plots_dir: Path, kind: str) -> Dict[str, Path]:
    return {
        "explained_variance": plots_dir / f"row_pca_explained_variance_{kind}.png",
        "pc1_pc2": plots_dir / f"row_pca_pc1_pc2_{kind}.png",
        "pc1_pc2_norm": plots_dir / f"row_pca_pc1_pc2_colored_by_norm_{kind}.png",
        "pc1_pc2_user_count": plots_dir / f"row_pca_pc1_pc2_colored_by_user_count_{kind}.png",
    }


def _parse_shard_id(path: Path, split: str, suffix: str) -> Optional[int]:
    pattern = re.compile(rf"^{re.escape(split)}_shard(\d+)_{suffix}$")
    match = pattern.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _validate_vector_meta_pair(vector_path: Path, meta_path: Path, split: str, shard_id: int) -> ShardInfo:
    try:
        vector = np.load(vector_path, mmap_mode="r")
    except Exception as exc:
        raise ValueError(
            f"Failed to load vector shard: split={split}, shard_id={shard_id}, "
            f"path={vector_path}, error={exc}"
        ) from exc

    if vector.ndim != 2 or vector.shape[1] != EXPECTED_EMB_DIM:
        raise ValueError(
            "Invalid vector shape: "
            f"split={split}, shard_id={shard_id}, path={vector_path}, "
            f"expected=[rows,{EXPECTED_EMB_DIM}], actual={tuple(vector.shape)}"
        )

    try:
        parquet_file = pq.ParquetFile(meta_path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read metadata parquet: split={split}, shard_id={shard_id}, "
            f"path={meta_path}, error={exc}"
        ) from exc

    num_meta_rows = int(parquet_file.metadata.num_rows)
    required_meta_columns = {"global_row_id", "user_id", "mean_vec_norm"}
    meta_columns = set(parquet_file.schema_arrow.names)
    missing = sorted(required_meta_columns - meta_columns)
    if missing:
        raise ValueError(
            "Metadata parquet is missing required column(s): "
            f"split={split}, shard_id={shard_id}, path={meta_path}, missing={missing}"
        )
    if int(vector.shape[0]) != num_meta_rows:
        raise ValueError(
            "Vector/meta row count mismatch: "
            f"split={split}, shard_id={shard_id}, vector_rows={vector.shape[0]}, "
            f"meta_rows={num_meta_rows}, vector_path={vector_path}, meta_path={meta_path}"
        )
    return ShardInfo(
        split=split,
        shard_id=shard_id,
        vector_path=vector_path,
        meta_path=meta_path,
        num_rows=int(vector.shape[0]),
    )


def _discover_selection(
    vector_root: Path,
    split: str,
    shard_start: Optional[int],
    shard_end: Optional[int],
) -> Selection:
    split_dir = vector_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Vector split directory not found: {split_dir}")

    shard_infos: List[ShardInfo] = []
    if shard_start is not None:
        for shard_id in range(shard_start, shard_end):
            vector_path = _vector_path(vector_root, split, shard_id)
            meta_path = _meta_path(vector_root, split, shard_id)
            missing = [str(path) for path in (vector_path, meta_path) if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Missing vector/meta pair for split={split}, shard_id={shard_id}: {missing}"
                )
            shard_infos.append(_validate_vector_meta_pair(vector_path, meta_path, split, shard_id))
    else:
        vector_ids = {
            shard_id: path
            for path in split_dir.glob(f"{split}_shard*_mean.npy")
            if (shard_id := _parse_shard_id(path, split, "mean\\.npy")) is not None
        }
        meta_ids = {
            shard_id: path
            for path in split_dir.glob(f"{split}_shard*_meta.parquet")
            if (shard_id := _parse_shard_id(path, split, "meta\\.parquet")) is not None
        }
        one_sided_vectors = sorted(set(vector_ids) - set(meta_ids))
        one_sided_meta = sorted(set(meta_ids) - set(vector_ids))
        if one_sided_vectors or one_sided_meta:
            raise ValueError(
                "Found one-sided vector/meta shard output(s): "
                f"vectors_without_meta={one_sided_vectors}, meta_without_vectors={one_sided_meta}"
            )
        if not vector_ids:
            raise FileNotFoundError(f"No complete vector/meta shards found for split={split} in {split_dir}")
        for shard_id in sorted(vector_ids):
            shard_infos.append(
                _validate_vector_meta_pair(vector_ids[shard_id], meta_ids[shard_id], split, shard_id)
            )

    total_rows = int(sum(info.num_rows for info in shard_infos))
    if total_rows <= 0:
        raise ValueError(f"No rows selected for split={split}.")
    return Selection(
        split=split,
        range_tag=_range_tag(split, shard_start, shard_end),
        shard_start=shard_start,
        shard_end=shard_end,
        shards=shard_infos,
        total_rows=total_rows,
    )


def _check_no_existing(paths: Iterable[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        preview = "\n".join(existing[:20])
        suffix = "" if len(existing) <= 20 else f"\n... and {len(existing) - 20} more"
        raise FileExistsError(f"Output file(s) already exist. Pass --overwrite:\n{preview}{suffix}")


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            np.save(handle, array)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
        df.to_parquet(tmp_path, engine="pyarrow", index=False)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_joblib(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
        joblib.dump(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _atomic_savefig(path: Path, fig: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
        fig.savefig(tmp_path, format="png", dpi=150, bbox_inches="tight")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    finally:
        plt.close(fig)


def _package_versions() -> Dict[str, str]:
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
    }


def _selection_summary(selection: Selection) -> Dict[str, Any]:
    return {
        "split": selection.split,
        "range_tag": selection.range_tag,
        "shard_start": selection.shard_start,
        "shard_end": selection.shard_end,
        "total_rows": selection.total_rows,
        "shards": [
            {
                "shard_id": info.shard_id,
                "num_rows": info.num_rows,
                "vector_path": str(info.vector_path),
                "meta_path": str(info.meta_path),
            }
            for info in selection.shards
        ],
    }


def _sample_global_indices(total_rows: int, max_rows: int, seed: int) -> np.ndarray:
    sample_rows = min(total_rows, max_rows)
    if sample_rows == total_rows:
        return np.arange(total_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_rows, size=sample_rows, replace=False).astype(np.int64))


def _shard_offsets(selection: Selection) -> List[int]:
    offsets: List[int] = []
    offset = 0
    for info in selection.shards:
        offsets.append(offset)
        offset += info.num_rows
    return offsets


def _load_sampled_rows(
    selection: Selection,
    sampled_indices: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    offsets = _shard_offsets(selection)
    chunks: List[np.ndarray] = []
    sampled_row_ids: List[str] = []
    for info, offset in tqdm(
        list(zip(selection.shards, offsets)),
        desc=f"Collect sampled rows ({selection.split})",
    ):
        start_pos = int(np.searchsorted(sampled_indices, offset, side="left"))
        end_pos = int(np.searchsorted(sampled_indices, offset + info.num_rows, side="left"))
        if start_pos == end_pos:
            continue
        local_rows = sampled_indices[start_pos:end_pos] - offset
        vector = np.load(info.vector_path, mmap_mode="r")
        chunks.append(np.asarray(vector[local_rows], dtype=np.float32))
        ids = pd.read_parquet(info.meta_path, columns=["global_row_id"]).iloc[local_rows][
            "global_row_id"
        ]
        sampled_row_ids.extend(ids.astype(str).tolist())

    if not chunks:
        raise ValueError("No sampled rows were collected.")
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False), sampled_row_ids


def _l2_normalize(x: np.ndarray, eps: float = L2_EPS) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps).astype(x.dtype, copy=False)


def _preprocess_vectors(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "raw":
        return x
    if kind == "l2":
        return _l2_normalize(x.astype(np.float32, copy=False))
    raise ValueError(f"Unsupported PCA kind: {kind}")


def _fit_pca(
    sampled_x: np.ndarray,
    kind: str,
    effective_n_components: int,
    seed: int,
) -> PCA:
    x_fit = _preprocess_vectors(sampled_x, kind)
    pca = PCA(
        n_components=effective_n_components,
        svd_solver="randomized",
        random_state=seed,
    )
    pca.fit(x_fit)
    return pca


def _build_combined_meta(selection: Selection) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for info in tqdm(selection.shards, desc=f"Load metadata ({selection.split})"):
        frames.append(pd.read_parquet(info.meta_path))
    if not frames:
        raise ValueError("No metadata frames loaded.")
    meta = pd.concat(frames, ignore_index=True)
    if len(meta) != selection.total_rows:
        raise ValueError(
            f"Combined metadata row count mismatch: expected={selection.total_rows}, got={len(meta)}"
        )
    meta["user_row_count_in_output"] = (
        meta.groupby("user_id", dropna=False)["user_id"].transform("size").astype("int64")
    )
    return meta


def _global_row_ids_for_selection(selection: Selection) -> List[str]:
    ids: List[str] = []
    for info in selection.shards:
        ids.extend(pd.read_parquet(info.meta_path, columns=["global_row_id"])["global_row_id"].astype(str))
    return ids


def _write_or_reuse_metadata(
    path: Path,
    selection: Selection,
    *,
    mode: str,
    overwrite: bool,
) -> Tuple[bool, Optional[pd.DataFrame]]:
    if path.exists() and mode == "transform":
        existing_ids = pd.read_parquet(path, columns=["global_row_id"])["global_row_id"].astype(str).tolist()
        current_ids = _global_row_ids_for_selection(selection)
        if existing_ids != current_ids:
            raise ValueError(
                "Existing metadata global_row_id order differs from current transform input: "
                f"path={path}, split={selection.split}, range_tag={selection.range_tag}"
            )
        print(f"[INFO] Reusing existing metadata parquet with matching global_row_id order: {path}")
        return True, None

    if path.exists() and not overwrite:
        raise FileExistsError(f"Metadata output already exists. Pass --overwrite: {path}")

    meta = _build_combined_meta(selection)
    _atomic_write_parquet(path, meta)
    return False, meta


def _load_or_get_meta_for_plots(meta_path: Path, meta: Optional[pd.DataFrame]) -> pd.DataFrame:
    if meta is not None:
        return meta
    return pd.read_parquet(meta_path)


def _transform_selection(
    selection: Selection,
    payload: Mapping[str, Any],
) -> np.ndarray:
    pca = payload["pca"]
    kind = str(payload["kind"])
    n_components = int(payload["effective_n_components"])
    scores = np.empty((selection.total_rows, n_components), dtype=np.float32)
    cursor = 0
    for info in tqdm(selection.shards, desc=f"Transform {selection.split}/{kind}"):
        x = np.asarray(np.load(info.vector_path, mmap_mode="r"), dtype=np.float32)
        x = _preprocess_vectors(x, kind)
        shard_scores = pca.transform(x).astype(np.float32, copy=False)
        scores[cursor : cursor + info.num_rows] = shard_scores
        cursor += info.num_rows
    return scores


def _score_stats(
    *,
    scores: np.ndarray,
    pca: PCA,
    kind: str,
    requested_n_components: int,
    effective_n_components: int,
    split: str,
    range_tag: str,
) -> Dict[str, Any]:
    explained_variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    return {
        "kind": kind,
        "split": split,
        "range_tag": range_tag,
        "row_count": int(scores.shape[0]),
        "requested_n_components": int(requested_n_components),
        "effective_n_components": int(effective_n_components),
        "score_mean": scores.astype(np.float64).mean(axis=0).tolist(),
        "score_std": scores.astype(np.float64).std(axis=0).tolist(),
        "explained_variance_ratio": explained_variance_ratio.tolist(),
        "cumulative_explained_variance_ratio": np.cumsum(explained_variance_ratio).tolist(),
    }


def _explained_variance_frame(pca: PCA) -> pd.DataFrame:
    explained_variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    return pd.DataFrame(
        {
            "component": np.arange(1, len(explained_variance_ratio) + 1, dtype=np.int64),
            "explained_variance": np.asarray(pca.explained_variance_, dtype=np.float64),
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_explained_variance_ratio": np.cumsum(explained_variance_ratio),
            "singular_value": np.asarray(pca.singular_values_, dtype=np.float64),
        }
    )


def _plot_explained_variance(ev_df: pd.DataFrame, path: Path, kind: str) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = ev_df["component"].to_numpy()
    ratio = ev_df["explained_variance_ratio"].to_numpy()
    cumulative = ev_df["cumulative_explained_variance_ratio"].to_numpy()
    ax1.bar(x, ratio, color="#4c78a8", alpha=0.75, label="Explained variance ratio")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio")
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color="#f58518", marker="o", linewidth=1.5, label="Cumulative")
    ax2.set_ylabel("Cumulative explained variance ratio")
    ax1.set_title(f"Row PCA explained variance ({kind})")
    fig.tight_layout()
    _atomic_savefig(path, fig)


def _plot_scatter(
    *,
    scores: np.ndarray,
    path: Path,
    title: str,
    seed: int,
    plot_max_points: int,
    color_values: Optional[Sequence[Any]] = None,
    colorbar_label: Optional[str] = None,
) -> None:
    n_rows = scores.shape[0]
    if n_rows > plot_max_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(n_rows, size=plot_max_points, replace=False))
    else:
        indices = np.arange(n_rows)

    x = scores[indices, 0]
    if scores.shape[1] >= 2:
        y = scores[indices, 1]
        y_label = "PC2"
    else:
        y = np.zeros_like(x)
        y_label = "PC2 unavailable"

    fig, ax = plt.subplots(figsize=(7, 6))
    if color_values is None:
        ax.scatter(x, y, s=8, alpha=0.65, color="#4c78a8", linewidths=0)
    else:
        colors = np.asarray(color_values)[indices]
        scatter = ax.scatter(x, y, c=colors, s=8, alpha=0.75, cmap="viridis", linewidths=0)
        fig.colorbar(scatter, ax=ax, label=colorbar_label)
    ax.set_xlabel("PC1")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _atomic_savefig(path, fig)


def _write_plots(
    *,
    plot_paths: Mapping[str, Path],
    ev_df: pd.DataFrame,
    scores: np.ndarray,
    meta: pd.DataFrame,
    kind: str,
    seed: int,
    plot_max_points: int,
) -> None:
    _plot_explained_variance(ev_df, plot_paths["explained_variance"], kind)
    _plot_scatter(
        scores=scores,
        path=plot_paths["pc1_pc2"],
        title=f"Row PCA PC1/PC2 ({kind})",
        seed=seed,
        plot_max_points=plot_max_points,
    )
    _plot_scatter(
        scores=scores,
        path=plot_paths["pc1_pc2_norm"],
        title=f"Row PCA PC1/PC2 colored by mean vector norm ({kind})",
        seed=seed,
        plot_max_points=plot_max_points,
        color_values=meta["mean_vec_norm"].to_numpy(),
        colorbar_label="mean_vec_norm",
    )
    _plot_scatter(
        scores=scores,
        path=plot_paths["pc1_pc2_user_count"],
        title=f"Row PCA PC1/PC2 colored by user row count ({kind})",
        seed=seed,
        plot_max_points=plot_max_points,
        color_values=meta["user_row_count_in_output"].to_numpy(),
        colorbar_label="user_row_count_in_output",
    )


def _effective_n_components(requested: int, sampled_rows: int, feature_dim: int) -> int:
    if sampled_rows < 2:
        raise ValueError(f"PCA requires at least 2 sampled rows, got {sampled_rows}.")
    effective = min(requested, sampled_rows, feature_dim)
    if effective < requested:
        warnings.warn(
            "Reducing n_components because requested value exceeds available samples "
            f"or feature dimension: requested={requested}, effective={effective}, "
            f"sampled_rows={sampled_rows}, feature_dim={feature_dim}",
            RuntimeWarning,
        )
    return int(effective)


def _model_payload(
    *,
    pca: PCA,
    kind: str,
    requested_n_components: int,
    effective_n_components: int,
    selection: Selection,
    sampled_global_indices: np.ndarray,
    sampled_global_row_ids: Sequence[str],
    seed: int,
    created_at_utc: str,
) -> Dict[str, Any]:
    return {
        "pca": pca,
        "kind": kind,
        "preprocessing": {
            "l2_normalize": kind == "l2",
            "l2_eps": L2_EPS,
            "feature_dim": EXPECTED_EMB_DIM,
        },
        "requested_n_components": int(requested_n_components),
        "effective_n_components": int(effective_n_components),
        "fit_split": selection.split,
        "fit_range_tag": selection.range_tag,
        "fit_shard_start": selection.shard_start,
        "fit_shard_end": selection.shard_end,
        "fit_shards": [info.shard_id for info in selection.shards],
        "fit_total_rows": int(selection.total_rows),
        "sampled_rows": int(len(sampled_global_indices)),
        "sampled_global_indices": sampled_global_indices.astype(np.int64).tolist(),
        "sampled_global_row_ids": list(sampled_global_row_ids),
        "seed": int(seed),
        "created_at_utc": created_at_utc,
        "versions": _package_versions(),
    }


def _load_pca_payload(path: Path) -> Dict[str, Any]:
    payload = joblib.load(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected PCA payload mapping at {path}, got {type(payload)}")
    required = {"pca", "kind", "effective_n_components", "preprocessing"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise ValueError(f"PCA payload is missing required keys at {path}: {missing}")
    kind = str(payload["kind"])
    if kind not in VALID_KINDS:
        raise ValueError(f"Unsupported PCA payload kind at {path}: {kind}")
    if int(payload["preprocessing"].get("feature_dim", EXPECTED_EMB_DIM)) != EXPECTED_EMB_DIM:
        raise ValueError(
            f"Unsupported PCA feature_dim at {path}: {payload['preprocessing'].get('feature_dim')}"
        )
    return dict(payload)


def _fit_mode(args: argparse.Namespace, vector_root: Path, output_root: Path) -> None:
    selection = _discover_selection(vector_root, args.fit_split, args.shard_start, args.shard_end)
    pca_output_dir = _pca_dir(output_root, args.run_name)
    plot_output_dir = _plots_dir(output_root, args.run_name)
    pca_output_dir.mkdir(parents=True, exist_ok=True)
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    sampled_indices = _sample_global_indices(selection.total_rows, args.max_rows, args.seed)
    sampled_x, sampled_row_ids = _load_sampled_rows(selection, sampled_indices)
    effective = _effective_n_components(args.n_components, sampled_x.shape[0], sampled_x.shape[1])
    kinds = [kind for kind, enabled in (("raw", args.run_raw), ("l2", args.run_l2)) if enabled]

    meta_path = _combined_meta_path(pca_output_dir, selection.range_tag)
    target_paths: List[Path] = [_fit_config_path(pca_output_dir), meta_path]
    for kind in kinds:
        target_paths.extend(
            [
                _model_path(pca_output_dir, kind, effective),
                _scores_path(pca_output_dir, selection.range_tag, kind),
                _stats_path(pca_output_dir, kind),
                _explained_variance_csv_path(pca_output_dir, kind),
                *_plot_paths(plot_output_dir, kind).values(),
            ]
        )
    _check_no_existing(target_paths, args.overwrite)

    metadata_reused, meta = _write_or_reuse_metadata(
        meta_path,
        selection,
        mode="fit",
        overwrite=args.overwrite,
    )
    if metadata_reused:
        raise AssertionError("Fit mode should not reuse metadata.")
    assert meta is not None

    fit_started_at = _utc_now_iso()
    outputs: Dict[str, Any] = {}
    for kind in kinds:
        print(
            f"[INFO] Fitting {kind} PCA: sampled_rows={sampled_x.shape[0]}, "
            f"effective_n_components={effective}",
            flush=True,
        )
        pca = _fit_pca(sampled_x, kind, effective, args.seed)
        payload = _model_payload(
            pca=pca,
            kind=kind,
            requested_n_components=args.n_components,
            effective_n_components=effective,
            selection=selection,
            sampled_global_indices=sampled_indices,
            sampled_global_row_ids=sampled_row_ids,
            seed=args.seed,
            created_at_utc=_utc_now_iso(),
        )
        model_path = _model_path(pca_output_dir, kind, effective)
        _atomic_write_joblib(model_path, payload)

        scores = _transform_selection(selection, payload)
        scores_path = _scores_path(pca_output_dir, selection.range_tag, kind)
        _atomic_write_npy(scores_path, scores)

        ev_df = _explained_variance_frame(pca)
        ev_csv_path = _explained_variance_csv_path(pca_output_dir, kind)
        _atomic_write_csv(ev_csv_path, ev_df)

        stats = _score_stats(
            scores=scores,
            pca=pca,
            kind=kind,
            requested_n_components=args.n_components,
            effective_n_components=effective,
            split=selection.split,
            range_tag=selection.range_tag,
        )
        stats_path = _stats_path(pca_output_dir, kind)
        _atomic_write_json(stats_path, stats)

        plot_paths = _plot_paths(plot_output_dir, kind)
        _write_plots(
            plot_paths=plot_paths,
            ev_df=ev_df,
            scores=scores,
            meta=meta,
            kind=kind,
            seed=args.seed,
            plot_max_points=args.plot_max_points,
        )
        outputs[kind] = {
            "model_path": str(model_path),
            "scores_path": str(scores_path),
            "stats_path": str(stats_path),
            "explained_variance_csv_path": str(ev_csv_path),
            "plot_paths": {name: str(path) for name, path in plot_paths.items()},
            "score_shape": list(scores.shape),
        }
        print(json.dumps({"fit_kind_summary": outputs[kind]}, ensure_ascii=False), flush=True)

    config = {
        "mode": "fit",
        "argv": sys.argv,
        "python_executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "created_at_utc": _utc_now_iso(),
        "fit_started_at_utc": fit_started_at,
        "versions": _package_versions(),
        "args": {
            "vector_root": str(args.vector_root),
            "output_root": str(args.output_root),
            "run_name": args.run_name,
            "fit_split": args.fit_split,
            "shard_start": args.shard_start,
            "shard_end": args.shard_end,
            "requested_n_components": args.n_components,
            "effective_n_components": effective,
            "max_rows": args.max_rows,
            "seed": args.seed,
            "run_raw": bool(args.run_raw),
            "run_l2": bool(args.run_l2),
            "overwrite": bool(args.overwrite),
        },
        "selection": _selection_summary(selection),
        "sampled_rows": int(len(sampled_indices)),
        "metadata_path": str(meta_path),
        "outputs": outputs,
    }
    _atomic_write_json(_fit_config_path(pca_output_dir), config)
    print(
        json.dumps(
            {
                "fit_run_summary": {
                    "run_name": args.run_name,
                    "range_tag": selection.range_tag,
                    "kinds": kinds,
                    "effective_n_components": effective,
                    "metadata_path": str(meta_path),
                    "fit_config_path": str(_fit_config_path(pca_output_dir)),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _transform_mode(args: argparse.Namespace, vector_root: Path, output_root: Path) -> None:
    payload = _load_pca_payload(_resolve_path(args.load_pca))
    kind = str(payload["kind"])
    selection = _discover_selection(vector_root, args.transform_split, args.shard_start, args.shard_end)
    pca_output_dir = _pca_dir(output_root, args.run_name)
    pca_output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = _combined_meta_path(pca_output_dir, selection.range_tag)
    scores_path = _scores_path(pca_output_dir, selection.range_tag, kind)
    transform_config_path = _transform_config_path(pca_output_dir, selection.range_tag, kind)
    _check_no_existing([scores_path, transform_config_path], args.overwrite)

    metadata_reused, meta = _write_or_reuse_metadata(
        meta_path,
        selection,
        mode="transform",
        overwrite=args.overwrite,
    )
    scores = _transform_selection(selection, payload)
    _atomic_write_npy(scores_path, scores)

    config = {
        "mode": "transform",
        "argv": sys.argv,
        "python_executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "created_at_utc": _utc_now_iso(),
        "versions": _package_versions(),
        "args": {
            "vector_root": str(args.vector_root),
            "output_root": str(args.output_root),
            "run_name": args.run_name,
            "load_pca": str(args.load_pca),
            "transform_split": args.transform_split,
            "shard_start": args.shard_start,
            "shard_end": args.shard_end,
            "seed": args.seed,
            "overwrite": bool(args.overwrite),
        },
        "selection": _selection_summary(selection),
        "pca_payload": {
            "kind": kind,
            "requested_n_components": int(payload.get("requested_n_components", -1)),
            "effective_n_components": int(payload["effective_n_components"]),
            "fit_split": payload.get("fit_split"),
            "fit_range_tag": payload.get("fit_range_tag"),
        },
        "metadata_path": str(meta_path),
        "metadata_reused": bool(metadata_reused),
        "scores_path": str(scores_path),
        "score_shape": list(scores.shape),
    }
    _atomic_write_json(transform_config_path, config)
    print(
        json.dumps(
            {
                "transform_run_summary": {
                    "run_name": args.run_name,
                    "range_tag": selection.range_tag,
                    "kind": kind,
                    "metadata_path": str(meta_path),
                    "metadata_reused": bool(metadata_reused),
                    "scores_path": str(scores_path),
                    "score_shape": list(scores.shape),
                    "transform_config_path": str(transform_config_path),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    vector_root = _resolve_path(args.vector_root)
    output_root = _resolve_path(args.output_root)
    if args.mode == "fit":
        _fit_mode(args, vector_root, output_root)
    else:
        _transform_mode(args, vector_root, output_root)


if __name__ == "__main__":
    main()
