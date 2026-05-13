#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


EXPECTED_EMB_DIM = 3584
NUM_SUPPORT_PAIRS = 4
VALID_SPLITS = ("train", "validation", "test")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required_columns() -> List[str]:
    columns = ["user_id", "text", "emb"]
    for idx in range(NUM_SUPPORT_PAIRS):
        columns.extend(
            [
                f"caption_{idx}",
                f"preferred_image_uid_{idx}",
                f"dispreferred_image_uid_{idx}",
            ]
        )
    return columns


REQUIRED_COLUMNS = _required_columns()
META_COLUMNS = [
    "global_row_id",
    "split",
    "shard_id",
    "row_in_shard",
    "user_id",
    "text",
    "caption_0",
    "caption_1",
    "caption_2",
    "caption_3",
    "preferred_image_uid_0",
    "preferred_image_uid_1",
    "preferred_image_uid_2",
    "preferred_image_uid_3",
    "dispreferred_image_uid_0",
    "dispreferred_image_uid_1",
    "dispreferred_image_uid_2",
    "dispreferred_image_uid_3",
    "emb_shape",
    "emb_token_count",
    "emb_dim",
    "emb_norm",
    "mean_vec_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract row-level mean-pooled vectors from Stage 1 user embedding shards."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/roycecho/PPD/data/user_emb_7b_full"),
        help="Directory containing {split}_shard{i}.json files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Data_Storage/roycecho/PPD/pca_data/user_row_pca"),
        help="Root directory for PCA extraction outputs.",
    )
    parser.add_argument("--split", required=True, choices=VALID_SPLITS)
    parser.add_argument("--shard-start", type=int, required=True)
    parser.add_argument("--shard-end", type=int, required=True)
    parser.add_argument("--pooling", choices=("mean",), default="mean")
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shard output pairs.",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _validate_args(args: argparse.Namespace) -> None:
    if args.shard_start < 0:
        raise ValueError(f"--shard-start must be >= 0, got {args.shard_start}")
    if args.shard_end <= args.shard_start:
        raise ValueError(
            "--shard-end must be greater than --shard-start, "
            f"got start={args.shard_start}, end={args.shard_end}"
        )


def _shard_ids(args: argparse.Namespace) -> List[int]:
    return list(range(args.shard_start, args.shard_end))


def _input_shard_path(input_root: Path, split: str, shard_id: int) -> Path:
    return input_root / f"{split}_shard{shard_id}.json"


def _output_paths(output_root: Path, split: str, shard_id: int) -> Dict[str, Path]:
    output_dir = output_root / "vectors" / split
    stem = f"{split}_shard{shard_id}"
    return {
        "vector": output_dir / f"{stem}_mean.npy",
        "meta": output_dir / f"{stem}_meta.parquet",
    }


def _config_path(output_root: Path, split: str, shard_start: int, shard_end: int) -> Path:
    return (
        output_root
        / "vectors"
        / split
        / f"extract_row_vectors_config_{split}_shard{shard_start}_{shard_end}.json"
    )


def _preflight(
    *,
    input_root: Path,
    output_root: Path,
    split: str,
    shard_ids: Sequence[int],
    overwrite: bool,
) -> None:
    missing_inputs = [
        str(_input_shard_path(input_root, split, shard_id))
        for shard_id in shard_ids
        if not _input_shard_path(input_root, split, shard_id).exists()
    ]
    if missing_inputs:
        preview = "\n".join(missing_inputs[:20])
        suffix = "" if len(missing_inputs) <= 20 else f"\n... and {len(missing_inputs) - 20} more"
        raise FileNotFoundError(f"Missing input shard(s):\n{preview}{suffix}")

    if not overwrite:
        existing_outputs: List[str] = []
        for shard_id in shard_ids:
            paths = _output_paths(output_root, split, shard_id)
            for path in (paths["vector"], paths["meta"]):
                if path.exists():
                    existing_outputs.append(str(path))
        if existing_outputs:
            preview = "\n".join(existing_outputs[:20])
            suffix = (
                ""
                if len(existing_outputs) <= 20
                else f"\n... and {len(existing_outputs) - 20} more"
            )
            raise FileExistsError(
                "Output file(s) already exist. Pass --overwrite to replace them:\n"
                f"{preview}{suffix}"
            )


def _validate_required_columns(df: pd.DataFrame, shard_path: Path, split: str, shard_id: int) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Input shard is missing required column(s): "
            f"split={split}, shard_id={shard_id}, path={shard_path}, missing={missing}"
        )


def _shape_to_string(shape: Sequence[int]) -> str:
    return "x".join(str(dim) for dim in shape)


def _invalid_emb_shape_error(
    *,
    split: str,
    shard_id: int,
    row_in_shard: int,
    user_id: Any,
    shape: Sequence[int],
) -> ValueError:
    return ValueError(
        "Invalid emb shape: "
        f"split={split}, shard_id={shard_id}, row_in_shard={row_in_shard}, "
        f"user_id={user_id}, expected=[L,{EXPECTED_EMB_DIM}] with L>0, "
        f"actual={tuple(shape)}"
    )


def _as_float32_embedding(
    emb_raw: Any,
    *,
    split: str,
    shard_id: int,
    row_in_shard: int,
    user_id: Any,
) -> np.ndarray:
    try:
        arr = np.asarray(emb_raw, dtype=np.float32)
    except Exception as exc:
        raise ValueError(
            "Failed to convert emb to float32 ndarray: "
            f"split={split}, shard_id={shard_id}, row_in_shard={row_in_shard}, "
            f"user_id={user_id}, error={exc}"
        ) from exc

    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] != EXPECTED_EMB_DIM:
        raise _invalid_emb_shape_error(
            split=split,
            shard_id=shard_id,
            row_in_shard=row_in_shard,
            user_id=user_id,
            shape=arr.shape,
        )
    return arr


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
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
    tmp_path: Path | None = None
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
    tmp_path: Path | None = None
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


def _float_summary(values: Iterable[float]) -> Dict[str, float | None]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _int_summary(values: Iterable[int]) -> Dict[str, int | None]:
    arr = np.asarray(list(values), dtype=np.int64)
    if arr.size == 0:
        return {"min": None, "max": None}
    return {"min": int(arr.min()), "max": int(arr.max())}


def _extract_shard(
    *,
    shard_path: Path,
    vector_path: Path,
    meta_path: Path,
    split: str,
    shard_id: int,
) -> Dict[str, Any]:
    df = pd.read_json(shard_path)
    _validate_required_columns(df, shard_path, split, shard_id)

    vectors = np.empty((len(df), EXPECTED_EMB_DIM), dtype=np.float32)
    meta_rows: List[Dict[str, Any]] = []
    token_counts: List[int] = []
    emb_norms: List[float] = []
    mean_vec_norms: List[float] = []
    column_idx = {column: idx for idx, column in enumerate(df.columns)}

    for row_in_shard, row_values in enumerate(df.itertuples(index=False, name=None)):
        user_id = row_values[column_idx["user_id"]]
        emb = _as_float32_embedding(
            row_values[column_idx["emb"]],
            split=split,
            shard_id=shard_id,
            row_in_shard=row_in_shard,
            user_id=user_id,
        )

        mean_vec = emb.mean(axis=0).astype(np.float32, copy=False)
        vectors[row_in_shard] = mean_vec

        token_count = int(emb.shape[0])
        emb_dim = int(emb.shape[1])
        emb_norm = float(np.linalg.norm(emb))
        mean_vec_norm = float(np.linalg.norm(mean_vec))
        token_counts.append(token_count)
        emb_norms.append(emb_norm)
        mean_vec_norms.append(mean_vec_norm)

        meta_row: Dict[str, Any] = {
            "global_row_id": f"{split}_shard{shard_id:03d}_row{row_in_shard:06d}",
            "split": split,
            "shard_id": int(shard_id),
            "row_in_shard": int(row_in_shard),
            "user_id": user_id,
            "text": row_values[column_idx["text"]],
        }
        for idx in range(NUM_SUPPORT_PAIRS):
            for prefix in ("caption", "preferred_image_uid", "dispreferred_image_uid"):
                column = f"{prefix}_{idx}"
                meta_row[column] = row_values[column_idx[column]]
        meta_row.update(
            {
                "emb_shape": _shape_to_string(emb.shape),
                "emb_token_count": token_count,
                "emb_dim": emb_dim,
                "emb_norm": emb_norm,
                "mean_vec_norm": mean_vec_norm,
            }
        )
        meta_rows.append(meta_row)

    meta_df = pd.DataFrame(meta_rows, columns=META_COLUMNS)
    _atomic_write_npy(vector_path, vectors)
    _atomic_write_parquet(meta_path, meta_df)

    token_summary = _int_summary(token_counts)
    emb_norm_summary = _float_summary(emb_norms)
    mean_norm_summary = _float_summary(mean_vec_norms)
    summary = {
        "split": split,
        "shard_id": int(shard_id),
        "input_path": str(shard_path),
        "rows": int(len(df)),
        "vector_shape": list(vectors.shape),
        "vector_dtype": str(vectors.dtype),
        "token_length_min": token_summary["min"],
        "token_length_max": token_summary["max"],
        "emb_norm_min": emb_norm_summary["min"],
        "emb_norm_max": emb_norm_summary["max"],
        "emb_norm_mean": emb_norm_summary["mean"],
        "mean_vec_norm_min": mean_norm_summary["min"],
        "mean_vec_norm_max": mean_norm_summary["max"],
        "mean_vec_norm_mean": mean_norm_summary["mean"],
        "vector_path": str(vector_path),
        "meta_path": str(meta_path),
    }
    print(json.dumps({"shard_summary": summary}, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    _validate_args(args)

    start_time = _utc_now_iso()
    input_root = _resolve_path(args.input_root)
    output_root = _resolve_path(args.output_root)
    shard_ids = _shard_ids(args)

    _preflight(
        input_root=input_root,
        output_root=output_root,
        split=args.split,
        shard_ids=shard_ids,
        overwrite=bool(args.overwrite),
    )

    shard_summaries: List[Dict[str, Any]] = []
    for shard_id in shard_ids:
        paths = _output_paths(output_root, args.split, shard_id)
        shard_summary = _extract_shard(
            shard_path=_input_shard_path(input_root, args.split, shard_id),
            vector_path=paths["vector"],
            meta_path=paths["meta"],
            split=args.split,
            shard_id=shard_id,
        )
        shard_summaries.append(shard_summary)

    end_time = _utc_now_iso()
    config = {
        "script": str(Path(__file__).expanduser().resolve()),
        "python_executable": sys.executable,
        "argv": sys.argv,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "start_time_utc": start_time,
        "end_time_utc": end_time,
        "args": {
            "input_root": str(args.input_root),
            "output_root": str(args.output_root),
            "split": args.split,
            "shard_start": args.shard_start,
            "shard_end": args.shard_end,
            "pooling": args.pooling,
            "dtype": args.dtype,
            "overwrite": bool(args.overwrite),
        },
        "resolved_paths": {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "config_path": str(
                _config_path(output_root, args.split, args.shard_start, args.shard_end)
            ),
        },
        "expected_emb_dim": EXPECTED_EMB_DIM,
        "required_columns": REQUIRED_COLUMNS,
        "metadata_columns": META_COLUMNS,
        "processed_shards": shard_summaries,
    }
    config_output_path = _config_path(output_root, args.split, args.shard_start, args.shard_end)
    _atomic_write_json(config_output_path, config)
    print(
        json.dumps(
            {
                "run_summary": {
                    "split": args.split,
                    "shard_start": args.shard_start,
                    "shard_end": args.shard_end,
                    "num_shards": len(shard_ids),
                    "config_path": str(config_output_path),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
