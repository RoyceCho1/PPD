from __future__ import annotations

"""Index already-saved latent `.pt` files into `latent_manifest.jsonl`.

This script does not generate latents. It does not read raw images, run an
encoder forward pass, or write new latent tensors.

Its only job is to scan latent `.pt` files that were already produced by a
separate pipeline such as `prepare_stage_c_latents.py`, extract compact
metadata from each file, and write one JSONL manifest record per UID.
"""

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


LATENT_TENSOR_KEYS: Sequence[str] = ("latent", "image_embeds", "tensor")
UID_META_COMPACT_FIELDS: Sequence[str] = (
    "num_occurrences",
    "best_count",
    "non_best_count",
    "caption_samples",
    "partner_uid_samples",
    "source_paths",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan existing latent .pt files and build latent_manifest.jsonl."
    )
    parser.add_argument(
        "--latent-root",
        type=Path,
        required=True,
        help="Root directory containing saved latent .pt files.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Path to write latent_manifest.jsonl.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan .pt files directly under --latent-root.",
    )
    parser.add_argument(
        "--prefer-filename-uid",
        action="store_true",
        help="Use the .pt filename stem as UID even when the bundle contains `uid`.",
    )
    parser.add_argument(
        "--no-compute-stats-if-missing",
        action="store_true",
        help="Do not compute min/max/mean/std when the bundle does not contain `stats`.",
    )
    parser.add_argument(
        "--fail-on-duplicate-uid",
        action="store_true",
        help="Raise an error when two latent files resolve to the same UID.",
    )
    parser.add_argument(
        "--uid-to-path-json",
        type=Path,
        default=None,
        help="Optional uid_to_path.json for enrichment.",
    )
    parser.add_argument(
        "--uid-to-meta-json",
        type=Path,
        default=None,
        help="Optional uid_to_meta.json for enrichment.",
    )
    parser.add_argument(
        "--include-uid-path",
        action="store_true",
        help="Include `uid_to_path_value` from --uid-to-path-json.",
    )
    parser.add_argument(
        "--include-uid-meta",
        action="store_true",
        help="Include compact `uid_meta` from --uid-to-meta-json.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to save the final summary JSON.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N scanned .pt files. Set <=0 to disable periodic progress logs.",
    )
    return parser.parse_args()


def _normalize_uid(value: Any) -> str:
    uid = str(value).strip()
    if not uid:
        raise ValueError("Resolved UID is empty.")
    return uid


def _load_json_object(path: Path, *, label: str) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} file not found: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse {label} JSON: {resolved} ({exc})") from exc

    if not isinstance(data, Mapping):
        raise ValueError(f"{label} JSON must be an object/dict: {resolved}")

    return {str(key): value for key, value in data.items()}


def _resolve_optional_inputs(args: argparse.Namespace) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if args.include_uid_path and args.uid_to_path_json is None:
        raise ValueError("--include-uid-path requires --uid-to-path-json.")
    if args.include_uid_meta and args.uid_to_meta_json is None:
        raise ValueError("--include-uid-meta requires --uid-to-meta-json.")

    uid_to_path_map: Dict[str, Any] = {}
    uid_to_meta_map: Dict[str, Any] = {}

    if args.include_uid_path and args.uid_to_path_json is not None:
        uid_to_path_map = _load_json_object(args.uid_to_path_json, label="uid_to_path")
    if args.include_uid_meta and args.uid_to_meta_json is not None:
        uid_to_meta_map = _load_json_object(args.uid_to_meta_json, label="uid_to_meta")

    return uid_to_path_map, uid_to_meta_map


def _scan_pt_files(latent_root: Path, recursive: bool) -> List[Path]:
    root = latent_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"latent root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"latent root is not a directory: {root}")

    iterator = root.rglob("*.pt") if recursive else root.glob("*.pt")
    return sorted(path.resolve() for path in iterator if path.is_file())


def _load_pt(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load latent .pt file: {path}\n{exc}") from exc


def _extract_latent_tensor(loaded: Any, pt_path: Path) -> torch.Tensor:
    if torch.is_tensor(loaded):
        return loaded

    if not isinstance(loaded, Mapping):
        raise TypeError(
            f"Unsupported .pt payload at {pt_path}: expected Tensor or dict-like bundle, got {type(loaded)}"
        )

    for key in LATENT_TENSOR_KEYS:
        if key not in loaded:
            continue
        tensor = loaded[key]
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"Bundle key `{key}` exists but is not a torch.Tensor at {pt_path}: got {type(tensor)}"
            )
        return tensor

    available_keys = sorted(str(key) for key in loaded.keys())
    raise KeyError(
        f"Could not find latent tensor in bundle at {pt_path}. "
        f"Expected one of {list(LATENT_TENSOR_KEYS)}. Available keys: {available_keys}"
    )


def _coerce_bool(value: Any, *, field_name: str, pt_path: Path) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(
        f"Expected `{field_name}` to be boolean-like at {pt_path}, got value={value!r} type={type(value)}"
    )


def _coerce_number(value: Any, *, field_name: str, pt_path: Path) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                f"Expected scalar tensor for `{field_name}` at {pt_path}, got shape={tuple(value.shape)}"
            )
        return float(value.item())
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(
        f"Expected numeric value for `{field_name}` at {pt_path}, got value={value!r} type={type(value)}"
    )


def _coerce_stats_mapping(stats: Any, pt_path: Path) -> Mapping[str, Any]:
    if isinstance(stats, Mapping):
        return stats
    if hasattr(stats, "__dict__"):
        raw = vars(stats)
        if isinstance(raw, Mapping):
            return raw
    raise ValueError(
        f"Bundle `stats` must be mapping-like at {pt_path}, got {type(stats)}"
    )


def _extract_stats_from_bundle(stats: Any, pt_path: Path) -> Dict[str, float]:
    mapping = _coerce_stats_mapping(stats, pt_path)
    field_aliases = {
        "min": ("min", "min_value"),
        "max": ("max", "max_value"),
        "mean": ("mean", "mean_value"),
        "std": ("std", "std_value"),
    }

    result: Dict[str, float] = {}
    for output_key, candidates in field_aliases.items():
        selected: Any = None
        found = False
        for candidate in candidates:
            if candidate in mapping:
                selected = mapping[candidate]
                found = True
                break
        if not found:
            raise ValueError(
                f"Bundle `stats` at {pt_path} is missing required field for `{output_key}`. "
                f"Accepted aliases: {list(candidates)}"
            )
        result[output_key] = _coerce_number(selected, field_name=f"stats.{output_key}", pt_path=pt_path)
    return result


def _compute_tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    cpu_tensor = tensor.detach().float().cpu()
    return {
        "min": float(cpu_tensor.min().item()),
        "max": float(cpu_tensor.max().item()),
        "mean": float(cpu_tensor.mean().item()),
        "std": float(cpu_tensor.std(unbiased=False).item()),
    }


def _resolve_uid(loaded: Any, pt_path: Path, prefer_filename_uid: bool) -> str:
    filename_uid = _normalize_uid(pt_path.stem)
    if prefer_filename_uid:
        return filename_uid

    if isinstance(loaded, Mapping) and "uid" in loaded and loaded["uid"] is not None:
        try:
            return _normalize_uid(loaded["uid"])
        except ValueError as exc:
            raise ValueError(f"Invalid bundle `uid` at {pt_path}: {exc}") from exc

    return filename_uid


def _resolve_scaled(loaded: Any, pt_path: Path) -> Optional[bool]:
    if not isinstance(loaded, Mapping):
        return None
    if "scaled" in loaded and loaded["scaled"] is not None:
        return _coerce_bool(loaded["scaled"], field_name="scaled", pt_path=pt_path)
    if "apply_scaling" in loaded and loaded["apply_scaling"] is not None:
        return _coerce_bool(loaded["apply_scaling"], field_name="apply_scaling", pt_path=pt_path)
    return None


def _resolve_source_image_path(loaded: Any) -> Optional[str]:
    if not isinstance(loaded, Mapping):
        return None

    for key in ("source_image_path", "original_image_path"):
        if key in loaded and loaded[key] is not None:
            return str(loaded[key])
    return None


def _resolve_latent_format_version(loaded: Any) -> Optional[Any]:
    if isinstance(loaded, Mapping) and "latent_format_version" in loaded:
        return loaded["latent_format_version"]
    return None


def _resolve_latent_semantics(loaded: Any) -> Optional[Any]:
    if isinstance(loaded, Mapping) and "latent_semantics" in loaded:
        return loaded["latent_semantics"]
    return None


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _to_jsonable(value.item())
        return [_to_jsonable(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    return str(value)


def _build_uid_meta_compact(uid: str, meta_value: Any, source_path: Path) -> Optional[Dict[str, Any]]:
    if meta_value is None:
        return None
    if not isinstance(meta_value, Mapping):
        raise ValueError(
            f"uid_to_meta entry for uid={uid!r} must be an object/dict in {source_path}, got {type(meta_value)}"
        )

    compact: Dict[str, Any] = {}
    for field in UID_META_COMPACT_FIELDS:
        if field in meta_value:
            compact[field] = _to_jsonable(meta_value[field])
    return compact


def _build_manifest_record(
    pt_path: Path,
    loaded: Any,
    tensor: torch.Tensor,
    prefer_filename_uid: bool,
    compute_stats_if_missing: bool,
    uid_to_path_map: Dict[str, Any],
    uid_to_meta_map: Dict[str, Any],
    include_uid_path: bool,
    include_uid_meta: bool,
    uid_to_meta_json_path: Optional[Path],
) -> Dict[str, Any]:
    uid = _resolve_uid(loaded, pt_path, prefer_filename_uid=prefer_filename_uid)
    stats: Dict[str, Optional[float]]

    if isinstance(loaded, Mapping) and loaded.get("stats") is not None:
        stats = _extract_stats_from_bundle(loaded["stats"], pt_path)
    elif compute_stats_if_missing:
        stats = _compute_tensor_stats(tensor)
    else:
        stats = {"min": None, "max": None, "mean": None, "std": None}

    record: Dict[str, Any] = {
        "uid": uid,
        "latent_path": str(pt_path.resolve()),
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "scaled": _resolve_scaled(loaded, pt_path),
        "latent_semantics": _to_jsonable(_resolve_latent_semantics(loaded)),
        "source_image_path": _resolve_source_image_path(loaded),
        "latent_format_version": _to_jsonable(_resolve_latent_format_version(loaded)),
        "min": stats["min"],
        "max": stats["max"],
        "mean": stats["mean"],
        "std": stats["std"],
    }

    if include_uid_path:
        record["uid_to_path_value"] = _to_jsonable(uid_to_path_map.get(uid))
    if include_uid_meta:
        assert uid_to_meta_json_path is not None
        record["uid_meta"] = _build_uid_meta_compact(uid, uid_to_meta_map.get(uid), uid_to_meta_json_path)

    return _to_jsonable(record)


def _write_jsonl(records: List[Dict[str, Any]], output_jsonl: Path) -> None:
    output_path = output_jsonl.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_summary_json(summary: Dict[str, Any], summary_json: Path) -> None:
    summary_path = summary_json.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _print_progress(
    *,
    scanned: int,
    total: int,
    written: int,
    duplicates: int,
    start_time: float,
) -> None:
    elapsed = max(time.time() - start_time, 1e-9)
    rate = scanned / elapsed
    remaining = max(total - scanned, 0)
    eta_seconds = remaining / rate if rate > 0 else float("inf")
    eta_text = _format_duration(eta_seconds) if eta_seconds != float("inf") else "inf"

    print(
        "[progress] "
        f"scanned={scanned}/{total} "
        f"written={written} "
        f"duplicates={duplicates} "
        f"rate={rate:.2f} files/s "
        f"elapsed={_format_duration(elapsed)} "
        f"eta={eta_text}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    recursive = not args.no_recursive
    compute_stats_if_missing = not args.no_compute_stats_if_missing

    uid_to_path_map, uid_to_meta_map = _resolve_optional_inputs(args)
    pt_files = _scan_pt_files(args.latent_root, recursive=recursive)
    total_files = len(pt_files)

    print(
        "[start] "
        f"scanning {total_files} .pt files under {args.latent_root.expanduser().resolve()} "
        f"(recursive={recursive})",
        flush=True,
    )

    records: List[Dict[str, Any]] = []
    first_path_by_uid: Dict[str, str] = {}
    duplicate_uids: List[Dict[str, str]] = []
    start_time = time.time()

    for idx, pt_path in enumerate(pt_files, start=1):
        loaded = _load_pt(pt_path)
        tensor = _extract_latent_tensor(loaded, pt_path)
        record = _build_manifest_record(
            pt_path=pt_path,
            loaded=loaded,
            tensor=tensor,
            prefer_filename_uid=args.prefer_filename_uid,
            compute_stats_if_missing=compute_stats_if_missing,
            uid_to_path_map=uid_to_path_map,
            uid_to_meta_map=uid_to_meta_map,
            include_uid_path=args.include_uid_path,
            include_uid_meta=args.include_uid_meta,
            uid_to_meta_json_path=args.uid_to_meta_json,
        )

        uid = str(record["uid"])
        latent_path = str(record["latent_path"])
        if uid in first_path_by_uid:
            duplicate_info = {
                "uid": uid,
                "first_path": first_path_by_uid[uid],
                "next_path": latent_path,
            }
            duplicate_uids.append(duplicate_info)
            if args.fail_on_duplicate_uid:
                raise ValueError(
                    f"Duplicate UID detected: uid={uid!r}, "
                    f"first_path={first_path_by_uid[uid]}, next_path={latent_path}"
                )
        else:
            first_path_by_uid[uid] = latent_path
            records.append(record)

        should_log_progress = (
            args.progress_every > 0
            and (
                idx == 1
                or idx == total_files
                or idx % args.progress_every == 0
            )
        )
        if should_log_progress:
            _print_progress(
                scanned=idx,
                total=total_files,
                written=len(records),
                duplicates=len(duplicate_uids),
                start_time=start_time,
            )

    _write_jsonl(records, args.output_jsonl)

    summary: Dict[str, Any] = {
        "latent_root": str(args.latent_root.expanduser().resolve()),
        "output_jsonl": str(args.output_jsonl.expanduser().resolve()),
        "total_pt_files_scanned": len(pt_files),
        "total_manifest_records_written": len(records),
        "duplicate_uids": duplicate_uids,
        "recursive": recursive,
        "include_uid_path": bool(args.include_uid_path),
        "include_uid_meta": bool(args.include_uid_meta),
    }

    if args.summary_json is not None:
        _write_summary_json(summary, args.summary_json)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
