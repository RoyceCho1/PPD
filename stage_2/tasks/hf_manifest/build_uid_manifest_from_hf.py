from __future__ import annotations

"""Build UID manifest files from a Hugging Face dataset.

This script scans preference rows and builds:
1) uid_to_path.json: image_uid -> image file path (or optional HF locator string)
2) uid_to_meta.json: image_uid -> aggregated metadata

Design goal:
- Stage 2 dataset remains Stage1-JSON-centric.
- HF dataset is used only as an auxiliary source to build lookup manifests.
"""
import os
os.environ.setdefault("HF_HOME", "/Data_Storage/roycecho/PPD/hf_cache")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset


def _normalize_uid(value: Any) -> str:
    return str(value)


def _to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def _extract_bytes_and_path(cell: Any) -> Tuple[Optional[bytes], Optional[str]]:
    """Extract raw bytes and/or source path from various HF cell formats."""
    if cell is None:
        return None, None

    # Some datasets store binary as single-item list.
    if isinstance(cell, list) and len(cell) == 1:
        cell = cell[0]

    if isinstance(cell, (bytes, bytearray, memoryview)):
        return bytes(cell), None

    if isinstance(cell, str):
        return None, cell

    if isinstance(cell, Mapping):
        b = cell.get("bytes")
        p = cell.get("path")
        out_bytes: Optional[bytes] = None
        if isinstance(b, (bytes, bytearray, memoryview)):
            out_bytes = bytes(b)
        elif isinstance(b, list) and len(b) == 1 and isinstance(b[0], (bytes, bytearray, memoryview)):
            out_bytes = bytes(b[0])
        out_path = str(p) if p is not None else None
        return out_bytes, out_path

    # Optional PIL fallback without hard dependency.
    save = getattr(cell, "save", None)
    if callable(save):
        try:
            buf = io.BytesIO()
            cell.save(buf, format="JPEG")
            return buf.getvalue(), getattr(cell, "filename", None)
        except Exception:
            return None, getattr(cell, "filename", None)

    return None, None


def _safe_stem(uid: str) -> str:
    # UID is usually UUID-like, but keep path-safe anyway.
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in uid)


def _path_for_uid(base_dir: Path, uid: str, use_subdirs: bool, ext: str) -> Path:
    stem = _safe_stem(uid)
    if use_subdirs:
        shard = stem[:2] if len(stem) >= 2 else "xx"
        return base_dir / shard / f"{stem}.{ext}"
    return base_dir / f"{stem}.{ext}"


def _write_bytes(path: Path, payload: bytes, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    with path.open("wb") as f:
        f.write(payload)


def _update_limited_list(values: List[str], value: str, max_items: int) -> None:
    if max_items <= 0:
        return
    if value in values:
        return
    if len(values) < max_items:
        values.append(value)


def _iter_splits(ds_obj: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(ds_obj, (DatasetDict, IterableDatasetDict)):
        for split_name, split_ds in ds_obj.items():
            yield str(split_name), split_ds
        return

    # Single split dataset case
    if isinstance(ds_obj, (Dataset, IterableDataset)):
        split_name = getattr(ds_obj, "split", None)
        yield str(split_name) if split_name is not None else "unspecified", ds_obj
        return

    raise ValueError(f"Unsupported dataset object type: {type(ds_obj)}")


def _as_row_iterator(ds_split: Any) -> Iterable[Tuple[int, Mapping[str, Any]]]:
    # Works for Dataset and IterableDataset.
    for idx, row in enumerate(ds_split):
        if not isinstance(row, Mapping):
            continue
        yield idx, row


def build_manifest(
    dataset_name: str,
    dataset_config_name: Optional[str],
    split: Optional[str],
    output_uid_to_path: Path,
    output_uid_to_meta: Path,
    image_save_dir: Optional[Path],
    use_subdirs: bool,
    overwrite_images: bool,
    include_hf_locator_as_path: bool,
    caption_sample_size: int,
    partner_sample_size: int,
    include_fields: Sequence[str],
    extension: str,
) -> Dict[str, Any]:
    """Scan HF dataset and write UID manifests."""

    ds_obj = load_dataset(dataset_name, dataset_config_name, split=split)

    uid_to_path: Dict[str, str] = {}
    uid_to_meta: Dict[str, Dict[str, Any]] = {}

    rows_scanned = 0
    image_cells_processed = 0
    images_written = 0
    images_from_source_path = 0

    image_cols = [("image_0_uid", "jpg_0", "image_1_uid"), ("image_1_uid", "jpg_1", "image_0_uid")]

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kwargs: x

    for split_name, split_ds in _iter_splits(ds_obj):
        row_iterator = tqdm(
            _as_row_iterator(split_ds),
            desc=f"Scanning {split_name} split",
            total=len(split_ds) if hasattr(split_ds, "__len__") else None
        )
        for row_idx, row in row_iterator:
            rows_scanned += 1

            caption = row.get("caption")
            caption_str = str(caption) if caption is not None else ""
            best_uid = _normalize_uid(row.get("best_image_uid")) if row.get("best_image_uid") is not None else None

            for uid_key, image_key, partner_uid_key in image_cols:
                uid_raw = row.get(uid_key)
                if uid_raw is None:
                    continue
                uid = _normalize_uid(uid_raw)

                partner_uid_raw = row.get(partner_uid_key)
                partner_uid = _normalize_uid(partner_uid_raw) if partner_uid_raw is not None else None

                image_cell = row.get(image_key)
                payload, source_path = _extract_bytes_and_path(image_cell)
                image_cells_processed += 1

                if uid not in uid_to_meta:
                    uid_to_meta[uid] = {
                        "uid": uid,
                        "num_occurrences": 0,
                        "split_counts": defaultdict(int),
                        "best_count": 0,
                        "non_best_count": 0,
                        "first_seen": {
                            "split": split_name,
                            "row_idx": row_idx,
                            "image_column": image_key,
                        },
                        "last_seen": {
                            "split": split_name,
                            "row_idx": row_idx,
                            "image_column": image_key,
                        },
                        "caption_samples": [],
                        "partner_uid_samples": [],
                        "source_paths": [],
                    }

                meta = uid_to_meta[uid]
                meta["num_occurrences"] += 1
                meta["split_counts"][split_name] += 1
                meta["last_seen"] = {
                    "split": split_name,
                    "row_idx": row_idx,
                    "image_column": image_key,
                }
                if best_uid is not None and uid == best_uid:
                    meta["best_count"] += 1
                else:
                    meta["non_best_count"] += 1

                if caption_str:
                    _update_limited_list(meta["caption_samples"], caption_str, caption_sample_size)
                if partner_uid:
                    _update_limited_list(meta["partner_uid_samples"], partner_uid, partner_sample_size)
                if source_path:
                    _update_limited_list(meta["source_paths"], source_path, 4)

                for field in include_fields:
                    if field in row:
                        key = f"field__{field}"
                        if key not in meta:
                            meta[key] = _to_jsonable(row[field])

                resolved_path: Optional[str] = None
                if image_save_dir is not None and payload is not None:
                    out_path = _path_for_uid(image_save_dir, uid, use_subdirs=use_subdirs, ext=extension)
                    existed_before = out_path.exists()
                    _write_bytes(out_path, payload, overwrite=overwrite_images)
                    resolved_path = str(out_path.resolve())
                    if overwrite_images or not existed_before:
                        images_written += 1
                elif source_path:
                    p = Path(source_path)
                    if p.exists():
                        resolved_path = str(p.resolve())
                        images_from_source_path += 1

                if resolved_path is not None:
                    uid_to_path.setdefault(uid, resolved_path)
                elif include_hf_locator_as_path:
                    uid_to_path.setdefault(uid, f"hf://{dataset_name}/{split_name}/{row_idx}/{image_key}")

    # Convert defaultdicts before writing JSON.
    uid_to_meta_json = {}
    for uid, meta in uid_to_meta.items():
        converted = dict(meta)
        converted["split_counts"] = dict(converted["split_counts"])
        uid_to_meta_json[uid] = converted

    output_uid_to_path.parent.mkdir(parents=True, exist_ok=True)
    output_uid_to_meta.parent.mkdir(parents=True, exist_ok=True)

    with output_uid_to_path.open("w", encoding="utf-8") as f:
        json.dump(uid_to_path, f, ensure_ascii=False)

    with output_uid_to_meta.open("w", encoding="utf-8") as f:
        json.dump(uid_to_meta_json, f, ensure_ascii=False)

    return {
        "dataset_name": dataset_name,
        "dataset_config_name": dataset_config_name,
        "split": split,
        "rows_scanned": rows_scanned,
        "uids_total": len(uid_to_meta_json),
        "uid_to_path_entries": len(uid_to_path),
        "image_cells_processed": image_cells_processed,
        "images_written": images_written,
        "images_from_source_path": images_from_source_path,
        "output_uid_to_path": str(output_uid_to_path.resolve()),
        "output_uid_to_meta": str(output_uid_to_meta.resolve()),
        "image_save_dir": str(image_save_dir.resolve()) if image_save_dir else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build uid_to_path / uid_to_meta JSON from HF dataset")

    parser.add_argument("--dataset-name", type=str, default="liuhuohuo2/pick-a-pic-v2")
    parser.add_argument("--dataset-config-name", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="HF split name (e.g. train). If omitted, scans all available splits.",
    )

    parser.add_argument(
        "--output-uid-to-path",
        type=Path,
        default=Path("data/uid_to_path.json"),
    )
    parser.add_argument(
        "--output-uid-to-meta",
        type=Path,
        default=Path("data/uid_to_meta.json"),
    )

    parser.add_argument(
        "--image-save-dir",
        type=Path,
        default=None,
        help=(
            "Optional local directory to write images by UID. "
            "If omitted, script will try to use source paths when available."
        ),
    )
    parser.add_argument("--extension", type=str, default="jpg", help="Output image extension")
    parser.add_argument("--no-subdirs", action="store_true", help="Do not shard saved images into subdirs")
    parser.add_argument("--overwrite-images", action="store_true", help="Overwrite existing image files")
    parser.add_argument(
        "--include-hf-locator-as-path",
        action="store_true",
        help="If local file path is unavailable, store hf://dataset/split/row/column pseudo-path in uid_to_path",
    )

    parser.add_argument(
        "--caption-sample-size",
        type=int,
        default=3,
        help="Max distinct caption samples to keep per UID in uid_to_meta",
    )
    parser.add_argument(
        "--partner-sample-size",
        type=int,
        default=6,
        help="Max distinct partner UID samples to keep per UID in uid_to_meta",
    )
    parser.add_argument(
        "--include-fields",
        type=str,
        nargs="*",
        default=[],
        help="Additional row fields to include in uid_to_meta as field__<name>",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = build_manifest(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        split=args.split,
        output_uid_to_path=args.output_uid_to_path,
        output_uid_to_meta=args.output_uid_to_meta,
        image_save_dir=args.image_save_dir,
        use_subdirs=not args.no_subdirs,
        overwrite_images=args.overwrite_images,
        include_hf_locator_as_path=args.include_hf_locator_as_path,
        caption_sample_size=args.caption_sample_size,
        partner_sample_size=args.partner_sample_size,
        include_fields=args.include_fields,
        extension=args.extension,
    )

    print("[build_uid_manifest_from_hf summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
