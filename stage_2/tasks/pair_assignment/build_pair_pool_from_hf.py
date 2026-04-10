from __future__ import annotations

"""Build a local Stage 2 pair-pool JSONL from a Hugging Face preference dataset.

This script is an auxiliary data-preparation tool for the Stage 2 prototype.
It scans raw preference rows from a Hugging Face dataset and writes one local
pair-pool JSONL record per valid preference row.

The output format is designed to be consumed by
`stage_2/tasks/pair_assignment/build_stage2_pair_assignments.py` and contains at least:
- `user_id`
- `preferred_uid`
- `dispreferred_uid`
- `caption`

The script does not create latents, does not build latent manifests, and does
not perform training.
"""

import os

os.environ["HF_HOME"] = "/var/tmp/roycecho_hf"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset


def _normalize_key(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Resolved key is empty.")
    return text


def _to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Mapping):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return _to_jsonable(obj.item())
        except Exception:
            pass
    return str(obj)


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null", "nan"}:
            return True
    return False


def _coerce_bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", "none", "null", ""}:
            return False
    return bool(value)


def _iter_splits(ds_obj: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(ds_obj, (DatasetDict, IterableDatasetDict)):
        for split_name, split_ds in ds_obj.items():
            yield str(split_name), split_ds
        return

    if isinstance(ds_obj, (Dataset, IterableDataset)):
        split_name = getattr(ds_obj, "split", None)
        yield str(split_name) if split_name is not None else "unspecified", ds_obj
        return

    raise ValueError(f"Unsupported dataset object type: {type(ds_obj)}")


def _as_row_iterator(ds_split: Any) -> Iterable[Tuple[int, Mapping[str, Any]]]:
    for idx, row in enumerate(ds_split):
        if not isinstance(row, Mapping):
            continue
        yield idx, row


def _resolve_required_field(row: Mapping[str, Any], field_name: str, *, row_idx: int, split_name: str) -> Any:
    value = row.get(field_name)
    if _is_missing_value(value):
        raise ValueError(
            f"Missing required field `{field_name}` at split={split_name}, row_idx={row_idx}"
        )
    return value


def _resolve_caption(row: Mapping[str, Any], field_name: str, *, row_idx: int, split_name: str) -> str:
    value = _resolve_required_field(row, field_name, row_idx=row_idx, split_name=split_name)
    return str(value)


def _resolve_preference_pair(
    row: Mapping[str, Any],
    image0_uid_field: str,
    image1_uid_field: str,
    best_image_uid_field: str,
    *,
    row_idx: int,
    split_name: str,
) -> Tuple[str, str, str]:
    image0_uid = _normalize_key(
        _resolve_required_field(row, image0_uid_field, row_idx=row_idx, split_name=split_name)
    )
    image1_uid = _normalize_key(
        _resolve_required_field(row, image1_uid_field, row_idx=row_idx, split_name=split_name)
    )
    best_uid = _normalize_key(
        _resolve_required_field(row, best_image_uid_field, row_idx=row_idx, split_name=split_name)
    )

    if best_uid == image0_uid:
        return image0_uid, image1_uid, best_uid
    if best_uid == image1_uid:
        return image1_uid, image0_uid, best_uid

    raise ValueError(
        "Could not resolve preference direction because best UID does not match either image UID: "
        f"split={split_name}, row_idx={row_idx}, best_uid={best_uid}, "
        f"{image0_uid_field}={image0_uid}, {image1_uid_field}={image1_uid}"
    )


def _build_pair_record(
    row: Mapping[str, Any],
    *,
    row_idx: int,
    split_name: str,
    user_id_field: str,
    caption_field: str,
    image0_uid_field: str,
    image1_uid_field: str,
    best_image_uid_field: str,
    include_fields: Sequence[str],
) -> Dict[str, Any]:
    user_id = _normalize_key(
        _resolve_required_field(row, user_id_field, row_idx=row_idx, split_name=split_name)
    )
    caption = _resolve_caption(row, caption_field, row_idx=row_idx, split_name=split_name)
    preferred_uid, dispreferred_uid, best_uid = _resolve_preference_pair(
        row,
        image0_uid_field=image0_uid_field,
        image1_uid_field=image1_uid_field,
        best_image_uid_field=best_image_uid_field,
        row_idx=row_idx,
        split_name=split_name,
    )

    pair_record: Dict[str, Any] = {
        "user_id": user_id,
        "preferred_uid": preferred_uid,
        "dispreferred_uid": dispreferred_uid,
        "caption": caption,
        "pair_idx": row_idx,
        "pair_key": f"{split_name}:{row_idx}",
        "source_row_idx": row_idx,
        "source_split": split_name,
        "best_image_uid": best_uid,
        "image_0_uid": _normalize_key(row[image0_uid_field]),
        "image_1_uid": _normalize_key(row[image1_uid_field]),
    }

    for field in include_fields:
        if field in row:
            pair_record[f"field__{field}"] = _to_jsonable(row[field])

    return pair_record


def _row_passes_filters(
    row: Mapping[str, Any],
    *,
    row_idx: int,
    split_name: str,
    require_has_label: bool,
    has_label_field: str,
    require_are_different: bool,
    are_different_field: str,
) -> Tuple[bool, Optional[str]]:
    if require_has_label and has_label_field in row:
        if not _coerce_bool_like(row.get(has_label_field)):
            return False, (
                f"Filtered row because `{has_label_field}` is falsey at "
                f"split={split_name}, row_idx={row_idx}"
            )

    if require_are_different and are_different_field in row:
        if not _coerce_bool_like(row.get(are_different_field)):
            return False, (
                f"Filtered row because `{are_different_field}` is falsey at "
                f"split={split_name}, row_idx={row_idx}"
            )

    return True, None


def build_pair_pool(
    dataset_name: str,
    dataset_config_name: Optional[str],
    split: Optional[str],
    output_jsonl: Path,
    user_id_field: str,
    caption_field: str,
    image0_uid_field: str,
    image1_uid_field: str,
    best_image_uid_field: str,
    include_fields: Sequence[str],
    require_has_label: bool,
    has_label_field: str,
    require_are_different: bool,
    are_different_field: str,
    strict: bool,
) -> Dict[str, Any]:
    ds_obj = load_dataset(dataset_name, dataset_config_name, split=split)

    output_path = output_jsonl.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_scanned = 0
    rows_written = 0
    users_written: set[str] = set()
    split_counts: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as handle:
        for split_name, split_ds in _iter_splits(ds_obj):
            row_iterator = _as_row_iterator(split_ds)
            for row_idx, row in row_iterator:
                rows_scanned += 1
                keep_row, skip_reason = _row_passes_filters(
                    row,
                    row_idx=row_idx,
                    split_name=split_name,
                    require_has_label=require_has_label,
                    has_label_field=has_label_field,
                    require_are_different=require_are_different,
                    are_different_field=are_different_field,
                )
                if not keep_row:
                    skip_reason_counts[str(skip_reason)] += 1
                    continue
                try:
                    pair_record = _build_pair_record(
                        row,
                        row_idx=row_idx,
                        split_name=split_name,
                        user_id_field=user_id_field,
                        caption_field=caption_field,
                        image0_uid_field=image0_uid_field,
                        image1_uid_field=image1_uid_field,
                        best_image_uid_field=best_image_uid_field,
                        include_fields=include_fields,
                    )
                except ValueError as exc:
                    skip_reason_counts[str(exc)] += 1
                    if strict:
                        raise
                    continue

                handle.write(json.dumps(_to_jsonable(pair_record), ensure_ascii=False) + "\n")
                rows_written += 1
                users_written.add(pair_record["user_id"])
                split_counts[split_name] += 1

    return {
        "dataset_name": dataset_name,
        "dataset_config_name": dataset_config_name,
        "split": split,
        "output_jsonl": str(output_path),
        "rows_scanned": rows_scanned,
        "rows_written": rows_written,
        "unique_users_written": len(users_written),
        "split_counts_written": dict(split_counts),
        "user_id_field": user_id_field,
        "caption_field": caption_field,
        "image0_uid_field": image0_uid_field,
        "image1_uid_field": image1_uid_field,
        "best_image_uid_field": best_image_uid_field,
        "require_has_label": require_has_label,
        "has_label_field": has_label_field,
        "require_are_different": require_are_different,
        "are_different_field": are_different_field,
        "strict": strict,
        "skip_reason_counts": dict(skip_reason_counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local pair_pool.jsonl from a Hugging Face preference dataset."
    )
    parser.add_argument("--dataset-name", type=str, default="liuhuohuo2/pick-a-pic-v2")
    parser.add_argument("--dataset-config-name", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="HF split name (e.g. train). If omitted, scans all available splits.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("stage_2/artifacts/pair_pool/pair_pool.jsonl"),
        help="Output JSONL path for user-level preference pairs.",
    )
    parser.add_argument(
        "--user-id-field",
        type=str,
        default="user_id",
        help="Field name in the HF dataset that identifies the user.",
    )
    parser.add_argument(
        "--caption-field",
        type=str,
        default="caption",
        help="Field name containing the prompt/caption.",
    )
    parser.add_argument(
        "--image0-uid-field",
        type=str,
        default="image_0_uid",
        help="Field name for the first image UID.",
    )
    parser.add_argument(
        "--image1-uid-field",
        type=str,
        default="image_1_uid",
        help="Field name for the second image UID.",
    )
    parser.add_argument(
        "--best-image-uid-field",
        type=str,
        default="best_image_uid",
        help="Field name for the preferred/winning image UID.",
    )
    parser.add_argument(
        "--include-fields",
        type=str,
        nargs="*",
        default=[],
        help="Additional HF row fields to copy into output as field__<name>.",
    )
    parser.add_argument(
        "--has-label-field",
        type=str,
        default="has_label",
        help="Field name used to filter rows that do not have a valid preference label.",
    )
    parser.add_argument(
        "--are-different-field",
        type=str,
        default="are_different",
        help="Field name used to filter rows where the two images are not different.",
    )
    parser.add_argument(
        "--no-require-has-label",
        action="store_true",
        help="Do not filter out rows with falsey `has_label`.",
    )
    parser.add_argument(
        "--no-require-are-different",
        action="store_true",
        help="Do not filter out rows with falsey `are_different`.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise immediately on malformed rows instead of skipping them.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to save the final summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = build_pair_pool(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        split=args.split,
        output_jsonl=args.output_jsonl,
        user_id_field=args.user_id_field,
        caption_field=args.caption_field,
        image0_uid_field=args.image0_uid_field,
        image1_uid_field=args.image1_uid_field,
        best_image_uid_field=args.best_image_uid_field,
        include_fields=args.include_fields,
        require_has_label=not args.no_require_has_label,
        has_label_field=args.has_label_field,
        require_are_different=not args.no_require_are_different,
        are_different_field=args.are_different_field,
        strict=args.strict,
    )

    if args.summary_json is not None:
        summary_path = args.summary_json.expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("[build_pair_pool_from_hf summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
