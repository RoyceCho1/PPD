from __future__ import annotations

"""Download all images for one HF preference split into a local UID image directory.

This is a dedicated image-download utility derived from the same assumptions as
`build_uid_manifest_from_hf.py`, but focused only on materializing image files
to a chosen local disk.

Typical use case:
- keep train images on a larger secondary filesystem
- optionally write a matching uid_to_path JSON for later latent generation
"""

import os

os.environ.setdefault("HF_HOME", "/Data_Storage/roycecho/PPD/hf_cache")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import argparse
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset


def _normalize_uid(value: Any) -> str:
    uid = str(value).strip()
    if not uid:
        raise ValueError("Resolved UID is empty.")
    return uid


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
    if cell is None:
        return None, None

    if isinstance(cell, list) and len(cell) == 1:
        cell = cell[0]

    if isinstance(cell, (bytes, bytearray, memoryview)):
        return bytes(cell), None

    if isinstance(cell, str):
        return None, cell

    if isinstance(cell, Mapping):
        raw_bytes = cell.get("bytes")
        raw_path = cell.get("path")
        payload: Optional[bytes] = None
        if isinstance(raw_bytes, (bytes, bytearray, memoryview)):
            payload = bytes(raw_bytes)
        elif (
            isinstance(raw_bytes, list)
            and len(raw_bytes) == 1
            and isinstance(raw_bytes[0], (bytes, bytearray, memoryview))
        ):
            payload = bytes(raw_bytes[0])
        return payload, str(raw_path) if raw_path is not None else None

    save = getattr(cell, "save", None)
    if callable(save):
        try:
            buffer = io.BytesIO()
            cell.save(buffer, format="JPEG")
            return buffer.getvalue(), getattr(cell, "filename", None)
        except Exception:
            return None, getattr(cell, "filename", None)

    return None, None


def _safe_stem(uid: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in uid)


def _path_for_uid(base_dir: Path, uid: str, use_subdirs: bool, ext: str) -> Path:
    stem = _safe_stem(uid)
    if use_subdirs:
        shard = stem[:2] if len(stem) >= 2 else "xx"
        return base_dir / shard / f"{stem}.{ext}"
    return base_dir / f"{stem}.{ext}"


def _write_bytes(path: Path, payload: bytes, overwrite: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return False
    with path.open("wb") as handle:
        handle.write(payload)
    return True


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


def download_split_images(
    dataset_name: str,
    dataset_config_name: Optional[str],
    split: str,
    output_image_dir: Path,
    output_uid_to_path: Optional[Path],
    use_subdirs: bool,
    overwrite_images: bool,
    extension: str,
    progress_every_rows: int,
    progress_every_images: int,
    verbose: bool,
) -> Dict[str, Any]:
    ds_obj = load_dataset(dataset_name, dataset_config_name, split=split)

    output_dir = output_image_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    uid_to_path: Dict[str, str] = {}
    rows_scanned = 0
    image_cells_seen = 0
    unique_uids_seen = 0
    images_written = 0
    images_linked_from_source_path = 0
    missing_payload_count = 0
    linked_uids = 0

    image_cols: Sequence[Tuple[str, str]] = (
        ("image_0_uid", "jpg_0"),
        ("image_1_uid", "jpg_1"),
    )

    def maybe_print_progress() -> None:
        if progress_every_rows > 0 and rows_scanned > 0 and rows_scanned % progress_every_rows == 0:
            print(
                "[progress] "
                f"rows_scanned={rows_scanned} "
                f"image_cells_seen={image_cells_seen} "
                f"unique_uids_seen={unique_uids_seen} "
                f"uid_to_path_entries={len(uid_to_path)} "
                f"images_written={images_written} "
                f"images_linked_from_source_path={images_linked_from_source_path} "
                f"missing_payload_count={missing_payload_count}"
            )

        total_resolved_images = images_written + linked_uids
        if progress_every_images > 0 and total_resolved_images > 0 and total_resolved_images % progress_every_images == 0:
            print(
                "[progress] "
                f"resolved_images={total_resolved_images} "
                f"images_written={images_written} "
                f"linked_uids={linked_uids} "
                f"uid_to_path_entries={len(uid_to_path)}"
            )

    for split_name, split_ds in _iter_splits(ds_obj):
        for row_idx, row in _as_row_iterator(split_ds):
            rows_scanned += 1

            for uid_key, image_key in image_cols:
                uid_raw = row.get(uid_key)
                if uid_raw is None:
                    continue

                uid = _normalize_uid(uid_raw)
                image_cells_seen += 1

                payload, source_path = _extract_bytes_and_path(row.get(image_key))
                resolved_path: Optional[str] = None

                if payload is not None:
                    output_path = _path_for_uid(output_dir, uid, use_subdirs=use_subdirs, ext=extension)
                    did_write = _write_bytes(output_path, payload, overwrite=overwrite_images)
                    resolved_path = str(output_path.resolve())
                    if did_write:
                        images_written += 1
                        if verbose:
                            print(
                                "[write] "
                                f"split={split_name} row_idx={row_idx} uid={uid} path={resolved_path}"
                            )
                    elif verbose:
                        print(
                            "[exists] "
                            f"split={split_name} row_idx={row_idx} uid={uid} path={resolved_path}"
                        )
                elif source_path:
                    candidate = Path(source_path).expanduser()
                    if candidate.exists():
                        resolved_path = str(candidate.resolve())
                        images_linked_from_source_path += 1
                        linked_uids += 1
                        if verbose:
                            print(
                                "[link] "
                                f"split={split_name} row_idx={row_idx} uid={uid} path={resolved_path}"
                            )
                    else:
                        missing_payload_count += 1
                        if verbose:
                            print(
                                "[missing] "
                                f"split={split_name} row_idx={row_idx} uid={uid} "
                                f"source_path={source_path}"
                            )
                else:
                    missing_payload_count += 1
                    if verbose:
                        print(
                            "[missing] "
                            f"split={split_name} row_idx={row_idx} uid={uid} "
                            "reason=no_payload_or_source_path"
                        )

                if uid not in uid_to_path:
                    unique_uids_seen += 1
                if resolved_path is not None:
                    uid_to_path.setdefault(uid, resolved_path)
                maybe_print_progress()

    if output_uid_to_path is not None:
        output_uid_to_path = output_uid_to_path.expanduser().resolve()
        output_uid_to_path.parent.mkdir(parents=True, exist_ok=True)
        with output_uid_to_path.open("w", encoding="utf-8") as handle:
            json.dump(uid_to_path, handle, ensure_ascii=False)

    return {
        "dataset_name": dataset_name,
        "dataset_config_name": dataset_config_name,
        "split": split,
        "output_image_dir": str(output_dir),
        "output_uid_to_path": str(output_uid_to_path.expanduser().resolve()) if output_uid_to_path else None,
        "rows_scanned": rows_scanned,
        "image_cells_seen": image_cells_seen,
        "unique_uids_seen": unique_uids_seen,
        "uid_to_path_entries": len(uid_to_path),
        "images_written": images_written,
        "images_linked_from_source_path": images_linked_from_source_path,
        "missing_payload_count": missing_payload_count,
        "overwrite_images": overwrite_images,
        "use_subdirs": use_subdirs,
        "extension": extension,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all images for one HF split into a local UID image directory."
    )
    parser.add_argument("--dataset-name", type=str, default="liuhuohuo2/pick-a-pic-v2")
    parser.add_argument("--dataset-config-name", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="HF split name to materialize locally, for example train/validation/test.",
    )
    parser.add_argument(
        "--output-image-dir",
        type=Path,
        required=True,
        help="Directory to store UID-named image files.",
    )
    parser.add_argument(
        "--output-uid-to-path",
        type=Path,
        default=None,
        help="Optional path to write uid_to_path.json for the downloaded split.",
    )
    parser.add_argument("--extension", type=str, default="jpg", help="Output image extension.")
    parser.add_argument("--no-subdirs", action="store_true", help="Do not shard images into subdirectories.")
    parser.add_argument("--overwrite-images", action="store_true", help="Overwrite existing image files.")
    parser.add_argument(
        "--progress-every-rows",
        type=int,
        default=1000,
        help="Print a summary progress line every N scanned rows. Use 0 to disable.",
    )
    parser.add_argument(
        "--progress-every-images",
        type=int,
        default=1000,
        help="Print a summary progress line every N resolved images. Use 0 to disable.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one log line for each written/existing/linked/missing image.",
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

    summary = download_split_images(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        split=args.split,
        output_image_dir=args.output_image_dir,
        output_uid_to_path=args.output_uid_to_path,
        use_subdirs=not args.no_subdirs,
        overwrite_images=args.overwrite_images,
        extension=args.extension,
        progress_every_rows=args.progress_every_rows,
        progress_every_images=args.progress_every_images,
        verbose=args.verbose,
    )

    if args.summary_json is not None:
        summary_path = args.summary_json.expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(summary), handle, indent=2, ensure_ascii=False)

    print("[download_split_images_from_hf summary]")
    print(json.dumps(_to_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
