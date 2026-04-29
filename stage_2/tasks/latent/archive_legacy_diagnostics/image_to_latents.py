from __future__ import annotations

"""Prepare candidate Stage C clean latents from raw images.

This script is analysis/prototyping only. It does not implement training,
optimizer setup, DPO loss, or a noisy-sample training loop.

Current working hypothesis:
    raw image
    -> EfficientNet-style transform
    -> EfficientNetEncoder
    -> image_embeds
    -> optional scaling
    -> saveable latent tensor

The hypothesis is based on the local Wuerstchen prior training example:
`diffusers/examples/research_projects/wuerstchen/text_to_image/train_text_to_image_prior.py`
and its paired encoder definition:
`modeling_efficient_net_encoder.py`.
"""

import argparse
import io
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from torch import Tensor
from torchvision import transforms

os.environ.setdefault("HF_HOME", "/Data_Storage/roycecho/PPD/hf_cache")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


DEFAULT_ENCODER_SEARCH_ROOT = (
    "/data/roycecho/diffusers/examples/research_projects/wuerstchen/text_to_image"
)
DEFAULT_OUTPUT_DIR = "stage_2/artifacts/stage_c_latents"
DEFAULT_TARGET_IMAGE_SIZE = "512x512"
DEFAULT_PRIOR_RESOLUTION_MULTIPLE = 42.67
DEFAULT_CHECKPOINT_FILENAME = "model_v2_stage_b.pt"
HF_LOCATOR_PREFIX = "hf://"
_HF_DATASET_CACHE: Dict[Tuple[str, Optional[str], str], Any] = {}


@dataclass
class ImageRecord:
    """One resolved image input."""

    image_path: str
    uid: Optional[str] = None


@dataclass
class TensorStats:
    """Basic tensor summary."""

    shape: List[int]
    dtype: str
    device: str
    min_value: float
    max_value: float
    mean_value: float
    std_value: float


@dataclass
class SaveMetadata:
    """Metadata saved next to one latent file."""

    uid: Optional[str]
    original_image_path: str
    original_image_size: Dict[str, int]
    latent_path: str
    latent_shape: List[int]
    dtype: str
    target_image_size: List[int]
    prior_resolution_multiple: float
    expected_latent_shape: List[int]
    actual_latent_shape: List[int]
    shape_match: bool
    preprocess_resolution_mode: str
    effnet_input_size: List[int]
    effnet_preprocess_resolution_mode: str
    effnet_preprocess_resolution: List[int]
    transform_info: Dict[str, Any]
    scaling_applied: bool
    scaling_expression: Optional[str]
    encoder_class: str
    checkpoint_path: str
    notes: List[str]


@dataclass
class RunningStatAccumulator:
    """Streaming accumulator for tensor range/mean/std."""

    shape: Optional[List[int]] = None
    dtype: Optional[str] = None
    device: Optional[str] = None
    total_count: int = 0
    sum_value: float = 0.0
    sum_sq_value: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None


def _print_header(title: str) -> None:
    """Render a report header."""

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _add_sys_path(path: Path) -> None:
    """Add a directory to sys.path once."""

    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _resolve_encoder_module_root(root_arg: Optional[str]) -> Path:
    """Resolve the local directory that contains modeling_efficient_net_encoder.py."""

    root = Path(root_arg or DEFAULT_ENCODER_SEARCH_ROOT).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Encoder search root does not exist: {root}\n"
            "Pass --encoder-search-root with the local Wuerstchen text_to_image directory."
        )

    encoder_file = root / "modeling_efficient_net_encoder.py"
    if not encoder_file.exists():
        raise FileNotFoundError(
            f"Could not find modeling_efficient_net_encoder.py under: {root}\n"
            "Expected a local diffusers checkout at examples/research_projects/wuerstchen/text_to_image."
        )
    return root


def _import_efficientnet_encoder(root: Path) -> Any:
    """Import EfficientNetEncoder from the local Wuerstchen example."""

    _add_sys_path(root)
    try:
        from modeling_efficient_net_encoder import EfficientNetEncoder  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Failed to import EfficientNetEncoder from local Wuerstchen example.\n"
            f"root={root}\n"
            f"traceback:\n{traceback.format_exc()}"
        ) from exc
    return EfficientNetEncoder


def _resolve_checkpoint_path(checkpoint_arg: Optional[str]) -> Path:
    """Resolve the checkpoint path for the EfficientNet encoder weights."""

    if checkpoint_arg:
        checkpoint_path = Path(checkpoint_arg).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
        return checkpoint_path

    raise FileNotFoundError(
        "Checkpoint path was not provided and no safe default local checkpoint was found.\n"
        "Pass --checkpoint-path pointing to a local `model_v2_stage_b.pt` file."
    )


def _load_json_file(path: Path, *, label: str) -> Any:
    """Load a JSON file with a readable error message."""

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} file does not exist: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse {label} JSON: {resolved} ({exc})") from exc


def _normalize_uid(value: Any) -> str:
    """Normalize an image UID into a non-empty string."""

    uid = str(value).strip()
    if not uid:
        raise ValueError("Resolved UID is empty.")
    return uid


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    """Deduplicate strings while preserving insertion order."""

    ordered: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _load_needed_uids_json(path: Path) -> List[str]:
    """Load needed UIDs from a JSON file."""

    data = _load_json_file(path, label="needed_uids")
    raw_uids: Any

    if isinstance(data, Mapping):
        if "uids" not in data:
            raise ValueError(
                f"needed_uids JSON object must contain a `uids` field: {path.expanduser().resolve()}"
            )
        raw_uids = data["uids"]
    else:
        raw_uids = data

    if not isinstance(raw_uids, list):
        raise ValueError(
            f"needed_uids JSON must provide a list of UIDs: {path.expanduser().resolve()}"
        )

    return _dedupe_preserve_order([_normalize_uid(uid) for uid in raw_uids])


def _load_needed_uids_txt(path: Path) -> List[str]:
    """Load needed UIDs from a plain text file with one UID per line."""

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"needed_uids text file does not exist: {resolved}")

    uids: List[str] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            uids.append(_normalize_uid(stripped))
    return _dedupe_preserve_order(uids)


def _load_uid_to_path_json(path: Path) -> Dict[str, str]:
    """Load uid -> image path mapping."""

    data = _load_json_file(path, label="uid_to_path")
    if not isinstance(data, Mapping):
        raise ValueError(f"uid_to_path JSON must be an object/dict: {path.expanduser().resolve()}")

    mapping: Dict[str, str] = {}
    for key, value in data.items():
        uid = _normalize_uid(key)
        if value is None:
            continue
        mapping[uid] = str(value)
    return mapping


def _is_hf_locator(value: str) -> bool:
    """Return True when a path string uses the hf:// locator scheme."""

    return value.startswith(HF_LOCATOR_PREFIX)


def _parse_hf_locator(locator: str) -> Tuple[str, str, int, str]:
    """Parse hf://dataset_name/split/row_idx/image_column locator."""

    if not _is_hf_locator(locator):
        raise ValueError(f"Not an hf locator: {locator}")

    raw = locator[len(HF_LOCATOR_PREFIX) :]
    try:
        dataset_name, split_name, row_idx_text, image_key = raw.rsplit("/", 3)
    except ValueError as exc:
        raise ValueError(
            "Invalid hf locator. Expected hf://dataset_name/split/row_idx/image_key, "
            f"got: {locator}"
        ) from exc

    try:
        row_idx = int(row_idx_text)
    except ValueError as exc:
        raise ValueError(f"Invalid row_idx in hf locator: {locator}") from exc

    if not dataset_name or not split_name or not image_key:
        raise ValueError(f"Invalid hf locator with empty field: {locator}")

    return dataset_name, split_name, row_idx, image_key


def _extract_bytes_and_path(cell: Any) -> Tuple[Optional[bytes], Optional[str]]:
    """Extract raw bytes and/or source path from common HF dataset cell formats."""

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


def _load_hf_dataset_split(
    dataset_name: str,
    dataset_config_name: Optional[str],
    split_name: str,
) -> Any:
    """Load and cache one HF dataset split."""

    cache_key = (dataset_name, dataset_config_name, split_name)
    if cache_key not in _HF_DATASET_CACHE:
        _HF_DATASET_CACHE[cache_key] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=split_name,
        )
    return _HF_DATASET_CACHE[cache_key]


def _load_pil_image_from_bytes(payload: bytes, locator: str) -> Image.Image:
    """Decode image bytes into a PIL RGB image."""

    try:
        return Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to decode image bytes from hf locator: {locator}\n{traceback.format_exc()}"
        ) from exc


def _write_uid_list_txt(path: Path, values: Sequence[str]) -> None:
    """Write one string per line."""

    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def _write_uid_list_json(path: Path, values: Sequence[str], *, field_name: str) -> None:
    """Write a compact JSON payload for UID lists."""

    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        field_name: list(values),
        f"num_{field_name}": len(values),
    }
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _load_needed_uids(
    needed_uids_json: Optional[str],
    needed_uids_txt: Optional[str],
) -> List[str]:
    """Load and merge needed UIDs from JSON and/or text files."""

    merged: List[str] = []
    if needed_uids_json:
        merged.extend(_load_needed_uids_json(Path(needed_uids_json)))
    if needed_uids_txt:
        merged.extend(_load_needed_uids_txt(Path(needed_uids_txt)))
    return _dedupe_preserve_order(merged)


def _load_encoder(
    encoder_cls: Any,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate EfficientNetEncoder and load local weights."""

    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint: {checkpoint_path}\n{traceback.format_exc()}"
        ) from exc

    if not isinstance(state_dict, Mapping):
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} is not a mapping-like object. "
            "Expected a dict containing `effnet_state_dict`."
        )
    if "effnet_state_dict" not in state_dict:
        available_keys = list(state_dict.keys())[:20]
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} does not contain `effnet_state_dict`.\n"
            f"Available top-level keys: {available_keys}"
        )

    try:
        encoder = encoder_cls()
        encoder.load_state_dict(state_dict["effnet_state_dict"])
        encoder.eval()
        encoder.requires_grad_(False)
        encoder.to(device=device)
    except Exception as exc:
        raise RuntimeError(
            "Failed to instantiate/load EfficientNetEncoder.\n"
            f"traceback:\n{traceback.format_exc()}"
        ) from exc

    return encoder


def _parse_hw(value: str, *, arg_name: str) -> Tuple[int, int]:
    """Parse `HxW`/`H,W`/single-int resolution text."""

    text = value.strip().lower().replace(" ", "")
    if not text:
        raise ValueError(f"{arg_name} cannot be empty.")

    if "x" in text:
        parts = text.split("x")
    elif "," in text:
        parts = text.split(",")
    else:
        parts = [text]

    try:
        if len(parts) == 1:
            size = int(parts[0])
            height, width = size, size
        elif len(parts) == 2:
            height, width = int(parts[0]), int(parts[1])
        else:
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            f"Invalid {arg_name} value `{value}`. Use `HxW` (for example `512x512`) or a single integer."
        ) from exc

    if height <= 0 or width <= 0:
        raise ValueError(f"{arg_name} must be positive, got `{value}`.")
    return height, width


def _compute_expected_latent_spatial(
    target_height: int,
    target_width: int,
    prior_resolution_multiple: float,
) -> Tuple[int, int]:
    """Compute Stage C expected latent spatial size from target image size."""

    if prior_resolution_multiple <= 0:
        raise ValueError(
            f"--prior-resolution-multiple must be positive, got: {prior_resolution_multiple}"
        )
    expected_h = int(math.ceil(target_height / prior_resolution_multiple))
    expected_w = int(math.ceil(target_width / prior_resolution_multiple))
    return expected_h, expected_w


def _resolve_effnet_preprocess_resolution(
    mode: str,
    fixed_resolution: Optional[str],
    expected_h: int,
    expected_w: int,
) -> Tuple[int, int]:
    """Resolve EffNet preprocess resolution from CLI policy."""

    if mode == "auto":
        return expected_h * 32, expected_w * 32

    if mode == "fixed":
        if not fixed_resolution:
            raise ValueError(
                "--effnet-preprocess-resolution is required when "
                "--effnet-preprocess-resolution-mode=fixed."
            )
        return _parse_hw(fixed_resolution, arg_name="--effnet-preprocess-resolution")

    raise ValueError(f"Unsupported --effnet-preprocess-resolution-mode: {mode}")


def _build_effnet_transform(resolution: Tuple[int, int]) -> transforms.Compose:
    """Match the local Wuerstchen prior training transform."""

    height, width = resolution
    return transforms.Compose(
        [
            transforms.Resize(
                (height, width),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _collect_image_records(
    image_path: Optional[str],
    image_dir: Optional[str],
    needed_uids_json: Optional[str],
    needed_uids_txt: Optional[str],
    uid_to_path_json: Optional[str],
    skip_missing_uids: bool,
    max_images: Optional[int],
    missing_uids_txt: Optional[str],
    missing_uids_json: Optional[str],
    output_dir: Path,
    skip_existing: bool,
) -> Tuple[List[ImageRecord], List[str], int]:
    """Resolve input image paths from CLI options."""

    records: List[ImageRecord] = []
    missing_uids: List[str] = []
    skipped_existing_count = 0

    if image_path:
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--image-path does not exist: {path}")
        records.append(ImageRecord(image_path=str(path)))

    if image_dir:
        root = Path(image_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"--image-dir does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"--image-dir is not a directory: {root}")

        image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in image_suffixes:
                records.append(ImageRecord(image_path=str(path)))

    needed_uids = _load_needed_uids(needed_uids_json, needed_uids_txt)
    if needed_uids:
        if not uid_to_path_json:
            raise ValueError(
                "UID list input requires --uid-to-path-json so image UIDs can be resolved into file paths."
            )
        uid_to_path = _load_uid_to_path_json(Path(uid_to_path_json))
        for uid in needed_uids:
            mapped_path = uid_to_path.get(uid)
            if mapped_path is None:
                if skip_missing_uids:
                    missing_uids.append(uid)
                    continue
                raise KeyError(f"UID was not found in uid_to_path mapping: {uid}")

            if _is_hf_locator(mapped_path):
                records.append(ImageRecord(image_path=mapped_path, uid=uid))
            else:
                path = Path(mapped_path).expanduser()
                if not path.exists():
                    if skip_missing_uids:
                        missing_uids.append(uid)
                        continue
                    raise FileNotFoundError(
                        f"Resolved image path for uid={uid} does not exist: {path.resolve()}"
                    )
                records.append(ImageRecord(image_path=str(path.resolve()), uid=uid))

    deduped: List[ImageRecord] = []
    seen: set[Tuple[Optional[str], str]] = set()
    for record in records:
        record_key = (record.uid, record.image_path)
        if record_key in seen:
            continue
        seen.add(record_key)
        deduped.append(record)

    if skip_existing:
        filtered: List[ImageRecord] = []
        for record in deduped:
            latent_path, meta_path = _latent_bundle_paths(output_dir, record)
            if latent_path.exists() and meta_path.exists():
                skipped_existing_count += 1
                continue
            filtered.append(record)
        deduped = filtered

    if not deduped and not skip_existing:
        raise ValueError("No input images were resolved. Pass --image-path or --image-dir.")

    if max_images is not None and max_images > 0:
        deduped = deduped[:max_images]

    missing_uids = _dedupe_preserve_order(missing_uids)
    if missing_uids_txt and missing_uids:
        _write_uid_list_txt(Path(missing_uids_txt), missing_uids)
    if missing_uids_json and missing_uids:
        _write_uid_list_json(Path(missing_uids_json), missing_uids, field_name="missing_uids")

    return deduped, missing_uids, skipped_existing_count


def _load_pil_image(path: Path) -> Image.Image:
    """Load an image as RGB."""

    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read image file: {path}\n{traceback.format_exc()}"
        ) from exc


def _load_pil_image_for_record(
    record: ImageRecord,
    hf_dataset_config_name: Optional[str],
) -> Image.Image:
    """Load one image from a local path or an hf:// locator."""

    if not _is_hf_locator(record.image_path):
        return _load_pil_image(Path(record.image_path))

    dataset_name, split_name, row_idx, image_key = _parse_hf_locator(record.image_path)
    dataset = _load_hf_dataset_split(
        dataset_name=dataset_name,
        dataset_config_name=hf_dataset_config_name,
        split_name=split_name,
    )

    try:
        row = dataset[row_idx]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load HF dataset row for locator={record.image_path}\n{traceback.format_exc()}"
        ) from exc

    if not isinstance(row, Mapping):
        raise RuntimeError(
            f"HF dataset row is not mapping-like for locator={record.image_path}: type={type(row)}"
        )

    if image_key not in row:
        raise KeyError(
            f"Image column `{image_key}` was not found in HF dataset row for locator={record.image_path}"
        )

    payload, source_path = _extract_bytes_and_path(row[image_key])
    if payload is not None:
        return _load_pil_image_from_bytes(payload, record.image_path)

    if source_path:
        source = Path(source_path).expanduser()
        if source.exists():
            return _load_pil_image(source)

    raise RuntimeError(
        f"Could not resolve image bytes or a readable local source path for locator={record.image_path}"
    )


def _tensor_stats(tensor: Tensor) -> TensorStats:
    """Compute compact stats for a tensor."""

    cpu_tensor = tensor.detach().float().cpu()
    return TensorStats(
        shape=[int(x) for x in cpu_tensor.shape],
        dtype=str(tensor.dtype).replace("torch.", ""),
        device=str(tensor.device),
        min_value=float(cpu_tensor.min().item()),
        max_value=float(cpu_tensor.max().item()),
        mean_value=float(cpu_tensor.mean().item()),
        std_value=float(cpu_tensor.std(unbiased=False).item()),
    )


def _update_running_stats(acc: RunningStatAccumulator, tensor: Tensor) -> None:
    """Accumulate tensor stats without storing all batches."""

    cpu_tensor = tensor.detach().float().cpu()
    if acc.shape is None:
        acc.shape = [int(x) for x in cpu_tensor.shape]
        acc.dtype = str(tensor.dtype).replace("torch.", "")
        acc.device = str(tensor.device)

    batch_min = float(cpu_tensor.min().item())
    batch_max = float(cpu_tensor.max().item())
    batch_sum = float(cpu_tensor.sum().item())
    batch_sum_sq = float(cpu_tensor.square().sum().item())
    batch_count = int(cpu_tensor.numel())

    acc.total_count += batch_count
    acc.sum_value += batch_sum
    acc.sum_sq_value += batch_sum_sq
    acc.min_value = batch_min if acc.min_value is None else min(acc.min_value, batch_min)
    acc.max_value = batch_max if acc.max_value is None else max(acc.max_value, batch_max)


def _finalize_running_stats(acc: RunningStatAccumulator) -> TensorStats:
    """Convert accumulated stats into a TensorStats object."""

    if acc.shape is None or acc.dtype is None or acc.device is None or acc.total_count <= 0:
        raise ValueError("Cannot finalize running stats before any tensors were accumulated.")

    mean_value = acc.sum_value / acc.total_count
    variance = max((acc.sum_sq_value / acc.total_count) - (mean_value * mean_value), 0.0)
    std_value = variance ** 0.5
    return TensorStats(
        shape=acc.shape,
        dtype=acc.dtype,
        device=acc.device,
        min_value=float(acc.min_value if acc.min_value is not None else 0.0),
        max_value=float(acc.max_value if acc.max_value is not None else 0.0),
        mean_value=float(mean_value),
        std_value=float(std_value),
    )


def _iter_record_batches(
    records: Sequence[ImageRecord],
    batch_size: int,
) -> Iterable[Tuple[int, int, Sequence[ImageRecord]]]:
    """Yield record batches with 1-based batch indices."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got: {batch_size}")

    for batch_idx, start in enumerate(range(0, len(records), batch_size), start=1):
        end = min(start + batch_size, len(records))
        yield batch_idx, start, records[start:end]


def _check_latent_shape(
    shape: Sequence[int],
    expected_h: int,
    expected_w: int,
    shape_policy: str,
) -> Tuple[bool, Optional[str]]:
    """Validate latent shape against Stage C expectation."""

    expected_desc = f"[B, 16, {expected_h}, {expected_w}]"
    actual_desc = list(shape)
    error: Optional[str] = None

    if len(shape) != 4:
        error = (
            "Expected a 4D latent batch tensor. "
            f"expected={expected_desc}, actual={actual_desc}"
        )
    elif int(shape[1]) != 16:
        error = (
            "Expected latent channel dimension 16. "
            f"expected={expected_desc}, actual={actual_desc}"
        )
    elif int(shape[2]) != expected_h or int(shape[3]) != expected_w:
        error = (
            "Latent spatial size mismatch. "
            f"expected={expected_desc}, actual={actual_desc}"
        )

    if error is None:
        return True, None

    if shape_policy == "strict":
        raise RuntimeError(error)
    return False, f"[WARN] {error}"


def _make_batch(
    records: Sequence[ImageRecord],
    transform: transforms.Compose,
    device: torch.device,
    hf_dataset_config_name: Optional[str],
) -> Tuple[Tensor, List[Dict[str, Any]]]:
    """Load and transform a small image batch."""

    tensors: List[Tensor] = []
    infos: List[Dict[str, Any]] = []
    for record in records:
        image = _load_pil_image_for_record(record, hf_dataset_config_name=hf_dataset_config_name)
        try:
            pixel_values = transform(image)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to apply EffNet transform to image: {record.image_path}\n{traceback.format_exc()}"
            ) from exc

        tensors.append(pixel_values)
        infos.append(
            {
                "image_path": record.image_path,
                "uid": record.uid,
                "original_width": int(image.width),
                "original_height": int(image.height),
            }
        )

    batch = torch.stack(tensors, dim=0).to(device=device)
    return batch, infos


def _apply_optional_scaling(latents: Tensor, apply_scaling: bool) -> Tuple[Tensor, Optional[str]]:
    """Apply the current candidate scaling rule from local Wuerstchen training code."""

    if not apply_scaling:
        return latents, None
    # Candidate rule from local train_text_to_image_prior.py:
    # image_embeds = image_embeds.add(1.0).div(42.0)
    return latents.add(1.0).div(42.0), "image_embeds = image_embeds.add(1.0).div(42.0)"


def _safe_name(value: str) -> str:
    """Convert arbitrary text into a path-safe filename stem."""

    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return safe or "image"


def _output_stem_for_record(record: ImageRecord) -> str:
    """Choose the output stem for one latent artifact."""

    if record.uid:
        return _safe_name(record.uid)
    return _safe_name(Path(record.image_path).stem)


def _latent_bundle_paths(output_dir: Path, record: ImageRecord) -> Tuple[Path, Path]:
    """Shard latent outputs by the first two characters of the output stem."""

    stem = _output_stem_for_record(record)
    shard = stem[:2] if len(stem) >= 2 else "xx"
    bundle_dir = output_dir / shard
    return bundle_dir / f"{stem}.pt", bundle_dir / f"{stem}.json"


def _save_latent_bundle(
    output_dir: Path,
    record: ImageRecord,
    latent: Tensor,
    metadata: SaveMetadata,
) -> Tuple[Path, Path]:
    """Save one latent tensor and its metadata."""

    latent_path, meta_path = _latent_bundle_paths(output_dir, record)
    latent_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(latent.detach().cpu(), latent_path)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(metadata), handle, indent=2, ensure_ascii=False)

    return latent_path, meta_path


def _sanity_reload(latent_path: Path) -> List[int]:
    """Reload a saved latent and return its shape."""

    try:
        tensor = torch.load(latent_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Saved latent sanity check failed while loading: {latent_path}\n{traceback.format_exc()}"
        ) from exc
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"Saved latent sanity check expected a torch.Tensor, got: {type(tensor)}"
        )
    return [int(x) for x in tensor.shape]


def _print_environment(
    args: argparse.Namespace,
    encoder_root: Path,
    checkpoint_path: Path,
    device: torch.device,
    num_missing_uids: int,
    num_resolved_images: int,
    num_skipped_existing: int,
    target_image_size: Tuple[int, int],
    expected_latent_spatial: Tuple[int, int],
    effnet_preprocess_resolution: Tuple[int, int],
) -> None:
    """Print runtime diagnostics."""

    _print_header("Environment")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"device: {device}")
    print(f"encoder_search_root: {encoder_root}")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"output_dir: {Path(args.output_dir).expanduser().resolve()}")
    print(f"apply_scaling: {args.apply_scaling}")
    print(f"target_image_size: {list(target_image_size)}")
    print(f"prior_resolution_multiple: {args.prior_resolution_multiple}")
    print(f"expected_latent_spatial: {list(expected_latent_spatial)}")
    print(f"shape_policy: {args.shape_policy}")
    print(f"preprocess_resolution_mode: {args.effnet_preprocess_resolution_mode}")
    print(f"effnet_input_size: {list(effnet_preprocess_resolution)}")
    print(f"effnet_preprocess_resolution_mode: {args.effnet_preprocess_resolution_mode}")
    print(f"effnet_preprocess_resolution: {list(effnet_preprocess_resolution)}")
    print(f"needed_uids_json: {Path(args.needed_uids_json).expanduser().resolve() if args.needed_uids_json else None}")
    print(f"needed_uids_txt: {Path(args.needed_uids_txt).expanduser().resolve() if args.needed_uids_txt else None}")
    print(f"uid_to_path_json: {Path(args.uid_to_path_json).expanduser().resolve() if args.uid_to_path_json else None}")
    print(f"hf_dataset_config_name: {args.hf_dataset_config_name}")
    print(f"skip_missing_uids: {args.skip_missing_uids}")
    print(f"num_missing_uids: {num_missing_uids}")
    print(f"num_resolved_images: {num_resolved_images}")
    print(f"skip_existing: {args.skip_existing}")
    print(f"num_skipped_existing: {num_skipped_existing}")
    print(f"batch_size: {args.batch_size}")
    print(f"progress_every_batches: {args.progress_every_batches}")
    print(f"summary_only: {args.summary_only}")
    print(f"verbose: {args.verbose}")


def _print_batch_report(
    batch_stats: TensorStats,
    raw_stats: TensorStats,
    scaled_stats: Optional[TensorStats],
    warning: Optional[str],
    apply_scaling: bool,
) -> None:
    """Print tensor-level report."""

    _print_header("Latent Report")
    print(f"EffNet input batch shape: {batch_stats.shape}")
    print(
        "Raw image_embeds stats: "
        f"shape={raw_stats.shape} dtype={raw_stats.dtype} device={raw_stats.device} "
        f"min={raw_stats.min_value:.6f} max={raw_stats.max_value:.6f} "
        f"mean={raw_stats.mean_value:.6f} std={raw_stats.std_value:.6f}"
    )
    if scaled_stats is not None:
        print(
            "Scaled image_embeds stats: "
            f"shape={scaled_stats.shape} dtype={scaled_stats.dtype} device={scaled_stats.device} "
            f"min={scaled_stats.min_value:.6f} max={scaled_stats.max_value:.6f} "
            f"mean={scaled_stats.mean_value:.6f} std={scaled_stats.std_value:.6f}"
        )
    else:
        print(f"Scaled image_embeds stats: skipped (apply_scaling={apply_scaling})")
    if warning:
        print(warning)


def _print_saved_files(
    saved_pairs_preview: Sequence[Tuple[Path, Path]],
    saved_count: int,
    summary_only: bool,
) -> None:
    """Print saved artifact paths."""

    _print_header("Saved Files")
    if saved_count <= 0:
        print("No latent files were saved.")
        return
    if summary_only:
        print(f"Saved {saved_count} latent bundle(s).")
        return
    print(f"Saved latent bundle count: {saved_count}")
    preview_count = len(saved_pairs_preview)
    if preview_count < saved_count:
        print(f"Previewing first {preview_count} saved bundle(s):")
    for latent_path, meta_path in saved_pairs_preview:
        print(f"- latent: {latent_path}")
        print(f"  meta:   {meta_path}")


def _print_summary(
    success: bool,
    latent_stats: TensorStats,
    scaling_applied: bool,
    plausibility_note: str,
    saved_count: int,
    num_images_processed: int,
    num_batches_processed: int,
    elapsed_seconds: float,
) -> None:
    """Print the required final summary."""

    _print_header("Final Summary")
    print(f"1. latent 생성 성공 여부: {success}")
    print(f"2. output shape: {latent_stats.shape}")
    print(
        "3. dtype / range: "
        f"dtype={latent_stats.dtype}, min={latent_stats.min_value:.6f}, "
        f"max={latent_stats.max_value:.6f}, mean={latent_stats.mean_value:.6f}, std={latent_stats.std_value:.6f}"
    )
    print(f"4. scaling 적용 여부: {scaling_applied}")
    print(f"5. Stage C clean target 후보 경로로서의 타당성: {plausibility_note}")
    print(f"Processed images: {num_images_processed}")
    print(f"Processed batches: {num_batches_processed}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    print("6. 다음 단계 추천")
    print("- `stage2_dataset.py` latent 확장: 이 저장 포맷을 기준으로 latent-backed variant를 추가.")
    print("- latent manifest 생성: 이미지 경로와 latent 경로를 연결하는 json/jsonl 작성.")
    print("- minimal noisy sample prototype: scheduler.add_noise(image_embeds, noise, timesteps) 재현.")
    print("- `train_stage2_dpo.py` minimal step: clean target/noisy sample 정의가 확정된 뒤 연결.")
    if saved_count > 0:
        print(f"Saved latent bundle count: {saved_count}")


def _write_summary_json(summary_path: Path, payload: Mapping[str, Any]) -> None:
    """Write a machine-readable generation summary."""

    resolved = summary_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""

    parser = argparse.ArgumentParser(
        description="Prepare candidate Stage C clean latents via local EfficientNetEncoder."
    )
    parser.add_argument("--image-path", type=str, default=None, help="Single image path.")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory of images.")
    parser.add_argument(
        "--needed-uids-json",
        type=str,
        default=None,
        help="JSON file containing a `uids` list or a plain UID list to resolve through uid_to_path.json.",
    )
    parser.add_argument(
        "--needed-uids-txt",
        type=str,
        default=None,
        help="Text file containing one image UID per line.",
    )
    parser.add_argument(
        "--uid-to-path-json",
        type=str,
        default=None,
        help="JSON mapping from image UID to local image path. Required for UID-list input mode.",
    )
    parser.add_argument(
        "--hf-dataset-config-name",
        type=str,
        default=None,
        help="Optional HF dataset config name used when resolving hf:// locators.",
    )
    parser.add_argument(
        "--skip-missing-uids",
        action="store_true",
        help="Skip UIDs that are missing from uid_to_path.json or point to missing files.",
    )
    parser.add_argument(
        "--missing-uids-txt",
        type=str,
        default=None,
        help="Optional text output path for skipped/missing UIDs.",
    )
    parser.add_argument(
        "--missing-uids-json",
        type=str,
        default=None,
        help="Optional JSON output path for skipped/missing UIDs.",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save .pt/.json files.")
    parser.add_argument(
        "--target-image-size",
        type=str,
        default=DEFAULT_TARGET_IMAGE_SIZE,
        help="Target image size used for expected Stage C latent shape. Format: HxW (default: 512x512).",
    )
    parser.add_argument(
        "--prior-resolution-multiple",
        type=float,
        default=DEFAULT_PRIOR_RESOLUTION_MULTIPLE,
        help="Stage C prior resolution multiple for expected shape, usually 42.67.",
    )
    parser.add_argument(
        "--shape-policy",
        type=str,
        choices=["strict", "warn"],
        default="strict",
        help="How to handle latent shape mismatch against expected Stage C shape.",
    )
    parser.add_argument(
        "--effnet-preprocess-resolution-mode",
        type=str,
        choices=["auto", "fixed"],
        default="auto",
        help=(
            "EffNet preprocessing resolution policy. "
            "`auto`: expected_h*32, expected_w*32. "
            "`fixed`: use --effnet-preprocess-resolution."
        ),
    )
    parser.add_argument(
        "--effnet-preprocess-resolution",
        type=str,
        default=None,
        help="When preprocess mode is fixed, set resolution as HxW or single integer.",
    )
    parser.add_argument(
        "--encoder-search-root",
        type=str,
        default=DEFAULT_ENCODER_SEARCH_ROOT,
        help="Local directory containing modeling_efficient_net_encoder.py.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Local path to model_v2_stage_b.pt or a compatible checkpoint containing effnet_state_dict.",
    )
    parser.add_argument(
        "--apply-scaling",
        action="store_true",
        help="Apply candidate scaling from local Wuerstchen training code: add(1.0).div(42.0).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process. If omitted, process all resolved images.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images to process per forward pass.",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=10,
        help="Print one progress line every N completed batches. Use 1 for every batch.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose latent .pt and metadata .json already exist in --output-dir.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, for example cpu or cuda.")
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional path to save a machine-readable latent generation summary JSON.",
    )
    parser.add_argument("--summary-only", action="store_true", help="Print a short report.")
    parser.add_argument("--verbose", action="store_true", help="Print extra per-image information.")
    return parser.parse_args()


def main() -> int:
    """Run the latent preparation prototype."""

    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    try:
        encoder_root = _resolve_encoder_module_root(args.encoder_search_root)
        EfficientNetEncoder = _import_efficientnet_encoder(encoder_root)
        checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
        target_height, target_width = _parse_hw(
            args.target_image_size,
            arg_name="--target-image-size",
        )
        expected_h, expected_w = _compute_expected_latent_spatial(
            target_height=target_height,
            target_width=target_width,
            prior_resolution_multiple=args.prior_resolution_multiple,
        )
        preprocess_h, preprocess_w = _resolve_effnet_preprocess_resolution(
            mode=args.effnet_preprocess_resolution_mode,
            fixed_resolution=args.effnet_preprocess_resolution,
            expected_h=expected_h,
            expected_w=expected_w,
        )
        image_records, missing_uids, skipped_existing_count = _collect_image_records(
            args.image_path,
            args.image_dir,
            args.needed_uids_json,
            args.needed_uids_txt,
            args.uid_to_path_json,
            args.skip_missing_uids,
            args.max_images,
            args.missing_uids_txt,
            args.missing_uids_json,
            output_dir,
            args.skip_existing,
        )
        transform = _build_effnet_transform((preprocess_h, preprocess_w))
        encoder = _load_encoder(EfficientNetEncoder, checkpoint_path, device)
    except Exception as exc:
        print("[ERROR] Initialization failed.")
        print(str(exc))
        return 1

    _print_environment(
        args,
        encoder_root,
        checkpoint_path,
        device,
        num_missing_uids=len(missing_uids),
        num_resolved_images=len(image_records),
        num_skipped_existing=skipped_existing_count,
        target_image_size=(target_height, target_width),
        expected_latent_spatial=(expected_h, expected_w),
        effnet_preprocess_resolution=(preprocess_h, preprocess_w),
    )

    if len(image_records) == 0:
        print("[INFO] No remaining images to process after applying --skip-existing.")
        return 0

    try:
        start_time = time.time()
        selected_acc = RunningStatAccumulator()
        saved_pairs_preview: List[Tuple[Path, Path]] = []
        saved_count = 0
        num_batches_processed = 0
        num_images_processed = 0
        did_print_batch_report = False
        transform_info = {
            "target_image_size": [target_height, target_width],
            "prior_resolution_multiple": args.prior_resolution_multiple,
            "expected_latent_spatial": [expected_h, expected_w],
            "preprocess_resolution_mode": args.effnet_preprocess_resolution_mode,
            "effnet_input_size": [preprocess_h, preprocess_w],
            "effnet_preprocess_resolution_mode": args.effnet_preprocess_resolution_mode,
            "effnet_preprocess_resolution": [preprocess_h, preprocess_w],
            "resize": "bilinear+antialias",
            "crop": "center_crop",
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "source": "local diffusers Wuerstchen train_text_to_image_prior.py",
            "status": "candidate path; additional confirmation still needed",
        }

        for batch_idx, batch_start, batch_records in _iter_record_batches(image_records, args.batch_size):
            batch, image_infos = _make_batch(
                batch_records,
                transform,
                device,
                hf_dataset_config_name=args.hf_dataset_config_name,
            )
            batch_stats = _tensor_stats(batch)

            with torch.no_grad():
                raw_latents = encoder(batch)
                scaled_latents, scaling_expression = _apply_optional_scaling(raw_latents, args.apply_scaling)

            raw_stats = _tensor_stats(raw_latents)
            scaled_stats = _tensor_stats(scaled_latents) if args.apply_scaling else None
            selected_latents = scaled_latents if args.apply_scaling else raw_latents
            selected_stats = scaled_stats or raw_stats
            shape_match, warning = _check_latent_shape(
                selected_stats.shape,
                expected_h=expected_h,
                expected_w=expected_w,
                shape_policy=args.shape_policy,
            )
            _update_running_stats(selected_acc, selected_latents)

            if not args.summary_only and not did_print_batch_report:
                _print_batch_report(
                    batch_stats=batch_stats,
                    raw_stats=raw_stats,
                    scaled_stats=scaled_stats,
                    warning=warning,
                    apply_scaling=args.apply_scaling,
                )
                did_print_batch_report = True

            for local_idx, record in enumerate(batch_records):
                image_path_str = record.image_path
                latent = selected_latents[local_idx : local_idx + 1]
                latent_stats = _tensor_stats(latent)
                notes = [
                    "Hypothesis: this tensor is a candidate Stage C clean target latent/image_embeds.",
                    "Additional confirmation is still needed for exact Stable Cascade scaling and scheduler target semantics.",
                ]
                if warning:
                    notes.append(warning)

                latent_path, _ = _latent_bundle_paths(output_dir, record)
                original_width = int(image_infos[local_idx]["original_width"])
                original_height = int(image_infos[local_idx]["original_height"])
                actual_shape = latent_stats.shape
                expected_shape = [1, 16, expected_h, expected_w]
                per_item_match = (
                    len(actual_shape) == 4
                    and actual_shape[1] == 16
                    and actual_shape[2] == expected_h
                    and actual_shape[3] == expected_w
                )
                metadata = SaveMetadata(
                    uid=record.uid,
                    original_image_path=image_path_str,
                    original_image_size={
                        "width": original_width,
                        "height": original_height,
                    },
                    latent_path=str(latent_path),
                    latent_shape=actual_shape,
                    dtype=latent_stats.dtype,
                    target_image_size=[target_height, target_width],
                    prior_resolution_multiple=float(args.prior_resolution_multiple),
                    expected_latent_shape=expected_shape,
                    actual_latent_shape=actual_shape,
                    shape_match=per_item_match and shape_match,
                    preprocess_resolution_mode=args.effnet_preprocess_resolution_mode,
                    effnet_input_size=[preprocess_h, preprocess_w],
                    effnet_preprocess_resolution_mode=args.effnet_preprocess_resolution_mode,
                    effnet_preprocess_resolution=[preprocess_h, preprocess_w],
                    transform_info=transform_info,
                    scaling_applied=args.apply_scaling,
                    scaling_expression=scaling_expression,
                    encoder_class=encoder.__class__.__name__,
                    checkpoint_path=str(checkpoint_path),
                    notes=notes,
                )
                saved_pair = _save_latent_bundle(output_dir, record, latent, metadata)
                saved_count += 1
                if len(saved_pairs_preview) < 5:
                    saved_pairs_preview.append(saved_pair)

                reloaded_shape = _sanity_reload(saved_pair[0])
                if reloaded_shape != latent_stats.shape:
                    raise RuntimeError(
                        f"Sanity reload shape mismatch for {saved_pair[0]}: "
                        f"saved={latent_stats.shape}, reloaded={reloaded_shape}"
                    )

                if args.verbose and not args.summary_only:
                    global_idx = batch_start + local_idx
                    print(
                        f"[{global_idx}] uid={record.uid} path={image_path_str} -> shape={latent_stats.shape} "
                        f"dtype={latent_stats.dtype} min={latent_stats.min_value:.6f} "
                        f"max={latent_stats.max_value:.6f}"
                    )
                    if local_idx < len(image_infos):
                        print(
                            "    original_size="
                            f"{image_infos[local_idx]['original_width']}x"
                            f"{image_infos[local_idx]['original_height']}"
                        )

            num_batches_processed += 1
            num_images_processed += len(batch_records)
            if (
                batch_idx == 1
                or args.progress_every_batches == 1
                or (args.progress_every_batches > 1 and batch_idx % args.progress_every_batches == 0)
                or num_images_processed == len(image_records)
            ):
                elapsed = time.time() - start_time
                print(
                    "[progress] "
                    f"batch={batch_idx} "
                    f"images_processed={num_images_processed}/{len(image_records)} "
                    f"saved={saved_count} "
                    f"elapsed_seconds={elapsed:.2f}"
                )

            del batch
            del raw_latents
            del selected_latents
            if scaled_stats is not None:
                del scaled_latents
            if device.type == "cuda":
                torch.cuda.empty_cache()

        aggregated_selected_stats = _finalize_running_stats(selected_acc)
        _print_saved_files(saved_pairs_preview, saved_count, args.summary_only)
        plausibility_note = (
            "상당히 타당해 보임. 로컬 Wuerstchen prior training code의 "
            "`EfficientNetEncoder -> image_embeds -> optional scaling` 흐름을 직접 재현했다. "
            "다만 Stable Cascade 공식 학습 경로와 scaling/target semantics가 완전히 동일한지는 추가 확인이 필요하다."
        )
        _print_summary(
            success=True,
            latent_stats=aggregated_selected_stats,
            scaling_applied=args.apply_scaling,
            plausibility_note=plausibility_note,
            saved_count=saved_count,
            num_images_processed=num_images_processed,
            num_batches_processed=num_batches_processed,
            elapsed_seconds=time.time() - start_time,
        )
        if args.summary_json:
            summary_payload = {
                "success": True,
                "output_dir": str(output_dir),
                "target_image_size": [target_height, target_width],
                "prior_resolution_multiple": float(args.prior_resolution_multiple),
                "expected_latent_shape": [1, 16, expected_h, expected_w],
                "expected_latent_spatial": [expected_h, expected_w],
                "preprocess_resolution_mode": args.effnet_preprocess_resolution_mode,
                "effnet_input_size": [preprocess_h, preprocess_w],
                "effnet_preprocess_resolution_mode": args.effnet_preprocess_resolution_mode,
                "effnet_preprocess_resolution": [preprocess_h, preprocess_w],
                "encoder_class": encoder.__class__.__name__,
                "checkpoint_path": str(checkpoint_path),
                "scaling_applied": bool(args.apply_scaling),
                "scaling_expression": scaling_expression,
                "shape_policy": args.shape_policy,
                "batch_size": int(args.batch_size),
                "device": str(device),
                "num_images_processed": int(num_images_processed),
                "num_batches_processed": int(num_batches_processed),
                "num_skipped_existing": int(skipped_existing_count),
                "num_missing_uids": int(len(missing_uids)),
                "saved_count": int(saved_count),
                "latent_stats": asdict(aggregated_selected_stats),
                "saved_pairs_preview": [
                    {"latent_path": str(latent_path), "metadata_path": str(meta_path)}
                    for latent_path, meta_path in saved_pairs_preview
                ],
                "elapsed_seconds": float(time.time() - start_time),
            }
            _write_summary_json(Path(args.summary_json), summary_payload)
        return 0
    except Exception as exc:
        print("[ERROR] Latent preparation failed.")
        print(str(exc))
        if args.verbose:
            print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
