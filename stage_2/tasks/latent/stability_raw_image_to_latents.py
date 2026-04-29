from __future__ import annotations

"""Generate raw StableCascade Stage C latents from images.

This script intentionally follows the original Stability-AI StableCascade
`train_c.py` Stage C target path:

    PIL RGB
    -> ToTensor() in [0, 1]
    -> Resize(image_size)
    -> deterministic CenterCrop(image_size)
    -> ImageNet Normalize
    -> modules.effnet.EfficientNetEncoder
    -> raw latent

It does not apply the diffusers/Wuerstchen `add(1).div(42)` scaling.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("TORCH_HOME", "/Data_Storage/roycecho/PPD/torch_cache")


STABLE_CASCADE_ROOT = Path("/data/roycecho/StableCascade")
DEFAULT_CHECKPOINT_DIR = Path("/Data_Storage/roycecho/PPD/checkpoints/stable_cascade")
DEFAULT_EFFNET_FILENAME = "effnet_encoder.safetensors"
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
LATENT_SEMANTICS = "stability_train_c_raw_effnet"
LATENT_FORMAT_VERSION = "stability_train_c_raw_effnet_v1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageRecord:
    uid: str
    image_path: Path
    category: Optional[str] = None


@dataclass(frozen=True)
class TensorStats:
    min: float
    max: float
    mean: float
    std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate raw Stability-AI StableCascade Stage C latents."
    )
    parser.add_argument("--image-dir", type=Path, default=None, help="Directory of images to scan recursively.")
    parser.add_argument(
        "--samples-json",
        type=Path,
        default=None,
        help="Optional JSON containing `samples: [{uid, category?}]` or a list of UIDs.",
    )
    parser.add_argument(
        "--uid-to-path-json",
        type=Path,
        default=None,
        help="Required when --samples-json only provides UIDs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Latent output root.")
    parser.add_argument("--image-size", type=int, default=768, help="Square train_c image size.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto.")
    parser.add_argument(
        "--effnet-checkpoint",
        type=Path,
        default=None,
        help="Local effnet_encoder.safetensors path.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory used when resolving/downloading effnet_encoder.safetensors.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing Stability-AI checkpoints from Hugging Face.",
    )
    parser.add_argument(
        "--shape-policy",
        choices=("strict", "warn", "none"),
        default="strict",
        help="How to handle unexpected latent shapes.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--progress-every-batches", type=int, default=10)
    return parser.parse_args()


def _load_python_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path.expanduser().resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path, *, label: str) -> Any:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_uid(value: Any) -> str:
    uid = str(value).strip()
    if not uid:
        raise ValueError("UID is empty.")
    return uid


def _load_uid_to_path(path: Path) -> Dict[str, Path]:
    data = _load_json(path, label="uid_to_path JSON")
    if not isinstance(data, Mapping):
        raise ValueError(f"uid_to_path JSON must be an object: {path}")
    return {str(uid): Path(str(image_path)) for uid, image_path in data.items()}


def _records_from_samples_json(path: Path, uid_to_path_json: Optional[Path]) -> List[ImageRecord]:
    data = _load_json(path, label="samples JSON")
    raw_samples: Any
    if isinstance(data, Mapping) and "samples" in data:
        raw_samples = data["samples"]
    else:
        raw_samples = data
    if not isinstance(raw_samples, list):
        raise ValueError(f"samples JSON must be a list or contain `samples`: {path}")

    uid_to_path: Dict[str, Path] = {}
    if uid_to_path_json is not None:
        uid_to_path = _load_uid_to_path(uid_to_path_json)

    records: List[ImageRecord] = []
    for item in raw_samples:
        category: Optional[str] = None
        explicit_path: Optional[Path] = None
        if isinstance(item, Mapping):
            uid = _normalize_uid(item.get("uid"))
            if item.get("category") is not None:
                category = str(item["category"])
            if item.get("image_path") is not None:
                explicit_path = Path(str(item["image_path"]))
        else:
            uid = _normalize_uid(item)

        image_path = explicit_path or uid_to_path.get(uid)
        if image_path is None:
            raise ValueError(
                f"No image path found for uid={uid!r}. Pass --uid-to-path-json or include image_path."
            )
        records.append(ImageRecord(uid=uid, image_path=image_path, category=category))
    return records


def _records_from_image_dir(image_dir: Path) -> List[ImageRecord]:
    root = image_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"image dir does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"image dir is not a directory: {root}")

    paths = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS
    )
    return [ImageRecord(uid=path.stem, image_path=path) for path in paths]


def _resolve_records(args: argparse.Namespace) -> List[ImageRecord]:
    if args.samples_json is not None:
        records = _records_from_samples_json(args.samples_json, args.uid_to_path_json)
    elif args.image_dir is not None:
        records = _records_from_image_dir(args.image_dir)
    else:
        raise ValueError("Provide either --samples-json or --image-dir.")

    if args.max_images is not None:
        records = records[: max(args.max_images, 0)]
    if not records:
        raise ValueError("No image records resolved.")
    return records


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _import_effnet_encoder() -> Any:
    module_path = STABLE_CASCADE_ROOT / "modules" / "effnet.py"
    if not module_path.exists():
        raise FileNotFoundError(f"StableCascade effnet.py not found under {STABLE_CASCADE_ROOT}")
    try:
        module = _load_python_module("stablecascade_effnet", module_path)
        EfficientNetEncoder = module.EfficientNetEncoder
    except Exception as exc:
        raise ImportError(
            "Failed to import Stability-AI EfficientNetEncoder.\n"
            f"root={STABLE_CASCADE_ROOT}\ntraceback:\n{traceback.format_exc()}"
        ) from exc
    return EfficientNetEncoder


def _download_checkpoint(filename: str, checkpoint_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise ImportError("huggingface_hub is required for --download-missing.") from exc

    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id="stabilityai/stable-cascade",
            filename=filename,
            local_dir=str(checkpoint_dir),
            local_dir_use_symlinks=False,
        )
    ).resolve()


def _resolve_checkpoint(
    checkpoint_arg: Optional[Path],
    *,
    checkpoint_dir: Path,
    filename: str,
    download_missing: bool,
) -> Path:
    if checkpoint_arg is not None:
        resolved = checkpoint_arg.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {resolved}")
        return resolved

    candidate = checkpoint_dir.expanduser().resolve() / filename
    if candidate.exists():
        return candidate
    if download_missing:
        return _download_checkpoint(filename, checkpoint_dir)
    raise FileNotFoundError(
        f"Missing checkpoint: {candidate}\n"
        f"Download it first or rerun with --download-missing."
    )


def _load_safetensors(path: Path) -> Dict[str, Tensor]:
    if path.suffix != ".safetensors":
        raise ValueError(f"Expected a .safetensors checkpoint, got: {path}")
    try:
        import safetensors
    except Exception as exc:
        raise ImportError("safetensors is required to load StableCascade checkpoints.") from exc

    state: Dict[str, Tensor] = {}
    with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            state[key] = handle.get_tensor(key)
    return state


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.CenterCrop(image_size),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _output_paths(output_dir: Path, uid: str) -> tuple[Path, Path]:
    shard = uid[:2] if len(uid) >= 2 else "__"
    base = output_dir.expanduser().resolve() / shard
    return base / f"{uid}.pt", base / f"{uid}.json"


def _stats(tensor: Tensor) -> TensorStats:
    cpu = tensor.detach().float().cpu()
    return TensorStats(
        min=float(cpu.min().item()),
        max=float(cpu.max().item()),
        mean=float(cpu.mean().item()),
        std=float(cpu.std(unbiased=False).item()),
    )


def _save_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _iter_batches(records: Sequence[ImageRecord], batch_size: int) -> Iterable[Sequence[ImageRecord]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def _load_batch(records: Sequence[ImageRecord], transform: transforms.Compose) -> tuple[Tensor, List[Dict[str, Any]]]:
    tensors: List[Tensor] = []
    metadata: List[Dict[str, Any]] = []
    for record in records:
        image_path = record.image_path.expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"image does not exist for uid={record.uid}: {image_path}")
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            original_width, original_height = rgb.size
            tensors.append(transform(rgb))
        metadata.append(
            {
                "uid": record.uid,
                "category": record.category,
                "source_image_path": str(image_path),
                "original_image_size": {"width": original_width, "height": original_height},
            }
        )
    return torch.stack(tensors, dim=0), metadata


def _check_shape(
    latent: Tensor,
    *,
    uid: str,
    expected_shape: Sequence[int],
    shape_policy: str,
) -> bool:
    actual_shape = tuple(int(dim) for dim in latent.shape)
    expected_tuple = tuple(int(dim) for dim in expected_shape)
    shape_match = actual_shape == expected_tuple
    if shape_match or shape_policy == "none":
        return shape_match

    message = f"latent shape mismatch for uid={uid}: expected={expected_tuple}, actual={actual_shape}"
    if shape_policy == "strict":
        raise ValueError(message)
    print(f"[warn] {message}", flush=True)
    return False


def _build_sidecar(
    *,
    record_meta: Mapping[str, Any],
    latent_path: Path,
    checkpoint_path: Path,
    image_size: int,
    expected_shape: Sequence[int],
    actual_shape: Sequence[int],
    shape_match: bool,
    stats: TensorStats,
) -> Dict[str, Any]:
    return {
        "uid": record_meta["uid"],
        "category": record_meta.get("category"),
        "original_image_path": record_meta["source_image_path"],
        "source_image_path": record_meta["source_image_path"],
        "original_image_size": record_meta["original_image_size"],
        "latent_path": str(latent_path.resolve()),
        "latent_shape": [int(dim) for dim in actual_shape],
        "expected_latent_shape": [int(dim) for dim in expected_shape],
        "actual_latent_shape": [int(dim) for dim in actual_shape],
        "shape_match": bool(shape_match),
        "encoder_source": "Stability-AI/StableCascade/modules/effnet.py",
        "encoder_checkpoint": str(checkpoint_path.resolve()),
        "encoder_class": "EfficientNetEncoder",
        "previewer_checkpoint": None,
        "preprocess_source": "StableCascade/train/train_c.py",
        "image_size": [int(image_size), int(image_size)],
        "target_image_size": [int(image_size), int(image_size)],
        "effnet_input_size": [int(image_size), int(image_size)],
        "preprocess_resolution_mode": "train_c_square_resize_center_crop",
        "crop_policy": "deterministic_center_crop",
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "scaling_applied": False,
        "scaling_expression": None,
        "latent_semantics": LATENT_SEMANTICS,
        "latent_format_version": LATENT_FORMAT_VERSION,
        "stats": asdict(stats),
    }


def _save_latent(
    *,
    latent: Tensor,
    record_meta: Mapping[str, Any],
    latent_path: Path,
    sidecar_path: Path,
    checkpoint_path: Path,
    image_size: int,
    expected_shape: Sequence[int],
    shape_match: bool,
    stats: TensorStats,
) -> None:
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "latent": latent.detach().cpu(),
        "uid": str(record_meta["uid"]),
        "source_image_path": str(record_meta["source_image_path"]),
        "original_image_path": str(record_meta["source_image_path"]),
        "latent_format_version": LATENT_FORMAT_VERSION,
        "latent_semantics": LATENT_SEMANTICS,
        "scaled": False,
        "apply_scaling": False,
        "stats": asdict(stats),
    }
    torch.save(bundle, latent_path)

    sidecar = _build_sidecar(
        record_meta=record_meta,
        latent_path=latent_path,
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        expected_shape=expected_shape,
        actual_shape=tuple(latent.shape),
        shape_match=shape_match,
        stats=stats,
    )
    _save_json(sidecar_path, sidecar)


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    records = _resolve_records(args)
    output_dir = args.output_dir.expanduser().resolve()
    expected_shape = (1, 16, args.image_size // 32, args.image_size // 32)

    checkpoint_path = _resolve_checkpoint(
        args.effnet_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        filename=DEFAULT_EFFNET_FILENAME,
        download_missing=args.download_missing,
    )

    EfficientNetEncoder = _import_effnet_encoder()
    model = EfficientNetEncoder()
    state_dict = _load_safetensors(checkpoint_path)
    load_result = model.load_state_dict(state_dict, strict=True)
    model.eval().requires_grad_(False).to(device)

    transform = _build_transform(args.image_size)
    start_time = time.time()
    written = 0
    skipped = 0
    failed: List[Dict[str, str]] = []
    processed_records: List[Dict[str, Any]] = []

    print(
        "[start] "
        f"records={len(records)} output_dir={output_dir} image_size={args.image_size} "
        f"expected_shape={list(expected_shape)} checkpoint={checkpoint_path} device={device}",
        flush=True,
    )

    with torch.inference_mode():
        for batch_idx, batch_records in enumerate(_iter_batches(records, args.batch_size), start=1):
            pending_records: List[ImageRecord] = []
            for record in batch_records:
                latent_path, sidecar_path = _output_paths(output_dir, record.uid)
                if args.skip_existing and latent_path.exists() and sidecar_path.exists():
                    skipped += 1
                    continue
                pending_records.append(record)

            if pending_records:
                try:
                    image_batch, metadata = _load_batch(pending_records, transform)
                    latent_batch = model(image_batch.to(device))
                    for latent, meta in zip(latent_batch.detach().cpu(), metadata):
                        uid = str(meta["uid"])
                        latent = latent.unsqueeze(0).contiguous()
                        latent_path, sidecar_path = _output_paths(output_dir, uid)
                        shape_match = _check_shape(
                            latent,
                            uid=uid,
                            expected_shape=expected_shape,
                            shape_policy=args.shape_policy,
                        )
                        stats = _stats(latent)
                        _save_latent(
                            latent=latent,
                            record_meta=meta,
                            latent_path=latent_path,
                            sidecar_path=sidecar_path,
                            checkpoint_path=checkpoint_path,
                            image_size=args.image_size,
                            expected_shape=expected_shape,
                            shape_match=shape_match,
                            stats=stats,
                        )
                        written += 1
                        processed_records.append(
                            {
                                "uid": uid,
                                "latent_path": str(latent_path),
                                "sidecar_path": str(sidecar_path),
                                "shape": [int(dim) for dim in latent.shape],
                                "shape_match": bool(shape_match),
                                "stats": asdict(stats),
                            }
                        )
                except Exception as exc:
                    for record in pending_records:
                        failed.append({"uid": record.uid, "error": str(exc)})
                    if args.shape_policy == "strict":
                        raise
                    print(f"[warn] failed batch {batch_idx}: {exc}", flush=True)

            should_print = (
                args.progress_every_batches > 0
                and (
                    batch_idx == 1
                    or batch_idx % args.progress_every_batches == 0
                    or batch_idx == (len(records) + args.batch_size - 1) // args.batch_size
                )
            )
            if should_print:
                elapsed = max(time.time() - start_time, 1e-9)
                done = min(batch_idx * args.batch_size, len(records))
                print(
                    "[progress] "
                    f"records_seen={done}/{len(records)} written={written} skipped={skipped} "
                    f"failed={len(failed)} rate={(written + skipped) / elapsed:.2f} records/s",
                    flush=True,
                )

    elapsed = time.time() - start_time
    summary: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_input_records": len(records),
        "num_written": written,
        "num_skipped": skipped,
        "num_failed": len(failed),
        "failed": failed,
        "elapsed_seconds": elapsed,
        "image_size": [int(args.image_size), int(args.image_size)],
        "expected_latent_shape": list(expected_shape),
        "encoder_source": "Stability-AI/StableCascade/modules/effnet.py",
        "encoder_checkpoint": str(checkpoint_path.resolve()),
        "previewer_checkpoint": None,
        "preprocess_source": "StableCascade/train/train_c.py",
        "crop_policy": "deterministic_center_crop",
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "scaling_applied": False,
        "scaling_expression": None,
        "latent_semantics": LATENT_SEMANTICS,
        "latent_format_version": LATENT_FORMAT_VERSION,
        "strict_load_missing_keys": list(load_result.missing_keys),
        "strict_load_unexpected_keys": list(load_result.unexpected_keys),
        "processed_records": processed_records[:50],
    }

    if args.summary_json is not None:
        _save_json(args.summary_json.expanduser().resolve(), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
