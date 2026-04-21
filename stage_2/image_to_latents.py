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
import json
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms


DEFAULT_ENCODER_SEARCH_ROOT = (
    "/data/roycecho/diffusers/examples/research_projects/wuerstchen/text_to_image"
)
DEFAULT_OUTPUT_DIR = "stage_2/artifacts/stage_c_latents"
DEFAULT_RESOLUTION = 1024
DEFAULT_CHECKPOINT_FILENAME = "model_v2_stage_b.pt"


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
    latent_path: str
    latent_shape: List[int]
    dtype: str
    transform_info: Dict[str, Any]
    scaling_applied: bool
    scaling_expression: Optional[str]
    encoder_class: str
    checkpoint_path: str
    notes: List[str]


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


def _build_effnet_transform(resolution: int) -> transforms.Compose:
    """Match the local Wuerstchen prior training transform."""

    return transforms.Compose(
        [
            transforms.Resize(
                resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.CenterCrop(resolution),
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
) -> Tuple[List[ImageRecord], List[str]]:
    """Resolve input image paths from CLI options."""

    records: List[ImageRecord] = []
    missing_uids: List[str] = []

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

    if not deduped:
        raise ValueError("No input images were resolved. Pass --image-path or --image-dir.")

    if max_images is not None and max_images > 0:
        deduped = deduped[:max_images]

    missing_uids = _dedupe_preserve_order(missing_uids)
    if missing_uids_txt and missing_uids:
        _write_uid_list_txt(Path(missing_uids_txt), missing_uids)
    if missing_uids_json and missing_uids:
        _write_uid_list_json(Path(missing_uids_json), missing_uids, field_name="missing_uids")

    return deduped, missing_uids


def _load_pil_image(path: Path) -> Image.Image:
    """Load an image as RGB."""

    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read image file: {path}\n{traceback.format_exc()}"
        ) from exc


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


def _warn_if_shape_is_unexpected(shape: Sequence[int]) -> Optional[str]:
    """Return a warning string when latent shape looks inconsistent with expectations."""

    if len(shape) != 4:
        return f"[WARN] Expected a 4D latent batch tensor, got shape={list(shape)}"
    if shape[1] != 16:
        return f"[WARN] Expected channel dimension 16 for Stage C candidate latent, got shape={list(shape)}"
    return None


def _make_batch(
    records: Sequence[ImageRecord],
    transform: transforms.Compose,
    device: torch.device,
) -> Tuple[Tensor, List[Dict[str, Any]]]:
    """Load and transform a small image batch."""

    tensors: List[Tensor] = []
    infos: List[Dict[str, Any]] = []
    for record in records:
        image_path = Path(record.image_path)
        image = _load_pil_image(image_path)
        try:
            pixel_values = transform(image)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to apply EffNet transform to image: {image_path}\n{traceback.format_exc()}"
            ) from exc

        tensors.append(pixel_values)
        infos.append(
            {
                "image_path": str(image_path),
                "uid": record.uid,
                "original_size": list(image.size),
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


def _save_latent_bundle(
    output_dir: Path,
    record: ImageRecord,
    latent: Tensor,
    metadata: SaveMetadata,
) -> Tuple[Path, Path]:
    """Save one latent tensor and its metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _output_stem_for_record(record)
    latent_path = output_dir / f"{stem}.pt"
    meta_path = output_dir / f"{stem}.json"

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
    print(f"needed_uids_json: {Path(args.needed_uids_json).expanduser().resolve() if args.needed_uids_json else None}")
    print(f"needed_uids_txt: {Path(args.needed_uids_txt).expanduser().resolve() if args.needed_uids_txt else None}")
    print(f"uid_to_path_json: {Path(args.uid_to_path_json).expanduser().resolve() if args.uid_to_path_json else None}")
    print(f"skip_missing_uids: {args.skip_missing_uids}")
    print(f"num_missing_uids: {num_missing_uids}")
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


def _print_saved_files(saved_pairs: Sequence[Tuple[Path, Path]], summary_only: bool) -> None:
    """Print saved artifact paths."""

    _print_header("Saved Files")
    if not saved_pairs:
        print("No latent files were saved.")
        return
    if summary_only:
        print(f"Saved {len(saved_pairs)} latent bundle(s).")
        return
    for latent_path, meta_path in saved_pairs:
        print(f"- latent: {latent_path}")
        print(f"  meta:   {meta_path}")


def _print_summary(
    success: bool,
    latent_stats: TensorStats,
    scaling_applied: bool,
    plausibility_note: str,
    saved_pairs: Sequence[Tuple[Path, Path]],
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
    print("6. 다음 단계 추천")
    print("- `stage2_dataset.py` latent 확장: 이 저장 포맷을 기준으로 latent-backed variant를 추가.")
    print("- latent manifest 생성: 이미지 경로와 latent 경로를 연결하는 json/jsonl 작성.")
    print("- minimal noisy sample prototype: scheduler.add_noise(image_embeds, noise, timesteps) 재현.")
    print("- `train_stage2_dpo.py` minimal step: clean target/noisy sample 정의가 확정된 뒤 연결.")
    if saved_pairs:
        print(f"Saved latent bundle count: {len(saved_pairs)}")


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
    parser.add_argument("--max-images", type=int, default=8, help="Maximum number of images to process.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, for example cpu or cuda.")
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
        image_records, missing_uids = _collect_image_records(
            args.image_path,
            args.image_dir,
            args.needed_uids_json,
            args.needed_uids_txt,
            args.uid_to_path_json,
            args.skip_missing_uids,
            args.max_images,
            args.missing_uids_txt,
            args.missing_uids_json,
        )
        transform = _build_effnet_transform(DEFAULT_RESOLUTION)
        encoder = _load_encoder(EfficientNetEncoder, checkpoint_path, device)
    except Exception as exc:
        print("[ERROR] Initialization failed.")
        print(str(exc))
        return 1

    _print_environment(args, encoder_root, checkpoint_path, device, num_missing_uids=len(missing_uids))

    try:
        batch, image_infos = _make_batch(image_records, transform, device)
        batch_stats = _tensor_stats(batch)

        with torch.no_grad():
            raw_latents = encoder(batch)
            scaled_latents, scaling_expression = _apply_optional_scaling(raw_latents, args.apply_scaling)

        raw_stats = _tensor_stats(raw_latents)
        scaled_stats = _tensor_stats(scaled_latents) if args.apply_scaling else None
        selected_latents = scaled_latents if args.apply_scaling else raw_latents
        selected_stats = scaled_stats or raw_stats
        warning = _warn_if_shape_is_unexpected(selected_stats.shape)

        if not args.summary_only:
            _print_batch_report(
                batch_stats=batch_stats,
                raw_stats=raw_stats,
                scaled_stats=scaled_stats,
                warning=warning,
                apply_scaling=args.apply_scaling,
            )

        saved_pairs: List[Tuple[Path, Path]] = []
        transform_info = {
            "resolution": DEFAULT_RESOLUTION,
            "resize": "bilinear+antialias",
            "crop": "center_crop",
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "source": "local diffusers Wuerstchen train_text_to_image_prior.py",
            "status": "candidate path; additional confirmation still needed",
        }

        for idx, record in enumerate(image_records):
            image_path = Path(record.image_path)
            latent = selected_latents[idx : idx + 1]
            latent_stats = _tensor_stats(latent)
            notes = [
                "Hypothesis: this tensor is a candidate Stage C clean target latent/image_embeds.",
                "Additional confirmation is still needed for exact Stable Cascade scaling and scheduler target semantics.",
            ]
            if warning:
                notes.append(warning)

            latent_path = output_dir / f"{_output_stem_for_record(record)}.pt"
            metadata = SaveMetadata(
                uid=record.uid,
                original_image_path=str(image_path),
                latent_path=str(latent_path),
                latent_shape=latent_stats.shape,
                dtype=latent_stats.dtype,
                transform_info=transform_info,
                scaling_applied=args.apply_scaling,
                scaling_expression=scaling_expression,
                encoder_class=encoder.__class__.__name__,
                checkpoint_path=str(checkpoint_path),
                notes=notes,
            )
            saved_pair = _save_latent_bundle(output_dir, record, latent, metadata)
            saved_pairs.append(saved_pair)

            reloaded_shape = _sanity_reload(saved_pair[0])
            if reloaded_shape != latent_stats.shape:
                raise RuntimeError(
                    f"Sanity reload shape mismatch for {saved_pair[0]}: "
                    f"saved={latent_stats.shape}, reloaded={reloaded_shape}"
                )

            if args.verbose and not args.summary_only:
                print(
                    f"[{idx}] uid={record.uid} path={image_path} -> shape={latent_stats.shape} "
                    f"dtype={latent_stats.dtype} min={latent_stats.min_value:.6f} "
                    f"max={latent_stats.max_value:.6f}"
                )
                if idx < len(image_infos):
                    print(f"    original_size={image_infos[idx]['original_size']}")

        _print_saved_files(saved_pairs, args.summary_only)
        plausibility_note = (
            "상당히 타당해 보임. 로컬 Wuerstchen prior training code의 "
            "`EfficientNetEncoder -> image_embeds -> optional scaling` 흐름을 직접 재현했다. "
            "다만 Stable Cascade 공식 학습 경로와 scaling/target semantics가 완전히 동일한지는 추가 확인이 필요하다."
        )
        _print_summary(
            success=True,
            latent_stats=selected_stats,
            scaling_applied=args.apply_scaling,
            plausibility_note=plausibility_note,
            saved_pairs=saved_pairs,
        )
        return 0
    except Exception as exc:
        print("[ERROR] Latent preparation failed.")
        print(str(exc))
        if args.verbose:
            print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
