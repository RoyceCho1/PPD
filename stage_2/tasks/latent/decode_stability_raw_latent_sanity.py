from __future__ import annotations

"""Sanity-check raw Stability-AI StableCascade Stage C latents.

For each selected latent this script saves:
  - original image
  - original StableCascade Previewer(latent)
  - diffusers StableCascadeDecoderPipeline(image_embeddings=latent, prompt=caption)
  - a compact side-by-side grid
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torchvision.transforms.functional import to_pil_image

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


STABLE_CASCADE_ROOT = Path("/data/roycecho/StableCascade")
DEFAULT_CHECKPOINT_DIR = Path("/Data_Storage/roycecho/PPD/checkpoints/stable_cascade")
DEFAULT_PREVIEWER_FILENAME = "previewer.safetensors"
DEFAULT_PROMPT = "a high quality image"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview/decode Stability raw Stage C latents.")
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument(
        "--samples-json",
        type=Path,
        default=None,
        help="Optional JSON with `samples: [{uid, category?}]` or a list of UIDs.",
    )
    parser.add_argument("--uid", action="append", default=None, help="Explicit UID. Can be repeated.")
    parser.add_argument("--uid-to-path-json", type=Path, default=None)
    parser.add_argument("--uid-to-meta-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previewer-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decoder-steps", type=int, default=20)
    parser.add_argument("--decoder-guidance-scale", type=float, default=0.0)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--decoder-model-id", default="stabilityai/stable-cascade")
    parser.add_argument(
        "--allow-decoder-download",
        action="store_true",
        help="Allow diffusers to download decoder files if missing from local cache.",
    )
    parser.add_argument("--skip-diffusers-decoder", action="store_true")
    parser.add_argument("--grid-cell-size", type=int, default=384)
    parser.add_argument("--summary-json", type=Path, default=None)
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


def _load_samples(path: Path) -> List[Dict[str, Optional[str]]]:
    data = _load_json(path, label="samples JSON")
    raw_samples = data.get("samples") if isinstance(data, Mapping) and "samples" in data else data
    if not isinstance(raw_samples, list):
        raise ValueError(f"samples JSON must be a list or contain `samples`: {path}")
    samples: List[Dict[str, Optional[str]]] = []
    for item in raw_samples:
        if isinstance(item, Mapping):
            samples.append(
                {
                    "uid": _normalize_uid(item.get("uid")),
                    "category": str(item["category"]) if item.get("category") is not None else None,
                }
            )
        else:
            samples.append({"uid": _normalize_uid(item), "category": None})
    return samples


def _resolve_samples(args: argparse.Namespace) -> List[Dict[str, Optional[str]]]:
    samples: List[Dict[str, Optional[str]]] = []
    if args.samples_json is not None:
        samples.extend(_load_samples(args.samples_json))
    if args.uid:
        samples.extend({"uid": _normalize_uid(uid), "category": None} for uid in args.uid)
    if not samples:
        raise ValueError("Provide --samples-json or at least one --uid.")

    seen: set[str] = set()
    deduped: List[Dict[str, Optional[str]]] = []
    for sample in samples:
        uid = str(sample["uid"])
        if uid in seen:
            continue
        seen.add(uid)
        deduped.append(sample)
    return deduped


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _import_previewer() -> Any:
    module_path = STABLE_CASCADE_ROOT / "modules" / "previewer.py"
    if not module_path.exists():
        raise FileNotFoundError(f"StableCascade previewer.py not found under {STABLE_CASCADE_ROOT}")
    try:
        module = _load_python_module("stablecascade_previewer", module_path)
        Previewer = module.Previewer
    except Exception as exc:
        raise ImportError(
            "Failed to import Stability-AI Previewer.\n"
            f"root={STABLE_CASCADE_ROOT}\ntraceback:\n{traceback.format_exc()}"
        ) from exc
    return Previewer


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
    try:
        import safetensors
    except Exception as exc:
        raise ImportError("safetensors is required to load StableCascade checkpoints.") from exc

    state: Dict[str, Tensor] = {}
    with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            state[key] = handle.get_tensor(key)
    return state


def _latent_path(latent_root: Path, uid: str) -> Path:
    return latent_root.expanduser().resolve() / uid[:2] / f"{uid}.pt"


def _extract_latent(loaded: Any, path: Path) -> Tensor:
    if torch.is_tensor(loaded):
        tensor = loaded
    elif isinstance(loaded, Mapping):
        tensor = None
        for key in ("latent", "image_embeddings", "latents", "sample"):
            value = loaded.get(key)
            if torch.is_tensor(value):
                tensor = value
                break
        if tensor is None:
            raise TypeError(f"No latent tensor key found in {path}")
    else:
        raise TypeError(f"Unsupported latent payload in {path}: {type(loaded)}")

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tuple(int(dim) for dim in tensor.shape[1:]) != (16, 24, 24):
        raise ValueError(f"Expected [B,16,24,24] latent in {path}, got {tuple(tensor.shape)}")
    return tensor


def _load_latent(path: Path) -> Tensor:
    if not path.exists():
        raise FileNotFoundError(f"latent not found: {path}")
    return _extract_latent(torch.load(path, map_location="cpu"), path)


def _tensor_stats(tensor: Tensor) -> Dict[str, float]:
    cpu = tensor.detach().float().cpu()
    return {
        "min": float(cpu.min().item()),
        "max": float(cpu.max().item()),
        "mean": float(cpu.mean().item()),
        "std": float(cpu.std(unbiased=False).item()),
    }


def _load_mapping_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = _load_json(path, label=str(path))
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected object JSON: {path}")
    return {str(key): value for key, value in data.items()}


def _caption_for_uid(uid: str, uid_to_meta: Mapping[str, Any]) -> str:
    meta = uid_to_meta.get(uid)
    if isinstance(meta, Mapping):
        captions = meta.get("caption_samples")
        if isinstance(captions, list) and captions:
            return str(captions[0])
        caption = meta.get("caption")
        if caption is not None:
            return str(caption)
    return DEFAULT_PROMPT


def _original_path_for_uid(uid: str, uid_to_path: Mapping[str, Any], latent_path: Path) -> Optional[Path]:
    if uid in uid_to_path:
        return Path(str(uid_to_path[uid])).expanduser().resolve()
    sidecar_path = latent_path.with_suffix(".json")
    if sidecar_path.exists():
        sidecar = _load_json(sidecar_path, label="latent sidecar")
        if isinstance(sidecar, Mapping):
            for key in ("source_image_path", "original_image_path"):
                if sidecar.get(key):
                    return Path(str(sidecar[key])).expanduser().resolve()
    return None


def _save_original(path: Optional[Path], output_path: Path) -> Optional[str]:
    if path is None or not path.exists():
        return None
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(output_path)
    return str(output_path)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def _load_decoder_pipe(args: argparse.Namespace, device: torch.device) -> Any:
    from diffusers import StableCascadeDecoderPipeline

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableCascadeDecoderPipeline.from_pretrained(
        args.decoder_model_id,
        torch_dtype=dtype,
        local_files_only=not bool(args.allow_decoder_download),
    )
    return pipe.to(device)


def _decode_diffusers(
    *,
    pipe: Any,
    latent: Tensor,
    prompt: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Image.Image:
    dtype = getattr(pipe, "dtype", None)
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    with torch.inference_mode():
        output = pipe(
            image_embeddings=latent.to(device=device, dtype=dtype),
            prompt=prompt,
            num_inference_steps=int(args.decoder_steps),
            guidance_scale=float(args.decoder_guidance_scale),
            negative_prompt=args.negative_prompt,
            num_images_per_prompt=1,
            generator=_make_generator(device, args.seed),
            output_type="pil",
            return_dict=True,
        )
    images = getattr(output, "images", output)
    if isinstance(images, list):
        if len(images) != 1:
            raise RuntimeError(f"Expected one decoder image, got {len(images)}")
        return images[0].convert("RGB")
    return images.convert("RGB")


def _save_previewer(previewer: Any, latent: Tensor, output_path: Path, device: torch.device) -> str:
    with torch.inference_mode():
        preview = previewer(latent.to(device=device, dtype=torch.float32)).detach().cpu()[0].clamp(0, 1)
    image = to_pil_image(preview)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


def _placeholder(text: str, size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), text[:500], fill=(20, 20, 20))
    return image


def _fit_cell(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (size, size), color=(255, 255, 255))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    cell.paste(image, (x, y))
    return cell


def _save_grid(
    *,
    original_path: Optional[str],
    previewer_path: str,
    decoder_path: Optional[str],
    decoder_error: Optional[str],
    output_path: Path,
    cell_size: int,
) -> str:
    labels = ["original", "previewer_raw_latent", "diffusers_decoder"]
    images: List[Image.Image] = []
    if original_path is not None:
        images.append(Image.open(original_path).convert("RGB"))
    else:
        images.append(_placeholder("original missing", cell_size))
    images.append(Image.open(previewer_path).convert("RGB"))
    if decoder_path is not None:
        images.append(Image.open(decoder_path).convert("RGB"))
    else:
        images.append(_placeholder(decoder_error or "decoder skipped", cell_size))

    label_height = 28
    grid = Image.new("RGB", (cell_size * len(images), cell_size + label_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for idx, (label, image) in enumerate(zip(labels, images)):
        x = idx * cell_size
        draw.text((x + 8, 8), label, fill=(0, 0, 0))
        grid.paste(_fit_cell(image, cell_size), (x, label_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return str(output_path)


def _save_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    uid_to_path = _load_mapping_json(args.uid_to_path_json)
    uid_to_meta = _load_mapping_json(args.uid_to_meta_json)
    samples = _resolve_samples(args)

    checkpoint_path = _resolve_checkpoint(
        args.previewer_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        filename=DEFAULT_PREVIEWER_FILENAME,
        download_missing=args.download_missing,
    )
    Previewer = _import_previewer()
    previewer = Previewer()
    load_result = previewer.load_state_dict(_load_safetensors(checkpoint_path), strict=True)
    previewer.eval().requires_grad_(False).to(device)

    decoder_pipe = None
    decoder_load_error: Optional[str] = None
    if not args.skip_diffusers_decoder:
        try:
            decoder_pipe = _load_decoder_pipe(args, device)
        except Exception as exc:
            decoder_load_error = str(exc)
            print(f"[warn] failed to load diffusers decoder: {decoder_load_error}", flush=True)

    records: List[Dict[str, Any]] = []
    start_time = time.time()
    for sample in samples:
        uid = str(sample["uid"])
        category = sample.get("category")
        latent_path = _latent_path(args.latent_root, uid)
        latent = _load_latent(latent_path)
        prompt = _caption_for_uid(uid, uid_to_meta)
        sample_dir = output_dir / uid[:2] / uid
        original_source = _original_path_for_uid(uid, uid_to_path, latent_path)

        original_out = _save_original(original_source, sample_dir / f"{uid}_original.png")
        previewer_out = _save_previewer(previewer, latent, sample_dir / f"{uid}_previewer.png", device)

        decoder_out: Optional[str] = None
        decoder_error = decoder_load_error
        if decoder_pipe is not None:
            try:
                decoded = _decode_diffusers(
                    pipe=decoder_pipe,
                    latent=latent,
                    prompt=prompt,
                    args=args,
                    device=device,
                )
                decoder_path = sample_dir / f"{uid}_diffusers_decoder.png"
                decoded.save(decoder_path)
                decoder_out = str(decoder_path)
                decoder_error = None
            except Exception as exc:
                decoder_error = str(exc)
                print(f"[warn] decoder failed for uid={uid}: {decoder_error}", flush=True)

        grid_out = _save_grid(
            original_path=original_out,
            previewer_path=previewer_out,
            decoder_path=decoder_out,
            decoder_error=decoder_error,
            output_path=sample_dir / f"{uid}_grid.png",
            cell_size=int(args.grid_cell_size),
        )

        record = {
            "uid": uid,
            "category": category,
            "prompt": prompt,
            "latent_path": str(latent_path),
            "latent_shape": [int(dim) for dim in latent.shape],
            "latent_stats": _tensor_stats(latent),
            "original_source_path": str(original_source) if original_source is not None else None,
            "original_png": original_out,
            "previewer_png": previewer_out,
            "diffusers_decoder_png": decoder_out,
            "diffusers_decoder_error": decoder_error,
            "grid_png": grid_out,
        }
        records.append(record)
        print(
            f"[done] uid={uid} category={category} previewer={previewer_out} decoder={decoder_out}",
            flush=True,
        )

    summary = {
        "latent_root": str(args.latent_root.expanduser().resolve()),
        "output_dir": str(output_dir),
        "num_samples": len(samples),
        "elapsed_seconds": time.time() - start_time,
        "previewer_checkpoint": str(checkpoint_path.resolve()),
        "previewer_source": "Stability-AI/StableCascade/modules/previewer.py",
        "strict_load_missing_keys": list(load_result.missing_keys),
        "strict_load_unexpected_keys": list(load_result.unexpected_keys),
        "diffusers_decoder_model_id": None if args.skip_diffusers_decoder else args.decoder_model_id,
        "diffusers_decoder_load_error": decoder_load_error,
        "records": records,
    }
    summary_path = args.summary_json or (output_dir / "summary.json")
    _save_json(summary_path.expanduser().resolve(), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
