from __future__ import annotations

"""Decode precomputed latents before/after channel-wise affine calibration."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch
from PIL import Image, ImageDraw


DEFAULT_DECODER_MODEL_ID = "stabilityai/stable-cascade"


def _load_json(path: Path) -> Any:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_samples(path: Path) -> List[Dict[str, str]]:
    payload = _load_json(path)
    raw_samples = payload.get("samples") if isinstance(payload, Mapping) else payload
    if raw_samples is None and isinstance(payload, Mapping) and "uids" in payload:
        raw_samples = [{"uid": str(uid), "category": "sample"} for uid in payload["uids"]]
    if not isinstance(raw_samples, list):
        raise ValueError("Sample JSON must be a list or an object with `samples`/`uids`.")

    samples: List[Dict[str, str]] = []
    for item in raw_samples:
        if isinstance(item, str):
            samples.append({"uid": item, "category": "sample"})
        elif isinstance(item, Mapping):
            uid = str(item.get("uid", "")).strip()
            if not uid:
                raise ValueError(f"Sample item is missing uid: {item}")
            samples.append({"uid": uid, "category": str(item.get("category", "sample"))})
        else:
            raise TypeError(f"Unsupported sample item type: {type(item)}")
    return samples


def _load_latent(path: Path) -> torch.Tensor:
    loaded = torch.load(path, map_location="cpu")
    if not torch.is_tensor(loaded):
        raise TypeError(f"Expected tensor latent at {path}, got {type(loaded)}")
    if loaded.ndim == 3:
        loaded = loaded.unsqueeze(0)
    if loaded.ndim != 4 or loaded.shape[1] != 16:
        raise ValueError(f"Expected [B,16,H,W] latent at {path}, got shape={tuple(loaded.shape)}")
    return loaded.float()


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    x = tensor.detach().float().cpu()
    return {
        "shape": [int(dim) for dim in x.shape],
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "channel_mean": [float(v.item()) for v in x.mean(dim=(0, 2, 3))],
        "channel_std": [float(v.item()) for v in x.std(dim=(0, 2, 3), unbiased=False)],
    }


def _load_channel_calibration(summary_path: Path, eps: float) -> Dict[str, torch.Tensor]:
    summary = _load_json(summary_path)
    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError(f"Calibration summary is missing aggregate: {summary_path}")
    prior = aggregate.get("vanilla_prior")
    precomputed = aggregate.get("precomputed_24x24")
    if not isinstance(prior, Mapping) or not isinstance(precomputed, Mapping):
        raise ValueError("Calibration summary must contain aggregate.vanilla_prior and aggregate.precomputed_24x24.")

    src_mean = torch.tensor(precomputed["channel_mean"], dtype=torch.float32).view(1, -1, 1, 1)
    src_std = torch.tensor(precomputed["channel_std"], dtype=torch.float32).view(1, -1, 1, 1).clamp_min(float(eps))
    dst_mean = torch.tensor(prior["channel_mean"], dtype=torch.float32).view(1, -1, 1, 1)
    dst_std = torch.tensor(prior["channel_std"], dtype=torch.float32).view(1, -1, 1, 1).clamp_min(float(eps))
    if src_mean.shape[1] != 16 or dst_mean.shape[1] != 16:
        raise ValueError(f"Expected 16 channel calibration stats, got src={src_mean.shape} dst={dst_mean.shape}")
    return {
        "src_mean": src_mean,
        "src_std": src_std,
        "dst_mean": dst_mean,
        "dst_std": dst_std,
    }


def _apply_channel_affine(latent: torch.Tensor, calibration: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return (latent - calibration["src_mean"]) / calibration["src_std"] * calibration["dst_std"] + calibration["dst_mean"]


def _fit_tile(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _draw_label(tile: Image.Image, text: str, subtext: str = "") -> Image.Image:
    label_h = 48
    output = Image.new("RGB", (tile.width, tile.height + label_h), "white")
    output.paste(tile, (0, label_h))
    draw = ImageDraw.Draw(output)
    draw.text((6, 5), text[:42], fill=(0, 0, 0))
    if subtext:
        draw.text((6, 25), subtext[:42], fill=(60, 60, 60))
    return output


def _decode_one(
    pipe: Any,
    latent: torch.Tensor,
    prompt: str,
    seed: int,
    steps: int,
    device: torch.device,
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.inference_mode():
        output = pipe(
            image_embeddings=latent,
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=0.0,
            negative_prompt=None,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
            return_dict=True,
        )
    images = getattr(output, "images", output)
    if not isinstance(images, list) or len(images) != 1:
        raise RuntimeError(f"Expected one decoded image, got {type(images)}")
    return images[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-json", type=Path, required=True)
    parser.add_argument("--uid-to-path-json", type=Path, required=True)
    parser.add_argument("--uid-to-meta-json", type=Path, default=None)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--calibration-summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decoder-model-id", type=str, default=DEFAULT_DECODER_MODEL_ID)
    parser.add_argument("--prompt", type=str, default="a high quality image")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decoder-steps", type=int, default=20)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from diffusers import StableCascadeDecoderPipeline

    device = torch.device(args.device)
    samples = _load_samples(args.samples_json)
    uid_to_path = {str(key): str(value) for key, value in _load_json(args.uid_to_path_json).items()}
    uid_to_meta: Dict[str, Any] = {}
    if args.uid_to_meta_json is not None:
        uid_to_meta = {str(key): value for key, value in _load_json(args.uid_to_meta_json).items()}

    output_dir = args.output_dir.expanduser().resolve()
    full_dir = output_dir / "full_decodes"
    row_dir = output_dir / "rows"
    latent_dir = output_dir / "calibrated_latents"
    full_dir.mkdir(parents=True, exist_ok=True)
    row_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)

    calibration = _load_channel_calibration(args.calibration_summary_json, eps=float(args.eps))

    pipe = StableCascadeDecoderPipeline.from_pretrained(
        args.decoder_model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
    )
    pipe.to(device)
    decoder_dtype = next(pipe.decoder.parameters()).dtype
    tile_size = (int(args.tile_size), int(args.tile_size))

    rows: List[Image.Image] = []
    records: List[Dict[str, Any]] = []
    for sample in samples:
        uid = sample["uid"]
        category = sample["category"]
        latent_path = args.latent_root.expanduser().resolve() / uid[:2] / f"{uid}.pt"
        original_path = Path(uid_to_path[uid]).expanduser().resolve()
        if not latent_path.exists() or not original_path.exists():
            raise FileNotFoundError(
                f"Missing input for uid={uid}: latent={latent_path.exists()} original={original_path.exists()}"
            )

        original_latent = _load_latent(latent_path)
        calibrated_latent = _apply_channel_affine(original_latent, calibration)
        calibrated_path = latent_dir / f"{uid}_channel_affine.pt"
        torch.save(calibrated_latent.cpu(), calibrated_path)

        variants = (
            ("original_24x24", "precomputed", original_latent),
            ("channel_affine", "precomputed -> prior ch mean/std", calibrated_latent),
        )
        original = Image.open(original_path).convert("RGB")
        tiles = [_draw_label(_fit_tile(original, tile_size), f"{category} original", uid[:8])]
        record: Dict[str, Any] = {
            "category": category,
            "uid": uid,
            "caption": " | ".join(str(item) for item in (uid_to_meta.get(uid, {}).get("caption_samples") or [])),
            "original_path": str(original_path),
            "latent_path": str(latent_path),
            "calibrated_latent_path": str(calibrated_path),
            "variants": {},
        }

        for variant_name, display_name, latent_cpu in variants:
            latent = latent_cpu.to(device=device, dtype=decoder_dtype)
            image = _decode_one(
                pipe=pipe,
                latent=latent,
                prompt=args.prompt,
                seed=int(args.seed),
                steps=int(args.decoder_steps),
                device=device,
            )
            decode_path = full_dir / f"{uid}_{variant_name}.png"
            image.save(decode_path)
            Image.open(decode_path).verify()
            tiles.append(_draw_label(_fit_tile(image, tile_size), variant_name, display_name))
            record["variants"][variant_name] = {
                "display_name": display_name,
                "decode_path": str(decode_path),
                "decode_size": list(image.size),
                "latent_stats": _tensor_stats(latent_cpu),
            }
            del latent
            if device.type == "cuda":
                torch.cuda.empty_cache()

        row = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), "white")
        x = 0
        for tile in tiles:
            row.paste(tile, (x, 0))
            x += tile.width
        row_path = row_dir / f"{uid}_channel_calibration.png"
        row.save(row_path)
        rows.append(row)
        record["row_path"] = str(row_path)
        records.append(record)
        print(f"[decode_latent_channel_calibration] done {category} {uid}", flush=True)

    pad = 8
    grid_w = max(row.width for row in rows)
    grid_h = sum(row.height for row in rows) + pad * max(0, len(rows) - 1)
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    y = 0
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height + pad
    grid_path = output_dir / "channel_calibration_grid.png"
    grid.save(grid_path)

    summary = {
        "prompt": args.prompt,
        "seed": int(args.seed),
        "decoder_steps": int(args.decoder_steps),
        "calibration_summary_json": str(args.calibration_summary_json.expanduser().resolve()),
        "calibration_formula": "(latent - precomputed_channel_mean) / precomputed_channel_std * prior_channel_std + prior_channel_mean",
        "grid_path": str(grid_path),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[decode_latent_channel_calibration] grid {grid_path}", flush=True)


if __name__ == "__main__":
    main()
