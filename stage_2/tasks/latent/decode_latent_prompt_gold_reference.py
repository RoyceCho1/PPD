from __future__ import annotations

"""Compare decoder behavior for precomputed latents and vanilla prior gold outputs.

For each UID this script saves:
1. original image
2. precomputed 24x24 latent decoded with a generic prompt
3. precomputed 24x24 latent decoded with the image caption
4. vanilla prior(caption) image_embeddings decoded with the same caption
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch
from PIL import Image, ImageDraw


DEFAULT_PRIOR_MODEL_ID = "stabilityai/stable-cascade-prior"
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


def _caption_for_uid(uid: str, uid_to_meta: Mapping[str, Any], fallback: str) -> str:
    meta = uid_to_meta.get(uid, {})
    if isinstance(meta, Mapping):
        captions = meta.get("caption_samples")
        if isinstance(captions, list) and captions:
            return str(captions[0])
        caption = meta.get("caption")
        if caption:
            return str(caption)
    return fallback


def _load_latent(path: Path) -> torch.Tensor:
    loaded = torch.load(path, map_location="cpu")
    if not torch.is_tensor(loaded):
        raise TypeError(f"Expected tensor latent at {path}, got {type(loaded)}")
    if loaded.ndim == 3:
        loaded = loaded.unsqueeze(0)
    if loaded.ndim != 4 or loaded.shape[1] != 16:
        raise ValueError(f"Expected [B,16,H,W] latent at {path}, got shape={tuple(loaded.shape)}")
    return loaded.float()


def _extract_prior_embeddings(output: Any) -> torch.Tensor:
    image_embeddings = getattr(output, "image_embeddings", None)
    if torch.is_tensor(image_embeddings):
        return image_embeddings
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Could not extract image_embeddings from prior output type {type(output)}.")


def _pipeline_dtype(pipe: Any) -> torch.dtype:
    for name in ("prior", "decoder", "vqgan", "text_encoder"):
        module = getattr(pipe, name, None)
        if module is None:
            continue
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            continue
        except AttributeError:
            continue
    return torch.float32


def _sync_text_encoder_dtype(pipe: Any, device: torch.device) -> None:
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        return
    to = getattr(text_encoder, "to", None)
    if callable(to):
        to(device=device, dtype=_pipeline_dtype(pipe))


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    x = tensor.detach().float().cpu()
    return {
        "shape": [int(dim) for dim in x.shape],
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "l2_norm": float(torch.linalg.vector_norm(x.flatten()).item()),
    }


def _fit_tile(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _draw_label(tile: Image.Image, text: str, subtext: str = "") -> Image.Image:
    label_h = 58
    output = Image.new("RGB", (tile.width, tile.height + label_h), "white")
    output.paste(tile, (0, label_h))
    draw = ImageDraw.Draw(output)
    draw.text((6, 5), text[:44], fill=(0, 0, 0))
    if subtext:
        draw.text((6, 27), subtext[:44], fill=(60, 60, 60))
    return output


def _decode_one(
    pipe: Any,
    image_embeddings: torch.Tensor,
    prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    device: torch.device,
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.inference_mode():
        output = pipe(
            image_embeddings=image_embeddings,
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
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


def _run_prior(
    pipe: Any,
    prompt: str,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            negative_prompt=None,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pt",
            return_dict=True,
        )
    return _extract_prior_embeddings(output).detach().float().cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-json", type=Path, required=True)
    parser.add_argument("--uid-to-path-json", type=Path, required=True)
    parser.add_argument("--uid-to-meta-json", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-model-id", type=str, default=DEFAULT_PRIOR_MODEL_ID)
    parser.add_argument("--decoder-model-id", type=str, default=DEFAULT_DECODER_MODEL_ID)
    parser.add_argument("--generic-prompt", type=str, default="a high quality image")
    parser.add_argument("--caption-fallback", type=str, default="a high quality image")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--prior-steps", type=int, default=20)
    parser.add_argument("--prior-guidance-scale", type=float, default=4.0)
    parser.add_argument("--decoder-steps", type=int, default=20)
    parser.add_argument("--decoder-guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

    device = torch.device(args.device)
    samples = _load_samples(args.samples_json)
    uid_to_path = {str(key): str(value) for key, value in _load_json(args.uid_to_path_json).items()}
    uid_to_meta = {str(key): value for key, value in _load_json(args.uid_to_meta_json).items()}

    output_dir = args.output_dir.expanduser().resolve()
    full_dir = output_dir / "full_decodes"
    row_dir = output_dir / "rows"
    prior_dir = output_dir / "prior_embeddings"
    full_dir.mkdir(parents=True, exist_ok=True)
    row_dir.mkdir(parents=True, exist_ok=True)
    prior_dir.mkdir(parents=True, exist_ok=True)

    prior_pipe = StableCascadePriorPipeline.from_pretrained(args.prior_model_id, local_files_only=True)
    prior_pipe.to(device)
    _sync_text_encoder_dtype(prior_pipe, device)
    if getattr(prior_pipe, "prior", None) is not None:
        prior_pipe.prior.eval()
    if getattr(prior_pipe, "text_encoder", None) is not None:
        prior_pipe.text_encoder.eval()

    decoder_pipe = StableCascadeDecoderPipeline.from_pretrained(
        args.decoder_model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
    )
    decoder_pipe.to(device)
    decoder_dtype = _pipeline_dtype(decoder_pipe)
    tile_size = (int(args.tile_size), int(args.tile_size))

    rows: List[Image.Image] = []
    records: List[Dict[str, Any]] = []
    for sample in samples:
        uid = sample["uid"]
        category = sample["category"]
        caption = _caption_for_uid(uid, uid_to_meta, args.caption_fallback)
        latent_path = args.latent_root.expanduser().resolve() / uid[:2] / f"{uid}.pt"
        original_path = Path(uid_to_path[uid]).expanduser().resolve()
        if not latent_path.exists() or not original_path.exists():
            raise FileNotFoundError(
                f"Missing input for uid={uid}: latent={latent_path.exists()} original={original_path.exists()}"
            )

        precomputed = _load_latent(latent_path)
        prior_embeddings = _run_prior(
            prior_pipe,
            prompt=caption,
            height=int(args.height),
            width=int(args.width),
            steps=int(args.prior_steps),
            guidance_scale=float(args.prior_guidance_scale),
            seed=int(args.seed),
            device=device,
        )
        prior_path = prior_dir / f"{uid}_vanilla_prior_caption.pt"
        torch.save(prior_embeddings.cpu(), prior_path)

        variants = (
            ("precomputed_generic", "precomputed + generic", precomputed, args.generic_prompt),
            ("precomputed_caption", "precomputed + caption", precomputed, caption),
            ("prior_caption", "vanilla prior(caption) + caption", prior_embeddings, caption),
        )

        original = Image.open(original_path).convert("RGB")
        tiles = [_draw_label(_fit_tile(original, tile_size), f"{category} original", uid[:8])]
        record: Dict[str, Any] = {
            "category": category,
            "uid": uid,
            "caption": caption,
            "original_path": str(original_path),
            "precomputed_latent_path": str(latent_path),
            "prior_embedding_path": str(prior_path),
            "precomputed_stats": _tensor_stats(precomputed),
            "prior_embedding_stats": _tensor_stats(prior_embeddings),
            "variants": {},
        }

        for variant_name, display_name, embedding_cpu, prompt in variants:
            embedding = embedding_cpu.to(device=device, dtype=decoder_dtype)
            image = _decode_one(
                decoder_pipe,
                image_embeddings=embedding,
                prompt=prompt,
                seed=int(args.seed),
                steps=int(args.decoder_steps),
                guidance_scale=float(args.decoder_guidance_scale),
                device=device,
            )
            decode_path = full_dir / f"{uid}_{variant_name}.png"
            image.save(decode_path)
            Image.open(decode_path).verify()
            tiles.append(_draw_label(_fit_tile(image, tile_size), display_name, prompt))
            record["variants"][variant_name] = {
                "display_name": display_name,
                "prompt": prompt,
                "decode_path": str(decode_path),
                "decode_size": list(image.size),
                "embedding_stats": _tensor_stats(embedding_cpu),
            }
            del embedding
            if device.type == "cuda":
                torch.cuda.empty_cache()

        row = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), "white")
        x = 0
        for tile in tiles:
            row.paste(tile, (x, 0))
            x += tile.width
        row_path = row_dir / f"{uid}_prompt_gold_reference.png"
        row.save(row_path)
        record["row_path"] = str(row_path)
        records.append(record)
        rows.append(row)
        print(f"[decode_latent_prompt_gold_reference] done {category} {uid}", flush=True)

    pad = 8
    grid_w = max(row.width for row in rows)
    grid_h = sum(row.height for row in rows) + pad * max(0, len(rows) - 1)
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    y = 0
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height + pad
    grid_path = output_dir / "prompt_gold_reference_grid.png"
    grid.save(grid_path)

    summary = {
        "generic_prompt": args.generic_prompt,
        "seed": int(args.seed),
        "height": int(args.height),
        "width": int(args.width),
        "prior_steps": int(args.prior_steps),
        "prior_guidance_scale": float(args.prior_guidance_scale),
        "decoder_steps": int(args.decoder_steps),
        "decoder_guidance_scale": float(args.decoder_guidance_scale),
        "grid_path": str(grid_path),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[decode_latent_prompt_gold_reference] grid {grid_path}", flush=True)


if __name__ == "__main__":
    main()
