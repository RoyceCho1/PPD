from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from PIL import Image, ImageDraw


DEFAULT_DECODER_MODEL_ID = "stabilityai/stable-cascade"


def _load_json(path: Path) -> Any:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_samples(path: Path) -> List[Dict[str, str]]:
    payload = _load_json(path)
    if isinstance(payload, Mapping):
        raw_samples = payload.get("samples")
        if raw_samples is None and "uids" in payload:
            raw_samples = [{"uid": str(uid), "category": "sample"} for uid in payload["uids"]]
    else:
        raw_samples = payload
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


def _load_latent(path: Path, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    loaded = torch.load(path, map_location="cpu")
    if not torch.is_tensor(loaded):
        raise TypeError(f"Expected tensor latent at {path}, got {type(loaded)}")
    if loaded.ndim == 3:
        loaded = loaded.unsqueeze(0)
    return loaded.to(device=device, dtype=dtype)


def _fit_tile(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _draw_label(tile: Image.Image, text: str, subtext: str = "") -> Image.Image:
    label_h = 44
    output = Image.new("RGB", (tile.width, tile.height + label_h), "white")
    output.paste(tile, (0, label_h))
    draw = ImageDraw.Draw(output)
    draw.text((6, 5), text[:42], fill=(0, 0, 0))
    if subtext:
        draw.text((6, 23), subtext[:42], fill=(60, 60, 60))
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
        raise RuntimeError(f"Expected one decoded image, got {type(images)} length={len(images) if isinstance(images, list) else 'n/a'}")
    return images[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode 24x24/12x12 latents and build original/24/12 comparison grids.")
    parser.add_argument("--samples-json", type=Path, required=True)
    parser.add_argument("--uid-to-path-json", type=Path, required=True)
    parser.add_argument("--uid-to-meta-json", type=Path, default=None)
    parser.add_argument("--latent-root-24", type=Path, required=True)
    parser.add_argument("--latent-root-12", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decoder-model-id", type=str, default=DEFAULT_DECODER_MODEL_ID)
    parser.add_argument("--prompt", type=str, default="a high quality image")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decoder-steps", type=int, default=20)
    parser.add_argument("--tile-size", type=int, default=256)
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
    full_dir.mkdir(parents=True, exist_ok=True)
    row_dir.mkdir(parents=True, exist_ok=True)

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
        shard = uid[:2]
        latent_24_path = args.latent_root_24.expanduser().resolve() / shard / f"{uid}.pt"
        latent_12_path = args.latent_root_12.expanduser().resolve() / shard / f"{uid}.pt"
        original_path = Path(uid_to_path[uid]).expanduser().resolve()
        if not latent_24_path.exists() or not latent_12_path.exists() or not original_path.exists():
            raise FileNotFoundError(
                f"Missing input for uid={uid}: "
                f"24={latent_24_path.exists()} 12={latent_12_path.exists()} original={original_path.exists()}"
            )

        original = Image.open(original_path).convert("RGB")
        decoded: Dict[str, Tuple[Image.Image, Path, Tuple[int, ...]]] = {}
        for label, latent_path in (("24x24", latent_24_path), ("12x12", latent_12_path)):
            latent = _load_latent(latent_path, dtype=decoder_dtype, device=device)
            image = _decode_one(
                pipe=pipe,
                latent=latent,
                prompt=args.prompt,
                seed=int(args.seed),
                steps=int(args.decoder_steps),
                device=device,
            )
            decode_path = full_dir / f"{uid}_{label}.png"
            image.save(decode_path)
            Image.open(decode_path).verify()
            decoded[label] = (image, decode_path, tuple(int(dim) for dim in latent.shape))
            del latent
            if device.type == "cuda":
                torch.cuda.empty_cache()

        tiles = [
            _draw_label(_fit_tile(original, tile_size), f"{category} original", uid[:8]),
            _draw_label(_fit_tile(decoded["24x24"][0], tile_size), "decode 24x24", str(decoded["24x24"][0].size)),
            _draw_label(_fit_tile(decoded["12x12"][0], tile_size), "decode 12x12", str(decoded["12x12"][0].size)),
        ]
        row = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), "white")
        x = 0
        for tile in tiles:
            row.paste(tile, (x, 0))
            x += tile.width
        row_path = row_dir / f"{uid}_comparison.png"
        row.save(row_path)
        rows.append(row)

        caption = " | ".join(str(item) for item in (uid_to_meta.get(uid, {}).get("caption_samples") or []))
        records.append(
            {
                "category": category,
                "uid": uid,
                "caption": caption,
                "original_path": str(original_path),
                "latent_24x24_path": str(latent_24_path),
                "latent_12x12_path": str(latent_12_path),
                "decode_24x24_path": str(decoded["24x24"][1]),
                "decode_12x12_path": str(decoded["12x12"][1]),
                "row_path": str(row_path),
                "latent_24x24_shape": list(decoded["24x24"][2]),
                "latent_12x12_shape": list(decoded["12x12"][2]),
                "decode_24x24_size": list(decoded["24x24"][0].size),
                "decode_12x12_size": list(decoded["12x12"][0].size),
            }
        )
        print(f"[decode_latent_comparison_grid] done {category} {uid}", flush=True)

    pad = 8
    grid_w = max(row.width for row in rows)
    grid_h = sum(row.height for row in rows) + pad * max(0, len(rows) - 1)
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    y = 0
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height + pad
    grid_path = output_dir / "comparison_grid_original_24x24_12x12.png"
    grid.save(grid_path)
    summary = {
        "prompt": args.prompt,
        "seed": int(args.seed),
        "decoder_steps": int(args.decoder_steps),
        "grid_path": str(grid_path),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[decode_latent_comparison_grid] grid {grid_path}", flush=True)


if __name__ == "__main__":
    main()
