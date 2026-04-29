from __future__ import annotations

"""Debug Stable Cascade raw-image representation and decoder-ready embeddings.

This script separates two official Stable Cascade paths:

1. Raw image -> CLIP image embedding used by StableCascadePriorPipeline as
   image-conditioning. This is an official internal image representation, but
   it is not a decoder input.
2. Raw image + prompt -> StableCascadePriorPipeline output image_embeddings.
   This is decoder-ready and can be compared against precomputed 24x24 latents.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from PIL import Image, ImageDraw


DEFAULT_PRIOR_MODEL_ID = "stabilityai/stable-cascade-prior"
DEFAULT_DECODER_MODEL_ID = "stabilityai/stable-cascade"


SOURCE_AUDIT = {
    "questions": [
        "Stable Cascade에서 raw image를 semantic representation으로 바꾸는 공식 내부 경로는 무엇인가",
        "그 representation은 decoder input과 직접 연결되는가",
        "prior training target과 decoder input은 정말 같은 공간인가",
        "512/768 이미지는 내부에서 어떻게 전처리되는가",
    ],
    "answers": {
        "official_raw_image_representation": (
            "StableCascadePriorPipeline.encode_image: "
            "feature_extractor(image).pixel_values -> image_encoder(image).image_embeds.unsqueeze(1)."
        ),
        "direct_decoder_input": (
            "No. The official raw-image representation is CLIP image embedding with shape [B,N,D]. "
            "Decoder expects image_embeddings/effnet with shape [B,16,H,W]."
        ),
        "prior_training_target_vs_decoder_input": (
            "They are intended to be the same decoder-ready image_embeddings space in the Wuerstchen prior "
            "training example: EfficientNetEncoder(effnet_images).add(1).div(42). "
            "In diffusers inference, the decoder receives prior_outputs.image_embeddings."
        ),
        "image_preprocessing": (
            "StableCascadePriorPipeline image-conditioning uses CLIPImageProcessor. "
            "Wuerstchen prior training target uses Resize(resolution), CenterCrop(resolution), ToTensor, "
            "ImageNet Normalize. The local precompute script uses the Wuerstchen target-style transform."
        ),
    },
    "local_source_references": {
        "prior_encode_image": (
            "/data/roycecho/miniconda3/envs/ppd_stage2/lib/python3.10/site-packages/"
            "diffusers/pipelines/stable_cascade/pipeline_stable_cascade_prior.py:262"
        ),
        "prior_output_image_embeddings": (
            "/data/roycecho/miniconda3/envs/ppd_stage2/lib/python3.10/site-packages/"
            "diffusers/pipelines/stable_cascade/pipeline_stable_cascade_prior.py:646"
        ),
        "decoder_effnet_assignment": (
            "/data/roycecho/miniconda3/envs/ppd_stage2/lib/python3.10/site-packages/"
            "diffusers/pipelines/stable_cascade/pipeline_stable_cascade.py:443"
        ),
        "wuerstchen_prior_training_target": (
            "/data/roycecho/diffusers/examples/research_projects/wuerstchen/text_to_image/"
            "train_text_to_image_prior.py:802"
        ),
    },
}


def _load_json(path: Path) -> Any:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_samples(path: Optional[Path], image_path: Optional[Path], uid: Optional[str]) -> List[Dict[str, str]]:
    if image_path is not None:
        return [{"uid": uid or image_path.stem, "category": "image", "image_path": str(image_path)}]
    if path is None:
        raise ValueError("Provide --samples-json or --image-path.")
    payload = _load_json(path)
    raw_samples = payload.get("samples") if isinstance(payload, Mapping) else payload
    if raw_samples is None and isinstance(payload, Mapping) and "uids" in payload:
        raw_samples = [{"uid": str(item), "category": "sample"} for item in payload["uids"]]
    if not isinstance(raw_samples, list):
        raise ValueError("Sample JSON must be a list or an object with `samples`/`uids`.")
    samples: List[Dict[str, str]] = []
    for item in raw_samples:
        if isinstance(item, str):
            samples.append({"uid": item, "category": "sample"})
        elif isinstance(item, Mapping):
            sample_uid = str(item.get("uid", "")).strip()
            if not sample_uid:
                raise ValueError(f"Sample item is missing uid: {item}")
            sample = {"uid": sample_uid, "category": str(item.get("category", "sample"))}
            if "image_path" in item:
                sample["image_path"] = str(item["image_path"])
            samples.append(sample)
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


def _sync_image_encoder_dtype(pipe: Any, device: torch.device) -> None:
    image_encoder = getattr(pipe, "image_encoder", None)
    if image_encoder is None:
        return
    to = getattr(image_encoder, "to", None)
    if callable(to):
        to(device=device, dtype=_pipeline_dtype(pipe))


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    x = tensor.detach().float().cpu()
    return {
        "shape": [int(dim) for dim in x.shape],
        "numel": int(x.numel()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "l2_norm": float(torch.linalg.vector_norm(x.flatten()).item()),
    }


def _compare_same_shape(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    payload: Dict[str, Any] = {"same_shape": list(a.shape) == list(b.shape)}
    if list(a.shape) != list(b.shape):
        payload["a_shape"] = list(a.shape)
        payload["b_shape"] = list(b.shape)
        return payload
    diff = a - b
    payload.update(
        {
            "l2_distance": float(torch.linalg.vector_norm(diff.flatten()).item()),
            "mae": float(diff.abs().mean().item()),
            "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
            "cosine": float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()),
        }
    )
    return payload


def _extract_clip_image_embedding(prior_pipe: Any, image: Image.Image, device: torch.device) -> torch.Tensor:
    dtype = _pipeline_dtype(prior_pipe)
    if hasattr(prior_pipe, "encode_image"):
        with torch.inference_mode():
            embeds, _ = prior_pipe.encode_image(
                images=[image],
                device=device,
                dtype=dtype,
                batch_size=1,
                num_images_per_prompt=1,
            )
        return embeds.detach().float().cpu()
    raise RuntimeError("StableCascadePriorPipeline does not expose encode_image.")


def _run_prior(
    prior_pipe: Any,
    *,
    prompt: str,
    image: Optional[Image.Image],
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "height": int(height),
        "width": int(width),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "negative_prompt": None,
        "num_images_per_prompt": 1,
        "generator": generator,
        "output_type": "pt",
        "return_dict": True,
    }
    if image is not None:
        kwargs["images"] = [image]
    with torch.inference_mode():
        output = prior_pipe(**kwargs)
    return _extract_prior_embeddings(output).detach().float().cpu()


def _decode_one(
    decoder_pipe: Any,
    image_embeddings: torch.Tensor,
    *,
    prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    device: torch.device,
) -> Image.Image:
    dtype = _pipeline_dtype(decoder_pipe)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.inference_mode():
        output = decoder_pipe(
            image_embeddings=image_embeddings.to(device=device, dtype=dtype),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-json", type=Path)
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--uid", type=str)
    parser.add_argument("--uid-to-path-json", type=Path)
    parser.add_argument("--uid-to-meta-json", type=Path)
    parser.add_argument("--latent-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-model-id", type=str, default=DEFAULT_PRIOR_MODEL_ID)
    parser.add_argument("--decoder-model-id", type=str, default=DEFAULT_DECODER_MODEL_ID)
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
    samples = _load_samples(args.samples_json, args.image_path, args.uid)
    uid_to_path: Dict[str, str] = {}
    if args.uid_to_path_json is not None:
        uid_to_path = {str(key): str(value) for key, value in _load_json(args.uid_to_path_json).items()}
    uid_to_meta: Dict[str, Any] = {}
    if args.uid_to_meta_json is not None:
        uid_to_meta = {str(key): value for key, value in _load_json(args.uid_to_meta_json).items()}

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    full_dir = output_dir / "full_decodes"
    row_dir = output_dir / "rows"
    tensor_dir = output_dir / "embeddings"
    full_dir.mkdir(parents=True, exist_ok=True)
    row_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    prior_pipe = StableCascadePriorPipeline.from_pretrained(args.prior_model_id, local_files_only=True)
    prior_pipe.to(device)
    _sync_text_encoder_dtype(prior_pipe, device)
    _sync_image_encoder_dtype(prior_pipe, device)
    if getattr(prior_pipe, "prior", None) is not None:
        prior_pipe.prior.eval()
    if getattr(prior_pipe, "text_encoder", None) is not None:
        prior_pipe.text_encoder.eval()
    if getattr(prior_pipe, "image_encoder", None) is not None:
        prior_pipe.image_encoder.eval()

    decoder_pipe = StableCascadeDecoderPipeline.from_pretrained(
        args.decoder_model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
    )
    decoder_pipe.to(device)

    tile_size = (int(args.tile_size), int(args.tile_size))
    rows: List[Image.Image] = []
    records: List[Dict[str, Any]] = []
    for sample in samples:
        uid = sample["uid"]
        category = sample.get("category", "sample")
        image_path_str = sample.get("image_path") or uid_to_path.get(uid)
        if image_path_str is None:
            raise ValueError(f"No image path found for uid={uid}. Provide --image-path or --uid-to-path-json.")
        image_path = Path(image_path_str).expanduser().resolve()
        image = Image.open(image_path).convert("RGB")
        caption = _caption_for_uid(uid, uid_to_meta, args.caption_fallback)

        precomputed: Optional[torch.Tensor] = None
        latent_path: Optional[Path] = None
        if args.latent_root is not None:
            candidate = args.latent_root.expanduser().resolve() / uid[:2] / f"{uid}.pt"
            if candidate.exists():
                latent_path = candidate
                precomputed = _load_latent(candidate)

        clip_image_embedding = _extract_clip_image_embedding(prior_pipe, image, device)
        image_conditioned_prior = _run_prior(
            prior_pipe,
            prompt=caption,
            image=image,
            height=int(args.height),
            width=int(args.width),
            steps=int(args.prior_steps),
            guidance_scale=float(args.prior_guidance_scale),
            seed=int(args.seed),
            device=device,
        )
        text_only_prior = _run_prior(
            prior_pipe,
            prompt=caption,
            image=None,
            height=int(args.height),
            width=int(args.width),
            steps=int(args.prior_steps),
            guidance_scale=float(args.prior_guidance_scale),
            seed=int(args.seed),
            device=device,
        )

        torch.save(clip_image_embedding, tensor_dir / f"{uid}_official_clip_image_embedding.pt")
        torch.save(image_conditioned_prior, tensor_dir / f"{uid}_official_image_conditioned_prior_embedding.pt")
        torch.save(text_only_prior, tensor_dir / f"{uid}_text_only_prior_embedding.pt")

        tiles = [_draw_label(_fit_tile(image, tile_size), f"{category} original", uid[:8])]
        record: Dict[str, Any] = {
            "category": category,
            "uid": uid,
            "caption": caption,
            "image_path": str(image_path),
            "latent_path": None if latent_path is None else str(latent_path),
            "official_clip_image_embedding_path": str(tensor_dir / f"{uid}_official_clip_image_embedding.pt"),
            "official_image_conditioned_prior_embedding_path": str(
                tensor_dir / f"{uid}_official_image_conditioned_prior_embedding.pt"
            ),
            "text_only_prior_embedding_path": str(tensor_dir / f"{uid}_text_only_prior_embedding.pt"),
            "official_clip_image_embedding_stats": _tensor_stats(clip_image_embedding),
            "official_image_conditioned_prior_stats": _tensor_stats(image_conditioned_prior),
            "text_only_prior_stats": _tensor_stats(text_only_prior),
            "comparisons": {
                "clip_image_embedding_vs_precomputed": {
                    "directly_comparable": False,
                    "reason": "CLIP image embedding is prior image-conditioning [B,N,D], not decoder-ready [B,16,H,W].",
                },
                "image_conditioned_prior_vs_text_only_prior": _compare_same_shape(
                    image_conditioned_prior, text_only_prior
                ),
            },
            "decodes": {},
        }

        decode_variants: List[Tuple[str, str, torch.Tensor]] = [
            ("official_image_conditioned_prior", "official img-cond prior + caption", image_conditioned_prior),
            ("text_only_prior", "text-only prior + caption", text_only_prior),
        ]
        if precomputed is not None:
            record["precomputed_stats"] = _tensor_stats(precomputed)
            record["comparisons"]["image_conditioned_prior_vs_precomputed"] = _compare_same_shape(
                image_conditioned_prior, precomputed
            )
            record["comparisons"]["text_only_prior_vs_precomputed"] = _compare_same_shape(text_only_prior, precomputed)
            decode_variants.insert(0, ("precomputed", "precomputed 24x24 + caption", precomputed))
        else:
            record["precomputed_stats"] = None
            record["comparisons"]["image_conditioned_prior_vs_precomputed"] = {
                "available": False,
                "reason": "No matching precomputed latent found.",
            }

        for variant_name, display_name, embedding in decode_variants:
            decoded = _decode_one(
                decoder_pipe,
                embedding,
                prompt=caption,
                seed=int(args.seed),
                steps=int(args.decoder_steps),
                guidance_scale=float(args.decoder_guidance_scale),
                device=device,
            )
            decode_path = full_dir / f"{uid}_{variant_name}.png"
            decoded.save(decode_path)
            Image.open(decode_path).verify()
            record["decodes"][variant_name] = {
                "display_name": display_name,
                "decode_path": str(decode_path),
                "decode_size": list(decoded.size),
            }
            tiles.append(_draw_label(_fit_tile(decoded, tile_size), display_name, caption))

        row = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), "white")
        x = 0
        for tile in tiles:
            row.paste(tile, (x, 0))
            x += tile.width
        row_path = row_dir / f"{uid}_official_image_path_debug.png"
        row.save(row_path)
        record["row_path"] = str(row_path)
        records.append(record)
        rows.append(row)
        print(f"[debug_stable_cascade_image_embedding_path] done {category} {uid}", flush=True)

    pad = 8
    grid_w = max(row.width for row in rows)
    grid_h = sum(row.height for row in rows) + pad * max(0, len(rows) - 1)
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    y = 0
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height + pad
    grid_path = output_dir / "official_image_path_debug_grid.png"
    grid.save(grid_path)

    summary = {
        "source_audit": SOURCE_AUDIT,
        "config": {
            "prior_model_id": args.prior_model_id,
            "decoder_model_id": args.decoder_model_id,
            "height": int(args.height),
            "width": int(args.width),
            "prior_steps": int(args.prior_steps),
            "prior_guidance_scale": float(args.prior_guidance_scale),
            "decoder_steps": int(args.decoder_steps),
            "decoder_guidance_scale": float(args.decoder_guidance_scale),
            "seed": int(args.seed),
        },
        "grid_path": str(grid_path),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[debug_stable_cascade_image_embedding_path] grid {grid_path}", flush=True)


if __name__ == "__main__":
    main()
