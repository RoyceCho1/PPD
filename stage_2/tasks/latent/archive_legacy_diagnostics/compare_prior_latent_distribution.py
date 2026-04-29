from __future__ import annotations

"""Compare vanilla Stable Cascade prior embeddings against precomputed latents.

This is a diagnostic script for checking whether latents produced by
`image_to_latents.py` live in roughly the same value distribution as vanilla
Stable Cascade prior `image_embeddings`.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


DEFAULT_PRIOR_MODEL_ID = "stabilityai/stable-cascade-prior"
DEFAULT_NEGATIVE_PROMPT = ""


def _torch_load(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_json(path: Path) -> Any:
    with path.expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _extract_prior_embeddings(output: Any) -> Tensor:
    image_embeddings = getattr(output, "image_embeddings", None)
    if torch.is_tensor(image_embeddings):
        return image_embeddings
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Could not extract image_embeddings from prior output type {type(output)}.")


def _load_samples(samples_json: Optional[Path], latent_root: Path, max_latents: int) -> List[Dict[str, str]]:
    if samples_json is not None:
        raw = _load_json(samples_json)
        records = raw.get("samples", raw) if isinstance(raw, Mapping) else raw
        if not isinstance(records, list):
            raise TypeError(f"--samples-json must contain a list or {{'samples': list}}, got {type(records)}")
        samples: List[Dict[str, str]] = []
        for item in records:
            if not isinstance(item, Mapping) or "uid" not in item:
                raise ValueError(f"Each sample must contain uid, got: {item!r}")
            samples.append(
                {
                    "uid": str(item["uid"]),
                    "category": str(item.get("category", "")),
                }
            )
        return samples

    paths = sorted(latent_root.expanduser().rglob("*.pt"))[: int(max_latents)]
    return [{"uid": path.stem, "category": ""} for path in paths]


def _caption_for_uid(uid: str, uid_to_meta: Mapping[str, Any], fallback_prompt: str) -> str:
    meta = uid_to_meta.get(uid, {})
    if isinstance(meta, Mapping):
        captions = meta.get("caption_samples")
        if isinstance(captions, list) and captions:
            return str(captions[0])
        caption = meta.get("caption")
        if caption:
            return str(caption)
    return fallback_prompt


def _latent_path_for_uid(latent_root: Path, uid: str) -> Path:
    return latent_root.expanduser().resolve() / uid[:2] / f"{uid}.pt"


def _as_bchw(tensor: Tensor) -> Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] or [C,H,W], got shape={tuple(tensor.shape)}")
    return tensor.detach().float().cpu()


def _finite_flat(flat: Tensor) -> Tensor:
    if flat.is_floating_point() or flat.is_complex():
        return flat[torch.isfinite(flat)]
    return flat


def _quantiles(flat: Tensor) -> Dict[str, float]:
    flat = _finite_flat(flat)
    if flat.numel() == 0:
        names = ("p0", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p100")
        return {name: float("nan") for name in names}
    qs = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0], dtype=torch.float32)
    values = torch.quantile(flat, qs)
    names = ("p0", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p100")
    return {name: float(value.item()) for name, value in zip(names, values)}


def _tensor_stats(tensor: Tensor) -> Dict[str, Any]:
    x = _as_bchw(tensor)
    flat = x.flatten()
    finite_mask = torch.isfinite(flat)
    finite_flat = flat[finite_mask]
    finite_count = int(finite_flat.numel())
    nan_count = int(torch.isnan(flat).sum().item())
    posinf_count = int(torch.isposinf(flat).sum().item())
    neginf_count = int(torch.isneginf(flat).sum().item())
    finite_fraction = float(finite_count / int(flat.numel())) if flat.numel() else 0.0
    safe_flat = finite_flat if finite_count > 0 else flat.new_tensor([float("nan")])
    safe_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    channel_mean = safe_x.mean(dim=(0, 2, 3))
    channel_std = safe_x.std(dim=(0, 2, 3), unbiased=False)
    spatial_mean = safe_x.mean(dim=(0, 1))
    spatial_std = safe_x.std(dim=(0, 1), unbiased=False)
    per_sample_l2 = torch.linalg.vector_norm(safe_x.flatten(start_dim=1), dim=1)
    stats: Dict[str, Any] = {
        "shape": list(x.shape),
        "numel": int(flat.numel()),
        "finite_count": finite_count,
        "finite_fraction": finite_fraction,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "mean": float(safe_flat.mean().item()),
        "std": float(safe_flat.std(unbiased=False).item()),
        "min": float(safe_flat.min().item()),
        "max": float(safe_flat.max().item()),
        "abs_mean": float(safe_flat.abs().mean().item()),
        "rms": float(torch.sqrt(torch.mean(safe_flat * safe_flat)).item()),
        "l2_norm": float(torch.linalg.vector_norm(safe_flat).item()),
        "per_sample_l2_norm": [float(v.item()) for v in per_sample_l2],
        "quantiles": _quantiles(flat),
        "channel_mean": [float(v.item()) for v in channel_mean],
        "channel_std": [float(v.item()) for v in channel_std],
        "channel_mean_summary": {
            "mean": float(channel_mean.mean().item()),
            "std": float(channel_mean.std(unbiased=False).item()),
            "min": float(channel_mean.min().item()),
            "max": float(channel_mean.max().item()),
        },
        "channel_std_summary": {
            "mean": float(channel_std.mean().item()),
            "std": float(channel_std.std(unbiased=False).item()),
            "min": float(channel_std.min().item()),
            "max": float(channel_std.max().item()),
        },
        "spatial_mean_summary": {
            "shape": list(spatial_mean.shape),
            "mean": float(spatial_mean.mean().item()),
            "std": float(spatial_mean.std(unbiased=False).item()),
            "min": float(spatial_mean.min().item()),
            "max": float(spatial_mean.max().item()),
        },
        "spatial_std_summary": {
            "shape": list(spatial_std.shape),
            "mean": float(spatial_std.mean().item()),
            "std": float(spatial_std.std(unbiased=False).item()),
            "min": float(spatial_std.min().item()),
            "max": float(spatial_std.max().item()),
        },
    }
    return stats


def _concat_flat(tensors: Sequence[Tensor]) -> Tensor:
    if not tensors:
        raise ValueError("No tensors to concatenate.")
    return torch.cat([_as_bchw(t).flatten() for t in tensors], dim=0)


def _aggregate_stats(tensors: Sequence[Tensor]) -> Dict[str, Any]:
    stacked = torch.cat([_as_bchw(t) for t in tensors], dim=0)
    return _tensor_stats(stacked)


def _histogram(flat: Tensor, bin_edges: Tensor) -> List[int]:
    flat = _finite_flat(flat)
    counts = torch.histc(flat, bins=int(bin_edges.numel() - 1), min=float(bin_edges[0]), max=float(bin_edges[-1]))
    return [int(v.item()) for v in counts]


def _histogram_payload(prior_flat: Tensor, latent_flat: Tensor, bins: int) -> Dict[str, Any]:
    prior_flat = _finite_flat(prior_flat)
    latent_flat = _finite_flat(latent_flat)
    if prior_flat.numel() == 0 or latent_flat.numel() == 0:
        raise RuntimeError(
            "Cannot build histogram because one side has no finite values: "
            f"prior_finite={prior_flat.numel()}, precomputed_finite={latent_flat.numel()}"
        )
    low = float(min(prior_flat.min().item(), latent_flat.min().item()))
    high = float(max(prior_flat.max().item(), latent_flat.max().item()))
    if math.isclose(low, high):
        low -= 0.5
        high += 0.5
    edges = torch.linspace(low, high, int(bins) + 1)
    return {
        "bin_edges": [float(v.item()) for v in edges],
        "prior_counts": _histogram(prior_flat, edges),
        "precomputed_counts": _histogram(latent_flat, edges),
    }


def _write_sample_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "uid",
        "category",
        "prompt",
        "seed",
        "shape",
        "finite_fraction",
        "nan_count",
        "posinf_count",
        "neginf_count",
        "mean",
        "std",
        "min",
        "max",
        "abs_mean",
        "rms",
        "l2_norm",
        "p1",
        "p5",
        "p50",
        "p95",
        "p99",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            stats = row["stats"]
            q = stats["quantiles"]
            writer.writerow(
                {
                    "group": row["group"],
                    "uid": row.get("uid", ""),
                    "category": row.get("category", ""),
                    "prompt": row.get("prompt", ""),
                    "seed": row.get("seed", ""),
                    "shape": "x".join(str(v) for v in stats["shape"]),
                    "finite_fraction": stats["finite_fraction"],
                    "nan_count": stats["nan_count"],
                    "posinf_count": stats["posinf_count"],
                    "neginf_count": stats["neginf_count"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "abs_mean": stats["abs_mean"],
                    "rms": stats["rms"],
                    "l2_norm": stats["l2_norm"],
                    "p1": q["p1"],
                    "p5": q["p5"],
                    "p50": q["p50"],
                    "p95": q["p95"],
                    "p99": q["p99"],
                }
            )


def _write_histogram_csv(path: Path, histogram: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = histogram["bin_edges"]
    prior_counts = histogram["prior_counts"]
    latent_counts = histogram["precomputed_counts"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bin_left", "bin_right", "prior_count", "precomputed_count"])
        writer.writeheader()
        for index in range(len(prior_counts)):
            writer.writerow(
                {
                    "bin_left": edges[index],
                    "bin_right": edges[index + 1],
                    "prior_count": prior_counts[index],
                    "precomputed_count": latent_counts[index],
                }
            )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-json", type=Path)
    parser.add_argument("--uid-to-meta-json", type=Path)
    parser.add_argument("--max-latents", type=int, default=8)
    parser.add_argument("--fallback-prompt", type=str, default="a high quality image")
    parser.add_argument("--prior-model-id", type=str, default=DEFAULT_PRIOR_MODEL_ID)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--prior-steps", type=int, default=20)
    parser.add_argument("--prior-guidance-scale", type=float, default=4.0)
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--histogram-bins", type=int, default=80)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _resolve_dtype(name: str, device: torch.device) -> Optional[torch.dtype]:
    if name == "auto":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return None


def main() -> None:
    args = parse_args()
    from diffusers import StableCascadePriorPipeline

    seeds = list(args.seeds or [0])
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(args.samples_json, args.latent_root, args.max_latents)
    uid_to_meta: Dict[str, Any] = {}
    if args.uid_to_meta_json is not None:
        uid_to_meta = {str(k): v for k, v in _load_json(args.uid_to_meta_json).items()}

    torch_dtype = _resolve_dtype(args.torch_dtype, device)
    pipe_kwargs: Dict[str, Any] = {
        "local_files_only": bool(args.local_files_only),
    }
    if torch_dtype is not None:
        pipe_kwargs["torch_dtype"] = torch_dtype
    pipe = StableCascadePriorPipeline.from_pretrained(args.prior_model_id, **pipe_kwargs)
    pipe.to(device)
    _sync_text_encoder_dtype(pipe, device)
    if getattr(pipe, "prior", None) is not None:
        pipe.prior.eval()
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.eval()

    prior_tensors: List[Tensor] = []
    latent_tensors: List[Tensor] = []
    sample_rows: List[Dict[str, Any]] = []

    for sample in samples:
        uid = sample["uid"]
        category = sample.get("category", "")
        prompt = _caption_for_uid(uid, uid_to_meta, args.fallback_prompt)
        latent_path = _latent_path_for_uid(args.latent_root, uid)
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing precomputed latent for uid={uid}: {latent_path}")
        latent = _as_bchw(_torch_load(latent_path, map_location="cpu"))
        latent_tensors.append(latent)
        sample_rows.append(
            {
                "group": "precomputed_24x24",
                "uid": uid,
                "category": category,
                "prompt": prompt,
                "seed": "",
                "latent_path": str(latent_path),
                "stats": _tensor_stats(latent),
            }
        )

        for seed in seeds:
            generator_device = device if device.type == "cuda" else torch.device("cpu")
            generator = torch.Generator(device=generator_device).manual_seed(int(seed))
            with torch.inference_mode():
                output = pipe(
                    prompt=prompt,
                    height=int(args.height),
                    width=int(args.width),
                    num_inference_steps=int(args.prior_steps),
                    guidance_scale=float(args.prior_guidance_scale),
                    negative_prompt=args.negative_prompt or None,
                    num_images_per_prompt=1,
                    generator=generator,
                    output_type="pt",
                    return_dict=True,
                )
            embeddings = _as_bchw(_extract_prior_embeddings(output))
            prior_tensors.append(embeddings)
            sample_rows.append(
                {
                    "group": "vanilla_prior",
                    "uid": uid,
                    "category": category,
                    "prompt": prompt,
                    "seed": int(seed),
                    "stats": _tensor_stats(embeddings),
                }
            )
            print(
                "[compare_prior_latent_distribution] "
                f"uid={uid} seed={seed} prior_shape={tuple(embeddings.shape)} "
                f"latent_shape={tuple(latent.shape)}"
            )

    prior_flat = _concat_flat(prior_tensors)
    latent_flat = _concat_flat(latent_tensors)
    histogram = _histogram_payload(prior_flat, latent_flat, args.histogram_bins)
    summary: Dict[str, Any] = {
        "config": {
            "prior_model_id": args.prior_model_id,
            "height": int(args.height),
            "width": int(args.width),
            "prior_steps": int(args.prior_steps),
            "prior_guidance_scale": float(args.prior_guidance_scale),
            "negative_prompt": args.negative_prompt,
            "seeds": seeds,
            "latent_root": str(args.latent_root.expanduser().resolve()),
            "torch_dtype": args.torch_dtype,
        },
        "counts": {
            "samples": len(samples),
            "prior_tensors": len(prior_tensors),
            "precomputed_tensors": len(latent_tensors),
        },
        "aggregate": {
            "vanilla_prior": _aggregate_stats(prior_tensors),
            "precomputed_24x24": _aggregate_stats(latent_tensors),
        },
        "histogram": histogram,
        "records": sample_rows,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_sample_csv(output_dir / "sample_stats.csv", sample_rows)
    _write_histogram_csv(output_dir / "histogram.csv", histogram)
    print(f"[compare_prior_latent_distribution] wrote {output_dir / 'summary.json'}")
    print(f"[compare_prior_latent_distribution] wrote {output_dir / 'sample_stats.csv'}")
    print(f"[compare_prior_latent_distribution] wrote {output_dir / 'histogram.csv'}")


if __name__ == "__main__":
    main()
