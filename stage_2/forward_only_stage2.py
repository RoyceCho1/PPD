from __future__ import annotations

"""
Forward-only Stage 2 integration smoke test.

This script verifies that an assignment + latent-manifest backed
Stage2PreferenceDataset batch can be connected to a patched Stable Cascade Stage
C prior with text conditioning and user conditioning. It intentionally does not
create a training loop, DPO loss, noisy samples, reference model, or optimizer.
"""

import argparse
import contextlib
import importlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

try:
    from stage2_dataset import Stage2PreferenceDataset
    from patch_stage_c import (
        PatchedSDCascadeAttnBlock,
        freeze_stage_c_except_user_modules,
        patch_stage_c_with_user_adapter,
        summarize_trainable_parameters,
    )
except ImportError:  # pragma: no cover - useful when imported as package module
    from stage_2.stage2_dataset import Stage2PreferenceDataset
    from stage_2.patch_stage_c import (
        PatchedSDCascadeAttnBlock,
        freeze_stage_c_except_user_modules,
        patch_stage_c_with_user_adapter,
        summarize_trainable_parameters,
    )


DEFAULT_PATCH_PATHS = ("down_blocks.0.2", "down_blocks.0.5")
EXPECTED_USER_EMB_DIM = 3584
EXPECTED_LATENT_SHAPE = (16, 12, 12)
DEFAULT_EMBEDDING_JSON_PATH = Path("data/user_emb_7b_full/train_shard0.json")
DEFAULT_ASSIGNMENT_JSONL_PATH = Path(
    "artifacts/pair_assignments/train/stage2_pair_assignments_train_shard0.jsonl"
)
DEFAULT_LATENT_MANIFEST_JSONL_PATH = Path("artifacts/stage_c_latents/latent_manifest_train_v512.jsonl")
DEFAULT_UID_TO_PATH_JSON_PATH = Path("data/train_uid_to_path.json")


@dataclass
class TextConditioning:
    """Text conditioning tensors for Stable Cascade prior."""

    clip_text: Tensor
    clip_text_pooled: Tensor


@dataclass
class ForwardCaseResult:
    """Summary for one forward-only test case."""

    name: str
    caption_batch_size: int
    clip_text_shape: Tuple[int, ...]
    clip_text_pooled_shape: Tuple[int, ...]
    clip_img_shape: Optional[Tuple[int, ...]]
    user_emb_shape: Optional[Tuple[int, ...]]
    user_mask_shape: Optional[Tuple[int, ...]]
    sample_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    dtype: str
    device: str
    has_nan: bool
    has_inf: bool
    all_finite: bool


@dataclass
class TensorDiagnostics:
    """Compact diagnostics for tensor shape and numerical health."""

    name: str
    shape: Tuple[int, ...]
    dtype: str
    device: str
    min_value: Optional[float]
    max_value: Optional[float]
    mean_value: Optional[float]
    std_value: Optional[float]
    has_nan: bool
    has_inf: bool
    all_finite: bool


def _safe_import_module(module_name: str) -> Tuple[Optional[Any], Optional[str]]:
    """Import an optional module and return a traceback string on failure."""

    try:
        return importlib.import_module(module_name), None
    except Exception:
        return None, traceback.format_exc()


def _resolve_device(device_arg: str) -> torch.device:
    """Resolve CLI device string into a torch device."""

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _resolve_torch_dtype(dtype_arg: str) -> Optional[torch.dtype]:
    # CLI dtype string을 torch dtype로 변환

    if dtype_arg == "auto":
        return None
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def _parse_patch_paths(raw_paths: Optional[Sequence[str]]) -> List[str]:

    if not raw_paths:
        return list(DEFAULT_PATCH_PATHS)

    paths: List[str] = []
    for raw in raw_paths:
        for item in raw.split(","):
            item = item.strip()
            if item:
                paths.append(item)
    return paths


def _finite_flags(tensor: Tensor) -> Tuple[bool, bool, bool]:
    """Return has_nan, has_inf, all_finite for a tensor."""

    if not tensor.is_floating_point() and not tensor.is_complex():
        return False, False, True
    has_nan = bool(torch.isnan(tensor).any().item())
    has_inf = bool(torch.isinf(tensor).any().item())
    all_finite = bool(torch.isfinite(tensor).all().item())
    return has_nan, has_inf, all_finite


def _tensor_diagnostics(name: str, tensor: Tensor) -> TensorDiagnostics:
    """Build stable tensor diagnostics without mutating the input tensor."""

    has_nan, has_inf, all_finite = _finite_flags(tensor)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    if tensor.numel() > 0 and tensor.is_floating_point() and all_finite:
        detached = tensor.detach().float().cpu()
        min_value = float(detached.min().item())
        max_value = float(detached.max().item())
        mean_value = float(detached.mean().item())
        std_value = float(detached.std(unbiased=False).item())

    return TensorDiagnostics(
        name=name,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        min_value=min_value,
        max_value=max_value,
        mean_value=mean_value,
        std_value=std_value,
        has_nan=has_nan,
        has_inf=has_inf,
        all_finite=all_finite,
    )


def _require_tensor(batch: Mapping[str, Any], key: str) -> Tensor:
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"Batch key `{key}` must be a Tensor after collation, got: {type(value)}")
    return value


def _validate_finite(name: str, tensor: Tensor) -> None:
    has_nan, has_inf, all_finite = _finite_flags(tensor)
    if not all_finite:
        raise ValueError(f"{name} contains non-finite values: has_nan={has_nan}, has_inf={has_inf}")


def _validate_batch_shapes(batch: Mapping[str, Any]) -> int:
    """Validate the real-latent smoke batch against Stage 1 completion criteria."""

    captions = batch.get("caption")
    if not isinstance(captions, list) or not captions:
        raise ValueError("Batch key `caption` must be a non-empty list.")
    batch_size = len(captions)

    user_emb = _require_tensor(batch, "user_emb")
    if user_emb.ndim != 3:
        raise ValueError(f"user_emb must have shape [B,L,{EXPECTED_USER_EMB_DIM}], got {tuple(user_emb.shape)}")
    if int(user_emb.shape[0]) != batch_size or int(user_emb.shape[2]) != EXPECTED_USER_EMB_DIM:
        raise ValueError(
            f"user_emb must have shape [B,L,{EXPECTED_USER_EMB_DIM}] with B={batch_size}, "
            f"got {tuple(user_emb.shape)}"
        )
    _validate_finite("user_emb", user_emb)

    user_mask = _require_tensor(batch, "user_emb_attention_mask")
    expected_mask_shape = (batch_size, int(user_emb.shape[1]))
    if tuple(user_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"user_emb_attention_mask must have shape {expected_mask_shape}, got {tuple(user_mask.shape)}"
        )

    for key in ("preferred_latent", "dispreferred_latent"):
        latent = _require_tensor(batch, key)
        expected_shape = (batch_size, *EXPECTED_LATENT_SHAPE)
        if tuple(latent.shape) != expected_shape:
            raise ValueError(f"{key} must have shape {expected_shape}, got {tuple(latent.shape)}")
        if not latent.is_floating_point():
            raise TypeError(f"{key} must be floating point, got dtype={latent.dtype}")
        _validate_finite(key, latent)

    return batch_size


def _load_dataset(args: argparse.Namespace) -> Stage2PreferenceDataset:
    # Stage 2 preference dataset 로드

    uid_to_path = (
        args.uid_to_path_json_path
        if args.uid_to_path_json_path and args.uid_to_path_json_path.exists()
        else None
    )
    assignment_path = args.assignment_jsonl_path
    latent_manifest_path = args.latent_manifest_jsonl_path

    if assignment_path is None:
        raise ValueError("--assignment-jsonl-path is required for real-latent forward smoke.")
    if latent_manifest_path is None:
        raise ValueError("--latent-manifest-jsonl-path is required for real-latent forward smoke.")
    if not assignment_path.exists():
        raise FileNotFoundError(f"--assignment-jsonl-path not found: {assignment_path}")
    if not latent_manifest_path.exists():
        raise FileNotFoundError(f"--latent-manifest-jsonl-path not found: {latent_manifest_path}")

    return Stage2PreferenceDataset(
        embedding_json_path=args.embedding_json_path,
        assignment_jsonl_path=assignment_path,
        latent_manifest_jsonl_path=latent_manifest_path,
        uid_to_path_json_path=uid_to_path,
        uid_to_meta_json_path=None,
        load_images=False,
        load_latents=True,
        skip_missing_latents=args.skip_missing_latents,
        validate_assignment_support_pairs=args.validate_assignment_support_pairs,
        skip_malformed_pairs=True,
    )


def _extract_real_sample_from_batch(
    batch: Mapping[str, Any],
    latent_source: str,
    prior: nn.Module,
    device: torch.device,
) -> Tensor:
    """Build prior sample from real latent tensors in one dataset batch."""

    if latent_source not in {"preferred", "dispreferred"}:
        raise ValueError(f"Unsupported latent_source={latent_source}. Expected preferred or dispreferred.")

    key = f"{latent_source}_latent"
    if key not in batch:
        raise ValueError(
            f"Batch does not contain `{key}`. Ensure real-latent mode is enabled "
            "(assignment + latent manifest + load_latents)."
        )

    sample = batch[key]
    if not isinstance(sample, Tensor):
        raise TypeError(f"Batch key `{key}` must be a Tensor after collation, got: {type(sample)}")
    expected_tail = EXPECTED_LATENT_SHAPE
    if sample.ndim != 4 or tuple(sample.shape[1:]) != expected_tail:
        raise ValueError(
            f"Real latent sample must have shape [B,{expected_tail[0]},{expected_tail[1]},{expected_tail[2]}], "
            f"got {tuple(sample.shape)}"
        )
    _validate_finite(key, sample)

    prior_dtype = _get_prior_dtype(prior)
    return sample.to(device=device, dtype=prior_dtype if sample.is_floating_point() else torch.float32)


def _get_one_batch(dataset: Stage2PreferenceDataset, batch_size: int) -> Mapping[str, Any]:
    # Stage 2 preference dataset에서 한 batch 로드

    if len(dataset) == 0:
        raise ValueError("Stage2PreferenceDataset produced zero valid samples.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=Stage2PreferenceDataset.collate_fn,
    )
    return next(iter(loader))


def _load_prior_pipeline(args: argparse.Namespace) -> Any:
    # Stable Cascade prior pipeline 로드

    diffusers, diffusers_error = _safe_import_module("diffusers")
    if diffusers is None:
        raise ImportError("Failed to import diffusers:\n" + str(diffusers_error))

    try:
        pipeline_cls = getattr(diffusers, "StableCascadePriorPipeline")
    except Exception as exc:
        raise ImportError(
            "diffusers failed while resolving StableCascadePriorPipeline. "
            "Check diffusers/transformers version compatibility."
        ) from exc

    kwargs: Dict[str, Any] = {
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    torch_dtype = _resolve_torch_dtype(args.torch_dtype)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    return pipeline_cls.from_pretrained(args.model_id, **kwargs)


def _move_module_if_possible(module: Any, device: torch.device) -> None:
    """Move a module to device when it exposes .to()."""

    to = getattr(module, "to", None)
    if callable(to):
        to(device)


def _tokenize_and_encode_text(pipe: Any, captions: Sequence[str], device: torch.device) -> TextConditioning:
    # caption convert to text-conditioning tensors

    tokenizer = getattr(pipe, "tokenizer", None)
    text_encoder = getattr(pipe, "text_encoder", None)
    if tokenizer is None or text_encoder is None:
        raise ValueError("Pipeline must expose tokenizer and text_encoder for text conditioning.")

    _move_module_if_possible(text_encoder, device)
    text_encoder.eval()

    max_length = getattr(tokenizer, "model_max_length", None)
    tokenized = tokenizer(
        list(captions),
        padding="max_length" if max_length else True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}

    with torch.no_grad():
        outputs = text_encoder(**tokenized, output_hidden_states=True, return_dict=True)

    clip_text = getattr(outputs, "last_hidden_state", None)
    if clip_text is None and hasattr(outputs, "hidden_states") and outputs.hidden_states:
        clip_text = outputs.hidden_states[-1]
    if clip_text is None:
        raise ValueError("Could not extract token-level clip_text from text_encoder output.")

    clip_text_pooled = getattr(outputs, "text_embeds", None)
    if clip_text_pooled is None:
        clip_text_pooled = getattr(outputs, "pooler_output", None)
    if clip_text_pooled is None:
        raise ValueError("Could not extract pooled/projected clip_text_pooled from text_encoder output.")
    if clip_text_pooled.ndim == 2:
        # StableCascadeUNet.get_clip_embeddings expects pooled text as a
        # sequence-like tensor [B, N_pool, D]. Passing [B, D] triggers an
        # invalid internal view in current diffusers versions.
        clip_text_pooled = clip_text_pooled.unsqueeze(1)

    return TextConditioning(
        clip_text=clip_text,
        clip_text_pooled=clip_text_pooled,
    )


def _load_batch_images(batch: Mapping[str, Any], image_source: str) -> List[Any]:
    """Load preferred or dispreferred batch images from dataset-resolved paths."""

    if image_source not in ("preferred", "dispreferred"):
        raise ValueError(f"Unsupported image_source={image_source}. Expected preferred or dispreferred.")

    path_key = f"{image_source}_path"
    if path_key not in batch:
        raise ValueError(
            f"Batch does not contain {path_key}. Pass --uid-to-path-json-path with a valid manifest "
            "and ensure the selected split has image paths."
        )

    pil_module, pil_error = _safe_import_module("PIL.Image")
    if pil_module is None:
        raise ImportError("Failed to import PIL.Image:\n" + str(pil_error))

    images: List[Any] = []
    missing_paths: List[str] = []
    for raw_path in batch[path_key]:
        if raw_path is None:
            missing_paths.append("<None>")
            continue
        path = Path(str(raw_path))
        if not path.exists():
            missing_paths.append(str(path))
            continue
        images.append(pil_module.open(path).convert("RGB"))

    if missing_paths:
        raise FileNotFoundError(
            f"Could not load {len(missing_paths)} {image_source} image path(s): "
            + ", ".join(missing_paths[:5])
        )
    if not images:
        raise ValueError(f"No {image_source} images were loaded from batch key {path_key}.")
    return images


def _encode_clip_images(pipe: Any, images: Sequence[Any], device: torch.device) -> Tensor:
    """Encode raw PIL images into Stable Cascade prior clip_img conditioning."""

    feature_extractor = getattr(pipe, "feature_extractor", None)
    image_encoder = getattr(pipe, "image_encoder", None)
    if feature_extractor is None or image_encoder is None:
        raise ValueError("Pipeline must expose feature_extractor and image_encoder for clip_img conditioning.")

    _move_module_if_possible(image_encoder, device)
    image_encoder.eval()

    encoded = feature_extractor(images=list(images), return_tensors="pt")
    pixel_values = encoded["pixel_values"].to(device)
    try:
        image_dtype = next(image_encoder.parameters()).dtype
        if pixel_values.is_floating_point():
            pixel_values = pixel_values.to(dtype=image_dtype)
    except StopIteration:
        pass

    with torch.no_grad():
        outputs = image_encoder(pixel_values=pixel_values, return_dict=True)

    clip_img = getattr(outputs, "image_embeds", None)
    if clip_img is None:
        pooled = getattr(outputs, "pooler_output", None)
        clip_img = pooled
    if clip_img is None:
        raise ValueError("Could not extract image_embeds or pooler_output from image_encoder output.")
    return clip_img


def _config_get(config: Any, keys: Sequence[str], default: Optional[Any] = None) -> Optional[Any]:
    """Read one of several possible keys from a diffusers config object."""

    for key in keys:
        if isinstance(config, Mapping) and key in config:
            return config[key]
        if hasattr(config, key):
            return getattr(config, key)
    return default


def _normalize_spatial(value: Any) -> Tuple[int, int]:
    """Normalize config spatial size into (height, width)."""

    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return 24, 24


def infer_dummy_sample_shape(prior: nn.Module, batch_size: int) -> Tuple[int, int, int, int]:
    """Infer a safe dummy sample shape for Stable Cascade prior.

    TODO: Replace this with real Stage C latent/image feature shape once the
    Stage 2 image-latent pipeline is implemented. The fallback shape follows the
    common Stable Cascade prior latent layout [B, 16, 24, 24].
    """

    config = getattr(prior, "config", None)
    channels = _config_get(config, ("in_channels", "c_in", "num_channels", "latent_channels"), default=16)
    sample_size = _config_get(config, ("sample_size", "resolution", "latent_size"), default=(24, 24))
    height, width = _normalize_spatial(sample_size)
    return int(batch_size), int(channels), int(height), int(width)


def _get_prior_dtype(prior: nn.Module) -> torch.dtype:
    """Infer prior parameter dtype."""

    try:
        return next(prior.parameters()).dtype
    except StopIteration:
        return torch.float32


def make_dummy_sample(prior: nn.Module, batch_size: int, device: torch.device) -> Tensor:
    """Create dummy Stage C prior sample tensor."""

    shape = infer_dummy_sample_shape(prior=prior, batch_size=batch_size)
    dtype = _get_prior_dtype(prior)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    return torch.zeros(shape, device=device, dtype=dtype)


def _extract_model_output_tensor(output: Any) -> Tensor:
    """Extract tensor from Stable Cascade prior output."""

    if isinstance(output, Tensor):
        return output
    sample = getattr(output, "sample", None)
    if isinstance(sample, Tensor):
        return sample
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError(f"Could not extract tensor output from prior forward result type {type(output)}.")


@contextlib.contextmanager
def user_conditioning_hooks(
    prior: nn.Module,
    user_emb: Optional[Tensor],
    user_emb_attention_mask: Optional[Tensor],
) -> Iterator[None]:
    """Inject user residual into patched blocks during a forward-only smoke test.

    This avoids rewriting StableCascadeUNet.forward. The hook is only used when
    user_emb is provided; text-only case runs without additional hooks.
    """

    handles: List[Any] = []
    if user_emb is None:
        yield
        return

    def _hook(module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if not isinstance(module, PatchedSDCascadeAttnBlock):
            return output
        if not isinstance(output, Tensor):
            raise TypeError(
                "PatchedSDCascadeAttnBlock hook expected tensor output, "
                f"got {type(output)}"
            )
        query, restore = module._to_token_sequence(output)
        module._sync_user_modules(reference=query)

        local_user_emb = user_emb.to(device=query.device, dtype=query.dtype)
        local_mask = None
        if user_emb_attention_mask is not None:
            local_mask = user_emb_attention_mask.to(device=query.device)

        user_tokens = module.user_projection(
            user_emb=local_user_emb,
            user_emb_attention_mask=local_mask,
        )
        user_residual = module.user_adapter(
            query=query,
            user_tokens=user_tokens,
            user_attention_mask=local_mask,
        )
        user_residual = restore(user_residual)
        scale = module.user_scale.to(device=output.device, dtype=output.dtype)
        return output + scale * user_residual

    try:
        for module in prior.modules():
            if isinstance(module, PatchedSDCascadeAttnBlock):
                handles.append(module.register_forward_hook(_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def run_prior_forward_case(
    name: str,
    prior: nn.Module,
    text: TextConditioning,
    sample: Tensor,
    user_emb: Optional[Tensor],
    user_mask: Optional[Tensor],
    clip_img: Optional[Tensor] = None,
) -> ForwardCaseResult:
    """Run one Stage C prior forward case and summarize output."""

    batch_size = int(sample.shape[0])
    timestep_ratio = torch.full(
        (batch_size,),
        0.5,
        device=sample.device,
        dtype=sample.dtype if sample.is_floating_point() else torch.float32,
    )

    with torch.no_grad():
        with user_conditioning_hooks(prior, user_emb=user_emb, user_emb_attention_mask=user_mask):
            output = prior(
                sample=sample,
                timestep_ratio=timestep_ratio,
                clip_text_pooled=text.clip_text_pooled,
                clip_text=text.clip_text,
                clip_img=clip_img,
                return_dict=True,
            )

    output_tensor = _extract_model_output_tensor(output)
    has_nan, has_inf, all_finite = _finite_flags(output_tensor)
    return ForwardCaseResult(
        name=name,
        caption_batch_size=batch_size,
        clip_text_shape=tuple(text.clip_text.shape),
        clip_text_pooled_shape=tuple(text.clip_text_pooled.shape),
        clip_img_shape=tuple(clip_img.shape) if clip_img is not None else None,
        user_emb_shape=tuple(user_emb.shape) if user_emb is not None else None,
        user_mask_shape=tuple(user_mask.shape) if user_mask is not None else None,
        sample_shape=tuple(sample.shape),
        output_shape=tuple(output_tensor.shape),
        dtype=str(output_tensor.dtype),
        device=str(output_tensor.device),
        has_nan=has_nan,
        has_inf=has_inf,
        all_finite=all_finite,
    )


def _print_case_result(result: ForwardCaseResult, summary_only: bool) -> None:
    """Print one case summary."""

    print(f"\n[{result.name}]")
    print("caption batch size:", result.caption_batch_size)
    print("clip_text shape:", result.clip_text_shape)
    print("clip_text_pooled shape:", result.clip_text_pooled_shape)
    print("clip_img shape:", result.clip_img_shape)
    print("user_emb shape:", result.user_emb_shape)
    print("user_emb_attention_mask shape:", result.user_mask_shape)
    print("sample shape:", result.sample_shape)
    print("model output shape:", result.output_shape)
    print("dtype:", result.dtype)
    print("device:", result.device)
    print("has NaN:", result.has_nan)
    print("has Inf:", result.has_inf)
    print("all finite:", result.all_finite)
    if not summary_only:
        print("status:", "ok" if result.all_finite else "failed: non-finite output detected")


def _print_tensor_diagnostics(diag: TensorDiagnostics) -> None:
    """Print one tensor diagnostics block."""

    print(f"[forward_only_stage2] {diag.name} shape:", diag.shape)
    print(f"[forward_only_stage2] {diag.name} dtype:", diag.dtype)
    print(f"[forward_only_stage2] {diag.name} device:", diag.device)
    print(f"[forward_only_stage2] {diag.name} min/max:", (diag.min_value, diag.max_value))
    print(f"[forward_only_stage2] {diag.name} mean/std:", (diag.mean_value, diag.std_value))
    print(f"[forward_only_stage2] {diag.name} has NaN:", diag.has_nan)
    print(f"[forward_only_stage2] {diag.name} has Inf:", diag.has_inf)
    print(f"[forward_only_stage2] {diag.name} all finite:", diag.all_finite)


def _print_batch_diagnostics(batch: Mapping[str, Any], dataset_stats: Mapping[str, Any]) -> None:
    """Print real-latent batch diagnostics before the model forward."""

    selected_stats = {
        "num_samples": dataset_stats.get("num_samples"),
        "num_users": dataset_stats.get("num_users"),
        "num_assignment_records": dataset_stats.get("num_assignment_records"),
        "num_query_samples": dataset_stats.get("num_query_samples"),
        "missing_latents_skipped": dataset_stats.get("missing_latents_skipped"),
        "assignment_validation_failures_skipped": dataset_stats.get(
            "assignment_validation_failures_skipped"
        ),
        "embedding_length_min": dataset_stats.get("embedding_length_min"),
        "embedding_length_max": dataset_stats.get("embedding_length_max"),
    }
    print("[forward_only_stage2] dataset stats:", selected_stats)
    print("[forward_only_stage2] batch size:", len(batch["caption"]))
    print("[forward_only_stage2] first user ids:", batch.get("user_id", [])[:3])
    print("[forward_only_stage2] first query pair keys:", batch.get("query_pair_key", [])[:3])
    print("[forward_only_stage2] first captions:", batch.get("caption", [])[:2])
    print("[forward_only_stage2] first preferred latent paths:", batch.get("preferred_latent_path", [])[:2])
    print("[forward_only_stage2] first dispreferred latent paths:", batch.get("dispreferred_latent_path", [])[:2])
    for key in ("user_emb", "user_emb_attention_mask", "preferred_latent", "dispreferred_latent"):
        _print_tensor_diagnostics(_tensor_diagnostics(key, _require_tensor(batch, key)))


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description="Forward-only Stage 2 integration smoke test.")
    parser.add_argument(
        "--embedding-json-path",
        type=Path,
        default=DEFAULT_EMBEDDING_JSON_PATH,
    )
    parser.add_argument(
        "--assignment-jsonl-path",
        type=Path,
        default=DEFAULT_ASSIGNMENT_JSONL_PATH,
        help="Stage 2 assignment JSONL. Defaults to train shard0 real-latent smoke input.",
    )
    parser.add_argument(
        "--latent-manifest-jsonl-path",
        type=Path,
        default=DEFAULT_LATENT_MANIFEST_JSONL_PATH,
        help="Latent manifest JSONL. Defaults to the train v512 latent manifest.",
    )
    parser.add_argument("--uid-to-path-json-path", type=Path, default=DEFAULT_UID_TO_PATH_JSON_PATH)
    parser.add_argument(
        "--skip-missing-latents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip samples missing preferred/dispreferred latent entries (default: False).",
    )
    parser.add_argument(
        "--validate-assignment-support-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate assignment support pairs against Stage 1 embedding rows (default: True).",
    )
    parser.add_argument(
        "--latent-source",
        type=str,
        default="preferred",
        choices=("preferred", "dispreferred"),
        help="Which real latent tensor to use as the Stage C prior sample.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--patch-path",
        action="append",
        default=None,
        help="Patch path; may be repeated or comma-separated. Defaults to a small safe patch set.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--model-id", type=str, default="stabilityai/stable-cascade-prior")
    parser.add_argument("--torch-dtype", type=str, default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--user-scale", type=float, default=1.0)
    return parser


def main() -> int:
    """Run forward-only integration smoke test."""

    args = _build_parser().parse_args()
    passed = False
    mode_name = "real-latent mode"

    try:
        device = _resolve_device(args.device)
        dataset = _load_dataset(args)
        batch = _get_one_batch(dataset, batch_size=max(1, args.batch_size))
        _validate_batch_shapes(batch)
    except Exception:
        print("[forward_only_stage2] dataset load/batch failure")
        print(traceback.format_exc().strip())
        print("\nforward-only integration test failed")
        return 1

    try:
        pipe = _load_prior_pipeline(args)
        prior = pipe.prior
        _move_module_if_possible(prior, device)
        prior.eval()

        patch_paths = _parse_patch_paths(args.patch_path)
        patch_summary = patch_stage_c_with_user_adapter(
            model=prior,
            target_paths=patch_paths,
            max_blocks=None,
            user_emb_dim=3584,
            user_scale=args.user_scale,
        )
        freeze_stage_c_except_user_modules(prior)
        trainable_summary = summarize_trainable_parameters(prior, max_names=20)
    except Exception:
        print("[forward_only_stage2] model load/patch failure")
        print(traceback.format_exc().strip())
        print("\nforward-only integration test failed")
        return 2

    try:
        captions = [str(x) for x in batch["caption"]]
        text = _tokenize_and_encode_text(pipe, captions=captions, device=device)
        text = TextConditioning(
            clip_text=text.clip_text.to(device=device, dtype=_get_prior_dtype(prior)),
            clip_text_pooled=text.clip_text_pooled.to(device=device, dtype=_get_prior_dtype(prior)),
        )

        user_emb = batch["user_emb"].to(device=device, dtype=_get_prior_dtype(prior))
        user_mask = batch["user_emb_attention_mask"].to(device=device)
        zero_user_emb = torch.zeros_like(user_emb)
        sample = _extract_real_sample_from_batch(
            batch=batch,
            latent_source=args.latent_source,
            prior=prior,
            device=device,
        )
    except Exception:
        print("[forward_only_stage2] conditioning/sample preparation failure")
        print(traceback.format_exc().strip())
        print("\nforward-only integration test failed")
        return 3

    dataset_stats = dataset.get_stats()
    if not args.summary_only:
        _print_batch_diagnostics(batch=batch, dataset_stats=dataset_stats)
    print("[forward_only_stage2] patched paths:", patch_summary.patched_paths)
    print("[forward_only_stage2] trainable parameters:", trainable_summary.trainable_parameters)
    print("[forward_only_stage2] trainable tensors:", trainable_summary.trainable_tensors)
    print("[forward_only_stage2] mode:", mode_name)
    print("[forward_only_stage2] latent source:", args.latent_source)
    _print_tensor_diagnostics(_tensor_diagnostics("prior sample", sample))
    latent_path_key = f"{args.latent_source}_latent_path"
    latent_paths = batch.get(latent_path_key)
    if isinstance(latent_paths, list) and latent_paths:
        print("[forward_only_stage2] first prior sample latent paths:", latent_paths[:2])

    results: List[ForwardCaseResult] = []
    try:
        results.append(
            run_prior_forward_case(
                name="case A: text only + real sample",
                prior=prior,
                text=text,
                sample=sample,
                user_emb=None,
                user_mask=None,
                clip_img=None,
            )
        )
        results.append(
            run_prior_forward_case(
                name="case B: text + zero user embedding + real sample",
                prior=prior,
                text=text,
                sample=sample,
                user_emb=zero_user_emb,
                user_mask=user_mask,
                clip_img=None,
            )
        )
        results.append(
            run_prior_forward_case(
                name="case C: text + real user embedding + real sample",
                prior=prior,
                text=text,
                sample=sample,
                user_emb=user_emb,
                user_mask=user_mask,
                clip_img=None,
            )
        )
        passed = all(result.all_finite for result in results)
    except Exception:
        print("[forward_only_stage2] prior forward failure")
        print(traceback.format_exc().strip())
        print("\nforward-only integration test failed")
        return 4

    for result in results:
        _print_case_result(result, summary_only=args.summary_only)

    print("\nforward-only integration test", "passed" if passed else "failed")
    return 0 if passed else 5


if __name__ == "__main__":
    raise SystemExit(main())
