from __future__ import annotations

"""Short multi-step Stage 2 training smoke run.

This script performs actual optimizer updates for the patched Stable Cascade
Stage C user-conditioning branch only. It is intentionally a smoke test, not a
full training entrypoint.
"""

import argparse
import copy
import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

try:
    from forward_only_stage2 import (
        DEFAULT_ASSIGNMENT_JSONL_PATH,
        DEFAULT_EMBEDDING_JSON_PATH,
        DEFAULT_LATENT_MANIFEST_JSONL_PATH,
        DEFAULT_UID_TO_PATH_JSON_PATH,
        TextConditioning,
        _finite_flags,
        _load_dataset,
        _load_prior_pipeline,
        _move_module_if_possible,
        _parse_patch_paths,
        _require_tensor,
        _resolve_device,
        _tokenize_and_encode_text,
        _validate_batch_shapes,
        user_conditioning_hooks,
    )
    from patch_stage_c import (
        freeze_stage_c_except_user_modules,
        patch_stage_c_with_user_adapter,
        summarize_trainable_parameters,
    )
    from stage2_dataset import Stage2PreferenceDataset
    from train_step_smoke_stage2 import (
        USER_CONDITIONING_NAME_MARKERS,
        LossBundle,
        _effective_prior_dtype,
        _freeze_all,
        _is_user_conditioning_param,
        _per_sample_mse,
        _prepare_pair_tensors,
        _require_finite,
        _run_prior,
        _set_seed,
        _validate_reference_frozen,
        _validate_reference_no_grads,
        _validate_trainable_scope,
    )
except ImportError:  # pragma: no cover - useful when imported as package module
    from stage_2.forward_only_stage2 import (
        DEFAULT_ASSIGNMENT_JSONL_PATH,
        DEFAULT_EMBEDDING_JSON_PATH,
        DEFAULT_LATENT_MANIFEST_JSONL_PATH,
        DEFAULT_UID_TO_PATH_JSON_PATH,
        TextConditioning,
        _finite_flags,
        _load_dataset,
        _load_prior_pipeline,
        _move_module_if_possible,
        _parse_patch_paths,
        _require_tensor,
        _resolve_device,
        _tokenize_and_encode_text,
        _validate_batch_shapes,
        user_conditioning_hooks,
    )
    from stage_2.patch_stage_c import (
        freeze_stage_c_except_user_modules,
        patch_stage_c_with_user_adapter,
        summarize_trainable_parameters,
    )
    from stage_2.stage2_dataset import Stage2PreferenceDataset
    from stage_2.train_step_smoke_stage2 import (
        USER_CONDITIONING_NAME_MARKERS,
        LossBundle,
        _effective_prior_dtype,
        _freeze_all,
        _is_user_conditioning_param,
        _per_sample_mse,
        _prepare_pair_tensors,
        _require_finite,
        _run_prior,
        _set_seed,
        _validate_reference_frozen,
        _validate_reference_no_grads,
        _validate_trainable_scope,
    )


DEFAULT_OUTPUT_ROOT = Path("artifacts/stage2_train_smoke")


@dataclass
class ModelBundle:
    train_prior: nn.Module
    reference_prior: nn.Module
    patch_summary: Any
    trainable_summary: Any
    train_device: torch.device
    reference_device: torch.device


@dataclass
class GradStats:
    trainable_tensors: int
    tensors_with_grad: int
    tensors_with_nonzero_grad: int
    total_grad_norm: float
    max_grad_abs: float


@dataclass
class ParamSliceSnapshot:
    values: Dict[str, Tensor]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2 short optimizer-update smoke training run.")
    parser.add_argument("--embedding-json-path", type=Path, default=DEFAULT_EMBEDDING_JSON_PATH)
    parser.add_argument("--assignment-jsonl-path", type=Path, default=DEFAULT_ASSIGNMENT_JSONL_PATH)
    parser.add_argument("--latent-manifest-jsonl-path", type=Path, default=DEFAULT_LATENT_MANIFEST_JSONL_PATH)
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--model-id", type=str, default="stabilityai/stable-cascade-prior")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", type=str, default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--reference-device", type=str, default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--patch-path",
        action="append",
        default=None,
        help="Patch path; may be repeated or comma-separated. Defaults to Stage 1 patch paths.",
    )
    parser.add_argument("--user-scale", type=float, default=1.0)
    parser.add_argument("--dpo-beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fail-grad-norm", type=float, default=1000.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--frozen-check-every", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-only", action="store_true")
    return parser


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return [_jsonable(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), ensure_ascii=False) + "\n")


def _make_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root.expanduser().resolve() / timestamp


def _resolve_reference_device(raw: str, train_device: torch.device) -> torch.device:
    if raw == "cpu":
        return torch.device("cpu")
    if train_device.type == "cuda":
        return train_device
    return torch.device("cpu")


def _is_oom_error(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _cuda_memory_mb(device: torch.device) -> Tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return (
        float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
        float(torch.cuda.max_memory_reserved(device) / (1024 * 1024)),
    )


def _load_and_prepare_models(args: argparse.Namespace, pipe: Any, train_device: torch.device) -> ModelBundle:
    reference_device = _resolve_reference_device(args.reference_device, train_device)
    train_prior = pipe.prior
    reference_prior = copy.deepcopy(train_prior)
    _freeze_all(reference_prior)

    patch_summary = patch_stage_c_with_user_adapter(
        model=train_prior,
        target_paths=_parse_patch_paths(args.patch_path),
        max_blocks=None,
        user_emb_dim=3584,
        user_scale=args.user_scale,
        user_projection_bias=bool(getattr(args, "user_projection_bias", True)),
        user_projection_norm_affine=bool(getattr(args, "user_projection_norm_affine", True)),
        user_adapter_projection_bias=bool(getattr(args, "user_adapter_projection_bias", True)),
        user_adapter_zero_init_out=bool(getattr(args, "user_adapter_zero_init_out", False)),
    )
    freeze_stage_c_except_user_modules(train_prior)
    _validate_trainable_scope(train_prior)
    _validate_reference_frozen(reference_prior)

    _move_module_if_possible(train_prior, train_device)
    if reference_device.type == "cpu" and _effective_prior_dtype(reference_prior, reference_device) == torch.float32:
        reference_prior.to(device=reference_device, dtype=torch.float32)
    else:
        _move_module_if_possible(reference_prior, reference_device)

    train_prior.train()
    reference_prior.eval()
    trainable_summary = summarize_trainable_parameters(train_prior, max_names=20)
    return ModelBundle(
        train_prior=train_prior,
        reference_prior=reference_prior,
        patch_summary=patch_summary,
        trainable_summary=trainable_summary,
        train_device=train_device,
        reference_device=reference_device,
    )


def _build_loader(args: argparse.Namespace) -> Tuple[Any, DataLoader]:
    dataset = _load_dataset(args)
    if len(dataset) == 0:
        raise ValueError("Stage2PreferenceDataset produced zero valid samples.")
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        collate_fn=Stage2PreferenceDataset.collate_fn,
        drop_last=False,
    )
    return dataset, loader


def _iter_batches(loader: DataLoader) -> Iterable[Mapping[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def _encode_text_for_devices(
    pipe: Any,
    batch: Mapping[str, Any],
    train_prior: nn.Module,
    reference_prior: nn.Module,
    train_device: torch.device,
    reference_device: torch.device,
) -> Tuple[TextConditioning, TextConditioning]:
    captions = [str(x) for x in batch["caption"]]
    cpu = torch.device("cpu")
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is not None:
        _move_module_if_possible(text_encoder, cpu)
    raw_text = _tokenize_and_encode_text(pipe, captions=captions, device=cpu)

    train_dtype = _effective_prior_dtype(train_prior, train_device)
    reference_dtype = _effective_prior_dtype(reference_prior, reference_device)
    return (
        TextConditioning(
            clip_text=raw_text.clip_text.to(device=train_device, dtype=train_dtype),
            clip_text_pooled=raw_text.clip_text_pooled.to(device=train_device, dtype=train_dtype),
        ),
        TextConditioning(
            clip_text=raw_text.clip_text.to(device=reference_device, dtype=reference_dtype),
            clip_text_pooled=raw_text.clip_text_pooled.to(device=reference_device, dtype=reference_dtype),
        ),
    )


def _make_noisy_pair(
    scheduler: Any,
    preferred: Tensor,
    dispreferred: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch_size = int(preferred.shape[0])
    timesteps = torch.rand((batch_size,), device=preferred.device, dtype=preferred.dtype)
    preferred_noise = torch.randn_like(preferred)
    dispreferred_noise = torch.randn_like(dispreferred)
    noisy_preferred = scheduler.add_noise(preferred, preferred_noise, timesteps)
    noisy_dispreferred = scheduler.add_noise(dispreferred, dispreferred_noise, timesteps)
    for name, tensor in (
        ("timesteps", timesteps),
        ("preferred_noise", preferred_noise),
        ("dispreferred_noise", dispreferred_noise),
        ("noisy_preferred", noisy_preferred),
        ("noisy_dispreferred", noisy_dispreferred),
    ):
        _require_finite(name, tensor)
    return noisy_preferred, noisy_dispreferred, preferred_noise, dispreferred_noise, timesteps


def _compute_loss_from_errors(
    train_pref_err: Tensor,
    train_dispref_err: Tensor,
    ref_pref_err: Tensor,
    ref_dispref_err: Tensor,
    dpo_beta: float,
) -> LossBundle:
    score = (ref_pref_err - train_pref_err) - (ref_dispref_err - train_dispref_err)
    loss = -F.logsigmoid(float(dpo_beta) * score).mean()
    for name, tensor in (
        ("train_pref_err", train_pref_err),
        ("train_dispref_err", train_dispref_err),
        ("ref_pref_err", ref_pref_err),
        ("ref_dispref_err", ref_dispref_err),
        ("pairwise_score", score),
        ("loss", loss),
    ):
        _require_finite(name, tensor)
    return LossBundle(
        train_pref_err=train_pref_err,
        train_dispref_err=train_dispref_err,
        ref_pref_err=ref_pref_err,
        ref_dispref_err=ref_dispref_err,
        score=score,
        loss=loss,
    )


def _grad_stats(model: nn.Module, *, check_frozen_grads: bool = True) -> GradStats:
    total_norm_sq = 0.0
    max_grad_abs = 0.0
    trainable_tensors = 0
    tensors_with_grad = 0
    tensors_with_nonzero_grad = 0

    for name, param in model.named_parameters():
        grad = param.grad
        if not param.requires_grad:
            if check_frozen_grads and grad is not None and bool((grad.detach() != 0).any().item()):
                raise RuntimeError(f"Frozen train-prior parameter accumulated gradient: {name}")
            continue

        trainable_tensors += 1
        if grad is None:
            continue
        tensors_with_grad += 1
        _require_finite(f"grad[{name}]", grad)
        grad_float = grad.detach().float()
        if bool((grad_float != 0).any().item()):
            tensors_with_nonzero_grad += 1
        norm = float(torch.linalg.vector_norm(grad_float).item())
        total_norm_sq += norm * norm
        max_grad_abs = max(max_grad_abs, float(grad_float.abs().max().item()))

    if trainable_tensors == 0:
        raise RuntimeError("No trainable tensors found in train prior.")
    if tensors_with_nonzero_grad == 0:
        raise RuntimeError("No trainable user-conditioning tensor received a nonzero gradient.")

    return GradStats(
        trainable_tensors=trainable_tensors,
        tensors_with_grad=tensors_with_grad,
        tensors_with_nonzero_grad=tensors_with_nonzero_grad,
        total_grad_norm=total_norm_sq**0.5,
        max_grad_abs=max_grad_abs,
    )


def _trainable_named_parameters(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def _validate_optimizer_scope(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    allowed_ids = {id(param) for _, param in _trainable_named_parameters(model)}
    optimizer_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    if optimizer_ids != allowed_ids:
        raise RuntimeError("Optimizer parameter set does not exactly match trainable user-conditioning parameters.")
    unexpected = [name for name, param in _trainable_named_parameters(model) if not _is_user_conditioning_param(name)]
    if unexpected:
        raise RuntimeError(f"Optimizer includes non-user-conditioning parameters: {unexpected[:10]}")


def _validate_no_frozen_train_grads(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad:
            continue
        if param.grad is not None and bool((param.grad.detach() != 0).any().item()):
            raise RuntimeError(f"Frozen train-prior parameter accumulated gradient: {name}")


def _full_frozen_integrity_check(bundle: ModelBundle, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
    _validate_trainable_scope(bundle.train_prior)
    _validate_reference_frozen(bundle.reference_prior)
    _validate_reference_no_grads(bundle.reference_prior)
    _validate_no_frozen_train_grads(bundle.train_prior)
    if optimizer is not None:
        _validate_optimizer_scope(bundle.train_prior, optimizer)


def _light_frozen_grad_check(bundle: ModelBundle) -> None:
    _validate_no_frozen_train_grads(bundle.train_prior)
    _validate_reference_no_grads(bundle.reference_prior)


def _capture_param_slices(model: nn.Module, max_values_per_tensor: int = 16) -> ParamSliceSnapshot:
    values: Dict[str, Tensor] = {}
    for name, param in _trainable_named_parameters(model):
        flat = param.detach().float().flatten()
        values[name] = flat[: min(max_values_per_tensor, flat.numel())].cpu().clone()
    return ParamSliceSnapshot(values=values)


def _param_slice_delta(before: ParamSliceSnapshot, model: nn.Module) -> Tuple[bool, float]:
    max_delta = 0.0
    changed = False
    current = dict(_trainable_named_parameters(model))
    for name, old_values in before.values.items():
        if name not in current:
            continue
        flat = current[name].detach().float().flatten()
        new_values = flat[: old_values.numel()].cpu()
        delta = (new_values - old_values).abs()
        if delta.numel() == 0:
            continue
        local_max = float(delta.max().item())
        max_delta = max(max_delta, local_max)
        changed = changed or local_max > 0.0
    return changed, max_delta


def _save_user_adapter_state(model: nn.Module, output_path: Path) -> None:
    state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad and _is_user_conditioning_param(name)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)


def _tensor_mean(tensor: Tensor) -> float:
    return float(tensor.detach().float().mean().cpu().item())


def _tensor_min(tensor: Tensor) -> float:
    return float(tensor.detach().float().min().cpu().item())


def _tensor_max(tensor: Tensor) -> float:
    return float(tensor.detach().float().max().cpu().item())


def _run_step(
    *,
    args: argparse.Namespace,
    pipe: Any,
    bundle: ModelBundle,
    scheduler: Any,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    step: int,
) -> Dict[str, Any]:
    step_start = time.time()
    _validate_batch_shapes(batch)
    train_text, ref_text = _encode_text_for_devices(
        pipe=pipe,
        batch=batch,
        train_prior=bundle.train_prior,
        reference_prior=bundle.reference_prior,
        train_device=bundle.train_device,
        reference_device=bundle.reference_device,
    )
    preferred, dispreferred, user_emb, user_mask = _prepare_pair_tensors(
        batch=batch,
        prior=bundle.train_prior,
        device=bundle.train_device,
    )
    noisy_pref, noisy_dispref, pref_noise, dispref_noise, timesteps = _make_noisy_pair(
        scheduler=scheduler,
        preferred=preferred,
        dispreferred=dispreferred,
    )

    optimizer.zero_grad(set_to_none=True)
    bundle.train_prior.zero_grad(set_to_none=True)
    bundle.reference_prior.zero_grad(set_to_none=True)

    with torch.no_grad():
        ref_dtype = _effective_prior_dtype(bundle.reference_prior, bundle.reference_device)
        ref_timesteps = timesteps.to(device=bundle.reference_device, dtype=ref_dtype)
        ref_pref_noise = pref_noise.to(device=bundle.reference_device, dtype=ref_dtype)
        ref_dispref_noise = dispref_noise.to(device=bundle.reference_device, dtype=ref_dtype)
        ref_noisy_pref = noisy_pref.to(device=bundle.reference_device, dtype=ref_dtype)
        ref_noisy_dispref = noisy_dispref.to(device=bundle.reference_device, dtype=ref_dtype)
        ref_pref_pred = _run_prior(bundle.reference_prior, text=ref_text, sample=ref_noisy_pref, timesteps=ref_timesteps)
        ref_dispref_pred = _run_prior(
            bundle.reference_prior,
            text=ref_text,
            sample=ref_noisy_dispref,
            timesteps=ref_timesteps,
        )
        ref_pref_err = _per_sample_mse(ref_pref_pred, ref_pref_noise).detach().to(bundle.train_device)
        ref_dispref_err = _per_sample_mse(ref_dispref_pred, ref_dispref_noise).detach().to(bundle.train_device)

    del ref_pref_pred, ref_dispref_pred
    if bundle.train_device.type == "cuda":
        torch.cuda.empty_cache()

    with user_conditioning_hooks(
        bundle.train_prior,
        user_emb=user_emb,
        user_emb_attention_mask=user_mask,
    ):
        train_pref_pred = _run_prior(bundle.train_prior, text=train_text, sample=noisy_pref, timesteps=timesteps)
        train_dispref_pred = _run_prior(bundle.train_prior, text=train_text, sample=noisy_dispref, timesteps=timesteps)

    train_pref_err = _per_sample_mse(train_pref_pred, pref_noise)
    train_dispref_err = _per_sample_mse(train_dispref_pred, dispref_noise)
    loss_bundle = _compute_loss_from_errors(
        train_pref_err=train_pref_err,
        train_dispref_err=train_dispref_err,
        ref_pref_err=ref_pref_err,
        ref_dispref_err=ref_dispref_err,
        dpo_beta=args.dpo_beta,
    )
    loss_bundle.loss.backward()

    pre_clip = _grad_stats(bundle.train_prior, check_frozen_grads=True)
    if pre_clip.total_grad_norm > float(args.fail_grad_norm):
        raise RuntimeError(
            f"Pre-clip grad norm {pre_clip.total_grad_norm:.6f} exceeds --fail-grad-norm={args.fail_grad_norm}."
        )

    torch.nn.utils.clip_grad_norm_(
        [param for _, param in _trainable_named_parameters(bundle.train_prior)],
        max_norm=float(args.max_grad_norm),
    )
    post_clip = _grad_stats(bundle.train_prior, check_frozen_grads=True)
    optimizer.step()

    for name, param in _trainable_named_parameters(bundle.train_prior):
        _require_finite(f"updated_param[{name}]", param)

    allocated_mb, reserved_mb = _cuda_memory_mb(bundle.train_device)
    return {
        "step": step,
        "loss": float(loss_bundle.loss.detach().cpu().item()),
        "train_pref_err_mean": _tensor_mean(loss_bundle.train_pref_err),
        "train_dispref_err_mean": _tensor_mean(loss_bundle.train_dispref_err),
        "ref_pref_err_mean": _tensor_mean(loss_bundle.ref_pref_err),
        "ref_dispref_err_mean": _tensor_mean(loss_bundle.ref_dispref_err),
        "score_mean": _tensor_mean(loss_bundle.score),
        "timestep_min": _tensor_min(timesteps),
        "timestep_max": _tensor_max(timesteps),
        "timestep_mean": _tensor_mean(timesteps),
        "pre_clip_grad_norm": pre_clip.total_grad_norm,
        "post_clip_grad_norm": post_clip.total_grad_norm,
        "max_grad_abs": post_clip.max_grad_abs,
        "tensors_with_grad": post_clip.tensors_with_grad,
        "tensors_with_nonzero_grad": post_clip.tensors_with_nonzero_grad,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "step_time_sec": time.time() - step_start,
        "cuda_max_memory_allocated_mb": allocated_mb,
        "cuda_max_memory_reserved_mb": reserved_mb,
    }


def _initial_summary(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    return {
        "run_status": "failed",
        "num_steps_completed": 0,
        "nan_failures": 0,
        "frozen_integrity_failures": 0,
        "max_grad_norm_observed": 0.0,
        "max_cuda_mem_reserved_mb": 0.0,
        "final_loss": None,
        "parameter_delta_status": "not_checked",
        "parameter_delta_max_abs": 0.0,
        "failure_message": None,
        "run_dir": str(run_dir),
        "metrics_path": str(run_dir / "metrics.jsonl"),
        "summary_path": str(run_dir / "summary.json"),
        "user_adapter_path": str(run_dir / "user_adapter_final.pt"),
        "args": vars(args),
    }


def _print_memory_hint(exc: BaseException) -> None:
    print("[train_smoke_stage2] failure:", str(exc))
    print("[train_smoke_stage2] no automatic retry will be attempted.")
    print("[train_smoke_stage2] if this is memory-related, rerun with --reference-device cpu.")


def main() -> int:
    args = _build_parser().parse_args()
    run_dir = _make_run_dir(args.output_dir)
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"
    user_adapter_path = run_dir / "user_adapter_final.pt"
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = _initial_summary(args, run_dir)

    try:
        train_device = _resolve_device(args.device)
        _set_seed(args.seed, train_device)
        dataset, loader = _build_loader(args)
        pipe = _load_prior_pipeline(args)
        bundle = _load_and_prepare_models(args=args, pipe=pipe, train_device=train_device)
        scheduler = getattr(pipe, "scheduler", None)
        if scheduler is None or not hasattr(scheduler, "add_noise"):
            raise ValueError("Pipeline scheduler must expose add_noise(original_samples, noise, timesteps).")

        trainable_params = [param for _, param in _trainable_named_parameters(bundle.train_prior)]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        _validate_optimizer_scope(bundle.train_prior, optimizer)
        _full_frozen_integrity_check(bundle, optimizer=optimizer)
        before = _capture_param_slices(bundle.train_prior)

        frozen_check_every = int(args.frozen_check_every or args.log_every or 10)
        final_metrics: Optional[Dict[str, Any]] = None
        batch_iter = _iter_batches(loader)
        max_steps = max(1, int(args.max_steps))
        log_every = max(1, int(args.log_every))

        for step in range(max_steps):
            batch = next(batch_iter)
            try:
                metrics = _run_step(
                    args=args,
                    pipe=pipe,
                    bundle=bundle,
                    scheduler=scheduler,
                    optimizer=optimizer,
                    batch=batch,
                    step=step,
                )
                _light_frozen_grad_check(bundle)
                if step == 0 or (frozen_check_every > 0 and (step + 1) % frozen_check_every == 0):
                    _full_frozen_integrity_check(bundle, optimizer=optimizer)
            except Exception as exc:
                if _is_oom_error(exc):
                    if train_device.type == "cuda":
                        torch.cuda.empty_cache()
                    summary["failure_message"] = f"CUDA OOM: {exc}"
                    _print_memory_hint(exc)
                else:
                    summary["failure_message"] = str(exc)
                    if isinstance(exc, RuntimeError):
                        _print_memory_hint(exc)
                    else:
                        print("[train_smoke_stage2] step failure:", str(exc))
                if "non-finite" in str(exc).lower() or "nan" in str(exc).lower() or "inf" in str(exc).lower():
                    summary["nan_failures"] = int(summary["nan_failures"]) + 1
                if "frozen" in str(exc).lower() or "reference" in str(exc).lower():
                    summary["frozen_integrity_failures"] = int(summary["frozen_integrity_failures"]) + 1
                _write_json(summary_path, summary)
                return 3

            _append_jsonl(metrics_path, metrics)
            final_metrics = metrics
            summary["num_steps_completed"] = int(summary["num_steps_completed"]) + 1
            summary["max_grad_norm_observed"] = max(
                float(summary["max_grad_norm_observed"]),
                float(metrics["pre_clip_grad_norm"]),
            )
            summary["max_cuda_mem_reserved_mb"] = max(
                float(summary["max_cuda_mem_reserved_mb"]),
                float(metrics["cuda_max_memory_reserved_mb"]),
            )
            summary["final_loss"] = metrics["loss"]

            if (step + 1) % log_every == 0 or step == 0 or step + 1 == max_steps:
                print(
                    "[train_smoke_stage2] "
                    f"step={step + 1}/{max_steps} "
                    f"loss={metrics['loss']:.6f} "
                    f"pre_clip_grad_norm={metrics['pre_clip_grad_norm']:.6f} "
                    f"post_clip_grad_norm={metrics['post_clip_grad_norm']:.6f} "
                    f"cuda_reserved_mb={metrics['cuda_max_memory_reserved_mb']:.2f}"
                )

        changed, max_delta = _param_slice_delta(before, bundle.train_prior)
        summary["parameter_delta_status"] = "changed" if changed else "unchanged"
        summary["parameter_delta_max_abs"] = max_delta
        if not changed:
            summary["failure_message"] = "Trainable user-conditioning parameter slices did not change."
            _write_json(summary_path, summary)
            return 4

        _save_user_adapter_state(bundle.train_prior, user_adapter_path)
        summary["run_status"] = "success"
        summary["failure_message"] = None
        if final_metrics is not None:
            summary["final_loss"] = final_metrics["loss"]
        _write_json(summary_path, summary)
        print("[train_smoke_stage2] run status: success")
        print("[train_smoke_stage2] steps completed:", summary["num_steps_completed"])
        print("[train_smoke_stage2] metrics:", metrics_path)
        print("[train_smoke_stage2] summary:", summary_path)
        print("[train_smoke_stage2] user adapter:", user_adapter_path)
        return 0
    except Exception as exc:
        if _is_oom_error(exc) or isinstance(exc, RuntimeError):
            _print_memory_hint(exc)
        else:
            print("[train_smoke_stage2] setup/finalization failure:", str(exc))
        summary["failure_message"] = traceback.format_exc().strip()
        if "non-finite" in str(exc).lower() or "nan" in str(exc).lower() or "inf" in str(exc).lower():
            summary["nan_failures"] = int(summary["nan_failures"]) + 1
        if "frozen" in str(exc).lower() or "reference" in str(exc).lower():
            summary["frozen_integrity_failures"] = int(summary["frozen_integrity_failures"]) + 1
        _write_json(summary_path, summary)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
