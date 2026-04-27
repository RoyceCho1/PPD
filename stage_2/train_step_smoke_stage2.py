from __future__ import annotations

"""One-step personalized diffusion-DPO smoke test for Stage 2.

This script runs one real-latent preference batch through:

- train Stage C prior with user-conditioning adapters
- frozen reference Stage C prior without user conditioning
- DDPMWuerstchen noisy sample creation
- personalized pairwise diffusion-DPO loss
- one backward pass

It intentionally does not create an optimizer, optimizer step, checkpoint, or
multi-step training loop.
"""

import argparse
import copy
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from forward_only_stage2 import (
        DEFAULT_ASSIGNMENT_JSONL_PATH,
        DEFAULT_EMBEDDING_JSON_PATH,
        DEFAULT_LATENT_MANIFEST_JSONL_PATH,
        DEFAULT_UID_TO_PATH_JSON_PATH,
        TextConditioning,
        _extract_model_output_tensor,
        _finite_flags,
        _get_one_batch,
        _get_prior_dtype,
        _load_dataset,
        _load_prior_pipeline,
        _move_module_if_possible,
        _parse_patch_paths,
        _print_batch_diagnostics,
        _print_tensor_diagnostics,
        _require_tensor,
        _resolve_device,
        _tensor_diagnostics,
        _tokenize_and_encode_text,
        _validate_batch_shapes,
        user_conditioning_hooks,
    )
    from patch_stage_c import (
        freeze_stage_c_except_user_modules,
        patch_stage_c_with_user_adapter,
        summarize_trainable_parameters,
    )
except ImportError:  # pragma: no cover - useful when imported as package module
    from stage_2.forward_only_stage2 import (
        DEFAULT_ASSIGNMENT_JSONL_PATH,
        DEFAULT_EMBEDDING_JSON_PATH,
        DEFAULT_LATENT_MANIFEST_JSONL_PATH,
        DEFAULT_UID_TO_PATH_JSON_PATH,
        TextConditioning,
        _extract_model_output_tensor,
        _finite_flags,
        _get_one_batch,
        _get_prior_dtype,
        _load_dataset,
        _load_prior_pipeline,
        _move_module_if_possible,
        _parse_patch_paths,
        _print_batch_diagnostics,
        _print_tensor_diagnostics,
        _require_tensor,
        _resolve_device,
        _tensor_diagnostics,
        _tokenize_and_encode_text,
        _validate_batch_shapes,
        user_conditioning_hooks,
    )
    from stage_2.patch_stage_c import (
        freeze_stage_c_except_user_modules,
        patch_stage_c_with_user_adapter,
        summarize_trainable_parameters,
    )


USER_CONDITIONING_NAME_MARKERS = (
    ".user_projection.",
    ".user_adapter.k_proj.",
    ".user_adapter.v_proj.",
    ".user_adapter.out_proj.",
    ".user_scale",
)


@dataclass
class ModelBundle:
    train_prior: nn.Module
    reference_prior: nn.Module
    patch_summary: Any
    trainable_summary: Any


@dataclass
class LossBundle:
    train_pref_err: Tensor
    train_dispref_err: Tensor
    ref_pref_err: Tensor
    ref_dispref_err: Tensor
    score: Tensor
    loss: Tensor


@dataclass
class GradSummary:
    trainable_tensors: int
    tensors_with_grad: int
    tensors_with_nonzero_grad: int
    total_grad_norm: float
    max_grad_abs: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2 one-step diffusion-DPO smoke test.")
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
    parser.add_argument("--model-id", type=str, default="stabilityai/stable-cascade-prior")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", type=str, default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--patch-path",
        action="append",
        default=None,
        help="Patch path; may be repeated or comma-separated. Defaults to Stage 1 patch paths.",
    )
    parser.add_argument("--user-scale", type=float, default=1.0)
    parser.add_argument("--dpo-beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary-only", action="store_true")
    return parser


def _set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _effective_prior_dtype(prior: nn.Module, device: torch.device) -> torch.dtype:
    dtype = _get_prior_dtype(prior)
    if device.type == "cpu" and dtype == torch.float16:
        return torch.float32
    return dtype


def _require_finite(name: str, tensor: Tensor) -> None:
    has_nan, has_inf, all_finite = _finite_flags(tensor)
    if not all_finite:
        raise ValueError(f"{name} contains non-finite values: has_nan={has_nan}, has_inf={has_inf}")


def _freeze_all(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _is_user_conditioning_param(name: str) -> bool:
    return (
        ".user_projection." in name
        or ".user_adapter.k_proj." in name
        or ".user_adapter.v_proj." in name
        or ".user_adapter.out_proj." in name
        or name.endswith(".user_scale")
        or name == "user_scale"
    )


def _validate_trainable_scope(model: nn.Module) -> None:
    unexpected = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and not _is_user_conditioning_param(name)
    ]
    if unexpected:
        preview = ", ".join(unexpected[:10])
        raise RuntimeError(f"Unexpected trainable non-user-conditioning parameters: {preview}")


def _validate_reference_frozen(reference_prior: nn.Module) -> None:
    trainable = [name for name, param in reference_prior.named_parameters() if param.requires_grad]
    if trainable:
        preview = ", ".join(trainable[:10])
        raise RuntimeError(f"Reference prior has trainable parameters: {preview}")


def _load_and_prepare_models(args: argparse.Namespace, pipe: Any, device: torch.device) -> ModelBundle:
    train_prior = pipe.prior
    reference_prior = copy.deepcopy(train_prior)
    _freeze_all(reference_prior)

    patch_summary = patch_stage_c_with_user_adapter(
        model=train_prior,
        target_paths=_parse_patch_paths(args.patch_path),
        max_blocks=None,
        user_emb_dim=3584,
        user_scale=args.user_scale,
    )
    freeze_stage_c_except_user_modules(train_prior)
    _validate_trainable_scope(train_prior)
    _validate_reference_frozen(reference_prior)

    _move_module_if_possible(train_prior, device)
    _move_module_if_possible(reference_prior, device)
    train_prior.train()
    reference_prior.eval()

    trainable_summary = summarize_trainable_parameters(train_prior, max_names=20)
    return ModelBundle(
        train_prior=train_prior,
        reference_prior=reference_prior,
        patch_summary=patch_summary,
        trainable_summary=trainable_summary,
    )


def _prepare_text(pipe: Any, batch: Mapping[str, Any], prior: nn.Module, device: torch.device) -> TextConditioning:
    captions = [str(x) for x in batch["caption"]]
    text = _tokenize_and_encode_text(pipe, captions=captions, device=device)
    dtype = _effective_prior_dtype(prior, device)

    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is not None and device.type == "cuda":
        _move_module_if_possible(text_encoder, torch.device("cpu"))
        torch.cuda.empty_cache()

    return TextConditioning(
        clip_text=text.clip_text.to(device=device, dtype=dtype),
        clip_text_pooled=text.clip_text_pooled.to(device=device, dtype=dtype),
    )


def _prepare_pair_tensors(
    batch: Mapping[str, Any],
    prior: nn.Module,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    dtype = _effective_prior_dtype(prior, device)
    preferred = _require_tensor(batch, "preferred_latent").to(device=device, dtype=dtype)
    dispreferred = _require_tensor(batch, "dispreferred_latent").to(device=device, dtype=dtype)
    user_emb = _require_tensor(batch, "user_emb").to(device=device, dtype=dtype)
    user_mask = _require_tensor(batch, "user_emb_attention_mask").to(device=device)

    _require_finite("preferred_latent", preferred)
    _require_finite("dispreferred_latent", dispreferred)
    _require_finite("user_emb", user_emb)
    return preferred, dispreferred, user_emb, user_mask


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

    _require_finite("timesteps", timesteps)
    _require_finite("preferred_noise", preferred_noise)
    _require_finite("dispreferred_noise", dispreferred_noise)
    _require_finite("noisy_preferred", noisy_preferred)
    _require_finite("noisy_dispreferred", noisy_dispreferred)
    return noisy_preferred, noisy_dispreferred, preferred_noise, dispreferred_noise, timesteps


def _run_prior(
    prior: nn.Module,
    text: TextConditioning,
    sample: Tensor,
    timesteps: Tensor,
) -> Tensor:
    output = prior(
        sample=sample,
        timestep_ratio=timesteps,
        clip_text_pooled=text.clip_text_pooled,
        clip_text=text.clip_text,
        clip_img=None,
        return_dict=True,
    )
    output_tensor = _extract_model_output_tensor(output)
    if tuple(output_tensor.shape) != tuple(sample.shape):
        raise ValueError(f"Prior output shape mismatch: expected {tuple(sample.shape)}, got {tuple(output_tensor.shape)}")
    _require_finite("prior_output", output_tensor)
    return output_tensor


def _per_sample_mse(prediction: Tensor, target: Tensor) -> Tensor:
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError(f"MSE shape mismatch: prediction={tuple(prediction.shape)}, target={tuple(target.shape)}")
    return (prediction.float() - target.float()).pow(2).flatten(start_dim=1).mean(dim=1)


def _compute_loss(
    *,
    train_pref_pred: Tensor,
    train_dispref_pred: Tensor,
    ref_pref_pred: Tensor,
    ref_dispref_pred: Tensor,
    preferred_noise: Tensor,
    dispreferred_noise: Tensor,
    dpo_beta: float,
) -> LossBundle:
    train_pref_err = _per_sample_mse(train_pref_pred, preferred_noise)
    train_dispref_err = _per_sample_mse(train_dispref_pred, dispreferred_noise)
    ref_pref_err = _per_sample_mse(ref_pref_pred, preferred_noise)
    ref_dispref_err = _per_sample_mse(ref_dispref_pred, dispreferred_noise)

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


def _iter_named_params(module: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    yield from module.named_parameters()


def _summarize_gradients(train_prior: nn.Module) -> GradSummary:
    total_norm_sq = 0.0
    max_grad_abs = 0.0
    trainable_tensors = 0
    tensors_with_grad = 0
    tensors_with_nonzero_grad = 0

    for name, param in _iter_named_params(train_prior):
        if not param.requires_grad:
            if param.grad is not None and bool((param.grad.detach() != 0).any().item()):
                raise RuntimeError(f"Frozen train-prior parameter accumulated gradient: {name}")
            continue

        trainable_tensors += 1
        grad = param.grad
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

    return GradSummary(
        trainable_tensors=trainable_tensors,
        tensors_with_grad=tensors_with_grad,
        tensors_with_nonzero_grad=tensors_with_nonzero_grad,
        total_grad_norm=total_norm_sq**0.5,
        max_grad_abs=max_grad_abs,
    )


def _validate_reference_no_grads(reference_prior: nn.Module) -> None:
    for name, param in reference_prior.named_parameters():
        if param.requires_grad:
            raise RuntimeError(f"Reference prior parameter is trainable after backward: {name}")
        if param.grad is not None and bool((param.grad.detach() != 0).any().item()):
            raise RuntimeError(f"Reference prior accumulated gradient: {name}")


def _print_loss_bundle(loss_bundle: LossBundle) -> None:
    print("[train_step_smoke_stage2] train preferred error:", loss_bundle.train_pref_err.detach().cpu().tolist())
    print("[train_step_smoke_stage2] train dispreferred error:", loss_bundle.train_dispref_err.detach().cpu().tolist())
    print("[train_step_smoke_stage2] ref preferred error:", loss_bundle.ref_pref_err.detach().cpu().tolist())
    print("[train_step_smoke_stage2] ref dispreferred error:", loss_bundle.ref_dispref_err.detach().cpu().tolist())
    print("[train_step_smoke_stage2] pairwise score:", loss_bundle.score.detach().cpu().tolist())
    print("[train_step_smoke_stage2] loss:", float(loss_bundle.loss.detach().cpu().item()))


def _print_grad_summary(summary: GradSummary) -> None:
    print("[train_step_smoke_stage2] trainable tensors:", summary.trainable_tensors)
    print("[train_step_smoke_stage2] tensors with grad:", summary.tensors_with_grad)
    print("[train_step_smoke_stage2] tensors with nonzero grad:", summary.tensors_with_nonzero_grad)
    print("[train_step_smoke_stage2] total grad norm:", summary.total_grad_norm)
    print("[train_step_smoke_stage2] max grad abs:", summary.max_grad_abs)


def main() -> int:
    args = _build_parser().parse_args()

    try:
        device = _resolve_device(args.device)
        _set_seed(args.seed, device)
        dataset = _load_dataset(args)
        batch = _get_one_batch(dataset, batch_size=max(1, args.batch_size))
        _validate_batch_shapes(batch)
    except Exception:
        print("[train_step_smoke_stage2] dataset load/batch failure")
        print(traceback.format_exc().strip())
        print("\ntrain-step smoke test failed")
        return 1

    try:
        pipe = _load_prior_pipeline(args)
        text = _prepare_text(pipe=pipe, batch=batch, prior=pipe.prior, device=device)
        bundle = _load_and_prepare_models(args=args, pipe=pipe, device=device)
        scheduler = getattr(pipe, "scheduler", None)
        if scheduler is None or not hasattr(scheduler, "add_noise"):
            raise ValueError("Pipeline scheduler must expose add_noise(original_samples, noise, timesteps).")
    except Exception:
        print("[train_step_smoke_stage2] model/text/scheduler preparation failure")
        print(traceback.format_exc().strip())
        print("\ntrain-step smoke test failed")
        return 2

    try:
        preferred, dispreferred, user_emb, user_mask = _prepare_pair_tensors(
            batch=batch,
            prior=bundle.train_prior,
            device=device,
        )
        noisy_pref, noisy_dispref, pref_noise, dispref_noise, timesteps = _make_noisy_pair(
            scheduler=scheduler,
            preferred=preferred,
            dispreferred=dispreferred,
        )
        bundle.train_prior.zero_grad(set_to_none=True)
        bundle.reference_prior.zero_grad(set_to_none=True)

        with user_conditioning_hooks(
            bundle.train_prior,
            user_emb=user_emb,
            user_emb_attention_mask=user_mask,
        ):
            train_pref_pred = _run_prior(bundle.train_prior, text=text, sample=noisy_pref, timesteps=timesteps)
            train_dispref_pred = _run_prior(bundle.train_prior, text=text, sample=noisy_dispref, timesteps=timesteps)

        with torch.no_grad():
            ref_pref_pred = _run_prior(bundle.reference_prior, text=text, sample=noisy_pref, timesteps=timesteps)
            ref_dispref_pred = _run_prior(bundle.reference_prior, text=text, sample=noisy_dispref, timesteps=timesteps)

        loss_bundle = _compute_loss(
            train_pref_pred=train_pref_pred,
            train_dispref_pred=train_dispref_pred,
            ref_pref_pred=ref_pref_pred,
            ref_dispref_pred=ref_dispref_pred,
            preferred_noise=pref_noise,
            dispreferred_noise=dispref_noise,
            dpo_beta=args.dpo_beta,
        )
        loss_bundle.loss.backward()
        grad_summary = _summarize_gradients(bundle.train_prior)
        _validate_reference_no_grads(bundle.reference_prior)
    except Exception:
        print("[train_step_smoke_stage2] forward/loss/backward failure")
        print(traceback.format_exc().strip())
        print("\ntrain-step smoke test failed")
        return 3

    if not args.summary_only:
        _print_batch_diagnostics(batch=batch, dataset_stats=dataset.get_stats())
        _print_tensor_diagnostics(_tensor_diagnostics("preferred clean", preferred))
        _print_tensor_diagnostics(_tensor_diagnostics("dispreferred clean", dispreferred))
        _print_tensor_diagnostics(_tensor_diagnostics("noisy preferred", noisy_pref))
        _print_tensor_diagnostics(_tensor_diagnostics("noisy dispreferred", noisy_dispref))
        _print_tensor_diagnostics(_tensor_diagnostics("timesteps", timesteps))

    print("[train_step_smoke_stage2] patched paths:", bundle.patch_summary.patched_paths)
    print("[train_step_smoke_stage2] trainable parameters:", bundle.trainable_summary.trainable_parameters)
    print("[train_step_smoke_stage2] dpo beta:", args.dpo_beta)
    _print_loss_bundle(loss_bundle)
    _print_grad_summary(grad_summary)
    print("\ntrain-step smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
