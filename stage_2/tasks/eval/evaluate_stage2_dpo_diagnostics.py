from __future__ import annotations

"""Evaluate Stage 2 DPO denoising-loss diagnostics for user-condition variants."""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE2_DIR = REPO_ROOT / "stage_2"
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from train_stage2_full import (  # noqa: E402
    _build_multishard_dataset,
    _build_parser as _build_train_parser,
    _check_resume_compatibility,
    _jsonable,
    _load_and_prepare_models,
    _load_prior_pipeline,
    _load_trainable_state,
    _load_user_branch_state,
    _make_loader,
    _mean_metrics,
    _resolve_device,
    _run_pair_pass,
    _set_seed,
    _torch_load_checkpoint,
    _write_json,
)


CONDITIONS = ("real_user", "zero_user", "shuffled_user", "random_user")


def _clone_batch(batch: Mapping[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def _random_user_emb_like(user_emb: Tensor, *, seed: int) -> Tensor:
    generator = torch.Generator(device=user_emb.device)
    generator.manual_seed(int(seed))
    random_emb = torch.randn(
        tuple(user_emb.shape),
        generator=generator,
        device=user_emb.device,
        dtype=user_emb.dtype,
    )
    source = user_emb.detach().float()
    source_mean = source.mean(dim=tuple(range(1, source.ndim)), keepdim=True).to(device=user_emb.device, dtype=user_emb.dtype)
    source_std = (
        source.std(dim=tuple(range(1, source.ndim)), keepdim=True, unbiased=False)
        .clamp_min(1e-6)
        .to(device=user_emb.device, dtype=user_emb.dtype)
    )
    return random_emb * source_std + source_mean


def _condition_batch(batch: Mapping[str, Any], *, condition: str, seed: int) -> Dict[str, Any]:
    conditioned = _clone_batch(batch)
    user_emb = conditioned["user_emb"]
    if not torch.is_tensor(user_emb):
        raise TypeError("Expected batch['user_emb'] to be a tensor.")

    if condition == "real_user":
        return conditioned
    if condition == "zero_user":
        conditioned["user_emb"] = torch.zeros_like(user_emb)
        return conditioned
    if condition == "shuffled_user":
        if int(user_emb.shape[0]) < 2:
            raise ValueError("shuffled_user DPO diagnostic requires --batch-size >= 2.")
        conditioned["user_emb"] = torch.roll(user_emb, shifts=1, dims=0)
        return conditioned
    if condition == "random_user":
        conditioned["user_emb"] = _random_user_emb_like(user_emb, seed=seed)
        return conditioned
    raise ValueError(f"Unknown condition: {condition}")


def _loss_diff(metrics: Mapping[str, Any]) -> float:
    return float(metrics["train_dispref_err_mean"]) - float(metrics["train_pref_err_mean"])


def _summarize_conditions(rows_by_condition: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    real_loss_diff = None
    if rows_by_condition.get("real_user"):
        real_loss_diff = _loss_diff(_mean_metrics(rows_by_condition["real_user"]))
    for condition, rows in rows_by_condition.items():
        if not rows:
            continue
        averaged = _mean_metrics(rows)
        loss_diff = _loss_diff(averaged)
        summary[condition] = {
            **averaged,
            "train_loss_diff_dispref_minus_pref": loss_diff,
            "train_pref_minus_dispref_err_mean": float(averaged["train_pref_err_mean"])
            - float(averaged["train_dispref_err_mean"]),
            "real_user_loss_diff_advantage": None if real_loss_diff is None else real_loss_diff - loss_diff,
        }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = _build_train_parser()
    parser.description = "Evaluate Stage 2 DPO denoising-loss diagnostics from a checkpoint."
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if int(args.max_batches) < 1:
        raise ValueError("--max-batches must be >= 1.")
    if "shuffled_user" in args.conditions and int(args.batch_size) < 2:
        raise ValueError("shuffled_user DPO diagnostic requires --batch-size >= 2.")

    device = _resolve_device(args.device)
    _set_seed(int(args.seed), device)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    dataset = _build_multishard_dataset(args, split_name=str(args.split))
    pipe = _load_prior_pipeline(args)
    bundle = _load_and_prepare_models(args=args, pipe=pipe, train_device=device)
    checkpoint = _torch_load_checkpoint(args.checkpoint_path.expanduser().resolve(), map_location="cpu")
    _check_resume_compatibility(checkpoint, args)
    if "user_branch_state" in checkpoint:
        _load_user_branch_state(bundle.train_prior, checkpoint["user_branch_state"])
    else:
        _load_trainable_state(bundle.train_prior, checkpoint["trainable_state"])
    bundle.train_prior.eval()
    bundle.reference_prior.eval()

    scheduler = getattr(pipe, "scheduler", None)
    if scheduler is None or not hasattr(scheduler, "add_noise"):
        raise ValueError("Pipeline scheduler must expose add_noise(original_samples, noise, timesteps).")

    loader = _make_loader(
        dataset,
        args=args,
        start_batch=int(args.start_batch),
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    rows_by_condition: Dict[str, List[Dict[str, Any]]] = {str(condition): [] for condition in args.conditions}
    per_batch: List[Dict[str, Any]] = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= int(args.max_batches):
            break
        row: Dict[str, Any] = {
            "batch_index": int(args.start_batch) + batch_idx,
            "batch_size": len(batch.get("caption", [])),
            "conditions": {},
        }
        noise_seed = int(args.seed) + 10_000 * (int(args.start_batch) + batch_idx + 1)
        for condition in args.conditions:
            condition_batch = _condition_batch(batch, condition=str(condition), seed=noise_seed + 17)
            _set_seed(noise_seed, device)
            metrics = _run_pair_pass(
                args=args,
                pipe=pipe,
                bundle=bundle,
                scheduler=scheduler,
                batch=condition_batch,
                backward=False,
                loss_scale=1.0,
            )
            metrics["train_loss_diff_dispref_minus_pref"] = _loss_diff(metrics)
            rows_by_condition[str(condition)].append(metrics)
            row["conditions"][str(condition)] = metrics
        per_batch.append(row)

    summary = _summarize_conditions(rows_by_condition)
    payload = {
        "mode": "stage2_dpo_diagnostics",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_path": str(args.checkpoint_path.expanduser().resolve()),
        "split": str(args.split),
        "device": str(device),
        "reference_device": str(bundle.reference_device),
        "batch_size": int(args.batch_size),
        "start_batch": int(args.start_batch),
        "max_batches": int(args.max_batches),
        "conditions": list(args.conditions),
        "dataset_stats": dataset.get_stats(),
        "summary": summary,
        "batches": per_batch,
    }
    if args.output_json is not None:
        _write_json(args.output_json, payload)
        print(f"[evaluate_stage2_dpo_diagnostics] wrote {args.output_json}")
    else:
        print(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
