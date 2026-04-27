from __future__ import annotations

"""Epoch-based Stage 2 full training entrypoint.

This is the production-oriented counterpart to the Stage 2 smoke scripts. It
keeps the Stable Cascade prior backbone frozen and updates only the patched
user-conditioning branch with the personalized pairwise diffusion-DPO loss.
"""

import argparse
import bisect
import copy
import glob
import json
import math
import random
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from forward_only_stage2 import (
        _load_prior_pipeline,
        _resolve_device,
        _validate_batch_shapes,
        user_conditioning_hooks,
    )
    from stage2_dataset import Stage2PreferenceDataset
    from train_smoke_stage2 import (
        GradStats,
        ModelBundle,
        ParamSliceSnapshot,
        _append_jsonl,
        _capture_param_slices,
        _compute_loss_from_errors,
        _cuda_memory_mb,
        _effective_prior_dtype,
        _encode_text_for_devices,
        _full_frozen_integrity_check,
        _grad_stats,
        _is_oom_error,
        _jsonable,
        _light_frozen_grad_check,
        _load_and_prepare_models,
        _make_noisy_pair,
        _make_run_dir,
        _param_slice_delta,
        _print_memory_hint,
        _tensor_max,
        _tensor_mean,
        _tensor_min,
        _trainable_named_parameters,
        _validate_optimizer_scope,
        _write_json,
    )
    from train_step_smoke_stage2 import (
        USER_CONDITIONING_NAME_MARKERS,
        LossBundle,
        _is_user_conditioning_param,
        _per_sample_mse,
        _prepare_pair_tensors,
        _require_finite,
        _run_prior,
        _set_seed,
    )
except ImportError:  # pragma: no cover - useful when imported as package module
    from stage_2.forward_only_stage2 import (
        _load_prior_pipeline,
        _resolve_device,
        _validate_batch_shapes,
        user_conditioning_hooks,
    )
    from stage_2.stage2_dataset import Stage2PreferenceDataset
    from stage_2.train_smoke_stage2 import (
        GradStats,
        ModelBundle,
        ParamSliceSnapshot,
        _append_jsonl,
        _capture_param_slices,
        _compute_loss_from_errors,
        _cuda_memory_mb,
        _effective_prior_dtype,
        _encode_text_for_devices,
        _full_frozen_integrity_check,
        _grad_stats,
        _is_oom_error,
        _jsonable,
        _light_frozen_grad_check,
        _load_and_prepare_models,
        _make_noisy_pair,
        _make_run_dir,
        _param_slice_delta,
        _print_memory_hint,
        _tensor_max,
        _tensor_mean,
        _tensor_min,
        _trainable_named_parameters,
        _validate_optimizer_scope,
        _write_json,
    )
    from stage_2.train_step_smoke_stage2 import (
        USER_CONDITIONING_NAME_MARKERS,
        LossBundle,
        _is_user_conditioning_param,
        _per_sample_mse,
        _prepare_pair_tensors,
        _require_finite,
        _run_prior,
        _set_seed,
    )


DEFAULT_OUTPUT_ROOT = Path("artifacts/stage2_train_full")
DEFAULT_TRAIN_EMBEDDING_PATTERNS = ["data/user_emb_7b_full/train_shard*.json"]
DEFAULT_TRAIN_ASSIGNMENT_PATTERNS = [
    "artifacts/pair_assignments/train/stage2_pair_assignments_train_shard*.jsonl"
]
DEFAULT_TRAIN_LATENT_MANIFEST = Path("artifacts/stage_c_latents/latent_manifest_train_v512.jsonl")
DEFAULT_TRAIN_UID_TO_PATH = Path("data/train_uid_to_path.json")
DATASET_SCHEMA_VERSION = "stage2_preference_real_latent_v1"
EXPECTED_LATENT_SHAPE = (16, 12, 12)


@dataclass(frozen=True)
class ShardPair:
    shard_id: int
    embedding_json_path: Path
    assignment_jsonl_path: Path


@dataclass
class TrainState:
    micro_step: int = 0
    optimizer_step: int = 0
    samples_seen: int = 0
    best_val_loss: Optional[float] = None
    latest_checkpoint_path: Optional[str] = None
    final_loss: Optional[float] = None
    max_grad_norm_observed: float = 0.0
    max_cuda_mem_reserved_mb: float = 0.0
    nan_failures: int = 0
    frozen_integrity_failures: int = 0
    data_errors: int = 0


def _load_mapping_json(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"Mapping JSON must be an object/dict: {resolved}")
    return {str(key): value for key, value in data.items()}


def _load_latent_manifest_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    resolved = path.expanduser().resolve()
    manifest: Dict[str, Dict[str, Any]] = {}
    with resolved.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse latent manifest line {line_idx} in {resolved}: {exc}") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"Latent manifest line {line_idx} in {resolved} is not an object/dict.")
            uid = record.get("uid")
            if uid is None or (isinstance(uid, str) and uid.strip() == ""):
                raise ValueError(f"Latent manifest line {line_idx} in {resolved} is missing `uid`.")
            uid_key = str(uid)
            if uid_key in manifest:
                raise ValueError(f"Duplicate uid in latent manifest: {uid_key}")
            manifest[uid_key] = dict(record)
    return manifest


def _count_assignment_query_samples(path: Path) -> int:
    resolved = path.expanduser().resolve()
    total = 0
    with resolved.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse assignment line {line_idx} in {resolved}: {exc}") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"Assignment line {line_idx} in {resolved} is not an object/dict.")
            query_pairs = record.get("query_pairs")
            if not isinstance(query_pairs, list):
                raise ValueError(f"Assignment line {line_idx} in {resolved} has invalid `query_pairs`.")
            total += len(query_pairs)
    return total


class MultiShardStage2PreferenceDataset(Dataset):
    """Concatenate assignment-backed Stage2PreferenceDataset shards."""

    collate_fn = staticmethod(Stage2PreferenceDataset.collate_fn)

    def __init__(
        self,
        *,
        embedding_json_paths: Sequence[Path],
        assignment_jsonl_paths: Sequence[Path],
        latent_manifest_jsonl_path: Path,
        uid_to_path_json_path: Optional[Path],
        split_name: str,
        skip_missing_latents: bool,
        validate_assignment_support_pairs: bool,
    ) -> None:
        self.split_name = split_name
        self.latent_manifest_jsonl_path = latent_manifest_jsonl_path
        self.uid_to_path_json_path = uid_to_path_json_path
        self.shards = _match_shards_by_id(embedding_json_paths, assignment_jsonl_paths)
        if not self.shards:
            raise ValueError(f"{split_name}: no matched embedding/assignment shards.")

        self.shared_uid_to_path = _load_mapping_json(uid_to_path_json_path) if uid_to_path_json_path else {}
        self.shared_latent_manifest = _load_latent_manifest_jsonl(latent_manifest_jsonl_path)
        self.skip_missing_latents = skip_missing_latents
        self.validate_assignment_support_pairs = validate_assignment_support_pairs
        self._cached_shard_idx: Optional[int] = None
        self._cached_dataset: Optional[Stage2PreferenceDataset] = None
        self.shard_lengths: List[int] = []
        self.cumulative_lengths: List[int] = []
        total = 0
        for shard in self.shards:
            shard_len = _count_assignment_query_samples(shard.assignment_jsonl_path)
            self.shard_lengths.append(shard_len)
            total += shard_len
            self.cumulative_lengths.append(total)

        if total == 0:
            raise ValueError(f"{split_name}: all shards produced zero valid samples.")

    def _make_shard_dataset(self, shard: ShardPair) -> Stage2PreferenceDataset:
        return Stage2PreferenceDataset(
            embedding_json_path=shard.embedding_json_path,
            assignment_jsonl_path=shard.assignment_jsonl_path,
            latent_manifest_jsonl_path=self.latent_manifest_jsonl_path,
            uid_to_path_json_path=self.uid_to_path_json_path,
            load_images=False,
            load_latents=False,
            skip_malformed_pairs=True,
            skip_missing_latents=self.skip_missing_latents,
            validate_assignment_support_pairs=self.validate_assignment_support_pairs,
            preloaded_uid_to_path=self.shared_uid_to_path,
            preloaded_latent_manifest=self.shared_latent_manifest,
        )

    def _get_shard_dataset(self, shard_idx: int) -> Stage2PreferenceDataset:
        if self._cached_shard_idx == shard_idx and self._cached_dataset is not None:
            return self._cached_dataset
        dataset = self._make_shard_dataset(self.shards[shard_idx])
        self._cached_shard_idx = shard_idx
        self._cached_dataset = dataset
        return dataset

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_idx = bisect.bisect_right(self.cumulative_lengths, index)
        prev = self.cumulative_lengths[shard_idx - 1] if shard_idx > 0 else 0
        dataset = self._get_shard_dataset(shard_idx)
        item = dict(dataset[index - prev])
        if "preferred_latent" not in item:
            item["preferred_latent"] = dataset._load_latent_tensor(str(item["preferred_latent_path"]))
        if "dispreferred_latent" not in item:
            item["dispreferred_latent"] = dataset._load_latent_tensor(str(item["dispreferred_latent_path"]))
        for key in ("preferred_latent", "dispreferred_latent"):
            shape = tuple(item[key].shape)
            if shape != EXPECTED_LATENT_SHAPE:
                raise ValueError(f"{key} shape mismatch: expected={EXPECTED_LATENT_SHAPE}, got={shape}")
        shard = self.shards[shard_idx]
        item["source_shard_id"] = shard.shard_id
        item["source_embedding_json_path"] = str(shard.embedding_json_path)
        item["source_assignment_jsonl_path"] = str(shard.assignment_jsonl_path)
        return item

    def get_stats(self) -> Dict[str, Any]:
        return {
            "split_name": self.split_name,
            "num_shards": len(self.shards),
            "num_samples": len(self),
            "latent_manifest_jsonl_path": str(self.latent_manifest_jsonl_path),
            "uid_to_path_json_path": str(self.uid_to_path_json_path) if self.uid_to_path_json_path else None,
            "shards": [
                {
                    "shard_id": shard.shard_id,
                    "embedding_json_path": str(shard.embedding_json_path),
                    "assignment_jsonl_path": str(shard.assignment_jsonl_path),
                    "num_samples": self.shard_lengths[idx],
                }
                for idx, shard in enumerate(self.shards)
            ],
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2 full epoch-based diffusion-DPO training.")
    parser.add_argument("--train-embedding-json-paths", nargs="+", default=DEFAULT_TRAIN_EMBEDDING_PATTERNS)
    parser.add_argument("--train-assignment-jsonl-paths", nargs="+", default=DEFAULT_TRAIN_ASSIGNMENT_PATTERNS)
    parser.add_argument("--train-latent-manifest-jsonl-path", type=Path, default=DEFAULT_TRAIN_LATENT_MANIFEST)
    parser.add_argument("--train-uid-to-path-json-path", type=Path, default=DEFAULT_TRAIN_UID_TO_PATH)
    parser.add_argument("--val-embedding-json-paths", nargs="+", default=None)
    parser.add_argument("--val-assignment-jsonl-paths", nargs="+", default=None)
    parser.add_argument("--val-latent-manifest-jsonl-path", type=Path, default=None)
    parser.add_argument("--val-uid-to-path-json-path", type=Path, default=None)
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
        help="Validate assignment support pairs against embedding rows (default: True).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
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
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=("cosine", "linear", "constant"))
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fail-grad-norm", type=float, default=1000.0)
    parser.add_argument("--fail-on-nan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--val-every-steps", type=int, default=100)
    parser.add_argument("--max-val-batches", type=int, default=20)
    parser.add_argument("--max-consecutive-data-errors", type=int, default=0)
    parser.add_argument("--frozen-check-every", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--dataset-dry-run", action="store_true")
    return parser


def _natural_sort_key(path: Path) -> Tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path)))


def _expand_paths(values: Optional[Sequence[str]], *, label: str, required: bool) -> List[Path]:
    if not values:
        if required:
            raise ValueError(f"{label} requires at least one path or glob pattern.")
        return []
    paths: List[Path] = []
    for raw in values:
        pattern = str(Path(str(raw)).expanduser())
        matches = [Path(match).resolve() for match in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            path = Path(pattern).resolve()
            if any(ch in pattern for ch in "*?[]"):
                raise FileNotFoundError(f"{label} glob matched no files: {raw}")
            if not path.exists():
                raise FileNotFoundError(f"{label} file not found: {path}")
            paths.append(path)
    deduped = sorted(dict.fromkeys(paths), key=_natural_sort_key)
    if required and not deduped:
        raise ValueError(f"{label} resolved to zero paths.")
    return deduped


def _shard_id(path: Path) -> int:
    match = re.search(r"shard(\d+)", path.name)
    if match is None:
        raise ValueError(f"Could not infer shard id from path name: {path}")
    return int(match.group(1))


def _match_shards_by_id(embedding_paths: Sequence[Path], assignment_paths: Sequence[Path]) -> List[ShardPair]:
    emb_by_id = {_shard_id(path): path for path in embedding_paths}
    assign_by_id = {_shard_id(path): path for path in assignment_paths}
    if len(emb_by_id) != len(embedding_paths):
        raise ValueError("Duplicate embedding shard ids detected.")
    if len(assign_by_id) != len(assignment_paths):
        raise ValueError("Duplicate assignment shard ids detected.")
    missing_assignments = sorted(set(emb_by_id) - set(assign_by_id))
    missing_embeddings = sorted(set(assign_by_id) - set(emb_by_id))
    if missing_assignments or missing_embeddings:
        raise ValueError(
            "Embedding/assignment shard ids do not match: "
            f"missing_assignments={missing_assignments[:10]}, missing_embeddings={missing_embeddings[:10]}"
        )
    return [
        ShardPair(shard_id=shard_id, embedding_json_path=emb_by_id[shard_id], assignment_jsonl_path=assign_by_id[shard_id])
        for shard_id in sorted(emb_by_id)
    ]


def _validation_enabled(args: argparse.Namespace) -> bool:
    provided = [
        args.val_embedding_json_paths is not None,
        args.val_assignment_jsonl_paths is not None,
        args.val_latent_manifest_jsonl_path is not None,
    ]
    if any(provided) and not all(provided):
        raise ValueError(
            "Validation requires --val-embedding-json-paths, --val-assignment-jsonl-paths, "
            "and --val-latent-manifest-jsonl-path together."
        )
    return all(provided)


def _build_multishard_dataset(args: argparse.Namespace, *, split_name: str) -> MultiShardStage2PreferenceDataset:
    if split_name == "train":
        embedding_paths = _expand_paths(args.train_embedding_json_paths, label="train embeddings", required=True)
        assignment_paths = _expand_paths(args.train_assignment_jsonl_paths, label="train assignments", required=True)
        latent_manifest = args.train_latent_manifest_jsonl_path.expanduser().resolve()
        uid_to_path = args.train_uid_to_path_json_path.expanduser().resolve() if args.train_uid_to_path_json_path else None
    elif split_name == "val":
        embedding_paths = _expand_paths(args.val_embedding_json_paths, label="val embeddings", required=True)
        assignment_paths = _expand_paths(args.val_assignment_jsonl_paths, label="val assignments", required=True)
        latent_manifest = args.val_latent_manifest_jsonl_path.expanduser().resolve()
        uid_to_path = args.val_uid_to_path_json_path.expanduser().resolve() if args.val_uid_to_path_json_path else None
    else:
        raise ValueError(f"Unknown split_name: {split_name}")

    if not latent_manifest.exists():
        raise FileNotFoundError(f"{split_name} latent manifest not found: {latent_manifest}")
    if uid_to_path is not None and not uid_to_path.exists():
        raise FileNotFoundError(f"{split_name} uid-to-path JSON not found: {uid_to_path}")

    return MultiShardStage2PreferenceDataset(
        embedding_json_paths=embedding_paths,
        assignment_jsonl_paths=assignment_paths,
        latent_manifest_jsonl_path=latent_manifest,
        uid_to_path_json_path=uid_to_path,
        split_name=split_name,
        skip_missing_latents=bool(args.skip_missing_latents),
        validate_assignment_support_pairs=bool(args.validate_assignment_support_pairs),
    )


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader(
    dataset: Dataset,
    *,
    args: argparse.Namespace,
    start_batch: int = 0,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    batch_size = max(1, int(args.batch_size))
    start_index = max(0, int(start_batch)) * batch_size
    if start_index >= len(dataset):
        subset: Dataset = Subset(dataset, [])
    elif start_index > 0:
        subset = Subset(dataset, range(start_index, len(dataset)))
    else:
        subset = dataset
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=Stage2PreferenceDataset.collate_fn,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        generator=generator,
    )


def _num_batches(num_samples: int, batch_size: int) -> int:
    return int(math.ceil(float(num_samples) / float(max(1, batch_size))))


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    warmup_steps: int,
    total_optimizer_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(0, int(warmup_steps))
    total = max(1, int(total_optimizer_steps))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if scheduler_name == "constant":
            return 1.0
        denom = max(1, total - warmup)
        progress = min(1.0, max(0.0, float(step - warmup) / float(denom)))
        if scheduler_name == "linear":
            return max(0.0, 1.0 - progress)
        if scheduler_name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _run_pair_pass(
    *,
    args: argparse.Namespace,
    pipe: Any,
    bundle: ModelBundle,
    scheduler: Any,
    batch: Mapping[str, Any],
    backward: bool,
    loss_scale: float,
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

    if backward:
        with user_conditioning_hooks(
            bundle.train_prior,
            user_emb=user_emb,
            user_emb_attention_mask=user_mask,
        ):
            train_pref_pred = _run_prior(bundle.train_prior, text=train_text, sample=noisy_pref, timesteps=timesteps)
            train_dispref_pred = _run_prior(bundle.train_prior, text=train_text, sample=noisy_dispref, timesteps=timesteps)
    else:
        with torch.no_grad():
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
    if backward:
        (loss_bundle.loss / float(loss_scale)).backward()

    batch_size = len(batch["caption"]) if "caption" in batch else int(preferred.shape[0])
    return {
        "loss": float(loss_bundle.loss.detach().cpu().item()),
        "train_pref_err_mean": _tensor_mean(loss_bundle.train_pref_err),
        "train_dispref_err_mean": _tensor_mean(loss_bundle.train_dispref_err),
        "ref_pref_err_mean": _tensor_mean(loss_bundle.ref_pref_err),
        "ref_dispref_err_mean": _tensor_mean(loss_bundle.ref_dispref_err),
        "score_mean": _tensor_mean(loss_bundle.score),
        "timestep_min": _tensor_min(timesteps),
        "timestep_max": _tensor_max(timesteps),
        "timestep_mean": _tensor_mean(timesteps),
        "batch_size": batch_size,
        "step_time_sec": time.time() - step_start,
    }


def _mean_metrics(metrics: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not metrics:
        raise ValueError("Cannot average an empty metrics list.")
    keys = [
        "loss",
        "train_pref_err_mean",
        "train_dispref_err_mean",
        "ref_pref_err_mean",
        "ref_dispref_err_mean",
        "score_mean",
        "timestep_min",
        "timestep_max",
        "timestep_mean",
        "step_time_sec",
    ]
    return {key: float(sum(float(item[key]) for item in metrics) / len(metrics)) for key in keys}


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _collect_trainable_state(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad and _is_user_conditioning_param(name)
    }


def _load_trainable_state(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    current = dict(_trainable_named_parameters(model))
    expected = {name for name in current if _is_user_conditioning_param(name)}
    incoming = set(state.keys())
    missing = sorted(expected - incoming)
    extra = sorted(incoming - expected)
    if missing or extra:
        raise RuntimeError(f"Trainable state key mismatch: missing={missing[:10]}, extra={extra[:10]}")
    with torch.no_grad():
        for name in sorted(expected):
            current[name].copy_(state[name].to(device=current[name].device, dtype=current[name].dtype))


def _rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _critical_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "schema": DATASET_SCHEMA_VERSION,
        "model_id": args.model_id,
        "patch_path": _jsonable(args.patch_path),
        "user_scale": float(args.user_scale),
        "trainable_markers": list(USER_CONDITIONING_NAME_MARKERS),
        "latent_shape": list(EXPECTED_LATENT_SHAPE),
    }


def _check_resume_compatibility(checkpoint: Mapping[str, Any], args: argparse.Namespace) -> None:
    expected = _critical_config(args)
    found = checkpoint.get("critical_config")
    if found is None:
        raise RuntimeError("Checkpoint is missing critical_config.")
    mismatches = {key: (found.get(key), value) for key, value in expected.items() if found.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Checkpoint config mismatch: {mismatches}")


def _torch_load_checkpoint(path: Path, map_location: Any = "cpu") -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _save_checkpoint(
    *,
    path: Path,
    args: argparse.Namespace,
    bundle: ModelBundle,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    state: TrainState,
    dataset_stats: Mapping[str, Any],
    run_dir: Path,
) -> str:
    payload = {
        "trainable_state": _collect_trainable_state(bundle.train_prior),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "micro_step": int(state.micro_step),
        "optimizer_step": int(state.optimizer_step),
        "samples_seen": int(state.samples_seen),
        "best_val_loss": state.best_val_loss,
        "args": _jsonable(vars(args)),
        "critical_config": _critical_config(args),
        "dataset_stats": _jsonable(dataset_stats),
        "rng_state": _rng_state(),
        "run_dir": str(run_dir),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    state.latest_checkpoint_path = str(path)
    return str(path)


def _validate_updated_params(bundle: ModelBundle) -> None:
    for name, param in _trainable_named_parameters(bundle.train_prior):
        _require_finite(f"updated_param[{name}]", param)


def _apply_optimizer_step(
    *,
    args: argparse.Namespace,
    bundle: ModelBundle,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
) -> Tuple[GradStats, GradStats]:
    pre_clip = _grad_stats(bundle.train_prior, check_frozen_grads=True)
    if not math.isfinite(pre_clip.total_grad_norm):
        raise RuntimeError(f"Non-finite pre-clip grad norm: {pre_clip.total_grad_norm}")
    if pre_clip.total_grad_norm > float(args.fail_grad_norm):
        raise RuntimeError(
            f"Pre-clip grad norm {pre_clip.total_grad_norm:.6f} exceeds --fail-grad-norm={args.fail_grad_norm}."
        )
    torch.nn.utils.clip_grad_norm_(
        [param for _, param in _trainable_named_parameters(bundle.train_prior)],
        max_norm=float(args.max_grad_norm),
    )
    post_clip = _grad_stats(bundle.train_prior, check_frozen_grads=True)
    if not math.isfinite(post_clip.total_grad_norm):
        raise RuntimeError(f"Non-finite post-clip grad norm: {post_clip.total_grad_norm}")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    bundle.train_prior.zero_grad(set_to_none=True)
    _validate_updated_params(bundle)
    _light_frozen_grad_check(bundle)
    return pre_clip, post_clip


def _train_metrics_row(
    *,
    averaged: Mapping[str, float],
    pre_clip: GradStats,
    post_clip: GradStats,
    optimizer: torch.optim.Optimizer,
    state: TrainState,
    epoch: int,
    accumulation_count: int,
    bundle: ModelBundle,
) -> Dict[str, Any]:
    allocated_mb, reserved_mb = _cuda_memory_mb(bundle.train_device)
    return {
        "micro_step": state.micro_step,
        "optimizer_step": state.optimizer_step,
        "epoch": epoch,
        "samples_seen": state.samples_seen,
        "accumulation_count": accumulation_count,
        "loss": averaged["loss"],
        "score_mean": averaged["score_mean"],
        "train_pref_err_mean": averaged["train_pref_err_mean"],
        "ref_pref_err_mean": averaged["ref_pref_err_mean"],
        "train_dispref_err_mean": averaged["train_dispref_err_mean"],
        "ref_dispref_err_mean": averaged["ref_dispref_err_mean"],
        "timestep_min": averaged["timestep_min"],
        "timestep_max": averaged["timestep_max"],
        "timestep_mean": averaged["timestep_mean"],
        "grad_norm_pre_clip": pre_clip.total_grad_norm,
        "grad_norm_post_clip": post_clip.total_grad_norm,
        "max_grad_abs": post_clip.max_grad_abs,
        "tensors_with_grad": post_clip.tensors_with_grad,
        "tensors_with_nonzero_grad": post_clip.tensors_with_nonzero_grad,
        "lr": float(optimizer.param_groups[0]["lr"]),
        "step_time_sec": averaged["step_time_sec"],
        "cuda_allocated_mb": allocated_mb,
        "cuda_reserved_mb": reserved_mb,
    }


def _run_validation(
    *,
    args: argparse.Namespace,
    pipe: Any,
    bundle: ModelBundle,
    scheduler: Any,
    val_dataset: MultiShardStage2PreferenceDataset,
    generator: torch.Generator,
    trigger_optimizer_step: int,
) -> Dict[str, Any]:
    was_training = bundle.train_prior.training
    bundle.train_prior.eval()
    bundle.reference_prior.eval()
    max_batches = max(1, int(args.max_val_batches))
    val_loader = _make_loader(val_dataset, args=args, start_batch=0, generator=generator)
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            rows.append(
                _run_pair_pass(
                    args=args,
                    pipe=pipe,
                    bundle=bundle,
                    scheduler=scheduler,
                    batch=batch,
                    backward=False,
                    loss_scale=1.0,
                )
            )
    if was_training:
        bundle.train_prior.train()
    if not rows:
        raise RuntimeError("Validation produced zero batches.")
    averaged = _mean_metrics(rows)
    return {
        "trigger_optimizer_step": trigger_optimizer_step,
        "val_batches": len(rows),
        "val_loss": averaged["loss"],
        "val_score_mean": averaged["score_mean"],
        "val_train_pref_err_mean": averaged["train_pref_err_mean"],
        "val_ref_pref_err_mean": averaged["ref_pref_err_mean"],
        "val_train_dispref_err_mean": averaged["train_dispref_err_mean"],
        "val_ref_dispref_err_mean": averaged["ref_dispref_err_mean"],
        "val_step_time_sec": averaged["step_time_sec"],
    }


def _initial_summary(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    train_dataset: Optional[MultiShardStage2PreferenceDataset] = None,
    val_dataset: Optional[MultiShardStage2PreferenceDataset] = None,
) -> Dict[str, Any]:
    return {
        "run_status": "failed",
        "num_micro_steps_completed": 0,
        "num_optimizer_steps_completed": 0,
        "samples_seen": 0,
        "nan_failures": 0,
        "frozen_integrity_failures": 0,
        "data_errors": 0,
        "max_grad_norm_observed": 0.0,
        "max_cuda_mem_reserved_mb": 0.0,
        "final_loss": None,
        "best_val_loss": None,
        "parameter_delta_status": "not_checked",
        "parameter_delta_max_abs": 0.0,
        "failure_message": None,
        "latest_checkpoint_path": None,
        "run_dir": str(run_dir),
        "metrics_train_path": str(run_dir / "metrics_train.jsonl"),
        "metrics_val_path": str(run_dir / "metrics_val.jsonl"),
        "summary_path": str(run_dir / "summary.json"),
        "config_path": str(run_dir / "config.json"),
        "checkpoint_latest_path": str(run_dir / "checkpoint_latest.pt"),
        "args": _jsonable(vars(args)),
        "train_dataset_stats": _jsonable(train_dataset.get_stats()) if train_dataset else None,
        "val_dataset_stats": _jsonable(val_dataset.get_stats()) if val_dataset else None,
    }


def _update_summary_from_state(summary: Dict[str, Any], state: TrainState) -> None:
    summary["num_micro_steps_completed"] = state.micro_step
    summary["num_optimizer_steps_completed"] = state.optimizer_step
    summary["samples_seen"] = state.samples_seen
    summary["best_val_loss"] = state.best_val_loss
    summary["final_loss"] = state.final_loss
    summary["nan_failures"] = state.nan_failures
    summary["frozen_integrity_failures"] = state.frozen_integrity_failures
    summary["data_errors"] = state.data_errors
    summary["max_grad_norm_observed"] = state.max_grad_norm_observed
    summary["max_cuda_mem_reserved_mb"] = state.max_cuda_mem_reserved_mb
    summary["latest_checkpoint_path"] = state.latest_checkpoint_path


def _write_config(run_dir: Path, args: argparse.Namespace, train_dataset: MultiShardStage2PreferenceDataset, val_dataset: Optional[MultiShardStage2PreferenceDataset]) -> None:
    _write_json(
        run_dir / "config.json",
        {
            "args": vars(args),
            "critical_config": _critical_config(args),
            "train_dataset_stats": train_dataset.get_stats(),
            "val_dataset_stats": val_dataset.get_stats() if val_dataset else None,
        },
    )


def _maybe_dataset_dry_run(args: argparse.Namespace, train_dataset: MultiShardStage2PreferenceDataset, val_dataset: Optional[MultiShardStage2PreferenceDataset]) -> bool:
    if not args.dataset_dry_run:
        return False
    loader = _make_loader(train_dataset, args=args, start_batch=0, generator=torch.Generator().manual_seed(int(args.seed)))
    batch = next(iter(loader))
    _validate_batch_shapes(batch)
    print("[train_stage2_full] dataset dry run passed")
    print("[train_stage2_full] train samples:", len(train_dataset))
    print("[train_stage2_full] train shards:", len(train_dataset.shards))
    print("[train_stage2_full] user_emb shape:", tuple(batch["user_emb"].shape))
    print("[train_stage2_full] preferred_latent shape:", tuple(batch["preferred_latent"].shape))
    print("[train_stage2_full] dispreferred_latent shape:", tuple(batch["dispreferred_latent"].shape))
    if val_dataset is not None:
        print("[train_stage2_full] val samples:", len(val_dataset))
        print("[train_stage2_full] val shards:", len(val_dataset.shards))
    return True


def main() -> int:
    args = _build_parser().parse_args()
    run_dir = Path(args.resume_from).expanduser().resolve().parent if args.resume_from else _make_run_dir(args.output_dir)
    metrics_train_path = run_dir / "metrics_train.jsonl"
    metrics_val_path = run_dir / "metrics_val.jsonl"
    summary_path = run_dir / "summary.json"
    run_dir.mkdir(parents=True, exist_ok=bool(args.resume_from))

    summary: Dict[str, Any] = {}
    state = TrainState()

    try:
        if int(args.gradient_accumulation_steps) < 1:
            raise ValueError("--gradient-accumulation-steps must be >= 1.")
        if int(args.num_epochs) < 1:
            raise ValueError("--num-epochs must be >= 1.")

        train_device = _resolve_device(args.device)
        _set_seed(args.seed, train_device)
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))

        train_dataset = _build_multishard_dataset(args, split_name="train")
        val_dataset = _build_multishard_dataset(args, split_name="val") if _validation_enabled(args) else None
        summary = _initial_summary(args=args, run_dir=run_dir, train_dataset=train_dataset, val_dataset=val_dataset)
        _write_json(summary_path, summary)
        _write_config(run_dir, args, train_dataset, val_dataset)

        if _maybe_dataset_dry_run(args, train_dataset, val_dataset):
            summary["run_status"] = "success"
            _write_json(summary_path, summary)
            return 0

        train_batches_per_epoch = _num_batches(len(train_dataset), int(args.batch_size))
        total_micro_steps = int(args.num_epochs) * train_batches_per_epoch
        if args.max_train_steps is not None:
            total_micro_steps = min(total_micro_steps, max(1, int(args.max_train_steps)))
        total_optimizer_steps = _num_batches(total_micro_steps, int(args.gradient_accumulation_steps))

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
        lr_scheduler = _build_lr_scheduler(
            optimizer,
            scheduler_name=args.lr_scheduler,
            warmup_steps=int(args.warmup_steps),
            total_optimizer_steps=total_optimizer_steps,
        )
        _validate_optimizer_scope(bundle.train_prior, optimizer)
        _full_frozen_integrity_check(bundle, optimizer=optimizer)

        checkpoint: Optional[Mapping[str, Any]] = None
        if args.resume_from is not None:
            checkpoint = _torch_load_checkpoint(args.resume_from.expanduser().resolve(), map_location="cpu")
            _check_resume_compatibility(checkpoint, args)
            _load_trainable_state(bundle.train_prior, checkpoint["trainable_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            _optimizer_to_device(optimizer, bundle.train_device)
            lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
            state.micro_step = int(checkpoint.get("micro_step", 0))
            state.optimizer_step = int(checkpoint.get("optimizer_step", 0))
            state.samples_seen = int(checkpoint.get("samples_seen", 0))
            state.best_val_loss = checkpoint.get("best_val_loss")
            state.latest_checkpoint_path = str(args.resume_from.expanduser().resolve())
            _restore_rng_state(checkpoint.get("rng_state", {}))
            _full_frozen_integrity_check(bundle, optimizer=optimizer)

        before = _capture_param_slices(bundle.train_prior)
        generator = torch.Generator()
        generator.manual_seed(int(args.seed))
        optimizer.zero_grad(set_to_none=True)
        bundle.train_prior.zero_grad(set_to_none=True)
        bundle.reference_prior.zero_grad(set_to_none=True)

        accumulation: List[Dict[str, Any]] = []
        frozen_check_every = int(args.frozen_check_every or 0)
        consecutive_data_errors = 0
        latest_train_metrics: Optional[Dict[str, Any]] = None

        while state.micro_step < total_micro_steps:
            epoch = state.micro_step // train_batches_per_epoch
            start_batch = state.micro_step % train_batches_per_epoch
            loader = _make_loader(train_dataset, args=args, start_batch=start_batch, generator=generator)
            for batch in loader:
                if state.micro_step >= total_micro_steps:
                    break
                try:
                    raw_metrics = _run_pair_pass(
                        args=args,
                        pipe=pipe,
                        bundle=bundle,
                        scheduler=scheduler,
                        batch=batch,
                        backward=True,
                        loss_scale=float(args.gradient_accumulation_steps),
                    )
                    consecutive_data_errors = 0
                except Exception as exc:
                    if _is_oom_error(exc):
                        if train_device.type == "cuda":
                            torch.cuda.empty_cache()
                        _print_memory_hint(exc)
                    consecutive_data_errors += 1
                    state.data_errors += 1
                    if consecutive_data_errors > int(args.max_consecutive_data_errors):
                        raise
                    print("[train_stage2_full] data/step error skipped:", str(exc))
                    continue

                state.micro_step += 1
                state.samples_seen += int(raw_metrics["batch_size"])
                accumulation.append(raw_metrics)
                should_step = (
                    len(accumulation) >= int(args.gradient_accumulation_steps)
                    or state.micro_step >= total_micro_steps
                )
                if not should_step:
                    continue

                pre_clip, post_clip = _apply_optimizer_step(
                    args=args,
                    bundle=bundle,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                )
                state.optimizer_step += 1
                averaged = _mean_metrics(accumulation)
                accumulation_count = len(accumulation)
                accumulation.clear()
                train_metrics = _train_metrics_row(
                    averaged=averaged,
                    pre_clip=pre_clip,
                    post_clip=post_clip,
                    optimizer=optimizer,
                    state=state,
                    epoch=epoch,
                    accumulation_count=accumulation_count,
                    bundle=bundle,
                )
                _append_jsonl(metrics_train_path, train_metrics)
                latest_train_metrics = train_metrics
                state.final_loss = float(train_metrics["loss"])
                state.max_grad_norm_observed = max(state.max_grad_norm_observed, float(train_metrics["grad_norm_pre_clip"]))
                state.max_cuda_mem_reserved_mb = max(state.max_cuda_mem_reserved_mb, float(train_metrics["cuda_reserved_mb"]))

                if frozen_check_every > 0 and state.optimizer_step % frozen_check_every == 0:
                    _full_frozen_integrity_check(bundle, optimizer=optimizer)

                if state.optimizer_step % max(1, int(args.log_every)) == 0 or state.optimizer_step == 1:
                    print(
                        "[train_stage2_full] "
                        f"optimizer_step={state.optimizer_step}/{total_optimizer_steps} "
                        f"micro_step={state.micro_step}/{total_micro_steps} "
                        f"epoch={epoch + 1}/{args.num_epochs} "
                        f"loss={train_metrics['loss']:.6f} "
                        f"grad_norm={train_metrics['grad_norm_pre_clip']:.6f} "
                        f"cuda_reserved_mb={train_metrics['cuda_reserved_mb']:.2f}"
                    )

                if (
                    int(args.checkpoint_every_steps) > 0
                    and state.optimizer_step % int(args.checkpoint_every_steps) == 0
                ):
                    step_path = run_dir / f"checkpoint_step_{state.optimizer_step:06d}.pt"
                    _save_checkpoint(
                        path=step_path,
                        args=args,
                        bundle=bundle,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        state=state,
                        dataset_stats={"train": train_dataset.get_stats(), "val": val_dataset.get_stats() if val_dataset else None},
                        run_dir=run_dir,
                    )
                    _save_checkpoint(
                        path=run_dir / "checkpoint_latest.pt",
                        args=args,
                        bundle=bundle,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        state=state,
                        dataset_stats={"train": train_dataset.get_stats(), "val": val_dataset.get_stats() if val_dataset else None},
                        run_dir=run_dir,
                    )

                if (
                    val_dataset is not None
                    and int(args.val_every_steps) > 0
                    and state.optimizer_step % int(args.val_every_steps) == 0
                ):
                    val_metrics = _run_validation(
                        args=args,
                        pipe=pipe,
                        bundle=bundle,
                        scheduler=scheduler,
                        val_dataset=val_dataset,
                        generator=generator,
                        trigger_optimizer_step=state.optimizer_step,
                    )
                    _append_jsonl(metrics_val_path, val_metrics)
                    val_loss = float(val_metrics["val_loss"])
                    if state.best_val_loss is None or val_loss < float(state.best_val_loss):
                        state.best_val_loss = val_loss
                        _save_checkpoint(
                            path=run_dir / "checkpoint_best.pt",
                            args=args,
                            bundle=bundle,
                            optimizer=optimizer,
                            scheduler=lr_scheduler,
                            state=state,
                            dataset_stats={
                                "train": train_dataset.get_stats(),
                                "val": val_dataset.get_stats() if val_dataset else None,
                            },
                            run_dir=run_dir,
                        )

                _update_summary_from_state(summary, state)
                _write_json(summary_path, summary)

            if start_batch == 0 and state.micro_step % train_batches_per_epoch != 0:
                break

        _save_checkpoint(
            path=run_dir / "checkpoint_latest.pt",
            args=args,
            bundle=bundle,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            state=state,
            dataset_stats={"train": train_dataset.get_stats(), "val": val_dataset.get_stats() if val_dataset else None},
            run_dir=run_dir,
        )

        changed, max_delta = _param_slice_delta(before, bundle.train_prior)
        summary["parameter_delta_status"] = "changed" if changed else "unchanged"
        summary["parameter_delta_max_abs"] = max_delta
        if not changed:
            summary["failure_message"] = "Trainable user-conditioning parameter slices did not change."
            _update_summary_from_state(summary, state)
            _write_json(summary_path, summary)
            return 4

        summary["run_status"] = "success"
        summary["failure_message"] = None
        _update_summary_from_state(summary, state)
        if latest_train_metrics is not None:
            summary["last_train_metrics"] = latest_train_metrics
        _write_json(summary_path, summary)
        print("[train_stage2_full] run status: success")
        print("[train_stage2_full] micro steps completed:", state.micro_step)
        print("[train_stage2_full] optimizer steps completed:", state.optimizer_step)
        print("[train_stage2_full] metrics train:", metrics_train_path)
        print("[train_stage2_full] summary:", summary_path)
        print("[train_stage2_full] checkpoint latest:", run_dir / "checkpoint_latest.pt")
        return 0
    except Exception as exc:
        if _is_oom_error(exc) or isinstance(exc, RuntimeError):
            _print_memory_hint(exc)
        else:
            print("[train_stage2_full] failure:", str(exc))
        if "non-finite" in str(exc).lower() or "nan" in str(exc).lower() or "inf" in str(exc).lower():
            state.nan_failures += 1
        if "frozen" in str(exc).lower() or "reference" in str(exc).lower():
            state.frozen_integrity_failures += 1
        if not summary:
            summary = {
                "run_status": "failed",
                "summary_path": str(summary_path),
                "args": _jsonable(vars(args)),
            }
        summary["run_status"] = "failed"
        summary["failure_message"] = traceback.format_exc().strip()
        _update_summary_from_state(summary, state)
        _write_json(summary_path, summary)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
