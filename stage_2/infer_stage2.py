from __future__ import annotations

"""Stage 2 personalized Stable Cascade inference.

This script is intentionally user-centric: it loads one Stage 1 user embedding
row plus its assignment metadata, then compares base / zero-user / real-user
conditioning at the prior embedding level and, optionally, through the decoder.
"""

import argparse
import contextlib
import html
import json
import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

try:
    from forward_only_stage2 import (
        _move_module_if_possible,
        _parse_patch_paths,
        _resolve_device,
        _resolve_torch_dtype,
    )
    from patch_stage_c import (
        PatchedSDCascadeAttnBlock,
        freeze_stage_c_except_user_modules,
        patch_stage_c_with_user_adapter,
        summarize_trainable_parameters,
    )
except ImportError:  # pragma: no cover - useful when imported as a package
    from stage_2.forward_only_stage2 import (
        _move_module_if_possible,
        _parse_patch_paths,
        _resolve_device,
        _resolve_torch_dtype,
    )
    from stage_2.patch_stage_c import (
        PatchedSDCascadeAttnBlock,
        freeze_stage_c_except_user_modules,
        patch_stage_c_with_user_adapter,
        summarize_trainable_parameters,
    )


INFERENCE_CONFIG_VERSION = "stage2_personalized_inference_v1"
DEFAULT_CHECKPOINT_PATH = Path("artifacts/stage2_train_full/20260426_145425/checkpoint_latest.pt")
DEFAULT_EMBEDDING_JSON_PATH = Path("data/user_emb_7b_full/validation_shard0.json")
DEFAULT_ASSIGNMENT_JSONL_PATH = Path(
    "artifacts/pair_assignments/validation/stage2_pair_assignments_validation_shard0.jsonl"
)
DEFAULT_UID_TO_PATH_JSON_PATH = Path("data/validation_uid_to_path.json")
DEFAULT_OUTPUT_ROOT = Path("artifacts/stage2_inference")
DEFAULT_PRIOR_MODEL_ID = "stabilityai/stable-cascade-prior"
DEFAULT_DECODER_MODEL_ID = "stabilityai/stable-cascade"
DEFAULT_DECODE_LATENT_PROMPT = "a high quality image"
EXPECTED_USER_EMB_DIM = 3584
SUPPORTED_STANDALONE_LATENT_SHAPES = ((16, 12, 12), (16, 24, 24))
CONDITIONS = ("base", "branch_off", "zero_user", "zero_user_zero_mask", "real_user")
USER_CONDITIONING_NAME_MARKERS = (
    ".user_projection.",
    ".user_adapter.k_proj.",
    ".user_adapter.v_proj.",
    ".user_adapter.out_proj.",
    ".user_scale",
)
ZERO_USER_DEFINITION = "zero_user = torch.zeros_like(real_user_emb) with the real user attention mask"
ZERO_USER_ZERO_MASK_DEFINITION = "zero_user_zero_mask = torch.zeros_like(real_user_emb) with an all-zero attention mask"
BRANCH_OFF_DEFINITION = "branch_off = no user-conditioning hook; should match base"
DEFAULT_EXTRA_PROMPTS = (
    "A cozy reading room with warm afternoon light",
    "A sleek wooden sports car parked on a mountain road",
)


@dataclass
class UserInferenceContext:
    user_embedding_id: str
    user_id: str
    source_embedding_row_idx: Optional[int]
    user_profile_text: str
    user_emb: Tensor
    user_mask: Tensor
    support_pairs: List[Dict[str, Any]]
    query_pairs: List[Dict[str, Any]]
    selected_prompts: List[Dict[str, Any]]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _load_json(path: Path) -> Any:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSONL line {line_idx} in {resolved}: {exc}") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"JSONL line {line_idx} in {resolved} is not an object/dict.")
            yield dict(record)


def _sort_mixed_keys(keys: Sequence[Any]) -> List[Any]:
    def _sort_key(value: Any) -> Tuple[int, Any]:
        text = str(value)
        if text.isdigit():
            return 0, int(text)
        return 1, text

    return sorted(keys, key=_sort_key)


def _normalize_embedding_records(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        if not all(isinstance(item, Mapping) for item in raw):
            raise ValueError("Embedding JSON list must contain only objects/dicts.")
        return [dict(item) for item in raw]
    if not isinstance(raw, Mapping):
        raise ValueError("Unsupported embedding JSON format: expected list or mapping.")

    value_types = [isinstance(value, (Mapping, list, tuple)) for value in raw.values()]
    if not any(value_types):
        return [dict(raw)]

    row_keys: set[Any] = set()
    for value in raw.values():
        if isinstance(value, Mapping):
            row_keys.update(value.keys())
        elif isinstance(value, (list, tuple)):
            row_keys.update(range(len(value)))

    records: List[Dict[str, Any]] = []
    for row_key in _sort_mixed_keys(list(row_keys)):
        row_index = int(row_key) if str(row_key).isdigit() else None
        row_key_str = str(row_key)
        record: Dict[str, Any] = {}
        for column, values in raw.items():
            if isinstance(values, Mapping):
                if row_key in values:
                    record[column] = values[row_key]
                elif row_key_str in values:
                    record[column] = values[row_key_str]
                elif row_index is not None and row_index in values:
                    record[column] = values[row_index]
            elif isinstance(values, (list, tuple)):
                if row_index is not None and 0 <= row_index < len(values):
                    record[column] = values[row_index]
            else:
                record[column] = values
        records.append(record)
    return records


def _expected_user_embedding_id(user_id: Any, row_idx: int) -> str:
    return f"{user_id}_emb_{row_idx:06d}"


def _resolve_user_embedding_id(record: Mapping[str, Any], row_idx: int) -> str:
    for key in ("user_embedding_id", "embedding_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return _expected_user_embedding_id(record.get("user_id", row_idx), row_idx)


def _load_assignment(path: Path, user_embedding_id: Optional[str]) -> Dict[str, Any]:
    first_record: Optional[Dict[str, Any]] = None
    for record in _iter_jsonl(path):
        if first_record is None:
            first_record = record
        record_id = record.get("user_embedding_id")
        if user_embedding_id is not None and str(record_id) == str(user_embedding_id):
            return record
        if user_embedding_id is None:
            return record
    if first_record is None:
        raise ValueError(f"Assignment JSONL is empty: {path}")
    raise ValueError(f"user_embedding_id not found in assignment JSONL: {user_embedding_id}")


def _find_embedding_row(
    records: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, Any],
    user_embedding_id: str,
) -> Tuple[int, Mapping[str, Any]]:
    source_row_idx_raw = assignment.get("source_embedding_row_idx")
    if source_row_idx_raw is not None:
        source_row_idx = int(source_row_idx_raw)
        if not (0 <= source_row_idx < len(records)):
            raise IndexError(
                f"source_embedding_row_idx out of range: {source_row_idx} "
                f"(num_embedding_rows={len(records)})"
            )
        candidate = records[source_row_idx]
        candidate_id = _resolve_user_embedding_id(candidate, source_row_idx)
        if candidate_id == user_embedding_id:
            return source_row_idx, candidate

    for row_idx, record in enumerate(records):
        if _resolve_user_embedding_id(record, row_idx) == user_embedding_id:
            return row_idx, record
    raise ValueError(f"user_embedding_id not found in embedding JSON: {user_embedding_id}")


def _load_uid_to_path(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {}
    data = _load_json(resolved)
    if not isinstance(data, Mapping):
        raise ValueError(f"UID path JSON must be an object/dict: {resolved}")
    return {str(key): str(value) for key, value in data.items()}


def _with_support_paths(support_pairs: Sequence[Mapping[str, Any]], uid_to_path: Mapping[str, str]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for pair in support_pairs:
        item = dict(pair)
        preferred_uid = str(item.get("preferred_uid", ""))
        dispreferred_uid = str(item.get("dispreferred_uid", ""))
        item["preferred_path"] = uid_to_path.get(preferred_uid)
        item["dispreferred_path"] = uid_to_path.get(dispreferred_uid)
        enriched.append(item)
    return enriched


def _select_prompts(args: argparse.Namespace, query_pairs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add_prompt(prompt: str, source: str, meta: Optional[Mapping[str, Any]] = None) -> None:
        text = str(prompt).strip()
        if not text or text in seen:
            return
        seen.add(text)
        item: Dict[str, Any] = {"prompt": text, "source": source}
        if meta is not None:
            for key in ("pair_key", "pair_idx", "source_row_idx", "preferred_uid", "dispreferred_uid"):
                if key in meta:
                    item[key] = meta[key]
        prompts.append(item)

    if args.prompt:
        for prompt in args.prompt:
            add_prompt(prompt, "manual")
    else:
        for pair in query_pairs:
            if len([item for item in prompts if item["source"] == "query"]) >= args.max_query_prompts:
                break
            add_prompt(str(pair.get("caption", "")), "query", pair)

    if not args.no_default_extra_prompts:
        for prompt in DEFAULT_EXTRA_PROMPTS:
            add_prompt(prompt, "default_extra")

    for prompt in args.extra_prompt or []:
        add_prompt(prompt, "extra")

    if not prompts:
        raise ValueError("No prompts selected. Provide --prompt or ensure assignment query_pairs contain captions.")
    return prompts


def load_user_context(args: argparse.Namespace) -> UserInferenceContext:
    assignment = _load_assignment(args.assignment_jsonl_path, args.user_embedding_id)
    user_embedding_id = str(assignment.get("user_embedding_id"))
    if not user_embedding_id or user_embedding_id == "None":
        raise ValueError("Selected assignment is missing user_embedding_id.")

    raw_embeddings = _load_json(args.embedding_json_path)
    embedding_records = _normalize_embedding_records(raw_embeddings)
    source_row_idx, embedding_row = _find_embedding_row(embedding_records, assignment, user_embedding_id)
    row_user_id = str(embedding_row.get("user_id", assignment.get("user_id", "")))
    assignment_user_id = str(assignment.get("user_id", row_user_id))
    if assignment_user_id != row_user_id:
        raise ValueError(f"user_id mismatch: assignment={assignment_user_id}, embedding_row={row_user_id}")

    if "emb" not in embedding_row:
        raise ValueError(f"Embedding row is missing `emb` for user_embedding_id={user_embedding_id}")
    user_emb = torch.as_tensor(embedding_row["emb"], dtype=torch.float32)
    if user_emb.ndim != 2:
        raise ValueError(f"user_emb must have shape [L,D], got {tuple(user_emb.shape)}")
    if int(user_emb.shape[1]) != EXPECTED_USER_EMB_DIM:
        raise ValueError(f"user_emb last dim must be {EXPECTED_USER_EMB_DIM}, got {int(user_emb.shape[1])}")
    if not torch.isfinite(user_emb).all().item():
        raise ValueError(f"user_emb contains NaN/Inf for user_embedding_id={user_embedding_id}")

    support_pairs_raw = assignment.get("support_pairs", [])
    query_pairs_raw = assignment.get("query_pairs", [])
    if not isinstance(support_pairs_raw, list):
        raise ValueError("assignment support_pairs must be a list.")
    if not isinstance(query_pairs_raw, list):
        raise ValueError("assignment query_pairs must be a list.")

    uid_to_path = _load_uid_to_path(args.uid_to_path_json_path)
    support_pairs = _with_support_paths(support_pairs_raw, uid_to_path)
    query_pairs = [dict(pair) for pair in query_pairs_raw]

    batched_user_emb = user_emb.unsqueeze(0).contiguous()
    user_mask = torch.ones((1, int(user_emb.shape[0])), dtype=torch.long)
    selected_prompts = _select_prompts(args, query_pairs)

    return UserInferenceContext(
        user_embedding_id=user_embedding_id,
        user_id=row_user_id,
        source_embedding_row_idx=source_row_idx,
        user_profile_text=str(embedding_row.get("text", embedding_row.get("user_profile_text", ""))),
        user_emb=batched_user_emb,
        user_mask=user_mask,
        support_pairs=support_pairs,
        query_pairs=query_pairs,
        selected_prompts=selected_prompts,
    )


def _torch_load(path: Path, map_location: Any = "cpu") -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _is_user_conditioning_param(name: str) -> bool:
    return (
        ".user_projection." in name
        or ".user_adapter.k_proj." in name
        or ".user_adapter.v_proj." in name
        or ".user_adapter.out_proj." in name
        or name.endswith(".user_scale")
        or name == "user_scale"
    )


def _normalize_patch_paths_for_compare(raw: Any) -> List[str]:
    if raw is None:
        return list(_parse_patch_paths(None))
    if isinstance(raw, str):
        return list(_parse_patch_paths([raw]))
    if isinstance(raw, Sequence):
        return list(_parse_patch_paths([str(item) for item in raw]))
    return list(_parse_patch_paths([str(raw)]))


def _compare_patch_path(found: Any, requested_raw: Optional[Sequence[str]]) -> bool:
    return _normalize_patch_paths_for_compare(found) == _normalize_patch_paths_for_compare(requested_raw)


def _get_trainable_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Tensor]:
    state = checkpoint.get("trainable_state")
    if not isinstance(state, Mapping):
        raise RuntimeError("Checkpoint is missing mapping key `trainable_state`.")
    if not all(torch.is_tensor(value) for value in state.values()):
        raise RuntimeError("All checkpoint trainable_state values must be tensors.")
    return state  # type: ignore[return-value]


def _validate_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    args: argparse.Namespace,
    current_trainable_names: Sequence[str],
) -> Dict[str, Any]:
    warnings: List[str] = []
    critical = checkpoint.get("critical_config")
    if not isinstance(critical, Mapping):
        raise RuntimeError("Checkpoint is missing critical_config.")

    found_model_id = critical.get("model_id")
    if found_model_id != args.prior_model_id:
        raise RuntimeError(f"model_id mismatch: checkpoint={found_model_id}, requested={args.prior_model_id}")

    found_user_scale = float(critical.get("user_scale"))
    if not math.isclose(found_user_scale, float(args.user_scale), rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError(f"user_scale mismatch: checkpoint={found_user_scale}, requested={args.user_scale}")

    if not _compare_patch_path(critical.get("patch_path"), args.patch_path):
        raise RuntimeError(f"patch_path mismatch: checkpoint={critical.get('patch_path')}, requested={args.patch_path}")

    found_markers = list(critical.get("trainable_markers", []))
    if found_markers != list(USER_CONDITIONING_NAME_MARKERS):
        raise RuntimeError(
            "trainable marker mismatch: "
            f"checkpoint={found_markers}, expected={list(USER_CONDITIONING_NAME_MARKERS)}"
        )

    unexpected_trainable = [name for name in current_trainable_names if not _is_user_conditioning_param(name)]
    if unexpected_trainable:
        raise RuntimeError(f"Unexpected trainable non-user-conditioning parameters: {unexpected_trainable[:10]}")

    state = _get_trainable_state(checkpoint)
    state_names = set(state.keys())
    current_names = set(current_trainable_names)
    missing = sorted(current_names - state_names)
    extra = sorted(state_names - current_names)
    if missing or extra:
        raise RuntimeError(f"trainable_state key mismatch: missing={missing[:10]}, extra={extra[:10]}")

    latent_shape = critical.get("latent_shape")
    if latent_shape is None:
        warnings.append("checkpoint critical_config has no latent_shape.")

    return {
        "checkpoint_critical_config": dict(critical),
        "trainable_state_tensors": len(state_names),
        "trainable_scope_mode": "user_projection+k_proj+v_proj+out_proj+user_scale",
        "user_branch_structure_keys": sorted(state_names),
        "compatibility_warnings": warnings,
        "inference_config_version": INFERENCE_CONFIG_VERSION,
    }


def _load_state_into_prior(prior: nn.Module, trainable_state: Mapping[str, Tensor]) -> None:
    params = dict(prior.named_parameters())
    with torch.no_grad():
        for name, source in trainable_state.items():
            target = params[name]
            target.copy_(source.to(device=target.device, dtype=target.dtype))


def _load_diffusers_class(class_name: str) -> Any:
    try:
        import diffusers
    except Exception as exc:  # pragma: no cover - environment-specific
        raise ImportError("Failed to import diffusers. Activate the ppd_stage2 environment.") from exc
    try:
        return getattr(diffusers, class_name)
    except AttributeError as exc:
        raise ImportError(f"diffusers does not expose {class_name}.") from exc


def _load_pipeline(class_name: str, model_id: str, args: argparse.Namespace, device: torch.device) -> Any:
    pipeline_cls = _load_diffusers_class(class_name)
    kwargs: Dict[str, Any] = {
        "local_files_only": bool(args.local_files_only),
        "trust_remote_code": bool(args.trust_remote_code),
    }
    torch_dtype = _resolve_torch_dtype(args.torch_dtype)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    pipe = pipeline_cls.from_pretrained(model_id, **kwargs)
    _move_module_if_possible(pipe, device)
    _sync_text_encoder_dtype(pipe, device)
    return pipe


def _sync_text_encoder_dtype(pipe: Any, device: torch.device) -> None:
    """Keep text embeddings in the same dtype as the primary denoising module."""

    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        return
    dtype = _pipeline_dtype(pipe)
    to = getattr(text_encoder, "to", None)
    if callable(to):
        to(device=device, dtype=dtype)


def _prepare_prior_pipeline(args: argparse.Namespace, device: torch.device) -> Tuple[Any, Dict[str, Any]]:
    checkpoint = _torch_load(args.checkpoint_path, map_location="cpu")
    pipe = _load_pipeline("StableCascadePriorPipeline", args.prior_model_id, args, device)
    prior = pipe.prior
    patch_paths = _parse_patch_paths(args.patch_path)
    patch_summary = patch_stage_c_with_user_adapter(
        model=prior,
        target_paths=patch_paths,
        max_blocks=None,
        user_emb_dim=EXPECTED_USER_EMB_DIM,
        user_scale=float(args.user_scale),
        user_projection_bias=bool(getattr(args, "user_projection_bias", True)),
        user_projection_norm_affine=bool(getattr(args, "user_projection_norm_affine", True)),
        user_adapter_projection_bias=bool(getattr(args, "user_adapter_projection_bias", True)),
        user_adapter_zero_init_out=bool(getattr(args, "user_adapter_zero_init_out", False)),
    )
    freeze_stage_c_except_user_modules(prior)
    trainable_summary = summarize_trainable_parameters(prior, max_names=40)
    current_trainable_names = [name for name, param in prior.named_parameters() if param.requires_grad]
    compatibility = _validate_checkpoint_metadata(checkpoint, args, current_trainable_names)
    _load_state_into_prior(prior, _get_trainable_state(checkpoint))

    prior.eval()
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is not None:
        text_encoder.eval()

    compatibility.update(
        {
            "patch_summary": _jsonable(patch_summary.__dict__),
            "trainable_summary": _jsonable(trainable_summary.__dict__),
            "checkpoint_path": str(args.checkpoint_path.expanduser().resolve()),
        }
    )
    return pipe, compatibility


def _prepare_vanilla_prior_pipeline(args: argparse.Namespace, device: torch.device) -> Tuple[Any, Dict[str, Any]]:
    pipe = _load_pipeline("StableCascadePriorPipeline", args.prior_model_id, args, device)
    prior = getattr(pipe, "prior", None)
    if prior is not None:
        prior.eval()
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is not None:
        text_encoder.eval()
    return pipe, {
        "prior_mode": "vanilla",
        "checkpoint_loaded": False,
        "patch_loaded": False,
        "prior_model_id": args.prior_model_id,
    }


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


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def _extract_prior_embeddings(output: Any) -> Tensor:
    image_embeddings = getattr(output, "image_embeddings", None)
    if torch.is_tensor(image_embeddings):
        return image_embeddings
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Could not extract image_embeddings from prior output type {type(output)}.")


def _finite_flags(tensor: Tensor) -> Tuple[bool, bool, bool]:
    if not tensor.is_floating_point() and not tensor.is_complex():
        return False, False, True
    has_nan = bool(torch.isnan(tensor).any().item())
    has_inf = bool(torch.isinf(tensor).any().item())
    all_finite = bool(torch.isfinite(tensor).all().item())
    return has_nan, has_inf, all_finite


def _tensor_diagnostics(tensor: Tensor) -> Dict[str, Any]:
    has_nan, has_inf, all_finite = _finite_flags(tensor)
    payload: Dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "all_finite": all_finite,
    }
    if tensor.numel() > 0 and tensor.is_floating_point() and all_finite:
        detached = tensor.detach().float().cpu()
        payload.update(
            {
                "min": float(detached.min().item()),
                "max": float(detached.max().item()),
                "mean": float(detached.mean().item()),
                "std": float(detached.std(unbiased=False).item()),
                "l2_norm": float(torch.linalg.vector_norm(detached).item()),
            }
        )
    return payload


def _pairwise_metrics(a: Tensor, b: Tensor) -> Dict[str, float]:
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"Pairwise tensor shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    a_f = a.detach().float().flatten().cpu()
    b_f = b.detach().float().flatten().cpu()
    delta = a_f - b_f
    l2 = float(torch.linalg.vector_norm(delta).item())
    denom = float(torch.linalg.vector_norm(a_f).item()) + 1e-12
    cosine = float(torch.nn.functional.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0), dim=1).item())
    return {
        "l2_norm": l2,
        "relative_delta_norm": l2 / denom,
        "mean_abs_diff": float(delta.abs().mean().item()),
        "max_abs_diff": float(delta.abs().max().item()),
        "cosine_similarity": cosine,
    }


def _scale_label(scale: float) -> str:
    text = f"{float(scale):.6g}".replace("-", "m").replace(".", "p")
    return text


def _inference_scales(args: argparse.Namespace) -> List[float]:
    if args.inference_user_scale_sweep:
        return [float(item) for item in args.inference_user_scale_sweep]
    return [float(args.inference_user_scale)]


def _selected_conditions(args: argparse.Namespace) -> Tuple[str, ...]:
    raw_conditions = getattr(args, "conditions", None)
    if bool(getattr(args, "vanilla_prior", False)) and raw_conditions is None:
        selected = ("base",)
    else:
        selected = tuple(raw_conditions or CONDITIONS)
    unknown = [condition for condition in selected if condition not in CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown condition(s): {unknown}. Valid choices: {list(CONDITIONS)}")
    if bool(getattr(args, "vanilla_prior", False)):
        unsupported = [condition for condition in selected if condition not in ("base", "branch_off")]
        if unsupported:
            raise ValueError(
                "--vanilla-prior supports only no-user conditions: base, branch_off. "
                f"Unsupported: {unsupported}"
            )
    return selected


def _pairwise_if_present(tensors: Mapping[str, Tensor], left: str, right: str) -> Optional[Dict[str, float]]:
    if left not in tensors or right not in tensors:
        return None
    return _pairwise_metrics(tensors[left], tensors[right])


def _compact_pairwise(pairwise: Mapping[str, Optional[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    return {key: value for key, value in pairwise.items() if value is not None}


def _summarize_residual_diagnostics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "num_hook_calls": 0,
            "max_residual_ratio": 0.0,
            "mean_residual_ratio": 0.0,
            "per_block": {},
        }
    ratios = [float(row["residual_ratio"]) for row in rows]
    per_block: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("module_path", row.get("module", "unknown"))), []).append(row)
    for path, items in grouped.items():
        block_ratios = [float(item["residual_ratio"]) for item in items]
        per_block[path] = {
            "num_calls": len(items),
            "max_residual_ratio": max(block_ratios),
            "mean_residual_ratio": float(sum(block_ratios) / len(block_ratios)),
            "checkpoint_user_scale": float(items[-1]["checkpoint_user_scale"]),
            "inference_user_scale": float(items[-1]["inference_user_scale"]),
        }
    return {
        "num_hook_calls": len(rows),
        "max_residual_ratio": max(ratios),
        "mean_residual_ratio": float(sum(ratios) / len(ratios)),
        "per_block": per_block,
    }


def _expand_user_conditioning_for_query(
    user_emb: Tensor,
    user_mask: Optional[Tensor],
    query_batch_size: int,
) -> Tuple[Tensor, Optional[Tensor]]:
    user_batch_size = int(user_emb.shape[0])
    if user_batch_size == query_batch_size:
        return user_emb, user_mask

    if query_batch_size == user_batch_size * 2:
        zeros = torch.zeros_like(user_emb)
        expanded_user = torch.cat([user_emb, zeros], dim=0)
        expanded_mask = None
        if user_mask is not None:
            expanded_mask = torch.cat([user_mask, torch.zeros_like(user_mask)], dim=0)
        return expanded_user, expanded_mask

    if query_batch_size % user_batch_size == 0:
        repeats = query_batch_size // user_batch_size
        expanded_user = user_emb.repeat_interleave(repeats, dim=0)
        expanded_mask = user_mask.repeat_interleave(repeats, dim=0) if user_mask is not None else None
        return expanded_user, expanded_mask

    raise ValueError(
        f"Cannot align user conditioning batch size {user_batch_size} to query batch size {query_batch_size}."
    )


@contextlib.contextmanager
def inference_user_conditioning_hooks(
    prior: nn.Module,
    user_emb: Optional[Tensor],
    user_emb_attention_mask: Optional[Tensor],
    inference_user_scale: float = 1.0,
    residual_diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """Inject user conditioning while handling prior CFG batch expansion."""

    handles: List[Any] = []
    if user_emb is None:
        yield
        return

    module_paths = {id(module): path for path, module in prior.named_modules()}

    def _hook(module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if not isinstance(module, PatchedSDCascadeAttnBlock):
            return output
        if not isinstance(output, Tensor):
            raise TypeError(f"Expected patched block tensor output, got {type(output)}")

        query, restore = module._to_token_sequence(output)
        module._sync_user_modules(reference=query)
        local_user_emb = user_emb.to(device=query.device, dtype=query.dtype)
        local_mask = user_emb_attention_mask.to(device=query.device) if user_emb_attention_mask is not None else None
        local_user_emb, local_mask = _expand_user_conditioning_for_query(
            local_user_emb,
            local_mask,
            query_batch_size=int(query.shape[0]),
        )

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
        scale = module.user_scale.to(device=output.device, dtype=output.dtype) * float(inference_user_scale)
        scaled_residual = scale * user_residual
        if residual_diagnostics is not None:
            with torch.no_grad():
                base_norm = float(torch.linalg.vector_norm(output.detach().float()).item())
                residual_norm = float(torch.linalg.vector_norm(scaled_residual.detach().float()).item())
                residual_diagnostics.append(
                    {
                        "module_path": module_paths.get(id(module), module.__class__.__name__),
                        "module": module.__class__.__name__,
                        "base_norm": base_norm,
                        "scaled_residual_norm": residual_norm,
                        "residual_ratio": float(residual_norm / (base_norm + 1e-12)),
                        "checkpoint_user_scale": float(module.user_scale.detach().float().cpu().item()),
                        "inference_user_scale": float(inference_user_scale),
                    }
                )
        return output + scaled_residual

    try:
        for module in prior.modules():
            if isinstance(module, PatchedSDCascadeAttnBlock):
                handles.append(module.register_forward_hook(_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _run_prior_condition(
    *,
    pipe: Any,
    prompt: str,
    condition: str,
    user_emb: Tensor,
    user_mask: Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> Tuple[Tensor, Dict[str, Any]]:
    hook_user_emb: Optional[Tensor]
    hook_user_mask: Optional[Tensor]
    if condition == "base" or condition == "branch_off":
        hook_user_emb = None
        hook_user_mask = None
    elif condition == "zero_user":
        hook_user_emb = torch.zeros_like(user_emb)
        hook_user_mask = user_mask
    elif condition == "zero_user_zero_mask":
        hook_user_emb = torch.zeros_like(user_emb)
        hook_user_mask = torch.zeros_like(user_mask)
    elif condition == "real_user":
        hook_user_emb = user_emb
        hook_user_mask = user_mask
    else:
        raise ValueError(f"Unknown condition: {condition}")

    generator = _make_generator(device, seed)
    residual_rows: List[Dict[str, Any]] = []
    with torch.inference_mode():
        with inference_user_conditioning_hooks(
            pipe.prior,
            user_emb=hook_user_emb,
            user_emb_attention_mask=hook_user_mask,
            inference_user_scale=float(getattr(args, "_active_inference_user_scale", args.inference_user_scale)),
            residual_diagnostics=residual_rows,
        ):
            output = pipe(
                prompt=prompt,
                height=int(args.height),
                width=int(args.width),
                num_inference_steps=int(args.prior_steps),
                guidance_scale=float(args.prior_guidance_scale),
                negative_prompt=args.negative_prompt,
                num_images_per_prompt=1,
                generator=generator,
                output_type="pt",
                return_dict=True,
            )
    embeddings = _extract_prior_embeddings(output)
    if not torch.isfinite(embeddings).all().item():
        has_nan, has_inf, _ = _finite_flags(embeddings)
        raise RuntimeError(f"{condition} prior output is non-finite: has_nan={has_nan}, has_inf={has_inf}")
    return embeddings.detach(), _summarize_residual_diagnostics(residual_rows)


def _run_dir(args: argparse.Namespace) -> Path:
    if args.inference_run_dir is not None:
        path = args.inference_run_dir
    else:
        stamp = args.run_name or time.strftime("%Y%m%d_%H%M%S")
        path = args.output_dir / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def _context_metadata(context: UserInferenceContext) -> Dict[str, Any]:
    return {
        "user_embedding_id": context.user_embedding_id,
        "user_id": context.user_id,
        "source_embedding_row_idx": context.source_embedding_row_idx,
        "user_emb_shape": list(context.user_emb.shape),
        "user_mask_shape": list(context.user_mask.shape),
        "support_pairs": context.support_pairs,
        "query_pairs_selected": context.selected_prompts,
        "num_query_pairs_available": len(context.query_pairs),
    }


def run_prior_smoke(args: argparse.Namespace, run_dir: Optional[Path] = None) -> Path:
    device = _resolve_device(args.device)
    run_dir = run_dir or _run_dir(args)
    embeddings_dir = run_dir / "inference_embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    warnings: List[str] = []
    sample_index = 0
    scales = _inference_scales(args)
    selected_conditions = _selected_conditions(args)
    context: Optional[UserInferenceContext]
    if bool(args.vanilla_prior):
        context = None
        selected_prompts = _select_prompts(args, query_pairs=[])
        pipe, compatibility = _prepare_vanilla_prior_pipeline(args, device)
        user_emb = torch.empty((1, 1, EXPECTED_USER_EMB_DIM), device=device, dtype=_pipeline_dtype(pipe))
        user_mask = torch.ones((1, 1), device=device, dtype=torch.long)
    else:
        context = load_user_context(args)
        selected_prompts = context.selected_prompts
        pipe, compatibility = _prepare_prior_pipeline(args, device)
        user_emb = context.user_emb.to(device=device, dtype=_pipeline_dtype(pipe))
        user_mask = context.user_mask.to(device=device)

    for prompt_index, prompt_item in enumerate(selected_prompts):
        for seed in args.seeds:
            for scale in scales:
                args._active_inference_user_scale = float(scale)
                prompt = str(prompt_item["prompt"])
                tensors: Dict[str, Tensor] = {}
                tensor_paths: Dict[str, str] = {}
                diagnostics: Dict[str, Any] = {}
                residual_diagnostics: Dict[str, Any] = {}
                sample_id = f"sample_{sample_index:02d}_scale_{_scale_label(float(scale))}"
                for condition in selected_conditions:
                    embeddings, residual_summary = _run_prior_condition(
                        pipe=pipe,
                        prompt=prompt,
                        condition=condition,
                        user_emb=user_emb,
                        user_mask=user_mask,
                        args=args,
                        device=device,
                        seed=int(seed),
                    )
                    tensors[condition] = embeddings
                    file_name = f"{sample_id}_{condition}.pt"
                    output_path = embeddings_dir / file_name
                    torch.save(embeddings.detach().cpu(), output_path)
                    tensor_paths[condition] = str(output_path.relative_to(run_dir))
                    diagnostics[condition] = _tensor_diagnostics(embeddings)
                    residual_diagnostics[condition] = residual_summary

                shape_set = {tuple(tensor.shape) for tensor in tensors.values()}
                if len(shape_set) != 1:
                    raise RuntimeError(f"Condition output shapes differ for {sample_id}: {shape_set}")

                pairwise = _compact_pairwise(
                    {
                        "base_vs_branch_off": _pairwise_if_present(tensors, "base", "branch_off"),
                        "base_vs_zero_user": _pairwise_if_present(tensors, "base", "zero_user"),
                        "base_vs_zero_user_zero_mask": _pairwise_if_present(tensors, "base", "zero_user_zero_mask"),
                        "base_vs_real_user": _pairwise_if_present(tensors, "base", "real_user"),
                        "zero_user_vs_zero_user_zero_mask": _pairwise_if_present(
                            tensors,
                            "zero_user",
                            "zero_user_zero_mask",
                        ),
                        "zero_user_vs_real_user": _pairwise_if_present(tensors, "zero_user", "real_user"),
                        "zero_user_zero_mask_vs_real_user": _pairwise_if_present(
                            tensors,
                            "zero_user_zero_mask",
                            "real_user",
                        ),
                    }
                )
                zero_vs_real = pairwise.get("zero_user_vs_real_user")
                if zero_vs_real is not None and zero_vs_real["l2_norm"] <= float(args.near_identical_l2_threshold):
                    warnings.append(
                        f"{sample_id}: zero_user and real_user are nearly identical "
                        f"(l2={zero_vs_real['l2_norm']})."
                    )

                records.append(
                    {
                        "sample_id": sample_id,
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "prompt_source": prompt_item.get("source"),
                        "prompt_meta": prompt_item,
                        "seed": int(seed),
                        "inference_user_scale": float(scale),
                        "tensor_paths": tensor_paths,
                        "diagnostics": diagnostics,
                        "residual_diagnostics": residual_diagnostics,
                        "pairwise": pairwise,
                    }
                )
                sample_index += 1

    summary = {
        "mode": "prior-smoke",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inference_config_version": INFERENCE_CONFIG_VERSION,
        "zero_user_definition": ZERO_USER_DEFINITION,
        "zero_user_zero_mask_definition": ZERO_USER_ZERO_MASK_DEFINITION,
        "branch_off_definition": BRANCH_OFF_DEFINITION,
        "condition_definitions": {
            "base": "no user-conditioning hook",
            "branch_off": BRANCH_OFF_DEFINITION,
            "zero_user": ZERO_USER_DEFINITION,
            "zero_user_zero_mask": ZERO_USER_ZERO_MASK_DEFINITION,
            "real_user": "actual user_emb with the real user attention mask",
        },
        "selected_conditions": list(selected_conditions),
        "runtime": {
            "device": str(device),
            "torch_dtype": args.torch_dtype,
            "vanilla_prior": bool(args.vanilla_prior),
            "height": int(args.height),
            "width": int(args.width),
            "prior_steps": int(args.prior_steps),
            "prior_guidance_scale": float(args.prior_guidance_scale),
            "negative_prompt": args.negative_prompt,
            "local_files_only": bool(args.local_files_only),
            "inference_user_scale": float(args.inference_user_scale),
            "inference_user_scale_sweep": scales,
        },
        "inputs": {
            "checkpoint_path": None if bool(args.vanilla_prior) else str(args.checkpoint_path),
            "embedding_json_path": None if bool(args.vanilla_prior) else str(args.embedding_json_path),
            "assignment_jsonl_path": None if bool(args.vanilla_prior) else str(args.assignment_jsonl_path),
            "uid_to_path_json_path": None if bool(args.vanilla_prior) else str(args.uid_to_path_json_path),
            "prior_model_id": args.prior_model_id,
        },
        "user_context": (
            {
                "mode": "vanilla_prior",
                "query_pairs_selected": selected_prompts,
                "num_query_pairs_available": 0,
            }
            if context is None
            else _context_metadata(context)
        ),
        "compatibility": compatibility,
        "records": records,
        "warnings": warnings + compatibility.get("compatibility_warnings", []),
    }
    _write_json(embeddings_dir / "summary.json", summary)
    return run_dir


def _load_prior_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "inference_embeddings" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Prior summary not found: {summary_path}")
    data = _load_json(summary_path)
    if not isinstance(data, Mapping):
        raise ValueError(f"Prior summary must be a JSON object: {summary_path}")
    return dict(data)


def _load_tensor_for_decoder(run_dir: Path, rel_path: str, device: torch.device, dtype: torch.dtype) -> Tensor:
    tensor_path = run_dir / rel_path
    loaded = torch.load(tensor_path, map_location="cpu")
    if not torch.is_tensor(loaded):
        raise TypeError(f"Expected tensor payload in {tensor_path}, got {type(loaded)}")
    return loaded.to(device=device, dtype=dtype)


def _extract_latent_payload(loaded: Any, path: Path) -> Tensor:
    if torch.is_tensor(loaded):
        return loaded
    if isinstance(loaded, Mapping):
        for key in ("image_embeddings", "latent", "latents", "sample"):
            value = loaded.get(key)
            if torch.is_tensor(value):
                return value
    raise TypeError(f"Expected tensor latent payload in {path}, got {type(loaded)}")


def _load_standalone_latent(path: Path, device: torch.device, dtype: torch.dtype) -> Tensor:
    resolved = path.expanduser().resolve()
    loaded = torch.load(resolved, map_location="cpu")
    tensor = _extract_latent_payload(loaded, resolved)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    latent_shape = tuple(int(dim) for dim in tensor.shape[1:])
    if latent_shape not in SUPPORTED_STANDALONE_LATENT_SHAPES:
        expected = " or ".join(f"[B,{','.join(str(dim) for dim in shape)}]" for shape in SUPPORTED_STANDALONE_LATENT_SHAPES)
        raise ValueError(f"Expected latent shape {expected}, got {tuple(tensor.shape)} from {resolved}")
    if tensor.shape[0] < 1:
        raise ValueError(f"Latent batch must be non-empty: {resolved}")
    return tensor.to(device=device, dtype=dtype)


def _extract_images(output: Any) -> List[Any]:
    images = getattr(output, "images", output)
    if isinstance(images, list):
        return images
    return [images]


def _save_grid(image_paths: Sequence[Path], output_path: Path, labels: Sequence[str], *, show_labels: bool = True) -> None:
    from PIL import Image, ImageDraw

    images = [Image.open(path).convert("RGB") for path in image_paths]
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    label_height = 32 if show_labels else 0
    grid = Image.new("RGB", (cell_width * len(images), cell_height + label_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for idx, image in enumerate(images):
        x = idx * cell_width
        grid.paste(image, (x, label_height))
        if show_labels:
            draw.text((x + 8, 8), str(labels[idx]), fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def _decode_one(
    *,
    decoder_pipe: Any,
    image_embeddings: Tensor,
    prompt: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    num_inference_steps: int,
) -> Any:
    generator = _make_generator(device, seed)
    with torch.inference_mode():
        output = decoder_pipe(
            image_embeddings=image_embeddings,
            prompt=prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(args.decoder_guidance_scale),
            negative_prompt=args.negative_prompt,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
            return_dict=True,
        )
    images = _extract_images(output)
    if len(images) != 1:
        raise RuntimeError(f"Expected one decoded image, got {len(images)}")
    return images[0]


def run_image_mode(args: argparse.Namespace) -> Path:
    device = _resolve_device(args.device)
    run_dir = _run_dir(args)
    summary_path = run_dir / "inference_embeddings" / "summary.json"
    if not summary_path.exists():
        run_prior_smoke(args, run_dir=run_dir)
    prior_summary = _load_prior_summary(run_dir)
    summary_vanilla_prior = bool(prior_summary.get("runtime", {}).get("vanilla_prior", False))
    if summary_vanilla_prior != bool(args.vanilla_prior):
        raise ValueError(
            "Existing prior summary was created with a different prior mode. "
            f"summary_vanilla_prior={summary_vanilla_prior}, requested_vanilla_prior={bool(args.vanilla_prior)}. "
            "Use a new --run-name or --inference-run-dir."
        )

    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    decoder_pipe = _load_pipeline("StableCascadeDecoderPipeline", args.decoder_model_id, args, device)
    decoder_dtype = _pipeline_dtype(decoder_pipe)

    records = list(prior_summary.get("records", []))
    if not records:
        raise ValueError("Prior summary contains no records.")

    first = records[0]
    selected_conditions = _selected_conditions(args)
    available_conditions = tuple(first.get("tensor_paths", {}).keys())
    decode_conditions = tuple(condition for condition in selected_conditions if condition in available_conditions)
    if not decode_conditions:
        raise ValueError(
            "None of the requested conditions are available in the prior summary: "
            f"requested={list(selected_conditions)}, available={list(available_conditions)}"
        )
    first_condition = "base" if "base" in decode_conditions else decode_conditions[0]
    first_tensor_path = first["tensor_paths"][first_condition]
    first_embeddings = _load_tensor_for_decoder(run_dir, first_tensor_path, device, decoder_dtype)
    preflight: Dict[str, Any] = {
        "prior_output_shape": list(first_embeddings.shape),
        "decoder_model_id": args.decoder_model_id,
        "decoder_preflight_steps": int(args.decoder_preflight_steps),
    }
    preflight_image = _decode_one(
        decoder_pipe=decoder_pipe,
        image_embeddings=first_embeddings,
        prompt=str(first["prompt"]),
        seed=int(first["seed"]),
        args=args,
        device=device,
        num_inference_steps=max(1, int(args.decoder_preflight_steps)),
    )
    preflight_path = images_dir / "preflight.png"
    preflight_image.save(preflight_path)
    preflight["preflight_image_path"] = str(preflight_path.relative_to(run_dir))

    image_records: List[Dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        prompt = str(record["prompt"])
        seed = int(record["seed"])
        condition_paths: Dict[str, str] = {}
        ordered_condition_image_paths: List[Path] = []
        for condition in decode_conditions:
            embeddings = _load_tensor_for_decoder(run_dir, record["tensor_paths"][condition], device, decoder_dtype)
            image = _decode_one(
                decoder_pipe=decoder_pipe,
                image_embeddings=embeddings,
                prompt=prompt,
                seed=seed,
                args=args,
                device=device,
                num_inference_steps=int(args.decoder_steps),
            )
            output_path = images_dir / f"{sample_id}_{condition}.png"
            image.save(output_path)
            condition_paths[condition] = str(output_path.relative_to(run_dir))
            ordered_condition_image_paths.append(output_path)

        grid_path: Optional[Path] = None
        if args.save_grid:
            grid_path = images_dir / f"{sample_id}_grid.png"
            _save_grid(
                ordered_condition_image_paths,
                grid_path,
                labels=decode_conditions,
                show_labels=bool(args.grid_labels),
            )
        image_record: Dict[str, Any] = {
            "sample_id": sample_id,
            "prompt": prompt,
            "seed": seed,
            "condition_image_paths": condition_paths,
        }
        if grid_path is not None:
            image_record["grid_path"] = str(grid_path.relative_to(run_dir))
        image_records.append(
            image_record
        )

    image_summary = {
        "mode": "image",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inference_config_version": INFERENCE_CONFIG_VERSION,
        "runtime": {
            "device": str(device),
            "decoder_steps": int(args.decoder_steps),
            "decoder_guidance_scale": float(args.decoder_guidance_scale),
            "negative_prompt": args.negative_prompt,
            "save_grid": bool(args.save_grid),
            "grid_labels": bool(args.grid_labels),
        },
        "selected_conditions": list(decode_conditions),
        "preflight": preflight,
        "records": image_records,
    }
    _write_json(images_dir / "summary.json", image_summary)
    return run_dir


def run_decode_latent_mode(args: argparse.Namespace) -> Path:
    if args.latent_path is None:
        raise ValueError("decode-latent mode requires --latent-path")

    device = _resolve_device(args.device)
    run_dir = _run_dir(args)
    decoded_dir = run_dir / "decoded_latents"
    decoded_dir.mkdir(parents=True, exist_ok=True)

    decoder_pipe = _load_pipeline("StableCascadeDecoderPipeline", args.decoder_model_id, args, device)
    decoder_dtype = _pipeline_dtype(decoder_pipe)
    latent = _load_standalone_latent(args.latent_path, device=device, dtype=decoder_dtype)
    prompts = list(args.prompt or [DEFAULT_DECODE_LATENT_PROMPT])

    records: List[Dict[str, Any]] = []
    latent_stem = args.latent_path.expanduser().resolve().stem
    for prompt_index, prompt in enumerate(prompts):
        for seed in args.seeds:
            for batch_index in range(int(latent.shape[0])):
                image_embeddings = latent[batch_index : batch_index + 1]
                image = _decode_one(
                    decoder_pipe=decoder_pipe,
                    image_embeddings=image_embeddings,
                    prompt=str(prompt),
                    seed=int(seed),
                    args=args,
                    device=device,
                    num_inference_steps=int(args.decoder_steps),
                )
                output_name = (
                    f"{latent_stem}_prompt_{prompt_index:02d}_seed_{int(seed)}_batch_{batch_index:02d}.png"
                )
                output_path = decoded_dir / output_name
                image.save(output_path)
                records.append(
                    {
                        "latent_batch_index": batch_index,
                        "prompt_index": prompt_index,
                        "prompt": str(prompt),
                        "seed": int(seed),
                        "image_path": str(output_path.relative_to(run_dir)),
                    }
                )

    summary = {
        "mode": "decode-latent",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inference_config_version": INFERENCE_CONFIG_VERSION,
        "runtime": {
            "device": str(device),
            "decoder_model_id": args.decoder_model_id,
            "decoder_steps": int(args.decoder_steps),
            "decoder_guidance_scale": float(args.decoder_guidance_scale),
            "negative_prompt": args.negative_prompt,
            "local_files_only": bool(args.local_files_only),
        },
        "inputs": {
            "latent_path": str(args.latent_path.expanduser().resolve()),
            "latent_diagnostics": _tensor_diagnostics(latent.detach().float().cpu()),
        },
        "records": records,
    }
    _write_json(decoded_dir / "summary.json", summary)
    return run_dir


def _latest_run_dir(output_root: Path) -> Path:
    candidates = [
        path
        for path in output_root.expanduser().resolve().iterdir()
        if path.is_dir() and (path / "inference_embeddings" / "summary.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed inference run directories found under {output_root}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def _html_img(src: Optional[str], alt: str) -> str:
    if not src:
        return "<div class='missing'>missing</div>"
    path = Path(src)
    uri = path.expanduser().resolve().as_uri() if path.is_absolute() else html.escape(src)
    return f"<img src='{uri}' alt='{html.escape(alt)}'>"


def _rel_or_uri(path: Path, base_dir: Path) -> str:
    try:
        return html.escape(str(path.resolve().relative_to(base_dir.resolve())))
    except ValueError:
        return path.resolve().as_uri()


def run_report_mode(args: argparse.Namespace) -> Path:
    run_dir = args.inference_run_dir or _latest_run_dir(args.output_dir)
    prior_summary = _load_prior_summary(run_dir)
    image_summary_path = run_dir / "images" / "summary.json"
    if not image_summary_path.exists():
        raise FileNotFoundError(f"Image summary not found. Run image mode first: {image_summary_path}")
    image_summary = _load_json(image_summary_path)
    if not isinstance(image_summary, Mapping):
        raise ValueError(f"Image summary must be a JSON object: {image_summary_path}")

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    user_context = prior_summary.get("user_context", {})
    user_embedding_id = str(user_context.get("user_embedding_id", "unknown_user"))
    output_path = reports_dir / f"user_{user_embedding_id}.html"

    support_rows: List[str] = []
    for pair in user_context.get("support_pairs", []):
        caption = html.escape(str(pair.get("caption", "")))
        support_rows.append(
            "<tr>"
            f"<td>{_html_img(pair.get('preferred_path'), 'preferred')}</td>"
            f"<td>{_html_img(pair.get('dispreferred_path'), 'dispreferred')}</td>"
            f"<td>{caption}</td>"
            "</tr>"
        )

    image_records_by_id = {record["sample_id"]: record for record in image_summary.get("records", [])}
    result_rows: List[str] = []
    for record in prior_summary.get("records", []):
        sample_id = str(record.get("sample_id"))
        image_record = image_records_by_id.get(sample_id, {})
        grid_rel = image_record.get("grid_path")
        grid_html = ""
        if grid_rel:
            grid_abs = run_dir / str(grid_rel)
            grid_html = _html_img(_rel_or_uri(grid_abs, reports_dir), f"{sample_id} grid")
        pairwise = html.escape(json.dumps(record.get("pairwise", {}), indent=2))
        result_rows.append(
            "<section class='sample'>"
            f"<h2>{html.escape(sample_id)} | seed {html.escape(str(record.get('seed')))}</h2>"
            f"<p>{html.escape(str(record.get('prompt')))}</p>"
            f"{grid_html}"
            f"<pre>{pairwise}</pre>"
            "</section>"
        )

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage 2 Inference Report - {html.escape(user_embedding_id)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    h1 {{ font-size: 24px; margin-bottom: 4px; }}
    h2 {{ font-size: 18px; margin-top: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    td, th {{ border: 1px solid #d9dee5; padding: 8px; vertical-align: top; }}
    img {{ max-width: 260px; height: auto; display: block; }}
    .sample img {{ max-width: 960px; }}
    pre {{ background: #f6f8fa; padding: 12px; overflow-x: auto; }}
    .missing {{ color: #8a1f11; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Stage 2 Inference Report</h1>
  <p>User embedding: {html.escape(user_embedding_id)} | User: {html.escape(str(user_context.get('user_id')))}</p>
  <p>Zero-user definition: {html.escape(str(prior_summary.get('zero_user_definition')))}</p>
  <h2>Support Pairs</h2>
  <table>
    <thead><tr><th>Preferred</th><th>Dispreferred</th><th>Caption</th></tr></thead>
    <tbody>{''.join(support_rows)}</tbody>
  </table>
  <h2>Generated Results</h2>
  {''.join(result_rows)}
</body>
</html>
"""
    output_path.write_text(doc, encoding="utf-8")
    return output_path


def _parse_seeds(raw: Optional[Sequence[int]]) -> List[int]:
    if not raw:
        return [0]
    return [int(seed) for seed in raw]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2 personalized prior/image/report inference.")
    parser.add_argument("mode", choices=("prior-smoke", "image", "report", "decode-latent"))
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--latent-path", type=Path, default=None, help="Decode an existing [B,16,12,12] or [B,16,24,24] latent .pt file.")
    parser.add_argument("--embedding-json-path", type=Path, default=DEFAULT_EMBEDDING_JSON_PATH)
    parser.add_argument("--assignment-jsonl-path", type=Path, default=DEFAULT_ASSIGNMENT_JSONL_PATH)
    parser.add_argument("--uid-to-path-json-path", type=Path, default=DEFAULT_UID_TO_PATH_JSON_PATH)
    parser.add_argument("--user-embedding-id", type=str, default=None)
    parser.add_argument("--prior-model-id", type=str, default=DEFAULT_PRIOR_MODEL_ID)
    parser.add_argument("--decoder-model-id", type=str, default=DEFAULT_DECODER_MODEL_ID)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", type=str, default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--patch-path", action="append", default=None)
    parser.add_argument(
        "--vanilla-prior",
        action="store_true",
        help="Use the unpatched diffusers Stable Cascade prior. Skips checkpoint/user embedding loading.",
    )
    parser.add_argument("--user-scale", type=float, default=1.0)
    parser.add_argument("--user-projection-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--user-projection-norm-affine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--user-adapter-projection-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--user-adapter-zero-init-out", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--inference-user-scale",
        type=float,
        default=1.0,
        help="Inference-only multiplier applied on top of checkpoint user_scale.",
    )
    parser.add_argument(
        "--inference-user-scale-sweep",
        nargs="+",
        type=float,
        default=None,
        help="Run each prompt/seed across multiple inference-only user scale values.",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--prior-steps", type=int, default=20)
    parser.add_argument("--prior-guidance-scale", type=float, default=4.0)
    parser.add_argument("--decoder-steps", type=int, default=10)
    parser.add_argument("--decoder-preflight-steps", type=int, default=1)
    parser.add_argument("--decoder-guidance-scale", type=float, default=0.0)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument(
        "--save-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a per-sample image grid across decoded conditions. Use --no-save-grid to skip it.",
    )
    parser.add_argument(
        "--grid-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw condition labels above grid cells. Use --no-grid-labels for image-only grids.",
    )
    parser.add_argument("--seed", dest="seeds", action="append", type=int, default=None)
    parser.add_argument(
        "--condition",
        dest="conditions",
        action="append",
        choices=CONDITIONS,
        default=None,
        help="Condition to run/decode. Repeat to select multiple. Defaults to all conditions.",
    )
    parser.add_argument("--max-query-prompts", type=int, default=2)
    parser.add_argument("--prompt", action="append", default=None, help="Manual prompt. Replaces query prompt selection.")
    parser.add_argument("--extra-prompt", action="append", default=None)
    parser.add_argument("--no-default-extra-prompts", action="store_true")
    parser.add_argument("--near-identical-l2-threshold", type=float, default=1e-8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--inference-run-dir",
        type=Path,
        default=None,
        help="Existing or explicit inference run directory. Image mode uses it if provided; report mode defaults to latest.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.seeds = _parse_seeds(args.seeds)

    try:
        if args.mode == "prior-smoke":
            run_dir = run_prior_smoke(args)
            print(f"[infer_stage2] prior-smoke complete: {run_dir}")
        elif args.mode == "image":
            run_dir = run_image_mode(args)
            print(f"[infer_stage2] image inference complete: {run_dir}")
        elif args.mode == "decode-latent":
            run_dir = run_decode_latent_mode(args)
            print(f"[infer_stage2] decode-latent complete: {run_dir}")
        elif args.mode == "report":
            report_path = run_report_mode(args)
            print(f"[infer_stage2] report written: {report_path}")
        else:  # pragma: no cover
            raise ValueError(f"Unsupported mode: {args.mode}")
        return 0
    except Exception:
        print("[infer_stage2] failure")
        print(traceback.format_exc().strip())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
