from __future__ import annotations

"""Patch helpers for adding a decoupled user-conditioning branch to Stage C.

This module intentionally does not implement training, DPO loss, or optimizer
setup. It only provides a safe wrapper around Stable Cascade SDCascadeAttnBlock
modules, helper functions for replacing those blocks, freezing the backbone, and
small smoke-test diagnostics.
"""

import argparse
import importlib
import inspect
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

try:
    from user_adapter import UserCrossAttentionAdapter, UserProjection
except ImportError:  # pragma: no cover - useful when imported as package module
    from stage_2.user_adapter import UserCrossAttentionAdapter, UserProjection


_COMPATIBLE_WRAPPER_CLASS_CACHE: Dict[type, type] = {}


@dataclass
class PatchCandidate:
    """One patchable Stable Cascade attention block."""

    path: str
    module: nn.Module
    cls_name: str
    forward_signature: str


@dataclass
class PatchSummary:
    """Summary returned after replacing Stage C attention blocks."""

    patched_paths: List[str]
    skipped_paths: List[str]
    d_model: int
    d_cross: int
    num_heads: int


@dataclass
class TrainableParameterSummary:
    """Trainable parameter counts after freezing the Stage C backbone."""

    trainable_tensors: int
    trainable_parameters: int
    total_parameters: int
    trainable_names: List[str]


class PatchedSDCascadeAttnBlock(nn.Module):
    """Wrap an SDCascadeAttnBlock with a decoupled user cross-attention branch.

    The original text-conditioning path is preserved by calling ``base_block``.
    If ``user_emb`` is omitted, the wrapper returns the original block output.
    If ``user_emb`` is provided, a parallel user attention residual is computed
    and added to the original output:

    ``out = base_block(x, kv) + user_scale * user_attention(base_block(x, kv))``
    """

    def __init__(
        self,
        base_block: nn.Module,
        d_model: int,
        d_cross: int,
        user_emb_dim: int = 3584,
        num_heads: int = 8,
        dropout: float = 0.0,
        user_scale: float = 0.0, # user embedding cross-attention의 scale
        trainable_user_scale: bool = True, # user scale을 학습할지 여부
        user_projection_bias: bool = True,
        user_projection_norm_affine: bool = True,
        user_adapter_projection_bias: bool = True,
        user_adapter_zero_init_out: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_cross <= 0:
            raise ValueError(f"d_cross must be positive, got {d_cross}")

        self.base_block = base_block
        self.d_model = d_model
        self.d_cross = d_cross
        self.user_emb_dim = user_emb_dim
        self.num_heads = num_heads

        self.user_projection = UserProjection(
            d_cross=d_cross,
            in_dim=user_emb_dim,
            bias=user_projection_bias,
            norm_elementwise_affine=user_projection_norm_affine,
        )
        self.user_adapter = UserCrossAttentionAdapter(
            d_model=d_model,
            d_cross=d_cross,
            num_heads=num_heads,
            dropout=dropout,
            projection_bias=user_adapter_projection_bias,
            zero_init_out=user_adapter_zero_init_out,
        )

        scale = torch.tensor(float(user_scale), dtype=torch.float32)
        if trainable_user_scale:
            self.user_scale = nn.Parameter(scale)
        else:
            self.register_buffer("user_scale", scale)

    def forward(
        self,
        x: Tensor, # input image feature
        kv: Tensor, # conditioning feature(text)
        user_emb: Optional[Tensor] = None, # user embedding
        user_emb_attention_mask: Optional[Tensor] = None, # user embedding attention mask
        **kwargs: Any,
    ) -> Tensor:
        """Run the original block and optionally add user-conditioning residual."""

        base_out = self._run_base_block(x=x, kv=kv, kwargs=kwargs) # original block output(x kv cross-attention)
        if user_emb is None:
            return base_out

        query, restore = self._to_token_sequence(base_out) # base_out을 token sequence로 변환
        self._sync_user_modules(reference=query) # user branch를 query tensor의 device와 dtype으로 동기화

        user_emb = user_emb.to(device=query.device, dtype=query.dtype)
        if user_emb_attention_mask is not None:
            user_emb_attention_mask = user_emb_attention_mask.to(device=query.device)

        user_tokens = self.user_projection(
            user_emb=user_emb,
            user_emb_attention_mask=user_emb_attention_mask,
        ) # user embedding을 cross-attention token space로 projection
        user_residual = self.user_adapter(
            query=query,
            user_tokens=user_tokens,
            user_attention_mask=user_emb_attention_mask,
        ) # user embedding을 cross-attention으로 주입
        user_residual = restore(user_residual) # user residual을 base_out과 같은 shape으로 복원
        return base_out + self.user_scale.to(device=base_out.device, dtype=base_out.dtype) * user_residual

    def _run_base_block(self, x: Tensor, kv: Tensor, kwargs: Mapping[str, Any]) -> Tensor:
        """Run the wrapped block while preserving its original path."""

        if not kwargs:
            return self.base_block(x, kv)
        try:
            return self.base_block(x, kv, **kwargs)
        except TypeError as exc:
            unsupported = ", ".join(sorted(kwargs.keys()))
            raise TypeError(
                "base_block did not accept extra keyword arguments passed to "
                f"PatchedSDCascadeAttnBlock: {unsupported}"
            ) from exc

    def _sync_user_modules(self, reference: Tensor) -> None:
        """Move the user branch to the query tensor's device and dtype."""

        dtype = reference.dtype if reference.is_floating_point() else torch.float32
        self.user_projection.to(device=reference.device, dtype=dtype)
        self.user_adapter.to(device=reference.device, dtype=dtype)

    def _to_token_sequence(self, hidden: Tensor) -> Tuple[Tensor, Any]:
        """Flatten supported hidden-state layouts to [B, N, D]."""

        if hidden.ndim == 3:
            if hidden.shape[-1] != self.d_model:
                raise ValueError(
                    "Expected 3D hidden state layout [B, N, D] with "
                    f"D={self.d_model}, got shape={tuple(hidden.shape)}"
                )
            return hidden, lambda y: y

        if hidden.ndim != 4:
            raise ValueError(
                "PatchedSDCascadeAttnBlock supports hidden states with ndim 3 or 4, "
                f"got shape={tuple(hidden.shape)}"
            )

        if hidden.shape[1] == self.d_model:
            bsz, channels, height, width = hidden.shape
            query = hidden.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)

            def restore_bchw(tokens: Tensor) -> Tensor:
                return tokens.reshape(bsz, height, width, channels).permute(0, 3, 1, 2).contiguous()

            return query, restore_bchw

        if hidden.shape[-1] == self.d_model:
            bsz, height, width, channels = hidden.shape
            query = hidden.reshape(bsz, height * width, channels)

            def restore_bhwc(tokens: Tensor) -> Tensor:
                return tokens.reshape(bsz, height, width, channels).contiguous()

            return query, restore_bhwc

        raise ValueError(
            "Could not infer hidden-state channel layout for user branch. "
            f"Expected channel dim {self.d_model} at axis 1 or -1, got shape={tuple(hidden.shape)}"
        )


def _compatible_wrapper_class(base_block: nn.Module) -> type:
    # wrapper class를 동적으로 생성
    # StableCascadeUNet은 isinstance(block, SDCascadeAttnBlock)으로 block을 찾기 때문에
    # PatchedSDCascadeAttnBlock만으로는 block을 찾을 수 없음
    # 따라서 PatchedSDCascadeAttnBlock과 base_block의 자식 클래스를 생성하여
    # isinstance(block, SDCascadeAttnBlock)이 true가 되도록 함

    base_cls = base_block.__class__
    if base_cls in _COMPATIBLE_WRAPPER_CLASS_CACHE:
        return _COMPATIBLE_WRAPPER_CLASS_CACHE[base_cls]

    wrapper_cls = type(
        f"Patched{base_cls.__name__}",
        (PatchedSDCascadeAttnBlock, base_cls),
        {"__doc__": PatchedSDCascadeAttnBlock.__doc__},
    )
    _COMPATIBLE_WRAPPER_CLASS_CACHE[base_cls] = wrapper_cls
    return wrapper_cls


def _format_forward_signature(module: nn.Module) -> str:
    """Return a compact forward signature string."""

    try:
        return f"forward{inspect.signature(module.forward)}"
    except Exception:
        return "<signature unavailable>"


def _get_child_module(root: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted module path without relying on newer PyTorch helpers."""

    current: nn.Module = root
    if path == "":
        return current
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]  # type: ignore[index]
        else:
            current = getattr(current, part)
    return current


def _set_child_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    """Replace a module at a dotted path."""

    if not path:
        raise ValueError("Refusing to replace the root module.")
    parent_path, child_name = path.rsplit(".", 1) if "." in path else ("", path)
    parent = _get_child_module(root, parent_path)
    if child_name.isdigit():
        parent[int(child_name)] = replacement  # type: ignore[index]
    else:
        setattr(parent, child_name, replacement)


def _infer_linear_in_features(module: Any) -> Optional[int]:
    # Linear layer의 in_features 추론

    value = getattr(module, "in_features", None)
    return int(value) if isinstance(value, int) and value > 0 else None


def infer_attention_dimensions(block: nn.Module) -> Tuple[int, int]:
    # d_model, d_cross 추론

    attention = getattr(block, "attention", None)
    if attention is None:
        raise ValueError(f"Block {block.__class__.__name__} has no child named 'attention'.")

    to_q = getattr(attention, "to_q", None) # query projection
    to_k = getattr(attention, "to_k", None) # key projection
    d_model = _infer_linear_in_features(to_q)
    d_cross = _infer_linear_in_features(to_k)

    if d_model is None:
        raise ValueError(f"Could not infer d_model from {block.__class__.__name__}.attention.to_q.")
    if d_cross is None:
        raise ValueError(f"Could not infer d_cross from {block.__class__.__name__}.attention.to_k.")

    return d_model, d_cross


def _choose_num_heads(d_model: int, requested_heads: int, attention: Optional[Any] = None) -> int:
    # num_heads 선택, d_model을 num_heads로 나눈 나머지가 0이어야 함

    for attr_name in ("heads", "num_heads"):
        value = getattr(attention, attr_name, None)
        if isinstance(value, int) and value > 0 and d_model % value == 0:
            return value

    if requested_heads > 0 and d_model % requested_heads == 0:
        return requested_heads

    for candidate in (32, 24, 16, 12, 8, 4, 2, 1):
        if d_model % candidate == 0:
            return candidate

    raise ValueError(f"Could not choose a valid num_heads for d_model={d_model}.")


def find_patch_candidate_blocks(model: nn.Module) -> List[PatchCandidate]:
    # stage c의 attention block을 찾음
    candidates: List[PatchCandidate] = []
    for path, module in model.named_modules():
        if isinstance(module, PatchedSDCascadeAttnBlock): # 이미 patching된 block은 제외
            continue
        if module.__class__.__name__ != "SDCascadeAttnBlock": # SDCascadeAttnBlock만 선택
            continue
        signature = _format_forward_signature(module)
        if "kv" not in signature: # kv를 인자로 받는 block만 선택 -> cross attention block만 찾는다
            continue
        if not hasattr(module, "attention"): # attention을 가지고 있는 block만 선택
            continue
        candidates.append(
            PatchCandidate(
                path=path,
                module=module,
                cls_name=module.__class__.__name__,
                forward_signature=signature,
            )
        )
    return candidates


def patch_stage_c_with_user_adapter(
    model: nn.Module,
    target_paths: Optional[Sequence[str]] = None,
    max_blocks: Optional[int] = None,
    user_emb_dim: int = 3584,
    d_cross: Optional[int] = None,
    num_heads: int = 8,
    dropout: float = 0.0,
    user_scale: float = 0.0,
    trainable_user_scale: bool = True,
    user_projection_bias: bool = True,
    user_projection_norm_affine: bool = True,
    user_adapter_projection_bias: bool = True,
    user_adapter_zero_init_out: bool = False,
) -> PatchSummary:
    # stage c의 attention block을 patching

    candidates = find_patch_candidate_blocks(model)
    candidate_by_path = {candidate.path: candidate for candidate in candidates}

    if target_paths is None:
        selected = candidates
    else:
        missing = [path for path in target_paths if path not in candidate_by_path]
        if missing:
            raise ValueError(
                "Requested patch paths were not found among SDCascadeAttnBlock candidates: "
                + ", ".join(missing)
            )
        selected = [candidate_by_path[path] for path in target_paths]

    if max_blocks is not None:
        selected = selected[: max(0, max_blocks)]

    patched_paths: List[str] = []
    skipped_paths: List[str] = []
    last_d_model: Optional[int] = None
    last_d_cross: Optional[int] = None
    last_num_heads: Optional[int] = None

    for candidate in selected:
        if isinstance(candidate.module, PatchedSDCascadeAttnBlock): # 이미 patching된 block은 제외
            skipped_paths.append(candidate.path)
            continue

        inferred_d_model, inferred_d_cross = infer_attention_dimensions(candidate.module) # d_model, d_cross 추론
        resolved_d_cross = int(d_cross) if d_cross is not None else inferred_d_cross # d_cross 설정
        resolved_num_heads = _choose_num_heads(
            d_model=inferred_d_model,
            requested_heads=num_heads,
            attention=getattr(candidate.module, "attention", None),
        )
        wrapper_cls = _compatible_wrapper_class(candidate.module)
        wrapper = wrapper_cls(
            base_block=candidate.module,
            d_model=inferred_d_model,
            d_cross=resolved_d_cross,
            user_emb_dim=user_emb_dim,
            num_heads=resolved_num_heads,
            dropout=dropout,
            user_scale=user_scale,
            trainable_user_scale=trainable_user_scale,
            user_projection_bias=user_projection_bias,
            user_projection_norm_affine=user_projection_norm_affine,
            user_adapter_projection_bias=user_adapter_projection_bias,
            user_adapter_zero_init_out=user_adapter_zero_init_out,
        )
        _set_child_module(model, candidate.path, wrapper) # 기존 attention block을 wrapper로 교체
        patched_paths.append(candidate.path) # patching된 block의 path 저장
        last_d_model = inferred_d_model # d_model 저장
        last_d_cross = resolved_d_cross # d_cross 저장
        last_num_heads = resolved_num_heads # num_heads 저장

    if not patched_paths:
        raise RuntimeError("No Stage C attention blocks were patched.")

    return PatchSummary(
        patched_paths=patched_paths,
        skipped_paths=skipped_paths,
        d_model=int(last_d_model),
        d_cross=int(last_d_cross),
        num_heads=int(last_num_heads),
    )


def freeze_stage_c_except_user_modules(model: nn.Module) -> None:
    # Freeze the Stage C backbone and keep only the user-conditioning trainable
    # subset needed for preference smoke training.
    for param in model.parameters():
        param.requires_grad = False # freeze everything

    for module in model.modules():
        if not isinstance(module, PatchedSDCascadeAttnBlock):
            continue
        for param in module.user_projection.parameters(): # user projection parameter
            param.requires_grad = True
        for param in module.user_adapter.k_proj.parameters(): # user W_k
            param.requires_grad = True
        for param in module.user_adapter.v_proj.parameters(): # user W_v
            param.requires_grad = True
        for param in module.user_adapter.out_proj.parameters(): # user branch output projection
            param.requires_grad = True
        if isinstance(module.user_scale, nn.Parameter):
            module.user_scale.requires_grad = True # user scale parameter


def summarize_trainable_parameters(model: nn.Module, max_names: int = 80) -> TrainableParameterSummary:
    # trainable parameter 요약

    trainable_names: List[str] = []
    trainable_tensors = 0
    trainable_parameters = 0
    total_parameters = 0

    for name, param in model.named_parameters():
        numel = int(param.numel())
        total_parameters += numel
        if not param.requires_grad:
            continue
        trainable_tensors += 1
        trainable_parameters += numel
        if len(trainable_names) < max_names:
            trainable_names.append(name)

    return TrainableParameterSummary(
        trainable_tensors=trainable_tensors,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        trainable_names=trainable_names,
    )


def iter_patched_blocks(model: nn.Module) -> Iterable[Tuple[str, PatchedSDCascadeAttnBlock]]:
    """Yield patched block paths and modules."""

    for path, module in model.named_modules():
        if isinstance(module, PatchedSDCascadeAttnBlock):
            yield path, module


def smoke_test_user_branch(
    model: nn.Module,
    user_seq_len: int = 4,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run the new user branch on dummy tensors without invoking full Stage C."""

    try:
        first_path, first_block = next(iter_patched_blocks(model))
    except StopIteration as exc:
        raise RuntimeError("No PatchedSDCascadeAttnBlock was found for smoke testing.") from exc

    if device is None:
        try:
            device = next(first_block.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    dtype = torch.float32
    query = torch.zeros(batch_size, 2, first_block.d_model, device=device, dtype=dtype)
    user_emb = torch.zeros(batch_size, user_seq_len, first_block.user_emb_dim, device=device, dtype=dtype)
    mask = torch.ones(batch_size, user_seq_len, device=device, dtype=torch.long)

    first_block._sync_user_modules(query)
    with torch.no_grad():
        user_tokens = first_block.user_projection(user_emb, user_emb_attention_mask=mask)
        user_out = first_block.user_adapter(query=query, user_tokens=user_tokens, user_attention_mask=mask)

    return {
        "patched_block": first_path,
        "query_shape": tuple(query.shape),
        "user_emb_shape": tuple(user_emb.shape),
        "user_tokens_shape": tuple(user_tokens.shape),
        "user_out_shape": tuple(user_out.shape),
    }


def _safe_import_module(module_name: str) -> Tuple[Optional[Any], Optional[str]]:
    """Import helper for optional smoke-test dependencies."""

    try:
        return importlib.import_module(module_name), None
    except Exception:
        return None, traceback.format_exc()


def _load_stage_c_prior(
    model_id: str,
    local_files_only: bool,
    torch_dtype: Optional[torch.dtype],
    trust_remote_code: bool,
) -> Any:
    """Load Stable Cascade prior pipeline for smoke testing."""

    diffusers, diffusers_error = _safe_import_module("diffusers")
    if diffusers is None:
        raise ImportError("Failed to import diffusers:\n" + str(diffusers_error))

    kwargs: Dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    try:
        pipeline_cls = getattr(diffusers, "StableCascadePriorPipeline")
    except Exception as exc:
        raise ImportError(
            "diffusers failed while resolving StableCascadePriorPipeline. "
            "This often indicates a diffusers/transformers version mismatch."
        ) from exc

    return pipeline_cls.from_pretrained(model_id, **kwargs)


def _resolve_torch_dtype(dtype_name: str) -> Optional[torch.dtype]:
    """Resolve CLI dtype into torch dtype."""

    if dtype_name == "auto":
        return None
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _parse_paths(raw_paths: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated list of module paths."""

    if raw_paths is None or raw_paths.strip() == "":
        return None
    return [item.strip() for item in raw_paths.split(",") if item.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for smoke testing patch application."""

    parser = argparse.ArgumentParser(description="Patch Stable Cascade Stage C attention blocks with user adapters.")
    parser.add_argument("--model-id", type=str, default="stabilityai/stable-cascade-prior")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", type=str, default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--target-paths", type=str, default=None, help="Comma-separated SDCascadeAttnBlock paths.")
    parser.add_argument("--max-blocks", type=int, default=2, help="Limit patched blocks for smoke testing.")
    parser.add_argument("--user-emb-dim", type=int, default=3584)
    parser.add_argument("--d-cross", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--user-scale", type=float, default=0.0)
    parser.add_argument("--frozen-scale", action="store_true", help="Use a non-trainable user_scale buffer.")
    parser.add_argument("--skip-smoke-branch", action="store_true", help="Skip dummy user branch smoke test.")
    return parser


def main() -> int:
    """Load Stage C, patch candidate blocks, freeze, and print diagnostics."""

    args = _build_arg_parser().parse_args()

    try:
        torch_dtype = _resolve_torch_dtype(args.torch_dtype)
        pipe = _load_stage_c_prior(
            model_id=args.model_id,
            local_files_only=args.local_files_only,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception:
        print("[patch_stage_c] Failed to load Stable Cascade Stage C prior.")
        print(traceback.format_exc().strip())
        return 1

    if not hasattr(pipe, "prior"):
        print("[patch_stage_c] Loaded pipeline does not expose a .prior component.")
        return 2

    model = pipe.prior
    candidates = find_patch_candidate_blocks(model)
    print("[patch_stage_c] Stage C core model:", model.__class__.__name__)
    print("[patch_stage_c] candidate SDCascadeAttnBlock count:", len(candidates))
    for candidate in candidates[: min(10, len(candidates))]:
        print(f"- candidate: {candidate.path} | {candidate.forward_signature}")

    try:
        patch_summary = patch_stage_c_with_user_adapter(
            model=model,
            target_paths=_parse_paths(args.target_paths),
            max_blocks=args.max_blocks,
            user_emb_dim=args.user_emb_dim,
            d_cross=args.d_cross,
            num_heads=args.num_heads,
            dropout=args.dropout,
            user_scale=args.user_scale,
            trainable_user_scale=not args.frozen_scale,
        )
        freeze_stage_c_except_user_modules(model)
    except Exception:
        print("[patch_stage_c] Failed while patching/freezing Stage C.")
        print(traceback.format_exc().strip())
        return 3

    trainable_summary = summarize_trainable_parameters(model)
    print("\n[patch_stage_c] patch summary")
    print("patched count:", len(patch_summary.patched_paths))
    print("patched paths:", patch_summary.patched_paths)
    print("skipped paths:", patch_summary.skipped_paths)
    print("d_model:", patch_summary.d_model)
    print("d_cross:", patch_summary.d_cross)
    print("num_heads:", patch_summary.num_heads)

    print("\n[patch_stage_c] trainable parameter summary")
    print("trainable tensors:", trainable_summary.trainable_tensors)
    print("trainable parameters:", trainable_summary.trainable_parameters)
    print("total parameters:", trainable_summary.total_parameters)
    print("first trainable names:")
    for name in trainable_summary.trainable_names[:30]:
        print("-", name)

    if not args.skip_smoke_branch:
        try:
            branch_summary = smoke_test_user_branch(model)
            print("\n[patch_stage_c] dummy user branch smoke test")
            for key, value in branch_summary.items():
                print(f"{key}: {value}")
        except Exception:
            print("\n[patch_stage_c] Dummy user branch smoke test failed.")
            print(traceback.format_exc().strip())
            return 4

    print("\n[patch_stage_c] Done. No training loop, DPO loss, or optimizer was created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
