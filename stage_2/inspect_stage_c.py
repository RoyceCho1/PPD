from __future__ import annotations

import argparse
import importlib
import inspect
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


ATTN_FILTER_RE = re.compile(r"(attn|attention|transformer|cross)", re.IGNORECASE)
CROSS_HINT_RE = re.compile(
    r"(cross|attn2|encoder_hidden_states|context|crossattn|add_k_proj|add_v_proj|to_k|to_v)",
    re.IGNORECASE,
)
SELF_HINT_RE = re.compile(r"(self|attn1)", re.IGNORECASE)


@dataclass
class ModuleRecord:
    name: str
    parent: str
    cls_name: str
    classification: str
    forward_params: List[str]


def _safe_import(module_name: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return importlib.import_module(module_name), None
    except Exception:
        return None, traceback.format_exc()


def _resolve_dtype(torch_mod: Any, dtype_str: str) -> Optional[Any]:
    if dtype_str == "auto":
        return None
    table = {
        "float16": torch_mod.float16,
        "bfloat16": torch_mod.bfloat16,
        "float32": torch_mod.float32,
    }
    if dtype_str not in table:
        raise ValueError(f"Unsupported --torch-dtype={dtype_str}")
    return table[dtype_str]


def _pipeline_load_attempts(
    diffusers_mod: Any,
    model_id: str,
    revision: Optional[str],
    variant: Optional[str],
    torch_dtype: Optional[Any],
    local_files_only: bool,
    trust_remote_code: bool,
) -> Tuple[Optional[Any], List[Tuple[str, str]]]:
    errors: List[Tuple[str, str]] = []

    common_kwargs: Dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if revision is not None:
        common_kwargs["revision"] = revision
    if variant is not None:
        common_kwargs["variant"] = variant
    if torch_dtype is not None:
        common_kwargs["torch_dtype"] = torch_dtype

    # Attempt 1: StableCascadePriorPipeline (most explicit and preferred)
    pipe = None
    if hasattr(diffusers_mod, "StableCascadePriorPipeline"):
        try:
            cls = getattr(diffusers_mod, "StableCascadePriorPipeline")
            pipe = cls.from_pretrained(model_id, **common_kwargs)
            return pipe, errors
        except Exception:
            errors.append(("StableCascadePriorPipeline.from_pretrained", traceback.format_exc()))
    else:
        errors.append(("StableCascadePriorPipeline import", "Class not found in installed diffusers version."))

    # Attempt 2: DiffusionPipeline fallback
    if hasattr(diffusers_mod, "DiffusionPipeline"):
        try:
            cls = getattr(diffusers_mod, "DiffusionPipeline")
            pipe = cls.from_pretrained(model_id, **common_kwargs)
            return pipe, errors
        except Exception:
            errors.append(("DiffusionPipeline.from_pretrained", traceback.format_exc()))

    return None, errors


def _choose_core_component(pipeline: Any, preferred_component: Optional[str]) -> Tuple[str, Any]:
    if preferred_component:
        if hasattr(pipeline, preferred_component):
            return preferred_component, getattr(pipeline, preferred_component)
        raise ValueError(
            f"Requested component '{preferred_component}' was not found on pipeline. "
            f"Available attrs include: {sorted([k for k in dir(pipeline) if not k.startswith('_')])[:80]}"
        )

    preferred_order = [
        "prior",
        "transformer",
        "unet",
        "model",
        "decoder",
    ]

    for key in preferred_order:
        if hasattr(pipeline, key):
            return key, getattr(pipeline, key)

    # diffusers pipelines usually expose .components dict
    components = getattr(pipeline, "components", None)
    if isinstance(components, dict):
        for key in preferred_order:
            if key in components:
                return key, components[key]
        # fallback: first nn.Module-like object
        for key, value in components.items():
            if hasattr(value, "named_modules"):
                return str(key), value

    raise ValueError("Could not identify Stage C core component (prior/transformer/unet/model).")


def _module_forward_params(module: Any) -> List[str]:
    try:
        sig = inspect.signature(module.forward)
        return [p.name for p in sig.parameters.values()]
    except Exception:
        return []


def _classify_module(name: str, cls_name: str, forward_params: Sequence[str]) -> str:
    text = " ".join([name, cls_name] + list(forward_params))
    if CROSS_HINT_RE.search(text):
        return "cross_candidate"
    if SELF_HINT_RE.search(text):
        return "self_candidate"
    return "attention_related"


def _collect_attention_records(core_model: Any) -> List[ModuleRecord]:
    records: List[ModuleRecord] = []
    if not hasattr(core_model, "named_modules"):
        return records

    for name, module in core_model.named_modules():
        if name == "":
            continue
        cls_name = module.__class__.__name__
        if not (ATTN_FILTER_RE.search(name) or ATTN_FILTER_RE.search(cls_name)):
            continue

        parent = name.rsplit(".", 1)[0] if "." in name else "<root>"
        forward_params = _module_forward_params(module)
        classification = _classify_module(name, cls_name, forward_params)
        records.append(
            ModuleRecord(
                name=name,
                parent=parent,
                cls_name=cls_name,
                classification=classification,
                forward_params=forward_params,
            )
        )

    return records


def _print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _print_load_failures(errors: Sequence[Tuple[str, str]]) -> None:
    _print_header("Load Failures")
    if not errors:
        print("No detailed errors captured.")
        return
    for i, (stage, detail) in enumerate(errors, start=1):
        print(f"[{i}] {stage}")
        print(detail.strip())
        print("-" * 88)


def _print_pipeline_overview(pipeline: Any, model_id: str, component_name: str, core_model: Any) -> None:
    _print_header("Pipeline Overview")
    print(f"Model ID: {model_id}")
    print(f"Pipeline class: {pipeline.__class__.__name__}")
    print(f"Selected Stage C core component: {component_name}")
    print(f"Core model class: {core_model.__class__.__name__}")

    components = getattr(pipeline, "components", None)
    if isinstance(components, dict):
        print("\nPipeline components:")
        for k, v in components.items():
            print(f"- {k}: {v.__class__.__name__}")


def _print_forward_info(core_model: Any) -> None:
    _print_header("Core Model Forward Signature")
    try:
        sig = inspect.signature(core_model.forward)
        print(f"forward{sig}")
        print("\nMain input args:")
        for p in sig.parameters.values():
            ann = str(p.annotation) if p.annotation is not inspect._empty else "Any"
            default = "<required>" if p.default is inspect._empty else repr(p.default)
            print(f"- {p.name}: annotation={ann}, default={default}")
    except Exception:
        print("Could not inspect core_model.forward signature.")
        print(traceback.format_exc().strip())


def _print_attention_records(records: Sequence[ModuleRecord], top_n: int) -> None:
    _print_header(f"Attention-Related Modules (filtered, top {top_n})")
    if not records:
        print("No attention-related modules found using name/class filters.")
        return

    for i, r in enumerate(records[:top_n], start=1):
        fwd = ",".join(r.forward_params[:8])
        if len(r.forward_params) > 8:
            fwd += ",..."
        print(
            f"[{i:03d}] name={r.name} | class={r.cls_name} | parent={r.parent} "
            f"| type={r.classification} | fwd_args=[{fwd}]"
        )


def _print_candidate_summary(records: Sequence[ModuleRecord], top_n: int) -> None:
    cross = [r for r in records if r.classification == "cross_candidate"]
    self_attn = [r for r in records if r.classification == "self_candidate"]

    _print_header("Cross-Attention Candidate Modules")
    if not cross:
        print("No cross-attention candidates were detected by heuristic.")
    else:
        for i, r in enumerate(cross[:top_n], start=1):
            print(f"[{i:03d}] {r.name} ({r.cls_name}) | parent={r.parent}")

    _print_header("Self-Attention Candidate Modules")
    if not self_attn:
        print("No self-attention candidates were detected by heuristic.")
    else:
        for i, r in enumerate(self_attn[:top_n], start=1):
            print(f"[{i:03d}] {r.name} ({r.cls_name}) | parent={r.parent}")

    _print_header("Patch Location Hints (Human-Readable)")
    print("Goal: locate where text-conditioning cross-attention lives in Stage C before patching.")
    print("Priority order:")
    print("1. Modules classified as cross_candidate with names like attn2/cross/context/encoder_hidden_states.")
    print("2. Parent paths containing transformer blocks where both attn1 (self) and attn2 (cross) co-exist.")
    print("3. Core model forward args that expose encoder_hidden_states/context arguments.")

    if cross:
        print("\nTop suggested patch candidates:")
        for i, r in enumerate(cross[: min(10, len(cross))], start=1):
            print(f"- {i}. {r.name} (parent: {r.parent}, class: {r.cls_name})")
    else:
        print("\nNo direct cross candidates found. Inspect transformer block modules from filtered list.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Stable Cascade Stage C (prior) structure for patch-point discovery."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-cascade-prior",
        help="HF model id for Stage C prior.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--component",
        type=str,
        default=None,
        help="Optional explicit component name to inspect (e.g. prior, transformer).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=120,
        help="How many filtered module rows to print.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    _print_header("Environment Diagnostics")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Model ID: {args.model_id}")
    print(f"Requested dtype: {args.torch_dtype}")
    print(f"local_files_only: {args.local_files_only}")

    torch_mod, torch_err = _safe_import("torch")
    if torch_mod is None:
        print("[ERROR] Failed to import torch")
        print(torch_err.strip())
        return 1
    print(f"torch version: {getattr(torch_mod, '__version__', 'unknown')}")

    diffusers_mod, diffusers_err = _safe_import("diffusers")
    if diffusers_mod is None:
        print("[ERROR] Failed to import diffusers")
        print(diffusers_err.strip())
        return 1
    print(f"diffusers version: {getattr(diffusers_mod, '__version__', 'unknown')}")

    try:
        torch_dtype = _resolve_dtype(torch_mod, args.torch_dtype)
    except Exception:
        print("[ERROR] Invalid dtype configuration")
        print(traceback.format_exc().strip())
        return 1

    _print_header("Model Loading")
    pipeline, load_errors = _pipeline_load_attempts(
        diffusers_mod=diffusers_mod,
        model_id=args.model_id,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )

    if pipeline is None:
        print("[ERROR] Failed to load Stage C pipeline/model.")
        _print_load_failures(load_errors)
        return 2

    print("Successfully loaded pipeline.")

    try:
        component_name, core_model = _choose_core_component(pipeline, args.component)
    except Exception:
        print("[ERROR] Loaded pipeline but failed to choose core Stage C component.")
        print(traceback.format_exc().strip())
        return 3

    _print_pipeline_overview(pipeline, args.model_id, component_name, core_model)
    _print_forward_info(core_model)

    try:
        records = _collect_attention_records(core_model)
    except Exception:
        print("[ERROR] Failed while collecting named_modules attention records.")
        print(traceback.format_exc().strip())
        return 4

    _print_attention_records(records, top_n=max(1, args.top_n))
    _print_candidate_summary(records, top_n=max(1, args.top_n))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
