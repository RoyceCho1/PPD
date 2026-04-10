from __future__ import annotations

"""Inspect Stable Cascade Stage C prior structure for future patch planning.

This script does not patch the model. It only loads the Stage C prior, scans
attention-like modules, classifies likely self-attention and text-conditioning
cross-attention candidates, and prints a human-readable patch-location summary.
"""

import argparse
import importlib
import inspect
import re
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_FILTER_KEYWORDS = ("attn", "attention", "cross", "transformer", "block")
CONDITIONING_HINTS = (
    "context",
    "conditioning",
    "encoder_hidden_states",
    "clip",
    "cross",
    "crossattn",
    "attn2",
    "to_k",
    "to_v",
    "add_k_proj",
    "add_v_proj",
    "kv",
)
SELF_ATTENTION_HINTS = ("self", "attn1", "to_q")
PATCH_PARENT_CLASSES = ("Attention", "SDCascadeAttnBlock")


@dataclass
class ParamSample:
    """Small parameter preview for an inspected module."""

    name: str
    shape: Tuple[int, ...]
    dtype: str


@dataclass
class ModuleRecord:
    """Structured description of one attention-related module."""

    name: str
    parent: str
    pattern: str
    cls_name: str
    classification: str
    forward_signature: str
    forward_params: List[str]
    hint_matches: List[str] = field(default_factory=list)
    child_hint_matches: List[str] = field(default_factory=list)
    param_samples: List[ParamSample] = field(default_factory=list)
    param_count: int = 0
    patch_score: int = 0


def _safe_import(module_name: str) -> Tuple[Optional[Any], Optional[str]]:
    """Import a module and return a traceback string instead of raising."""

    try:
        return importlib.import_module(module_name), None
    except Exception:
        return None, traceback.format_exc()


def _resolve_dtype(torch_mod: Any, dtype_str: str) -> Optional[Any]:
    """Map a CLI dtype string to a torch dtype."""

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


def _format_signature(obj: Any) -> str:
    """Return a compact forward signature for a module-like object."""

    forward = getattr(obj, "forward", None)
    if forward is None:
        return "<no forward>"
    try:
        return f"forward{inspect.signature(forward)}"
    except Exception:
        return "<signature unavailable>"


def _forward_params(obj: Any) -> List[str]:
    """Return forward parameter names for a module-like object."""

    forward = getattr(obj, "forward", None)
    if forward is None:
        return []
    try:
        return [p.name for p in inspect.signature(forward).parameters.values()]
    except Exception:
        return []


def _normalize_pattern(name: str) -> str:
    """Replace numeric path segments with '*' to reveal repeated layer patterns."""

    return ".".join("*" if part.isdigit() else part for part in name.split("."))


def _compile_filter_regex(extra_keywords: Sequence[str]) -> re.Pattern[str]:
    """Build the attention-candidate filter regex from defaults and CLI keywords."""

    keywords = list(DEFAULT_FILTER_KEYWORDS) + [kw for kw in extra_keywords if kw]
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile("(" + "|".join(escaped) + ")", re.IGNORECASE)


def _matches_any(text: str, hints: Sequence[str]) -> List[str]:
    """Return hint tokens that appear in text, case-insensitively."""

    lower = text.lower()
    return [hint for hint in hints if hint.lower() in lower]


def _iter_child_names(module: Any, max_children: int = 80) -> Iterable[str]:
    """Yield a bounded set of child module names for local structural hints."""

    named_modules = getattr(module, "named_modules", None)
    if not callable(named_modules):
        return
    for idx, (child_name, child_module) in enumerate(named_modules()):
        if idx >= max_children:
            break
        if child_name == "":
            continue
        yield f"{child_name}:{child_module.__class__.__name__}"


def _parameter_samples(module: Any, show_params: bool, max_params: int = 8) -> Tuple[List[ParamSample], int]:
    """Collect a bounded parameter preview from a module."""

    if not show_params:
        return [], 0

    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        return [], 0

    samples: List[ParamSample] = []
    total = 0
    try:
        for param_name, param in named_parameters(recurse=True):
            total += 1
            if len(samples) >= max_params:
                continue
            shape = tuple(int(x) for x in getattr(param, "shape", ()))
            dtype = str(getattr(param, "dtype", "unknown")).replace("torch.", "")
            samples.append(ParamSample(name=param_name, shape=shape, dtype=dtype))
    except Exception:
        return samples, total

    return samples, total


def _classify_module(
    name: str,
    cls_name: str,
    forward_params: Sequence[str],
    child_hint_matches: Sequence[str],
) -> Tuple[str, List[str], int]:
    """Classify a module and estimate how useful it is as a future patch target."""

    text = " ".join([name, cls_name] + list(forward_params))
    direct_hints = _matches_any(text, CONDITIONING_HINTS)
    self_hints = _matches_any(text, SELF_ATTENTION_HINTS)
    child_hints = list(child_hint_matches)

    is_attention_parent = "attention" in cls_name.lower() or "attnblock" in cls_name.lower()
    has_external_arg = any(p in forward_params for p in ("encoder_hidden_states", "context", "kv", "clip_text"))
    has_kv_child = any(h in child_hints for h in ("to_k", "to_v", "kv", "add_k_proj", "add_v_proj"))

    patch_score = 0
    if is_attention_parent:
        patch_score += 3
    if has_external_arg:
        patch_score += 5
    if direct_hints:
        patch_score += 2
    if has_kv_child:
        patch_score += 2
    if name.endswith((".to_k", ".to_v")):
        patch_score -= 2
    if cls_name == "Linear":
        patch_score -= 1

    if has_external_arg or (is_attention_parent and (direct_hints or has_kv_child)):
        return "text_cross_attention_candidate", sorted(set(direct_hints)), patch_score
    if name.endswith((".to_k", ".to_v")) or any(h in direct_hints for h in ("to_k", "to_v", "add_k_proj", "add_v_proj")):
        return "conditioning_projection_candidate", sorted(set(direct_hints)), patch_score
    if any(h in self_hints for h in ("self", "attn1")):
        return "self_attention_candidate", sorted(set(self_hints)), patch_score
    if is_attention_parent:
        return "attention_block_candidate", sorted(set(direct_hints + self_hints)), patch_score
    return "attention_related", sorted(set(direct_hints + self_hints)), patch_score


def _load_pipeline(
    diffusers_mod: Any,
    model_id: str,
    revision: Optional[str],
    variant: Optional[str],
    torch_dtype: Optional[Any],
    local_files_only: bool,
    trust_remote_code: bool,
) -> Tuple[Optional[Any], List[Tuple[str, str]]]:
    """Try StableCascadePriorPipeline first, then DiffusionPipeline as fallback."""

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

    try:
        stable_cascade_cls = getattr(diffusers_mod, "StableCascadePriorPipeline")
    except Exception:
        stable_cascade_cls = None
        errors.append(("StableCascadePriorPipeline lookup", traceback.format_exc()))

    if stable_cascade_cls is not None:
        try:
            return stable_cascade_cls.from_pretrained(model_id, **common_kwargs), errors
        except Exception:
            errors.append(("StableCascadePriorPipeline.from_pretrained", traceback.format_exc()))

    try:
        diffusion_cls = getattr(diffusers_mod, "DiffusionPipeline")
    except Exception:
        diffusion_cls = None
        errors.append(("DiffusionPipeline lookup", traceback.format_exc()))

    if diffusion_cls is not None:
        try:
            return diffusion_cls.from_pretrained(model_id, **common_kwargs), errors
        except Exception:
            errors.append(("DiffusionPipeline.from_pretrained", traceback.format_exc()))

    return None, errors


def _choose_core_component(pipeline: Any, preferred_component: Optional[str]) -> Tuple[str, Any]:
    """Select the Stage C core module from a loaded pipeline."""

    if preferred_component:
        if hasattr(pipeline, preferred_component):
            return preferred_component, getattr(pipeline, preferred_component)
        public_attrs = sorted(k for k in dir(pipeline) if not k.startswith("_"))[:80]
        raise ValueError(
            f"Requested component '{preferred_component}' was not found. "
            f"Available public attrs include: {public_attrs}"
        )

    preferred_order = ("prior", "transformer", "unet", "model", "decoder")
    for key in preferred_order:
        if hasattr(pipeline, key):
            return key, getattr(pipeline, key)

    components = getattr(pipeline, "components", None)
    if isinstance(components, Mapping):
        for key in preferred_order:
            if key in components:
                return key, components[key]
        for key, value in components.items():
            if hasattr(value, "named_modules"):
                return str(key), value

    raise ValueError("Could not identify Stage C core component (prior/transformer/unet/model).")


def _collect_attention_records(
    core_model: Any,
    filter_re: re.Pattern[str],
    show_params: bool,
) -> List[ModuleRecord]:
    """Scan named_modules and return attention-like structure records."""

    named_modules = getattr(core_model, "named_modules", None)
    if not callable(named_modules):
        raise TypeError(f"Core model {core_model.__class__.__name__} does not expose named_modules().")

    records: List[ModuleRecord] = []
    for name, module in named_modules():
        if not name:
            continue

        cls_name = module.__class__.__name__
        if not (filter_re.search(name) or filter_re.search(cls_name)):
            continue

        parent = name.rsplit(".", 1)[0] if "." in name else "<root>"
        forward_params = _forward_params(module)
        forward_signature = _format_signature(module)
        child_text = " ".join(_iter_child_names(module))
        child_hints = _matches_any(child_text, CONDITIONING_HINTS)
        classification, direct_hints, patch_score = _classify_module(
            name=name,
            cls_name=cls_name,
            forward_params=forward_params,
            child_hint_matches=child_hints,
        )
        param_samples, param_count = _parameter_samples(module, show_params=show_params)

        records.append(
            ModuleRecord(
                name=name,
                parent=parent,
                pattern=_normalize_pattern(name),
                cls_name=cls_name,
                classification=classification,
                forward_signature=forward_signature,
                forward_params=list(forward_params),
                hint_matches=direct_hints,
                child_hint_matches=sorted(set(child_hints)),
                param_samples=param_samples,
                param_count=param_count,
                patch_score=patch_score,
            )
        )

    return records


def _dedupe_patch_candidates(records: Sequence[ModuleRecord], max_items: int) -> List[ModuleRecord]:
    """Return high-level attention parent candidates, deduplicated by module path."""

    useful = [
        r
        for r in records
        if r.classification in ("text_cross_attention_candidate", "attention_block_candidate")
        and r.cls_name != "Linear"
        and not r.name.endswith((".to_k", ".to_v", ".to_q"))
    ]
    useful.sort(key=lambda r: (r.patch_score, r.classification == "text_cross_attention_candidate"), reverse=True)
    return useful[:max_items]


def _pattern_counts(records: Sequence[ModuleRecord]) -> List[Tuple[str, int]]:
    """Count repeated module path patterns."""

    counter = Counter(r.pattern for r in records if r.patch_score > 0)
    return counter.most_common()


def _classification_counts(records: Sequence[ModuleRecord]) -> Dict[str, int]:
    """Count module records by classification label."""

    return dict(Counter(r.classification for r in records))


def _group_by_parent(records: Sequence[ModuleRecord]) -> Dict[str, List[ModuleRecord]]:
    """Group records by parent path."""

    grouped: Dict[str, List[ModuleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.parent].append(record)
    return grouped


def _print_header(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def _print_load_failures(errors: Sequence[Tuple[str, str]]) -> None:
    """Print model-load errors with enough context for environment debugging."""

    _print_header("Load Failures")
    if not errors:
        print("No detailed loader errors were captured.")
        return
    for idx, (stage, detail) in enumerate(errors, start=1):
        print(f"[{idx}] {stage}")
        print(detail.strip())
        print("-" * 96)


def _print_environment(args: argparse.Namespace, torch_mod: Any, diffusers_mod: Any) -> None:
    """Print runtime diagnostics."""

    _print_header("Environment Diagnostics")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Model ID: {args.model_id}")
    print(f"Requested dtype: {args.torch_dtype}")
    print(f"local_files_only: {args.local_files_only}")
    print(f"torch version: {getattr(torch_mod, '__version__', 'unknown')}")
    print(f"diffusers version: {getattr(diffusers_mod, '__version__', 'unknown')}")


def _print_pipeline_overview(pipeline: Any, model_id: str, component_name: str, core_model: Any) -> None:
    """Print loaded pipeline and selected core model information."""

    _print_header("Pipeline Overview")
    print(f"Model ID: {model_id}")
    print(f"Pipeline class: {pipeline.__class__.__name__}")
    print(f"Selected Stage C core component: {component_name}")
    print(f"Core model class: {core_model.__class__.__name__}")

    components = getattr(pipeline, "components", None)
    if isinstance(components, Mapping):
        print("\nPipeline components:")
        for key, value in components.items():
            print(f"- {key}: {value.__class__.__name__}")


def _print_forward_info(core_model: Any) -> None:
    """Print top-level forward signature and highlight conditioning-like inputs."""

    _print_header("Core Model Forward Signature")
    print(_format_signature(core_model))
    params = _forward_params(core_model)
    if not params:
        print("No forward parameters could be inspected.")
        return

    conditioning = _matches_any(" ".join(params), CONDITIONING_HINTS)
    print("\nForward args:")
    for param in params:
        marker = "  <-- conditioning-like" if _matches_any(param, CONDITIONING_HINTS) else ""
        print(f"- {param}{marker}")
    if conditioning:
        print(f"\nConditioning-related forward hints: {', '.join(conditioning)}")


def _format_param_samples(record: ModuleRecord) -> str:
    """Format a module's parameter preview."""

    if not record.param_samples:
        return "params=<hidden or none>"
    entries = [
        f"{sample.name}:{list(sample.shape)}:{sample.dtype}"
        for sample in record.param_samples
    ]
    suffix = "" if record.param_count <= len(record.param_samples) else f", ... total={record.param_count}"
    return "params=[" + "; ".join(entries) + suffix + "]"


def _print_record(record: ModuleRecord, idx: int, verbose: bool, show_params: bool) -> None:
    """Print one module record in summary or verbose form."""

    hints = ",".join(record.hint_matches) if record.hint_matches else "-"
    child_hints = ",".join(record.child_hint_matches) if record.child_hint_matches else "-"
    print(
        f"[{idx:03d}] {record.name} | class={record.cls_name} | type={record.classification} "
        f"| score={record.patch_score} | hints={hints} | child_hints={child_hints}"
    )
    if verbose:
        print(f"      parent={record.parent}")
        print(f"      pattern={record.pattern}")
        print(f"      signature={record.forward_signature}")
    if show_params:
        print(f"      {_format_param_samples(record)}")


def _print_records(records: Sequence[ModuleRecord], max_items: int, verbose: bool, show_params: bool) -> None:
    """Print filtered attention-like module records."""

    _print_header(f"Attention-Related Module Records (showing up to {max_items})")
    if not records:
        print("No attention-related modules were found with the selected keyword filters.")
        return

    for idx, record in enumerate(records[:max_items], start=1):
        _print_record(record, idx=idx, verbose=verbose, show_params=show_params)

    if len(records) > max_items:
        print(f"\n... omitted {len(records) - max_items} records. Increase --max-items or use --summary-only.")


def _print_candidate_sections(records: Sequence[ModuleRecord], max_items: int, verbose: bool) -> None:
    """Print self-attention and cross-attention candidate sections."""

    cross = [r for r in records if r.classification == "text_cross_attention_candidate"]
    projections = [r for r in records if r.classification == "conditioning_projection_candidate"]
    self_attn = [r for r in records if r.classification == "self_attention_candidate"]

    cross.sort(key=lambda r: r.patch_score, reverse=True)
    projections.sort(key=lambda r: r.patch_score, reverse=True)

    _print_header("Text-Conditioning Cross-Attention Candidates")
    if not cross:
        print("No direct text-conditioning cross-attention candidates were detected.")
    else:
        for idx, record in enumerate(cross[:max_items], start=1):
            print(f"[{idx:03d}] {record.name} ({record.cls_name}) | score={record.patch_score}")
            if verbose:
                print(f"      signature={record.forward_signature}")
                print(f"      reason=hints={record.hint_matches}, child_hints={record.child_hint_matches}")

    _print_header("Conditioning Projection Submodules")
    if not projections:
        print("No to_k/to_v-style conditioning projection modules were detected.")
    else:
        for idx, record in enumerate(projections[:max_items], start=1):
            print(f"[{idx:03d}] {record.name} ({record.cls_name}) | parent={record.parent}")

    _print_header("Self-Attention Candidates")
    if not self_attn:
        print("No explicit self-attention candidates were detected by name/signature hints.")
        print("This can be normal for Stable Cascade if attention blocks are not named attn1/self.")
    else:
        for idx, record in enumerate(self_attn[:max_items], start=1):
            print(f"[{idx:03d}] {record.name} ({record.cls_name}) | score={record.patch_score}")


def _print_pattern_summary(records: Sequence[ModuleRecord], max_items: int) -> None:
    """Print repeated module path patterns that likely need repeated patching."""

    _print_header("Repeated Patch Pattern Summary")
    pattern_rows = _pattern_counts(records)
    if not pattern_rows:
        print("No repeated attention patterns were found.")
        return

    for idx, (pattern, count) in enumerate(pattern_rows[:max_items], start=1):
        print(f"[{idx:03d}] count={count:03d} pattern={pattern}")


def _print_final_patch_summary(records: Sequence[ModuleRecord], max_items: int) -> None:
    """Print final human-readable patch recommendations."""

    _print_header("Patch Candidate Summary")
    candidates = _dedupe_patch_candidates(records, max_items=max_items)
    counts = _classification_counts(records)

    print("Classification counts:")
    for key in sorted(counts):
        print(f"- {key}: {counts[key]}")

    if not candidates:
        print("\nNo high-confidence patch candidates were found.")
        print("Recommendation: inspect verbose records for modules with conditioning-like forward args.")
        return

    print("\nPatch candidate module paths:")
    for idx, record in enumerate(candidates, start=1):
        reason_parts = []
        if "encoder_hidden_states" in record.forward_params:
            reason_parts.append("accepts encoder_hidden_states")
        if "kv" in record.forward_params:
            reason_parts.append("accepts kv conditioning")
        if record.child_hint_matches:
            reason_parts.append(f"contains child hints {record.child_hint_matches}")
        if record.hint_matches:
            reason_parts.append(f"matches hints {record.hint_matches}")
        reason = "; ".join(reason_parts) if reason_parts else "attention-like module with conditioning hints"
        print(f"- {record.name} ({record.cls_name}, score={record.patch_score}): {reason}")

    print("\nRepeated layer patterns to consider patching:")
    shown = 0
    for pattern, count in _pattern_counts(candidates):
        if shown >= max_items:
            break
        print(f"- {pattern} appears {count} time(s) in top patch candidates")
        shown += 1

    print("\nRecommendation for future patch_stage_c.py:")
    print("- Patch the attention parent module, not the Linear to_k/to_v child alone.")
    print("- Prefer modules whose forward accepts encoder_hidden_states or kv, because those are the text/conditioning entry points.")
    print("- Add the user branch as a decoupled parallel attention path next to the text-conditioning path.")
    print("- Keep user_adapter.py disconnected for now; this inspection script only identifies candidate attachment points.")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""

    parser = argparse.ArgumentParser(
        description="Inspect Stable Cascade Stage C prior structure for patch-point discovery."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-cascade-prior",
        help="HF model id or local path for the Stable Cascade prior.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--component",
        type=str,
        default=None,
        help="Optional explicit pipeline component to inspect, e.g. prior or transformer.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print signatures, patterns, and reasons per record.")
    parser.add_argument(
        "--max-items",
        type=int,
        default=80,
        help="Maximum number of records/candidates to print per section.",
    )
    parser.add_argument("--show-params", action="store_true", help="Print bounded parameter name/shape previews.")
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Additional keyword for filtering named_modules; may be passed multiple times.",
    )
    parser.add_argument("--summary-only", action="store_true", help="Only print overview and final summaries.")
    return parser.parse_args()


def main() -> int:
    """Run Stage C inspection and print patch-location analysis."""

    args = parse_args()

    torch_mod, torch_err = _safe_import("torch")
    if torch_mod is None:
        _print_header("Environment Diagnostics")
        print("[ERROR] Failed to import torch.")
        print(torch_err.strip() if torch_err else "No traceback captured.")
        return 1

    diffusers_mod, diffusers_err = _safe_import("diffusers")
    if diffusers_mod is None:
        _print_header("Environment Diagnostics")
        print("[ERROR] Failed to import diffusers.")
        print(diffusers_err.strip() if diffusers_err else "No traceback captured.")
        return 1

    _print_environment(args=args, torch_mod=torch_mod, diffusers_mod=diffusers_mod)

    try:
        torch_dtype = _resolve_dtype(torch_mod, args.torch_dtype)
    except Exception:
        print("[ERROR] Invalid dtype configuration.")
        print(traceback.format_exc().strip())
        return 1

    _print_header("Model Loading")
    pipeline, load_errors = _load_pipeline(
        diffusers_mod=diffusers_mod,
        model_id=args.model_id,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )

    if pipeline is None:
        print("[ERROR] Failed to load Stable Cascade Stage C prior pipeline.")
        _print_load_failures(load_errors)
        return 2

    print("Successfully loaded pipeline.")

    try:
        component_name, core_model = _choose_core_component(pipeline, args.component)
    except Exception:
        print("[ERROR] Loaded pipeline, but failed to choose the Stage C core component.")
        print(traceback.format_exc().strip())
        return 3

    _print_pipeline_overview(
        pipeline=pipeline,
        model_id=args.model_id,
        component_name=component_name,
        core_model=core_model,
    )
    _print_forward_info(core_model)

    try:
        filter_re = _compile_filter_regex(args.keyword)
        records = _collect_attention_records(
            core_model=core_model,
            filter_re=filter_re,
            show_params=args.show_params,
        )
    except Exception:
        print("[ERROR] Failed while scanning named_modules for attention candidates.")
        print(traceback.format_exc().strip())
        return 4

    max_items = max(1, int(args.max_items))
    if not args.summary_only:
        _print_records(records, max_items=max_items, verbose=args.verbose, show_params=args.show_params)
        _print_candidate_sections(records, max_items=max_items, verbose=args.verbose)

    _print_pattern_summary(records, max_items=max_items)
    _print_final_patch_summary(records, max_items=max_items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
