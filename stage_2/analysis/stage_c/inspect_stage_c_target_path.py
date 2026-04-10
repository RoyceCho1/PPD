from __future__ import annotations

"""Inspect the Stable Cascade Stage C clean target latent path.

This script is an analysis tool only. It does not implement DPO loss, an
optimizer, a training loop, or a latent-backed dataset. Its purpose is to
separate CLIP image-conditioning from the Stage C denoising target path and to
collect concrete hints from diffusers pipeline/module/source structure.
"""

import argparse
import importlib
import inspect
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_PRIOR_MODEL_ID = "stabilityai/stable-cascade-prior"
DEFAULT_DECODER_MODEL_ID = "stabilityai/stable-cascade"

PIPELINE_CLASS_NAMES = (
    "StableCascadePriorPipeline",
    "StableCascadeDecoderPipeline",
    "StableCascadeCombinedPipeline",
)

COMPONENT_HINTS = (
    "prior",
    "decoder",
    "unet",
    "transformer",
    "image_encoder",
    "feature_extractor",
    "image_processor",
    "vqgan",
    "effnet",
    "stage_a",
    "stage_b",
    "stage_c",
    "text_encoder",
    "tokenizer",
    "scheduler",
)

LATENT_PATH_HINTS = (
    "image_embedding",
    "image_embeddings",
    "latent",
    "latents",
    "effnet",
    "vqgan",
    "stage_a",
    "stage_b",
    "stage_c",
    "encode",
    "encoder",
    "decoder",
    "compress",
    "quant",
    "sample",
    "clip_img",
    "clip_image",
    "feature_extractor",
    "image_encoder",
    "pixels",
    "sca",
    "crp",
)

SOURCE_HINTS = (
    "image_embeddings",
    "clip_img",
    "feature_extractor",
    "image_encoder",
    "effnet",
    "vqgan",
    "Stage",
    "stage",
    "latent",
    "latents",
    "sample",
    "decode",
    "encode",
)


@dataclass
class LoadResult:
    """Result of one optional pipeline load attempt."""

    label: str
    class_name: str
    model_id: str
    pipeline: Optional[Any] = None
    errors: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class ComponentRecord:
    """Small description of one pipeline component."""

    pipeline_label: str
    name: str
    class_name: str
    forward_signature: str
    config_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleRecord:
    """Small description of one module that may participate in image paths."""

    component: str
    path: str
    class_name: str
    forward_signature: str
    matched_hints: List[str] = field(default_factory=list)
    param_shapes: List[Tuple[str, Tuple[int, ...], str]] = field(default_factory=list)


@dataclass
class SourceHint:
    """One bounded source-code hint from an installed diffusers object."""

    object_name: str
    source_file: str
    line_no: int
    text: str


def _safe_import(module_name: str) -> Tuple[Optional[Any], Optional[str]]:
    """Import a module and return a traceback string on failure."""

    try:
        return importlib.import_module(module_name), None
    except Exception:
        return None, traceback.format_exc()


def _format_signature(callable_or_module: Any, attr: str = "forward") -> str:
    """Return a compact signature for a module method or callable."""

    target = getattr(callable_or_module, attr, None)
    if target is None and callable(callable_or_module):
        target = callable_or_module
    if target is None:
        return f"<no {attr}>"
    try:
        return f"{attr}{inspect.signature(target)}"
    except Exception:
        return "<signature unavailable>"


def _matches(text: str, hints: Sequence[str]) -> List[str]:
    """Return hint strings found in text, case-insensitively."""

    lowered = text.lower()
    return sorted({hint for hint in hints if hint.lower() in lowered})


def _resolve_device(torch_mod: Any, device_arg: str) -> Any:
    """Resolve a CLI device value without importing torch at module import time."""

    if device_arg == "auto":
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    return torch_mod.device(device_arg)


def _move_pipeline_if_possible(pipeline: Any, device: Any) -> None:
    """Move a loaded pipeline to device when it exposes .to()."""

    to = getattr(pipeline, "to", None)
    if callable(to):
        to(device)


def _config_hints(config: Any) -> Dict[str, Any]:
    """Extract shape/channel/range-like hints from a config object."""

    keys = (
        "in_channels",
        "out_channels",
        "latent_channels",
        "sample_size",
        "resolution",
        "c_in",
        "c_out",
        "block_out_channels",
        "clip_img_in_channels",
        "clip_txt_in_channels",
        "conditioning_dim",
        "image_size",
        "scaler",
    )
    hints: Dict[str, Any] = {}
    for key in keys:
        try:
            if isinstance(config, Mapping) and key in config:
                hints[key] = config[key]
            elif hasattr(config, key):
                hints[key] = getattr(config, key)
        except Exception:
            continue
    return hints


def _parameter_preview(module: Any, max_items: int = 4) -> List[Tuple[str, Tuple[int, ...], str]]:
    """Collect a bounded parameter shape preview."""

    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        return []

    rows: List[Tuple[str, Tuple[int, ...], str]] = []
    try:
        for name, param in named_parameters(recurse=False):
            rows.append(
                (
                    name,
                    tuple(int(x) for x in getattr(param, "shape", ())),
                    str(getattr(param, "dtype", "unknown")).replace("torch.", ""),
                )
            )
            if len(rows) >= max_items:
                break
    except Exception:
        return rows
    return rows


def _iter_pipeline_components(pipeline: Any) -> Iterable[Tuple[str, Any]]:
    """Yield public pipeline components with stable ordering."""

    components = getattr(pipeline, "components", None)
    if isinstance(components, Mapping):
        for name, value in components.items():
            yield str(name), value
        return

    for name in sorted(k for k in dir(pipeline) if not k.startswith("_")):
        try:
            value = getattr(pipeline, name)
        except Exception:
            continue
        if hasattr(value, "named_modules") or _matches(name, COMPONENT_HINTS):
            yield name, value


def _collect_component_records(label: str, pipeline: Any) -> List[ComponentRecord]:
    """Collect pipeline component summaries."""

    records: List[ComponentRecord] = []
    for name, value in _iter_pipeline_components(pipeline):
        records.append(
            ComponentRecord(
                pipeline_label=label,
                name=name,
                class_name=value.__class__.__name__,
                forward_signature=_format_signature(value),
                config_hints=_config_hints(getattr(value, "config", None)),
            )
        )
    return records


def _collect_module_records(
    component_name: str,
    component: Any,
    max_items: int,
    verbose: bool,
) -> List[ModuleRecord]:
    """Scan named_modules for latent/image/encoder path hints."""

    named_modules = getattr(component, "named_modules", None)
    if not callable(named_modules):
        return []

    records: List[ModuleRecord] = []
    for path, module in named_modules():
        if not path:
            continue
        cls_name = module.__class__.__name__
        forward_sig = _format_signature(module)
        text = " ".join((path, cls_name, forward_sig))
        matched = _matches(text, LATENT_PATH_HINTS)
        if not matched:
            continue
        records.append(
            ModuleRecord(
                component=component_name,
                path=path,
                class_name=cls_name,
                forward_signature=forward_sig,
                matched_hints=matched,
                param_shapes=_parameter_preview(module) if verbose else [],
            )
        )
        if len(records) >= max_items:
            break
    return records


def _source_hints_for_object(
    object_name: str,
    obj: Any,
    max_items: int,
    context_chars: int = 180,
) -> List[SourceHint]:
    """Collect bounded source-code lines matching Stable Cascade path hints."""

    hints: List[SourceHint] = []
    try:
        source_file = inspect.getsourcefile(obj) or "<unknown source file>"
        source_lines, start_line = inspect.getsourcelines(obj)
    except Exception:
        return hints

    for offset, line in enumerate(source_lines):
        if not _matches(line, SOURCE_HINTS):
            continue
        stripped = " ".join(line.strip().split())
        if len(stripped) > context_chars:
            stripped = stripped[: context_chars - 3] + "..."
        hints.append(
            SourceHint(
                object_name=object_name,
                source_file=source_file,
                line_no=start_line + offset,
                text=stripped,
            )
        )
        if len(hints) >= max_items:
            break
    return hints


def _load_pipeline(
    diffusers_mod: Any,
    label: str,
    class_name: str,
    model_id: str,
    local_files_only: bool,
) -> LoadResult:
    """Try loading one Stable Cascade pipeline and preserve failures."""

    result = LoadResult(label=label, class_name=class_name, model_id=model_id)
    try:
        pipeline_cls = getattr(diffusers_mod, class_name)
    except Exception:
        result.errors.append((f"{class_name} lookup", traceback.format_exc()))
        return result

    kwargs: Dict[str, Any] = {"local_files_only": local_files_only}
    try:
        result.pipeline = pipeline_cls.from_pretrained(model_id, **kwargs)
    except Exception:
        result.errors.append((f"{class_name}.from_pretrained({model_id})", traceback.format_exc()))
    return result


def _load_available_pipelines(args: argparse.Namespace, diffusers_mod: Any) -> List[LoadResult]:
    """Load prior, decoder, and combined pipelines when their classes exist."""

    results = [
        _load_pipeline(
            diffusers_mod=diffusers_mod,
            label="prior",
            class_name="StableCascadePriorPipeline",
            model_id=args.model_id_prior,
            local_files_only=args.local_files_only,
        ),
        _load_pipeline(
            diffusers_mod=diffusers_mod,
            label="decoder",
            class_name="StableCascadeDecoderPipeline",
            model_id=args.model_id_decoder,
            local_files_only=args.local_files_only,
        ),
    ]

    if hasattr(diffusers_mod, "StableCascadeCombinedPipeline"):
        results.append(
            _load_pipeline(
                diffusers_mod=diffusers_mod,
                label="combined",
                class_name="StableCascadeCombinedPipeline",
                model_id=args.model_id_decoder,
                local_files_only=args.local_files_only,
            )
        )
    else:
        missing = LoadResult(
            label="combined",
            class_name="StableCascadeCombinedPipeline",
            model_id=args.model_id_decoder,
        )
        missing.errors.append(("StableCascadeCombinedPipeline lookup", "Class is not exposed by this diffusers install."))
        results.append(missing)

    return results


def _print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _print_environment(args: argparse.Namespace, torch_mod: Optional[Any], diffusers_mod: Optional[Any]) -> None:
    """Print runtime and CLI diagnostics."""

    _print_header("Environment")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Prior model id: {args.model_id_prior}")
    print(f"Decoder model id: {args.model_id_decoder}")
    print(f"local_files_only: {args.local_files_only}")
    print(f"device: {args.device}")
    print(f"torch: {getattr(torch_mod, '__version__', '<not imported>') if torch_mod else '<not imported>'}")
    print(f"diffusers: {getattr(diffusers_mod, '__version__', '<not imported>') if diffusers_mod else '<not imported>'}")


def _print_class_level_inspection(diffusers_mod: Any, max_items: int) -> None:
    """Print class signatures and source hints available without model weights."""

    _print_header("Diffusers Class-Level Inspection")
    for class_name in PIPELINE_CLASS_NAMES:
        pipeline_cls = getattr(diffusers_mod, class_name, None)
        if pipeline_cls is None:
            print(f"- {class_name}: not available in this diffusers install")
            continue
        print(f"- {class_name}: {pipeline_cls}")
        print(f"  __call__ signature: {_format_signature(pipeline_cls, attr='__call__')}")
        try:
            print(f"  source: {inspect.getsourcefile(pipeline_cls) or '<unknown>'}")
        except Exception:
            print("  source: <unavailable>")

        hints = _source_hints_for_object(class_name, pipeline_cls, max_items=max(1, max_items // 3))
        for hint in hints:
            print(f"  source hint {hint.source_file}:{hint.line_no}: {hint.text}")


def _print_load_results(results: Sequence[LoadResult], summary_only: bool) -> None:
    """Print pipeline load success/failure state."""

    _print_header("Pipeline Loading")
    for result in results:
        status = "loaded" if result.pipeline is not None else "not loaded"
        print(f"- {result.label}: {result.class_name} from {result.model_id}: {status}")
        if result.errors and not summary_only:
            for stage, detail in result.errors:
                print(f"  failure at {stage}:")
                print("  " + detail.strip().replace("\n", "\n  "))


def _print_component_records(records: Sequence[ComponentRecord], max_items: int) -> None:
    """Print bounded component summaries."""

    _print_header("Pipeline Components")
    if not records:
        print("No pipeline components could be inspected. Class-level source hints are still useful.")
        return

    for idx, record in enumerate(records[:max_items], start=1):
        print(f"[{idx:03d}] {record.pipeline_label}.{record.name}: {record.class_name}")
        if record.forward_signature != "<no forward>":
            print(f"      {record.forward_signature}")
        if record.config_hints:
            print(f"      config hints: {record.config_hints}")
    if len(records) > max_items:
        print(f"... omitted {len(records) - max_items} component records")


def _print_module_records(records: Sequence[ModuleRecord], max_items: int, verbose: bool) -> None:
    """Print latent/image path module matches."""

    _print_header("Latent/Image Path Module Hints")
    if not records:
        print("No named_modules() records matched latent/image path hints in loaded components.")
        return

    for idx, record in enumerate(records[:max_items], start=1):
        print(
            f"[{idx:03d}] {record.component}.{record.path}: {record.class_name} "
            f"| hints={','.join(record.matched_hints)}"
        )
        if verbose:
            print(f"      {record.forward_signature}")
            if record.param_shapes:
                formatted = "; ".join(
                    f"{name}:{list(shape)}:{dtype}" for name, shape, dtype in record.param_shapes
                )
                print(f"      params: {formatted}")
    if len(records) > max_items:
        print(f"... omitted {len(records) - max_items} module records")


def _print_source_hints(hints: Sequence[SourceHint], max_items: int) -> None:
    """Print bounded source-level hints from loaded component classes."""

    _print_header("Loaded Component Source Hints")
    if not hints:
        print("No loaded component source hints were collected.")
        return

    for idx, hint in enumerate(hints[:max_items], start=1):
        print(f"[{idx:03d}] {hint.object_name} {hint.source_file}:{hint.line_no}")
        print(f"      {hint.text}")
    if len(hints) > max_items:
        print(f"... omitted {len(hints) - max_items} source hints")


def _print_path_interpretation() -> None:
    """Print the key semantic distinction being investigated."""

    _print_header("Path Interpretation")
    print("clip_img conditioning path:")
    print("- raw image -> feature_extractor -> image_encoder -> image embedding -> prior forward clip_img")
    print("- This path is conditioning. It helps guide Stage C, but it is not the noisy sample being denoised.")
    print("")
    print("Stage C clean latent target candidate path:")
    print("- raw image -> candidate Stable Cascade image compressor/encoder -> clean image_embeddings-like tensor")
    print("- The prior denoising input/output shape is expected to be latent-like, commonly [B, 16, 24, 24].")
    print("- The decoder pipeline's image_embeddings input is a strong hint for what the prior produces.")
    print("- The unresolved part is the exact offline training encoder used to turn raw RGB images into that clean target.")
    print("")
    print("Why these paths differ:")
    print("- clip_img comes from a CLIP image encoder style path and is passed as optional conditioning.")
    print("- Stage C clean target is the clean state before scheduler noise is added to the prior sample.")
    print("- Raw RGB resize into sample would bypass the Stable Cascade compression space and is likely semantically wrong.")


def _print_final_summary(
    component_records: Sequence[ComponentRecord],
    module_records: Sequence[ModuleRecord],
    source_hints: Sequence[SourceHint],
) -> None:
    """Print mandatory final human-readable summary."""

    _print_header("Final Summary")

    has_clip_path = any(
        record.name in ("feature_extractor", "image_encoder") or record.class_name.lower().find("clip") >= 0
        for record in component_records
    ) or any("clip_img" in hint.text or "image_encoder" in hint.text for hint in source_hints)

    target_related = [
        record
        for record in component_records
        if _matches(" ".join((record.name, record.class_name)), ("prior", "decoder", "effnet", "vqgan", "stage_a", "stage_b"))
    ]
    module_target_related = [
        record
        for record in module_records
        if any(h in record.matched_hints for h in ("image_embeddings", "effnet", "vqgan", "stage_a", "stage_b", "latent"))
    ]

    print("1. clip_img conditioning status:")
    print(f"- clip_img is treated as a conditioning path, not the Stage C clean target. Detected locally: {has_clip_path}.")
    print("- Expected path: raw image -> feature_extractor -> image_encoder -> image embedding -> clip_img.")
    print("")
    print("2. Stage C clean latent target candidate path:")
    print("- Candidate object name: image_embeddings / latent-like prior sample used by StableCascadeUNet.")
    print("- Candidate shape: [B, 16, 24, 24] unless model config/source inspection reports a different channel or sample_size.")
    print("- Candidate semantic path: raw image -> Stable Cascade image compression/encoder path -> clean image_embeddings.")
    print("- Strong components to inspect next:")
    if target_related:
        for record in target_related[:12]:
            print(f"  - {record.pipeline_label}.{record.name}: {record.class_name} config={record.config_hints}")
    else:
        print("  - No loaded prior/decoder/effnet/vqgan/stage_a/stage_b components were available in this run.")
    if module_target_related:
        print("- Module/source hints mentioning latent target candidates:")
        for record in module_target_related[:12]:
            print(f"  - {record.component}.{record.path}: {record.class_name} hints={record.matched_hints}")
    print("")
    print("3. Currently certain vs uncertain:")
    print("- Certain: clip_img is conditioning and should not be reused as the denoising target without proof.")
    print("- Certain: Stage C prior denoises a compressed latent/image_embeddings-like tensor, not raw RGB pixels.")
    print("- Uncertain: the exact raw image -> clean Stage C target encoder path unless source/model inspection identifies it.")
    print("- Uncertain: whether the relevant compressor is exposed in diffusers as effnet, Stage A/B, VQGAN, or only in training code.")
    print("")
    print("4. Next implementation step:")
    print("- Do not implement DPO loss, optimizer, full training loop, or latent-backed dataset yet.")
    print("- First write a latent precompute prototype only after confirming the raw image -> clean image_embeddings encoder path.")
    print("- If this script cannot resolve the encoder path, inspect decoder/prior source and upstream Stable Cascade training code next.")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""

    parser = argparse.ArgumentParser(
        description="Inspect Stable Cascade raw image to Stage C clean latent target path."
    )
    parser.add_argument("--model-id-prior", type=str, default=DEFAULT_PRIOR_MODEL_ID)
    parser.add_argument("--model-id-decoder", type=str, default=DEFAULT_DECODER_MODEL_ID)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--max-items", type=int, default=60)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> int:
    """Run the inspection script and print a human-readable report."""

    args = parse_args()
    max_items = max(1, int(args.max_items))

    torch_mod, torch_error = _safe_import("torch")
    diffusers_mod, diffusers_error = _safe_import("diffusers")
    _print_environment(args=args, torch_mod=torch_mod, diffusers_mod=diffusers_mod)

    if diffusers_mod is None:
        _print_header("Import Failure")
        print("[ERROR] Failed to import diffusers; cannot inspect Stable Cascade classes.")
        print(diffusers_error.strip() if diffusers_error else "No traceback captured.")
        print("")
        _print_path_interpretation()
        _print_final_summary(component_records=[], module_records=[], source_hints=[])
        return 1

    if torch_mod is None:
        _print_header("Torch Import Warning")
        print("[WARN] Failed to import torch. Pipeline loading may fail, but class/source inspection can continue.")
        print(torch_error.strip() if torch_error else "No traceback captured.")

    _print_class_level_inspection(diffusers_mod=diffusers_mod, max_items=max_items)

    results = _load_available_pipelines(args=args, diffusers_mod=diffusers_mod)
    if torch_mod is not None:
        try:
            device = _resolve_device(torch_mod, args.device)
            for result in results:
                if result.pipeline is not None:
                    _move_pipeline_if_possible(result.pipeline, device)
        except Exception:
            _print_header("Device Warning")
            print("[WARN] Failed to move loaded pipelines to requested device.")
            print(traceback.format_exc().strip())

    _print_load_results(results=results, summary_only=args.summary_only)

    component_records: List[ComponentRecord] = []
    module_records: List[ModuleRecord] = []
    source_hints: List[SourceHint] = []

    for result in results:
        if result.pipeline is None:
            continue
        component_records.extend(_collect_component_records(result.label, result.pipeline))
        source_hints.extend(
            _source_hints_for_object(
                object_name=f"{result.label}.{result.pipeline.__class__.__name__}",
                obj=result.pipeline.__class__,
                max_items=max(1, max_items // 2),
            )
        )
        for component_name, component in _iter_pipeline_components(result.pipeline):
            full_name = f"{result.label}.{component_name}"
            module_records.extend(
                _collect_module_records(
                    component_name=full_name,
                    component=component,
                    max_items=max_items,
                    verbose=args.verbose,
                )
            )
            if hasattr(component, "forward"):
                source_hints.extend(
                    _source_hints_for_object(
                        object_name=f"{full_name}.{component.__class__.__name__}",
                        obj=component.__class__,
                        max_items=max(1, max_items // 4),
                    )
                )

    _print_component_records(component_records, max_items=max_items)
    if not args.summary_only:
        _print_module_records(module_records, max_items=max_items, verbose=args.verbose)
        _print_source_hints(source_hints, max_items=max_items)
    _print_path_interpretation()
    _print_final_summary(
        component_records=component_records,
        module_records=module_records,
        source_hints=source_hints,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
