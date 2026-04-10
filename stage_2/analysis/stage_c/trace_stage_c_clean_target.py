from __future__ import annotations

"""Trace Stable Cascade Stage C clean target construction paths.

This script is for analysis only. It does not create a training loop, optimizer,
loss, or dataset migration. Its purpose is to answer a narrower question:

    raw image -> which encoder/compressor/path -> Stage C clean latent target?

It inspects two axes together:
1. Installed diffusers Stable Cascade / Wuerstchen classes and optional local caches.
2. Local source trees that may contain Stable Cascade or official upstream code.
"""

import argparse
import importlib
import inspect
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence


DEFAULT_PRIOR_MODEL_ID = "stabilityai/stable-cascade-prior"
DEFAULT_DECODER_MODEL_ID = "stabilityai/stable-cascade"
DEFAULT_MAX_ITEMS = 40

KEYWORDS = (
    "image_embeddings",
    "effnet",
    "vqgan",
    "stage_a",
    "stage_b",
    "encode",
    "latent",
    "compress",
    "targets",
    "target",
    "noise",
    "scheduler",
    "sample",
    "clip_img",
)

FILE_PATTERNS = (
    "*stable*cascade*.py",
    "*wuerstchen*.py",
    "*paella*.py",
    "*effnet*.py",
    "*vq*.py",
    "*train*.py",
    "*lora*.py",
    "*controlnet*.py",
    "*finetune*.py",
)

PATH_TOKENS = (
    "stable_cascade",
    "stable-cascade",
    "wuerstchen",
    "paella",
    "effnet",
    "efficient_net",
    "vqgan",
    "stage_a",
    "stage_b",
    "stage_c",
)

DIFFUSERS_CLASS_NAMES = (
    "StableCascadePriorPipeline",
    "StableCascadeDecoderPipeline",
    "StableCascadeCombinedPipeline",
    "StableCascadeUNet",
)

RELEVANT_ATTRS = (
    "prior",
    "decoder",
    "vqgan",
    "scheduler",
    "image_encoder",
    "feature_extractor",
    "tokenizer",
    "text_encoder",
    "prior_prior",
    "prior_scheduler",
    "prior_image_encoder",
)


@dataclass
class ImportResult:
    """Result for optional module import."""

    module: Optional[Any]
    error: Optional[str]


@dataclass
class ClassRecord:
    """Summary of one inspected class."""

    name: str
    qualified_name: str
    source_file: str
    init_signature: str
    call_signature: str
    forward_signature: str
    source_hints: List[str] = field(default_factory=list)


@dataclass
class PipelineLoadRecord:
    """Summary of one optional pipeline loading attempt."""

    label: str
    class_name: str
    model_id: str
    loaded: bool
    error: Optional[str] = None
    component_lines: List[str] = field(default_factory=list)


@dataclass
class SearchHit:
    """One bounded source hit."""

    path: str
    line_no: int
    keyword: str
    line: str
    snippet: Optional[str] = None


@dataclass
class PathCategory:
    """Grouped hits for one semantic path."""

    title: str
    description: str
    hits: List[SearchHit] = field(default_factory=list)


def _safe_import(module_name: str) -> ImportResult:
    """Import a module and preserve traceback on failure."""

    try:
        return ImportResult(importlib.import_module(module_name), None)
    except Exception:
        return ImportResult(None, traceback.format_exc())


def _format_signature(obj: Any, attr: str) -> str:
    """Return a compact signature for an attribute when possible."""

    target = getattr(obj, attr, None)
    if target is None:
        return "<unavailable>"
    try:
        return f"{attr}{inspect.signature(target)}"
    except Exception:
        return "<signature unavailable>"


def _truncate(text: str, limit: int = 220) -> str:
    """Normalize whitespace and truncate long lines."""

    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _iter_text_files(root: Path) -> Iterator[Path]:
    """Yield candidate Python/text files under one root."""

    if not root.exists():
        return
    if root.is_file():
        if any(token in str(root).lower() for token in PATH_TOKENS):
            yield root
        return
    seen: set[Path] = set()
    for pattern in FILE_PATTERNS:
        for path in root.rglob(pattern):
            if path in seen or not path.is_file():
                continue
            if not any(token in str(path).lower() for token in PATH_TOKENS):
                continue
            seen.add(path)
            yield path


def _source_hint_lines(obj: Any, max_items: int) -> List[str]:
    """Collect bounded in-source hint lines from an object."""

    hints: List[str] = []
    try:
        lines, start = inspect.getsourcelines(obj)
        source_file = inspect.getsourcefile(obj) or "<unknown>"
    except Exception:
        return hints

    pattern = re.compile("|".join(re.escape(word) for word in KEYWORDS), re.IGNORECASE)
    for offset, line in enumerate(lines):
        if not pattern.search(line):
            continue
        hints.append(f"{source_file}:{start + offset}: {_truncate(line)}")
        if len(hints) >= max_items:
            break
    return hints


def _collect_diffusers_classes(diffusers_mod: Any, max_items: int) -> List[ClassRecord]:
    """Inspect Stable Cascade related classes without loading model weights."""

    records: List[ClassRecord] = []
    for name in DIFFUSERS_CLASS_NAMES:
        obj = getattr(diffusers_mod, name, None)
        if obj is None:
            continue
        try:
            source_file = inspect.getsourcefile(obj) or "<unknown>"
        except Exception:
            source_file = "<unknown>"
        records.append(
            ClassRecord(
                name=name,
                qualified_name=f"{obj.__module__}.{obj.__qualname__}",
                source_file=source_file,
                init_signature=_format_signature(obj, "__init__"),
                call_signature=_format_signature(obj, "__call__"),
                forward_signature=_format_signature(obj, "forward"),
                source_hints=_source_hint_lines(obj, max(2, max_items // 4)),
            )
        )
    return records


def _resolve_device(torch_mod: Any, device_arg: str) -> Any:
    """Resolve CLI device lazily."""

    if device_arg == "auto":
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    return torch_mod.device(device_arg)


def _optional_pipeline_loads(
    args: argparse.Namespace,
    diffusers_mod: Any,
    torch_mod: Any,
    max_items: int,
) -> List[PipelineLoadRecord]:
    """Try to load prior/decoder/combined pipelines from local cache only when possible."""

    load_specs = (
        ("prior", "StableCascadePriorPipeline", args.model_id_prior),
        ("decoder", "StableCascadeDecoderPipeline", args.model_id_decoder),
        ("combined", "StableCascadeCombinedPipeline", args.model_id_decoder),
    )
    records: List[PipelineLoadRecord] = []
    device = _resolve_device(torch_mod, args.device)

    for label, class_name, model_id in load_specs:
        pipeline_cls = getattr(diffusers_mod, class_name, None)
        if pipeline_cls is None:
            records.append(
                PipelineLoadRecord(
                    label=label,
                    class_name=class_name,
                    model_id=model_id,
                    loaded=False,
                    error="class not exposed by installed diffusers",
                )
            )
            continue

        kwargs: dict[str, Any] = {"local_files_only": args.local_files_only}
        if args.local_files_only:
            kwargs["local_files_only"] = True

        try:
            pipe = pipeline_cls.from_pretrained(model_id, **kwargs)
            to = getattr(pipe, "to", None)
            if callable(to):
                to(device)

            component_lines: List[str] = []
            for attr in RELEVANT_ATTRS:
                if not hasattr(pipe, attr):
                    continue
                value = getattr(pipe, attr)
                if value is None:
                    continue
                line = f"{label}.{attr}: {value.__class__.__name__}"
                config = getattr(value, "config", None)
                if config is not None:
                    shape_hints = []
                    for key in ("in_channels", "out_channels", "latent_channels", "sample_size", "effnet_in_channels"):
                        if isinstance(config, Mapping) and key in config:
                            shape_hints.append(f"{key}={config[key]}")
                        elif hasattr(config, key):
                            shape_hints.append(f"{key}={getattr(config, key)}")
                    if shape_hints:
                        line += " [" + ", ".join(shape_hints) + "]"
                component_lines.append(line)
                if len(component_lines) >= max_items:
                    break

            records.append(
                PipelineLoadRecord(
                    label=label,
                    class_name=class_name,
                    model_id=model_id,
                    loaded=True,
                    component_lines=component_lines,
                )
            )
        except Exception:
            records.append(
                PipelineLoadRecord(
                    label=label,
                    class_name=class_name,
                    model_id=model_id,
                    loaded=False,
                    error=traceback.format_exc(),
                )
            )
    return records


def _default_search_roots() -> List[Path]:
    """Build conservative default roots from the current machine layout."""

    roots = [
        Path.cwd(),
        Path.cwd().parent / "diffusers",
        Path.home() / "diffusers",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    return roots


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    """Deduplicate while preserving order."""

    resolved: List[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path.expanduser())
    return resolved


def _collect_search_roots(args: argparse.Namespace, diffusers_mod: Optional[Any]) -> List[Path]:
    """Resolve local roots used for source discovery."""

    roots: List[Path] = []
    if diffusers_mod is not None:
        try:
            roots.append(Path(diffusers_mod.__file__).resolve().parent)
        except Exception:
            pass
    roots.extend(_default_search_roots())
    if args.search_root:
        roots.extend(Path(item).expanduser() for item in args.search_root)
    return _dedupe_paths(roots)


def _search_one_file(path: Path, keywords: Sequence[str], snippet_radius: int) -> List[SearchHit]:
    """Search a file for keyword hits with bounded context."""

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
    except Exception:
        return []

    lines = raw.splitlines()
    hits: List[SearchHit] = []
    lowered_keywords = tuple(word.lower() for word in keywords)

    for index, line in enumerate(lines):
        lowered = line.lower()
        matched = [word for word in lowered_keywords if word in lowered]
        if not matched:
            continue
        keyword = matched[0]
        snippet = None
        if snippet_radius > 0:
            start = max(0, index - snippet_radius)
            end = min(len(lines), index + snippet_radius + 1)
            rendered = []
            for line_no in range(start, end):
                marker = ">" if line_no == index else " "
                rendered.append(f"{marker} {line_no + 1}: {lines[line_no]}")
            snippet = "\n".join(rendered)

        hits.append(
            SearchHit(
                path=str(path),
                line_no=index + 1,
                keyword=keyword,
                line=_truncate(line),
                snippet=snippet,
            )
        )
    return hits


def _search_local_sources(
    roots: Sequence[Path],
    keywords: Sequence[str],
    max_items: int,
    show_source_snippets: bool,
) -> List[SearchHit]:
    """Search local code trees for relevant keyword hits."""

    all_hits: List[SearchHit] = []
    for root in roots:
        for path in _iter_text_files(root):
            all_hits.extend(
                _search_one_file(
                    path=path,
                    keywords=keywords,
                    snippet_radius=2 if show_source_snippets else 0,
                )
            )

    def sort_key(hit: SearchHit) -> tuple[int, str, int]:
        priority = 0
        path_lower = hit.path.lower()
        if "train_text_to_image_prior" in path_lower:
            priority = -4
        elif "stable_cascade" in path_lower or "wuerstchen" in path_lower:
            priority = -3
        elif "efficient_net" in path_lower or "effnet" in path_lower:
            priority = -2
        return (priority, hit.path, hit.line_no)

    all_hits.sort(key=sort_key)
    return all_hits[:max_items]


def _classify_hits(hits: Sequence[SearchHit]) -> List[PathCategory]:
    """Group hits into the semantic buckets requested by the user."""

    categories = [
        PathCategory(
            title="clip_img conditioning path",
            description="Raw image -> feature_extractor/image_encoder -> CLIP image embedding -> prior forward clip_img",
        ),
        PathCategory(
            title="Stage C clean latent target candidate path",
            description="Raw image -> encoder/compressor candidate -> clean image_embeddings-like tensor used before noise",
        ),
        PathCategory(
            title="decoder input image_embeddings path",
            description="What the decoder consumes as image_embeddings and how that relates to prior output",
        ),
        PathCategory(
            title="training noisy sample generation path",
            description="Where clean target, noise, scheduler.add_noise, and noisy sample are defined",
        ),
    ]

    for hit in hits:
        text = f"{hit.path.lower()} {hit.line.lower()}"

        if any(token in text for token in ("clip_img", "feature_extractor", "image_encoder(image)", "encode_image(")):
            categories[0].hits.append(hit)
        if any(
            token in text
            for token in ("image_embeds = image_encoder", "image_embeddings", "effnet", "efficientnet", "compress", "latent")
        ):
            categories[1].hits.append(hit)
        if any(token in text for token in ("decoder", "image_embeddings=image_embeddings", "image_embeddings=image_embeds", "vqgan")):
            categories[2].hits.append(hit)
        if any(token in text for token in ("noise =", "add_noise(", "noisy_latents", "timesteps", "scheduler.step(", "sample=torch.cat", "pred_noise")):
            categories[3].hits.append(hit)

    return categories


def _print_header(title: str) -> None:
    """Render a section header."""

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _print_environment(args: argparse.Namespace, torch_info: ImportResult, diffusers_info: ImportResult) -> None:
    """Print execution environment diagnostics."""

    _print_header("Environment")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"device: {args.device}")
    print(f"local_files_only: {args.local_files_only}")
    print(f"search_root: {[str(Path(item).expanduser()) for item in args.search_root] if args.search_root else 'default auto roots'}")
    print(f"torch: {getattr(torch_info.module, '__version__', '<not imported>') if torch_info.module else '<import failed>'}")
    print(f"diffusers: {getattr(diffusers_info.module, '__version__', '<not imported>') if diffusers_info.module else '<import failed>'}")
    if torch_info.error and not args.summary_only:
        print("\n[torch import error]")
        print(torch_info.error.strip())
    if diffusers_info.error:
        print("\n[diffusers import error]")
        print(diffusers_info.error.strip())


def _print_class_records(records: Sequence[ClassRecord], max_items: int, verbose: bool) -> None:
    """Print diffusers class inspection."""

    _print_header("Diffusers Classes")
    if not records:
        print("No Stable Cascade classes were found in installed diffusers.")
        return

    for record in records[:max_items]:
        print(f"- {record.name}: {record.qualified_name}")
        print(f"  source: {record.source_file}")
        print(f"  {record.init_signature}")
        print(f"  {record.call_signature}")
        if record.forward_signature != "<unavailable>":
            print(f"  {record.forward_signature}")
        if verbose and record.source_hints:
            for hint in record.source_hints:
                print(f"  hint: {hint}")


def _print_pipeline_loads(records: Sequence[PipelineLoadRecord], summary_only: bool) -> None:
    """Print optional local pipeline loading results."""

    _print_header("Optional Pipeline Loading")
    for record in records:
        status = "loaded" if record.loaded else "not loaded"
        print(f"- {record.label}: {record.class_name} from {record.model_id}: {status}")
        if record.component_lines:
            for line in record.component_lines:
                print(f"  {line}")
        if record.error and not summary_only:
            print("  error:")
            print("  " + record.error.strip().replace("\n", "\n  "))


def _print_roots(roots: Sequence[Path]) -> None:
    """Print local search roots."""

    _print_header("Local Search Roots")
    for root in roots:
        exists = "exists" if root.exists() else "missing"
        print(f"- {root} [{exists}]")


def _print_categories(
    categories: Sequence[PathCategory],
    max_items: int,
    show_source_snippets: bool,
) -> None:
    """Print grouped search results."""

    for category in categories:
        _print_header(category.title)
        print(category.description)
        if not category.hits:
            print("No local source hits matched this category.")
            continue
        for hit in category.hits[:max_items]:
            print(f"- {hit.path}:{hit.line_no} [{hit.keyword}] {hit.line}")
            if show_source_snippets and hit.snippet:
                print(hit.snippet)
        if len(category.hits) > max_items:
            print(f"... omitted {len(category.hits) - max_items} additional hits")


def _collect_summary_lines(
    categories: Sequence[PathCategory],
    hits: Sequence[SearchHit],
    load_records: Sequence[PipelineLoadRecord],
) -> List[str]:
    """Build the mandatory final summary checklist."""

    hit_text = "\n".join(f"{hit.path}:{hit.line_no}: {hit.line}" for hit in hits)
    has_clip = "clip_img" in hit_text or "encode_image(" in hit_text
    has_effnet = "effnet" in hit_text or "efficientnet" in hit_text
    has_add_noise = "add_noise(" in hit_text
    has_decoder_image_embeddings = "image_embeddings" in hit_text and "decoder" in hit_text
    loaded_labels = [record.label for record in load_records if record.loaded]

    lines = [
        "1. 현재 확실한 것",
        f"- clip_img conditioning 경로는 CLIP image encoder 기반이며 Stage C clean target과 분리되어 있다: {has_clip}.",
        f"- decoder는 image_embeddings를 입력으로 받으며 prior 출력과 연결된다: {has_decoder_image_embeddings}.",
        f"- 로컬 training 후보 코드에서 effnet/EfficientNet 기반 image encoder 경로가 확인된다: {has_effnet}.",
        f"- noisy sample은 scheduler.add_noise(...) 류 경로로 만들어질 가능성이 높다: {has_add_noise}.",
        f"- 이번 실행에서 로컬 캐시로 로드된 파이프라인: {loaded_labels if loaded_labels else '없음'}.",
        "",
        "2. 아직 불확실한 것",
        "- Stable Cascade 공식 학습이 raw RGB를 어떤 정규화/스케일 상수까지 포함해 clean target으로 만드는지 완전히 확정되었는지.",
        "- Stable Cascade가 Wuerstchen prior training 경로와 1:1로 동일한 encoder/scaling을 쓰는지.",
        "- Stage C clean latent target이 diffusers 추론 파이프라인 안에 직접 노출되는지, 아니면 training 코드에서만 구성되는지.",
        "",
        "3. clean latent target 생성 후보 경로",
        "- 후보 A: raw image -> EffNet/EfficientNet encoder -> image_embeds -> optional scaling -> clean target.",
        "- 후보 B: decoder가 받는 image_embeddings와 prior가 생성하는 latent-like tensor를 동일 공간으로 간주.",
        "- 후보 C: Stable Cascade 원본 코드에 Stage A/B, VQGAN, Paella, effnet 관련 별도 compressor 경로가 숨겨져 있는지 추가 확인.",
        "",
        "4. noisy sample 생성에 필요한 다음 정보",
        "- clean target tensor의 정확한 shape, dtype, scaling.",
        "- timestep sampling 방식이 uniform ratio인지, discrete scheduler timestep인지.",
        "- target이 noise prediction인지, v-prediction 변형인지.",
        "- scheduler.add_noise 입력이 image_embeddings인지 다른 latent인지.",
        "",
        "5. 다음 구현 단계 추천",
        "- `prepare_stage_c_latents.py`: 단일 이미지 기준 encoder path와 scaling을 검증하는 프로토타입 작성.",
        "- `stage2_dataset.py` latent 확장: 위 프로토타입으로 clean target 경로가 확정된 뒤에만 추가.",
        "- `train_stage2_dpo.py` minimal step: clean target/noisy sample 정의가 확정된 뒤 최소 학습 step만 연결.",
    ]

    category_map = {category.title: category for category in categories}
    clean_hits = category_map["Stage C clean latent target candidate path"].hits
    noisy_hits = category_map["training noisy sample generation path"].hits
    if clean_hits:
        lines.insert(13, f"- 대표 clean target 후보 근거: {clean_hits[0].path}:{clean_hits[0].line_no}")
    if noisy_hits:
        lines.insert(19, f"- 대표 noisy sample 후보 근거: {noisy_hits[0].path}:{noisy_hits[0].line_no}")
    return lines


def _print_summary(summary_lines: Sequence[str]) -> None:
    """Print final checklist-style summary."""

    _print_header("Final Summary")
    for line in summary_lines:
        print(line)


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""

    parser = argparse.ArgumentParser(
        description="Trace Stable Cascade raw image -> Stage C clean target construction paths."
    )
    parser.add_argument("--model-id-prior", type=str, default=DEFAULT_PRIOR_MODEL_ID)
    parser.add_argument("--model-id-decoder", type=str, default=DEFAULT_DECODER_MODEL_ID)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    parser.add_argument("--search-root", action="append", default=[], help="Additional local root to search.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument("--show-source-snippets", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> int:
    """Run the analysis report."""

    args = parse_args()
    args.max_items = max(1, int(args.max_items))

    torch_info = _safe_import("torch")
    diffusers_info = _safe_import("diffusers")

    _print_environment(args, torch_info, diffusers_info)

    if diffusers_info.module is None:
        summary_lines = [
            "1. 현재 확실한 것",
            "- diffusers import 실패로 설치된 Stable Cascade 코드는 검사하지 못했다.",
            "",
            "2. 아직 불확실한 것",
            "- Stable Cascade / Wuerstchen class 및 source path 전반.",
            "",
            "3. clean latent target 생성 후보 경로",
            "- 로컬 source 검색만으로 추적 필요.",
            "",
            "4. noisy sample 생성에 필요한 다음 정보",
            "- scheduler.add_noise 및 target construction이 있는 training code 위치.",
            "",
            "5. 다음 구현 단계 추천",
            "- 먼저 diffusers 환경 import 문제를 해결한 뒤 다시 실행.",
        ]
        _print_summary(summary_lines)
        return 1

    class_records = _collect_diffusers_classes(diffusers_info.module, args.max_items)
    roots = _collect_search_roots(args, diffusers_info.module)
    hits = _search_local_sources(
        roots=roots,
        keywords=KEYWORDS,
        max_items=max(args.max_items * 4, 80),
        show_source_snippets=args.show_source_snippets,
    )
    categories = _classify_hits(hits)

    load_records: List[PipelineLoadRecord] = []
    if torch_info.module is not None:
        load_records = _optional_pipeline_loads(
            args=args,
            diffusers_mod=diffusers_info.module,
            torch_mod=torch_info.module,
            max_items=args.max_items,
        )

    if not args.summary_only:
        _print_class_records(class_records, args.max_items, args.verbose)
        _print_pipeline_loads(load_records, args.summary_only)
        _print_roots(roots)
        _print_categories(categories, args.max_items, args.show_source_snippets)

    summary_lines = _collect_summary_lines(categories, hits, load_records)
    _print_summary(summary_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
