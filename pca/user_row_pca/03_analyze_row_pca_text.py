#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Pattern, Sequence, Tuple


INSTALL_COMMAND = (
    "conda activate ppd_pca && "
    "conda install -c conda-forge -y numpy pandas pyarrow tqdm"
)
REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "tqdm": "tqdm",
}


def _require_dependencies() -> None:
    missing = [
        package_name
        for module_name, package_name in REQUIRED_IMPORTS.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        raise SystemExit(
            "Missing required package(s) in the active environment: "
            f"{', '.join(missing)}\nInstall with:\n  {INSTALL_COMMAND}"
        )


_require_dependencies()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import tqdm as tqdm_module  # noqa: E402
from tqdm import tqdm  # noqa: E402


KEYWORD_CATEGORIES = {
    "style": [
        "anime",
        "cartoon",
        "realistic",
        "photorealistic",
        "naturalistic",
        "cinematic",
        "illustration",
        "stylized",
        "fantasy",
        "futuristic",
        "industrial",
        "surreal",
        "abstract",
        "minimalist",
        "vintage",
        "retro",
    ],
    "color_lighting": [
        "warm",
        "cool",
        "colorful",
        "vibrant",
        "muted",
        "dramatic",
        "soft",
        "bright",
        "dark",
        "moody",
        "intense",
        "diffuse",
        "high contrast",
        "low contrast",
        "spotlight",
    ],
    "composition": [
        "composition",
        "central",
        "centered",
        "foreground",
        "background",
        "close-up",
        "perspective",
        "depth",
        "scale",
        "large",
        "small",
        "minimal",
        "complex",
        "simple",
        "dynamic",
        "wide",
        "close up",
        "close-up",
        "angle",
        "viewpoint",
        "focus",
        "focused",
    ],
    "detail_sharpness": [
        "detailed",
        "intricate",
        "texture",
        "sharp",
        "soft",
        "blurry",
        "focused",
        "fine detail",
        "textures",
        "crisp",
        "smooth",
        "clear",
        "clarity",
    ],
    "aesthetic": [
        "aesthetic",
        "beautiful",
        "atmospheric",
        "engaging",
        "visually appealing",
        "appealing",
        "immersive",
    ],
}
DEFAULT_KEYWORDS = sorted({keyword for keywords in KEYWORD_CATEGORIES.values() for keyword in keywords})
VALID_SECTIONS = ("full_text", "user_profile", "preferred", "dispreferred", "differences")
DEFAULT_SECTIONS = list(VALID_SECTIONS)
DEFAULT_STAT_FRACS = [0.01, 0.05, 0.07, 0.10]
DEFAULT_FOCUS_PCS = [4, 5, 8]

CAPTION_COLUMNS = [f"caption_{i}" for i in range(4)]
PREFERRED_UID_COLUMNS = [f"preferred_image_uid_{i}" for i in range(4)]
DISPREFERRED_UID_COLUMNS = [f"dispreferred_image_uid_{i}" for i in range(4)]
REQUIRED_META_COLUMNS = [
    "global_row_id",
    "split",
    "shard_id",
    "row_in_shard",
    "user_id",
    "text",
    *CAPTION_COLUMNS,
    *PREFERRED_UID_COLUMNS,
    *DISPREFERRED_UID_COLUMNS,
    "mean_vec_norm",
    "user_row_count_in_output",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect row-level PCA axes using high/low LLaVA profile text examples."
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/roycecho/PPD/pca_data/user_row_pca"),
    )
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--num-pcs", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--stat-fracs",
        type=float,
        nargs="+",
        default=DEFAULT_STAT_FRACS,
        help="Fractions for top/bottom PCA group statistics.",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=VALID_SECTIONS,
        default=DEFAULT_SECTIONS,
        help="Text sections used for fractional keyword/category analysis.",
    )
    parser.add_argument(
        "--focus-pcs",
        type=int,
        nargs="+",
        default=DEFAULT_FOCUS_PCS,
        help="PC numbers to include in the compact focus summary.",
    )
    parser.add_argument(
        "--text-max-chars",
        type=int,
        default=0,
        help="Maximum characters to write per text block. Use 0 for full text.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_pcs <= 0:
        raise ValueError(f"--num-pcs must be > 0, got {args.num_pcs}")
    if args.top_k <= 0:
        raise ValueError(f"--top-k must be > 0, got {args.top_k}")
    if not args.stat_fracs:
        raise ValueError("--stat-fracs must contain at least one fraction.")
    for frac in args.stat_fracs:
        if frac <= 0.0 or frac >= 0.5:
            raise ValueError(f"--stat-fracs values must be in (0, 0.5), got {frac}")
    args.stat_fracs = sorted(set(float(frac) for frac in args.stat_fracs))
    args.sections = list(dict.fromkeys(args.sections))
    if not args.sections:
        raise ValueError("--sections must contain at least one section.")
    args.focus_pcs = sorted(set(int(pc) for pc in args.focus_pcs))
    for pc in args.focus_pcs:
        if pc <= 0:
            raise ValueError(f"--focus-pcs values must be positive PC numbers, got {pc}")
        if pc > args.num_pcs:
            raise ValueError(f"--focus-pcs cannot exceed --num-pcs: focus_pc={pc}, num_pcs={args.num_pcs}")
    if args.text_max_chars < 0:
        raise ValueError(f"--text-max-chars must be >= 0, got {args.text_max_chars}")
    if not args.output_name.strip():
        raise ValueError("--output-name must be non-empty.")


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _output_dir(output_root: Path, output_name: str) -> Path:
    return output_root / "text_inspection" / output_name


def _pc_side_md_path(output_dir: Path, pc_num: int, side: str) -> Path:
    return output_dir / f"pc{pc_num:02d}_{side}_texts.md"


def _summary_path(output_dir: Path) -> Path:
    return output_dir / "pc_top_bottom_summary.csv"


def _keyword_frequency_path(output_dir: Path) -> Path:
    return output_dir / "keyword_frequency_by_pc.csv"


def _keyword_delta_path(output_dir: Path) -> Path:
    return output_dir / "keyword_frequency_delta_by_pc.csv"


def _pc1_quantile_path(output_dir: Path) -> Path:
    return output_dir / "pc01_quantile_summary.csv"


def _section_parse_summary_path(output_dir: Path) -> Path:
    return output_dir / "section_parse_summary.csv"


def _pc_group_stats_path(output_dir: Path) -> Path:
    return output_dir / "pc_group_stats.csv"


def _keyword_group_section_path(output_dir: Path) -> Path:
    return output_dir / "keyword_frequency_by_pc_group_section.csv"


def _category_group_section_path(output_dir: Path) -> Path:
    return output_dir / "category_frequency_by_pc_group_section.csv"


def _keyword_group_section_delta_path(output_dir: Path) -> Path:
    return output_dir / "keyword_high_low_delta_by_pc_group_section.csv"


def _category_group_section_delta_path(output_dir: Path) -> Path:
    return output_dir / "category_high_low_delta_by_pc_group_section.csv"


def _preferred_dispreferred_contrast_path(output_dir: Path) -> Path:
    return output_dir / "preferred_dispreferred_contrast_by_pc_group.csv"


def _preferred_dispreferred_contrast_delta_path(output_dir: Path) -> Path:
    return output_dir / "preferred_dispreferred_contrast_delta_by_pc_group.csv"


def _focus_pc_summary_path(output_dir: Path) -> Path:
    return output_dir / "focus_pc_summary.csv"


def _config_path(output_dir: Path) -> Path:
    return output_dir / "analyze_row_pca_text_config.json"


def _check_no_existing(paths: Iterable[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        preview = "\n".join(existing[:30])
        suffix = "" if len(existing) <= 30 else f"\n... and {len(existing) - 30} more"
        raise FileExistsError(f"Output file(s) already exist. Pass --overwrite:\n{preview}{suffix}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _package_versions() -> Dict[str, str]:
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "tqdm": tqdm_module.__version__,
    }


def _load_inputs(scores_path: Path, meta_path: Path, num_pcs: int, top_k: int) -> Tuple[np.ndarray, pd.DataFrame]:
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata parquet not found: {meta_path}")

    try:
        scores = np.load(scores_path, mmap_mode="r")
    except Exception as exc:
        raise ValueError(f"Failed to load score file: path={scores_path}, error={exc}") from exc
    if scores.ndim != 2:
        raise ValueError(f"Scores must be a 2D array [rows, pcs], got shape={tuple(scores.shape)}")
    if scores.shape[1] < num_pcs:
        raise ValueError(
            f"--num-pcs exceeds score width: num_pcs={num_pcs}, score_shape={tuple(scores.shape)}"
        )
    if scores.shape[0] < top_k:
        raise ValueError(f"--top-k exceeds row count: top_k={top_k}, rows={scores.shape[0]}")

    meta = pd.read_parquet(meta_path)
    missing = sorted(set(REQUIRED_META_COLUMNS) - set(meta.columns))
    if missing:
        raise ValueError(f"Metadata parquet is missing required column(s): path={meta_path}, missing={missing}")
    if len(meta) != scores.shape[0]:
        raise ValueError(
            "Scores/meta row count mismatch: "
            f"scores_rows={scores.shape[0]}, meta_rows={len(meta)}, scores={scores_path}, meta={meta_path}"
        )

    score_subset = np.asarray(scores[:, :num_pcs])
    if not np.isfinite(score_subset).all():
        raise ValueError(f"Scores contain NaN or inf values in first {num_pcs} PC(s): {scores_path}")
    return scores, meta


def _value_to_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _row_text(row: pd.Series) -> str:
    return _value_to_text(row["text"])


def _text_char_len(text: str) -> int:
    return len(text)


def _text_word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _nonempty_unique_count(values: Sequence[Any]) -> int:
    normalized = []
    for value in values:
        text = _value_to_text(value).strip()
        if text:
            normalized.append(text)
    return len(set(normalized))


def _truncate_text(text: str, text_max_chars: int) -> str:
    if text_max_chars <= 0 or len(text) <= text_max_chars:
        return text
    return (
        text[:text_max_chars]
        + f"\n\n[TRUNCATED at {text_max_chars} chars; original length={len(text)} chars]"
    )


def _compile_keyword_patterns(keywords: Sequence[str]) -> Dict[str, Pattern[str]]:
    patterns: Dict[str, Pattern[str]] = {}
    for keyword in keywords:
        if re.fullmatch(r"[A-Za-z0-9_]+", keyword):
            pattern = rf"\b{re.escape(keyword)}\b"
        else:
            pattern = rf"(?<![A-Za-z0-9]){re.escape(keyword)}(?![A-Za-z0-9])"
        patterns[keyword] = re.compile(pattern, flags=re.IGNORECASE)
    return patterns


def _keyword_category_pairs() -> List[Tuple[str, str]]:
    return [
        (category, keyword)
        for category, keywords in KEYWORD_CATEGORIES.items()
        for keyword in keywords
    ]


def _duplicated_keywords() -> Dict[str, List[str]]:
    keyword_to_categories: Dict[str, List[str]] = {}
    for category, keyword in _keyword_category_pairs():
        keyword_to_categories.setdefault(keyword, []).append(category)
    return {
        keyword: categories
        for keyword, categories in keyword_to_categories.items()
        if len(categories) > 1
    }


def _keyword_counts(texts: Sequence[str], patterns: Mapping[str, Pattern[str]]) -> List[Dict[str, Any]]:
    total_examples = len(texts)
    records: List[Dict[str, Any]] = []
    for keyword, pattern in patterns.items():
        example_count = 0
        occurrence_count = 0
        for text in texts:
            matches = pattern.findall(text)
            if matches:
                example_count += 1
                occurrence_count += len(matches)
        records.append(
            {
                "keyword": keyword,
                "total_examples": int(total_examples),
                "example_count": int(example_count),
                "occurrence_count": int(occurrence_count),
                "example_rate": float(example_count / total_examples) if total_examples else 0.0,
                "occurrences_per_example": float(occurrence_count / total_examples)
                if total_examples
                else 0.0,
            }
        )
    return records


def _keyword_counts_for_section(
    texts: Sequence[str],
    patterns: Mapping[str, Pattern[str]],
) -> List[Dict[str, Any]]:
    group_size = len(texts)
    nonempty_texts = [text for text in texts if text.strip()]
    section_nonempty_rows = len(nonempty_texts)
    records: List[Dict[str, Any]] = []
    for category, keyword in _keyword_category_pairs():
        pattern = patterns[keyword]
        example_count = 0
        occurrence_count = 0
        for text in nonempty_texts:
            matches = pattern.findall(text)
            if matches:
                example_count += 1
                occurrence_count += len(matches)
        records.append(
            {
                "category": category,
                "keyword": keyword,
                "group_size": int(group_size),
                "section_nonempty_rows": int(section_nonempty_rows),
                "example_count": int(example_count),
                "occurrence_count": int(occurrence_count),
                "example_rate": float(example_count / section_nonempty_rows)
                if section_nonempty_rows
                else 0.0,
                "occurrences_per_nonempty_example": float(occurrence_count / section_nonempty_rows)
                if section_nonempty_rows
                else 0.0,
            }
        )
    return records


def _category_counts_for_section(
    texts: Sequence[str],
    patterns: Mapping[str, Pattern[str]],
) -> List[Dict[str, Any]]:
    group_size = len(texts)
    nonempty_texts = [text for text in texts if text.strip()]
    section_nonempty_rows = len(nonempty_texts)
    records: List[Dict[str, Any]] = []
    for category, keywords in KEYWORD_CATEGORIES.items():
        example_count = 0
        occurrence_count = 0
        for text in nonempty_texts:
            category_occurrences = 0
            for keyword in keywords:
                category_occurrences += len(patterns[keyword].findall(text))
            if category_occurrences:
                example_count += 1
                occurrence_count += category_occurrences
        records.append(
            {
                "category": category,
                "group_size": int(group_size),
                "section_nonempty_rows": int(section_nonempty_rows),
                "category_example_count": int(example_count),
                "category_occurrence_count": int(occurrence_count),
                "category_example_rate": float(example_count / section_nonempty_rows)
                if section_nonempty_rows
                else 0.0,
                "category_occurrences_per_nonempty_example": float(
                    occurrence_count / section_nonempty_rows
                )
                if section_nonempty_rows
                else 0.0,
            }
        )
    return records


def _precompute_section_keyword_counts(
    section_texts: Mapping[str, Sequence[str]],
    patterns: Mapping[str, Pattern[str]],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    counts_by_section: Dict[str, Dict[str, np.ndarray]] = {}
    nonempty_by_section: Dict[str, np.ndarray] = {}
    for section, texts in tqdm(section_texts.items(), desc="Precompute section keyword counts"):
        series = pd.Series(list(texts), dtype="string").fillna("")
        nonempty_by_section[section] = series.str.strip().ne("").to_numpy(dtype=bool)
        keyword_counts: Dict[str, np.ndarray] = {}
        for keyword, pattern in patterns.items():
            keyword_counts[keyword] = (
                series.str.count(pattern.pattern, flags=re.IGNORECASE)
                .fillna(0)
                .astype("int32")
                .to_numpy()
            )
        counts_by_section[section] = keyword_counts
    return counts_by_section, nonempty_by_section


def _keyword_counts_from_precomputed(
    *,
    indices: np.ndarray,
    section: str,
    counts_by_section: Mapping[str, Mapping[str, np.ndarray]],
    nonempty_by_section: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    group_size = int(len(indices))
    nonempty_mask = nonempty_by_section[section][indices]
    section_nonempty_rows = int(nonempty_mask.sum())
    records: List[Dict[str, Any]] = []
    for category, keyword in _keyword_category_pairs():
        selected_counts = counts_by_section[section][keyword][indices]
        if section_nonempty_rows:
            selected_counts = selected_counts[nonempty_mask]
        example_count = int((selected_counts > 0).sum())
        occurrence_count = int(selected_counts.sum())
        records.append(
            {
                "category": category,
                "keyword": keyword,
                "group_size": group_size,
                "section_nonempty_rows": section_nonempty_rows,
                "example_count": example_count,
                "occurrence_count": occurrence_count,
                "example_rate": float(example_count / section_nonempty_rows)
                if section_nonempty_rows
                else 0.0,
                "occurrences_per_nonempty_example": float(occurrence_count / section_nonempty_rows)
                if section_nonempty_rows
                else 0.0,
            }
        )
    return records


def _category_counts_from_precomputed(
    *,
    indices: np.ndarray,
    section: str,
    counts_by_section: Mapping[str, Mapping[str, np.ndarray]],
    nonempty_by_section: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    group_size = int(len(indices))
    nonempty_mask = nonempty_by_section[section][indices]
    section_nonempty_rows = int(nonempty_mask.sum())
    records: List[Dict[str, Any]] = []
    for category, keywords in KEYWORD_CATEGORIES.items():
        if keywords:
            selected_counts = np.zeros(group_size, dtype=np.int32)
            for keyword in keywords:
                selected_counts += counts_by_section[section][keyword][indices]
        else:
            selected_counts = np.zeros(group_size, dtype=np.int32)
        if section_nonempty_rows:
            selected_counts = selected_counts[nonempty_mask]
        example_count = int((selected_counts > 0).sum())
        occurrence_count = int(selected_counts.sum())
        records.append(
            {
                "category": category,
                "group_size": group_size,
                "section_nonempty_rows": section_nonempty_rows,
                "category_example_count": example_count,
                "category_occurrence_count": occurrence_count,
                "category_example_rate": float(example_count / section_nonempty_rows)
                if section_nonempty_rows
                else 0.0,
                "category_occurrences_per_nonempty_example": float(
                    occurrence_count / section_nonempty_rows
                )
                if section_nonempty_rows
                else 0.0,
            }
        )
    return records


PAIR_MARKER_RE = re.compile(r"^Pair\s+\d+\s*:", flags=re.IGNORECASE | re.MULTILINE)
USER_PROFILE_MARKER_RE = re.compile(r"User Profile\s*:", flags=re.IGNORECASE)
PREFERRED_MARKER_RE = re.compile(r"Preferred Image\s*:", flags=re.IGNORECASE)
DISPREFERRED_MARKER_RE = re.compile(r"Dispreferred Image\s*:", flags=re.IGNORECASE)
DIFFERENCES_MARKER_RE = re.compile(r"Differences\s*:", flags=re.IGNORECASE)


def _find_marker(pattern: Pattern[str], text: str, start: int = 0) -> Optional[re.Match[str]]:
    return pattern.search(text, pos=start)


def _between_markers(text: str, start_pattern: Pattern[str], end_patterns: Sequence[Pattern[str]]) -> str:
    start_match = _find_marker(start_pattern, text)
    if start_match is None:
        return ""
    start = start_match.end()
    end = len(text)
    for end_pattern in end_patterns:
        end_match = _find_marker(end_pattern, text, start)
        if end_match is not None:
            end = min(end, end_match.start())
    return text[start:end].strip()


def _split_pair_chunks(text: str) -> List[str]:
    matches = list(PAIR_MARKER_RE.finditer(text))
    if not matches:
        return [text]
    chunks: List[str] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunks.append(text[start:end])
    return chunks


def _parse_text_sections(text: str) -> Tuple[Dict[str, str], Dict[str, int]]:
    text = _value_to_text(text)
    user_profile_matches = list(USER_PROFILE_MARKER_RE.finditer(text))
    user_profile_marker_count = len(user_profile_matches)
    user_profile_start = user_profile_matches[-1].end() if user_profile_matches else None
    pair_text = text[: user_profile_matches[-1].start()] if user_profile_matches else text
    pair_chunks = _split_pair_chunks(pair_text)

    preferred_chunks: List[str] = []
    dispreferred_chunks: List[str] = []
    differences_chunks: List[str] = []
    for chunk in pair_chunks:
        preferred = _between_markers(
            chunk,
            PREFERRED_MARKER_RE,
            [DISPREFERRED_MARKER_RE, DIFFERENCES_MARKER_RE],
        )
        dispreferred = _between_markers(
            chunk,
            DISPREFERRED_MARKER_RE,
            [DIFFERENCES_MARKER_RE],
        )
        differences = _between_markers(chunk, DIFFERENCES_MARKER_RE, [])
        if preferred:
            preferred_chunks.append(preferred)
        if dispreferred:
            dispreferred_chunks.append(dispreferred)
        if differences:
            differences_chunks.append(differences)

    sections = {
        "full_text": text,
        "user_profile": text[user_profile_start:].strip() if user_profile_start is not None else "",
        "preferred": "\n\n".join(preferred_chunks).strip(),
        "dispreferred": "\n\n".join(dispreferred_chunks).strip(),
        "differences": "\n\n".join(differences_chunks).strip(),
    }
    marker_counts = {
        "pair_marker_count": len(pair_chunks) if pair_chunks != [text] else len(list(PAIR_MARKER_RE.finditer(text))),
        "preferred_marker_count": len(list(PREFERRED_MARKER_RE.finditer(text))),
        "dispreferred_marker_count": len(list(DISPREFERRED_MARKER_RE.finditer(text))),
        "differences_marker_count": len(list(DIFFERENCES_MARKER_RE.finditer(text))),
        "user_profile_marker_count": user_profile_marker_count,
    }
    return sections, marker_counts


def _parse_all_sections(
    meta: pd.DataFrame,
    sections: Sequence[str],
) -> Tuple[Dict[str, List[str]], pd.DataFrame, pd.DataFrame]:
    requested_sections = list(dict.fromkeys(sections))
    section_texts: Dict[str, List[str]] = {section: [] for section in requested_sections}
    parse_records: List[Dict[str, Any]] = []

    for row_idx, text_value in tqdm(
        enumerate(meta["text"].tolist()),
        total=len(meta),
        desc="Parse text sections",
    ):
        parsed_sections, marker_counts = _parse_text_sections(_value_to_text(text_value))
        row_record: Dict[str, Any] = {"row_index": int(row_idx), **marker_counts}
        for section in requested_sections:
            section_text = parsed_sections[section]
            section_texts[section].append(section_text)
            row_record[f"{section}_nonempty"] = bool(section_text.strip())
            row_record[f"{section}_char_len"] = _text_char_len(section_text)
            row_record[f"{section}_word_count"] = _text_word_count(section_text)
        parse_records.append(row_record)

    parse_df = pd.DataFrame(parse_records)
    summary_records: List[Dict[str, Any]] = []
    for section in requested_sections:
        char_lengths = parse_df[f"{section}_char_len"].to_numpy(dtype=np.float64)
        word_counts = parse_df[f"{section}_word_count"].to_numpy(dtype=np.float64)
        nonempty = parse_df[f"{section}_nonempty"].to_numpy(dtype=bool)
        summary_records.append(
            {
                "section": section,
                "total_rows": int(len(parse_df)),
                "nonempty_rows": int(nonempty.sum()),
                "nonempty_rate": float(nonempty.mean()) if len(nonempty) else 0.0,
                "char_len_mean": float(char_lengths.mean()) if len(char_lengths) else 0.0,
                "char_len_std": float(char_lengths.std(ddof=0)) if len(char_lengths) else 0.0,
                "word_count_mean": float(word_counts.mean()) if len(word_counts) else 0.0,
                "word_count_std": float(word_counts.std(ddof=0)) if len(word_counts) else 0.0,
                "pair_marker_count_mean": float(parse_df["pair_marker_count"].mean()),
                "preferred_marker_count_mean": float(parse_df["preferred_marker_count"].mean()),
                "dispreferred_marker_count_mean": float(parse_df["dispreferred_marker_count"].mean()),
                "differences_marker_count_mean": float(parse_df["differences_marker_count"].mean()),
                "user_profile_marker_count_mean": float(parse_df["user_profile_marker_count"].mean()),
                "user_profile_marker_count_zero_rows": int(
                    (parse_df["user_profile_marker_count"] == 0).sum()
                ),
                "user_profile_marker_count_one_rows": int(
                    (parse_df["user_profile_marker_count"] == 1).sum()
                ),
                "user_profile_marker_count_gt_one_rows": int(
                    (parse_df["user_profile_marker_count"] > 1).sum()
                ),
            }
        )
    return section_texts, parse_df, pd.DataFrame(summary_records)


def _selected_indices(scores_pc: np.ndarray, top_k: int) -> Dict[str, np.ndarray]:
    order_low = np.argsort(scores_pc, kind="mergesort")[:top_k]
    order_high = np.argsort(-scores_pc, kind="mergesort")[:top_k]
    return {"high": order_high.astype(np.int64), "low": order_low.astype(np.int64)}


def _summary_record(
    *,
    pc_num: int,
    side: str,
    rank: int,
    score: float,
    row: pd.Series,
) -> Dict[str, Any]:
    text = _row_text(row)
    record: Dict[str, Any] = {
        "pc": f"PC{pc_num}",
        "pc_index": pc_num - 1,
        "side": side,
        "rank": int(rank),
        "score": float(score),
        "global_row_id": _value_to_text(row["global_row_id"]),
        "split": _value_to_text(row["split"]),
        "shard_id": int(row["shard_id"]),
        "row_in_shard": int(row["row_in_shard"]),
        "user_id": _value_to_text(row["user_id"]),
        "mean_vec_norm": float(row["mean_vec_norm"]),
        "user_row_count_in_output": int(row["user_row_count_in_output"]),
        "text_char_len": _text_char_len(text),
        "text_word_count": _text_word_count(text),
        "caption_unique_count": _nonempty_unique_count([row[col] for col in CAPTION_COLUMNS]),
        "preferred_uid_unique_count": _nonempty_unique_count([row[col] for col in PREFERRED_UID_COLUMNS]),
        "dispreferred_uid_unique_count": _nonempty_unique_count([row[col] for col in DISPREFERRED_UID_COLUMNS]),
        "text_preview": text[:300].replace("\n", " "),
    }
    for col in CAPTION_COLUMNS + PREFERRED_UID_COLUMNS + DISPREFERRED_UID_COLUMNS:
        record[col] = _value_to_text(row[col])
    return record


def _markdown_for_examples(
    *,
    pc_num: int,
    side: str,
    indices: Sequence[int],
    scores_pc: np.ndarray,
    meta: pd.DataFrame,
    text_max_chars: int,
) -> str:
    lines = [f"# PC{pc_num} {side} examples", ""]
    for rank, row_idx in enumerate(indices, start=1):
        row = meta.iloc[int(row_idx)]
        text = _truncate_text(_row_text(row), text_max_chars)
        lines.extend(
            [
                f"## rank {rank}",
                f"score: {float(scores_pc[int(row_idx)]):.8g}",
                f"global_row_id: {_value_to_text(row['global_row_id'])}",
                f"split: {_value_to_text(row['split'])}",
                f"shard_id: {int(row['shard_id'])}",
                f"row_in_shard: {int(row['row_in_shard'])}",
                f"user_id: {_value_to_text(row['user_id'])}",
                f"mean_vec_norm: {float(row['mean_vec_norm']):.8g}",
                f"user_row_count_in_output: {int(row['user_row_count_in_output'])}",
                "",
                "captions:",
            ]
        )
        for col in CAPTION_COLUMNS:
            lines.append(f"- {col}: {_value_to_text(row[col])}")
        lines.extend(["", "preferred_image_uids:"])
        for col in PREFERRED_UID_COLUMNS:
            lines.append(f"- {col}: {_value_to_text(row[col])}")
        lines.extend(["", "dispreferred_image_uids:"])
        for col in DISPREFERRED_UID_COLUMNS:
            lines.append(f"- {col}: {_value_to_text(row[col])}")
        lines.extend(["", "text:", "", "````text", text, "````", "", "---", ""])
    return "\n".join(lines)


def _build_top_bottom_outputs(
    *,
    scores: np.ndarray,
    meta: pd.DataFrame,
    num_pcs: int,
    top_k: int,
    patterns: Mapping[str, Pattern[str]],
    text_max_chars: int,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    summary_records: List[Dict[str, Any]] = []
    keyword_records: List[Dict[str, Any]] = []
    markdown_outputs: Dict[str, str] = {}

    for pc_index in tqdm(range(num_pcs), desc="Analyze PC high/low text"):
        pc_num = pc_index + 1
        scores_pc = np.asarray(scores[:, pc_index], dtype=np.float64)
        indices_by_side = _selected_indices(scores_pc, top_k)
        for side, indices in indices_by_side.items():
            md_text = _markdown_for_examples(
                pc_num=pc_num,
                side=side,
                indices=indices,
                scores_pc=scores_pc,
                meta=meta,
                text_max_chars=text_max_chars,
            )
            markdown_outputs[str(_pc_side_md_path(output_dir, pc_num, side))] = md_text
            selected_texts = [_row_text(meta.iloc[int(row_idx)]) for row_idx in indices]
            for rank, row_idx in enumerate(indices, start=1):
                summary_records.append(
                    _summary_record(
                        pc_num=pc_num,
                        side=side,
                        rank=rank,
                        score=float(scores_pc[int(row_idx)]),
                        row=meta.iloc[int(row_idx)],
                    )
                )
            for counts in _keyword_counts(selected_texts, patterns):
                keyword_records.append(
                    {
                        "pc": f"PC{pc_num}",
                        "pc_index": pc_index,
                        "side": side,
                        **counts,
                    }
                )

    summary_df = pd.DataFrame(summary_records)
    keyword_df = pd.DataFrame(keyword_records)
    delta_df = _keyword_delta_frame(keyword_df)
    return summary_df, keyword_df, delta_df, markdown_outputs


def _keyword_delta_frame(keyword_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for (pc, pc_index, keyword), group in keyword_df.groupby(["pc", "pc_index", "keyword"], sort=True):
        by_side = {str(row["side"]): row for _, row in group.iterrows()}
        high = by_side.get("high")
        low = by_side.get("low")
        if high is None or low is None:
            continue
        records.append(
            {
                "pc": pc,
                "pc_index": int(pc_index),
                "keyword": keyword,
                "high_total_examples": int(high["total_examples"]),
                "low_total_examples": int(low["total_examples"]),
                "high_example_count": int(high["example_count"]),
                "low_example_count": int(low["example_count"]),
                "high_occurrence_count": int(high["occurrence_count"]),
                "low_occurrence_count": int(low["occurrence_count"]),
                "high_example_rate": float(high["example_rate"]),
                "low_example_rate": float(low["example_rate"]),
                "example_rate_delta_high_minus_low": float(high["example_rate"] - low["example_rate"]),
                "occurrence_count_delta_high_minus_low": int(
                    high["occurrence_count"] - low["occurrence_count"]
                ),
                "occurrences_per_example_delta_high_minus_low": float(
                    high["occurrences_per_example"] - low["occurrences_per_example"]
                ),
            }
        )
    return pd.DataFrame(records)


def _ceil_count(n_rows: int, fraction: float) -> int:
    return max(1, int(np.ceil(n_rows * fraction)))


def _pc1_quantile_groups(scores_pc1: np.ndarray) -> Dict[str, np.ndarray]:
    n_rows = int(scores_pc1.shape[0])
    order_asc = np.argsort(scores_pc1, kind="mergesort")
    order_desc = order_asc[::-1]
    middle_start = int(np.floor(n_rows * 0.45))
    middle_end = int(np.ceil(n_rows * 0.55))
    return {
        "top_1pct": order_desc[: _ceil_count(n_rows, 0.01)].astype(np.int64),
        "top_7pct": order_desc[: _ceil_count(n_rows, 0.07)].astype(np.int64),
        "middle_45_55pct": order_asc[middle_start:middle_end].astype(np.int64),
        "bottom_7pct": order_asc[: _ceil_count(n_rows, 0.07)].astype(np.int64),
        "bottom_1pct": order_asc[: _ceil_count(n_rows, 0.01)].astype(np.int64),
    }


def _mean_std(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std(ddof=0))


def _pc1_quantile_summary(
    *,
    scores: np.ndarray,
    meta: pd.DataFrame,
    patterns: Mapping[str, Pattern[str]],
) -> pd.DataFrame:
    scores_pc1 = np.asarray(scores[:, 0], dtype=np.float64)
    groups = _pc1_quantile_groups(scores_pc1)
    text_char_lengths = meta["text"].map(lambda value: _text_char_len(_value_to_text(value))).to_numpy()
    text_word_counts = meta["text"].map(lambda value: _text_word_count(_value_to_text(value))).to_numpy()
    mean_vec_norm = meta["mean_vec_norm"].astype("float64").to_numpy()

    records: List[Dict[str, Any]] = []
    for group_name, indices in groups.items():
        group_scores = scores_pc1[indices]
        norm_mean, norm_std = _mean_std(mean_vec_norm[indices])
        char_mean, char_std = _mean_std(text_char_lengths[indices])
        word_mean, word_std = _mean_std(text_word_counts[indices])
        texts = [_row_text(meta.iloc[int(row_idx)]) for row_idx in indices]
        base = {
            "pc": "PC1",
            "pc_index": 0,
            "group": group_name,
            "group_size": int(len(indices)),
            "score_min": float(group_scores.min()),
            "score_max": float(group_scores.max()),
            "score_mean": float(group_scores.mean()),
            "score_std": float(group_scores.std(ddof=0)),
            "mean_vec_norm_mean": norm_mean,
            "mean_vec_norm_std": norm_std,
            "text_char_len_mean": char_mean,
            "text_char_len_std": char_std,
            "text_word_count_mean": word_mean,
            "text_word_count_std": word_std,
        }
        for counts in _keyword_counts(texts, patterns):
            records.append({**base, **counts})
    return pd.DataFrame(records)


def _fraction_label(fraction: float) -> str:
    percent = fraction * 100.0
    if abs(percent - round(percent)) < 1e-9:
        return f"{int(round(percent))}pct"
    return f"{percent:g}".replace(".", "p") + "pct"


def _pc_fractional_groups(
    scores_pc: np.ndarray,
    stat_fracs: Sequence[float],
) -> List[Dict[str, Any]]:
    n_rows = int(scores_pc.shape[0])
    order_asc = np.argsort(scores_pc, kind="mergesort")
    order_desc = order_asc[::-1]
    groups: List[Dict[str, Any]] = []
    for fraction in stat_fracs:
        count = _ceil_count(n_rows, fraction)
        label = _fraction_label(fraction)
        groups.append(
            {
                "group": f"top_{label}",
                "group_side": "high",
                "fraction": float(fraction),
                "indices": order_desc[:count].astype(np.int64),
            }
        )
        groups.append(
            {
                "group": f"bottom_{label}",
                "group_side": "low",
                "fraction": float(fraction),
                "indices": order_asc[:count].astype(np.int64),
            }
        )

    middle_start = int(np.floor(n_rows * 0.45))
    middle_end = int(np.ceil(n_rows * 0.55))
    groups.append(
        {
            "group": "middle_45_55pct",
            "group_side": "middle",
            "fraction": np.nan,
            "indices": order_asc[middle_start:middle_end].astype(np.int64),
        }
    )
    return groups


def _base_group_stats(
    *,
    pc_num: int,
    scores_pc: np.ndarray,
    indices: np.ndarray,
    group_name: str,
    group_side: str,
    fraction: float,
    meta: pd.DataFrame,
    section_texts: Mapping[str, Sequence[str]],
    sections: Sequence[str],
) -> Dict[str, Any]:
    group_scores = scores_pc[indices]
    mean_vec_norm = meta["mean_vec_norm"].astype("float64").to_numpy()[indices]
    full_texts = [_value_to_text(meta.iloc[int(row_idx)]["text"]) for row_idx in indices]
    char_lengths = np.asarray([_text_char_len(text) for text in full_texts], dtype=np.float64)
    word_counts = np.asarray([_text_word_count(text) for text in full_texts], dtype=np.float64)
    norm_mean, norm_std = _mean_std(mean_vec_norm)
    char_mean, char_std = _mean_std(char_lengths)
    word_mean, word_std = _mean_std(word_counts)

    record: Dict[str, Any] = {
        "pc": f"PC{pc_num}",
        "pc_index": pc_num - 1,
        "group": group_name,
        "group_side": group_side,
        "fraction": float(fraction) if np.isfinite(fraction) else "",
        "group_size": int(len(indices)),
        "score_min": float(group_scores.min()),
        "score_max": float(group_scores.max()),
        "score_mean": float(group_scores.mean()),
        "score_std": float(group_scores.std(ddof=0)),
        "mean_vec_norm_mean": norm_mean,
        "mean_vec_norm_std": norm_std,
        "text_char_len_mean": char_mean,
        "text_char_len_std": char_std,
        "text_word_count_mean": word_mean,
        "text_word_count_std": word_std,
    }
    for section in sections:
        selected = [section_texts[section][int(row_idx)] for row_idx in indices]
        record[f"{section}_nonempty_rows"] = int(sum(bool(text.strip()) for text in selected))
    return record


def _build_fractional_outputs(
    *,
    scores: np.ndarray,
    meta: pd.DataFrame,
    num_pcs: int,
    stat_fracs: Sequence[float],
    sections: Sequence[str],
    section_texts: Mapping[str, Sequence[str]],
    patterns: Mapping[str, Pattern[str]],
    counts_by_section: Mapping[str, Mapping[str, np.ndarray]],
    nonempty_by_section: Mapping[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_stats_records: List[Dict[str, Any]] = []
    keyword_records: List[Dict[str, Any]] = []
    category_records: List[Dict[str, Any]] = []

    for pc_index in tqdm(range(num_pcs), desc="Analyze fractional PC groups"):
        pc_num = pc_index + 1
        scores_pc = np.asarray(scores[:, pc_index], dtype=np.float64)
        for group_info in _pc_fractional_groups(scores_pc, stat_fracs):
            indices = group_info["indices"]
            group_name = str(group_info["group"])
            group_side = str(group_info["group_side"])
            fraction = float(group_info["fraction"])
            group_stats_records.append(
                _base_group_stats(
                    pc_num=pc_num,
                    scores_pc=scores_pc,
                    indices=indices,
                    group_name=group_name,
                    group_side=group_side,
                    fraction=fraction,
                    meta=meta,
                    section_texts=section_texts,
                    sections=sections,
                )
            )
            for section in sections:
                common = {
                    "pc": f"PC{pc_num}",
                    "pc_index": pc_index,
                    "group": group_name,
                    "group_side": group_side,
                    "fraction": float(fraction) if np.isfinite(fraction) else "",
                    "section": section,
                }
                for counts in _keyword_counts_from_precomputed(
                    indices=indices,
                    section=section,
                    counts_by_section=counts_by_section,
                    nonempty_by_section=nonempty_by_section,
                ):
                    keyword_records.append({**common, **counts})
                for counts in _category_counts_from_precomputed(
                    indices=indices,
                    section=section,
                    counts_by_section=counts_by_section,
                    nonempty_by_section=nonempty_by_section,
                ):
                    category_records.append({**common, **counts})

    group_stats_df = pd.DataFrame(group_stats_records)
    keyword_df = pd.DataFrame(keyword_records)
    category_df = pd.DataFrame(category_records)
    keyword_delta_df = _high_low_delta_frame(keyword_df, level="keyword")
    category_delta_df = _high_low_delta_frame(category_df, level="category")
    return group_stats_df, keyword_df, category_df, keyword_delta_df, category_delta_df


def _high_low_delta_frame(df: pd.DataFrame, *, level: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if level == "keyword":
        group_cols = ["pc", "pc_index", "fraction", "section", "category", "keyword"]
        rate_col = "example_rate"
        example_count_col = "example_count"
        occurrence_count_col = "occurrence_count"
    elif level == "category":
        group_cols = ["pc", "pc_index", "fraction", "section", "category"]
        rate_col = "category_example_rate"
        example_count_col = "category_example_count"
        occurrence_count_col = "category_occurrence_count"
    else:
        raise ValueError(f"Unsupported delta level: {level}")

    records: List[Dict[str, Any]] = []
    comparable = df[df["group_side"].isin(["high", "low"])].copy()
    for keys, group in comparable.groupby(group_cols, sort=True, dropna=False):
        by_side = {str(row["group_side"]): row for _, row in group.iterrows()}
        high = by_side.get("high")
        low = by_side.get("low")
        if high is None or low is None:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        records.append(
            {
                **base,
                "metric_level": level,
                "high_group": str(high["group"]),
                "low_group": str(low["group"]),
                "high_group_size": int(high["group_size"]),
                "low_group_size": int(low["group_size"]),
                "high_section_nonempty_rows": int(high["section_nonempty_rows"]),
                "low_section_nonempty_rows": int(low["section_nonempty_rows"]),
                "high_example_count": int(high[example_count_col]),
                "low_example_count": int(low[example_count_col]),
                "high_occurrence_count": int(high[occurrence_count_col]),
                "low_occurrence_count": int(low[occurrence_count_col]),
                "high_example_rate": float(high[rate_col]),
                "low_example_rate": float(low[rate_col]),
                "high_minus_low": float(high[rate_col] - low[rate_col]),
            }
        )
    return pd.DataFrame(records)


def _preferred_dispreferred_contrast_frame(
    keyword_df: pd.DataFrame,
    category_df: pd.DataFrame,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    records.extend(_preferred_dispreferred_contrast_records(keyword_df, level="keyword"))
    records.extend(_preferred_dispreferred_contrast_records(category_df, level="category"))
    return pd.DataFrame(records)


def _preferred_dispreferred_contrast_records(df: pd.DataFrame, *, level: str) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    if level == "keyword":
        group_cols = ["pc", "pc_index", "group", "group_side", "fraction", "category", "keyword"]
        rate_col = "example_rate"
        count_col = "example_count"
    elif level == "category":
        group_cols = ["pc", "pc_index", "group", "group_side", "fraction", "category"]
        rate_col = "category_example_rate"
        count_col = "category_example_count"
    else:
        raise ValueError(f"Unsupported contrast level: {level}")

    records: List[Dict[str, Any]] = []
    subset = df[df["section"].isin(["preferred", "dispreferred"])].copy()
    for keys, group in subset.groupby(group_cols, sort=True, dropna=False):
        by_section = {str(row["section"]): row for _, row in group.iterrows()}
        preferred = by_section.get("preferred")
        dispreferred = by_section.get("dispreferred")
        if preferred is None or dispreferred is None:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        records.append(
            {
                **base,
                "metric_level": level,
                "keyword": base.get("keyword", "__category__"),
                "preferred_section_nonempty_rows": int(preferred["section_nonempty_rows"]),
                "dispreferred_section_nonempty_rows": int(dispreferred["section_nonempty_rows"]),
                "preferred_example_count": int(preferred[count_col]),
                "dispreferred_example_count": int(dispreferred[count_col]),
                "preferred_example_rate": float(preferred[rate_col]),
                "dispreferred_example_rate": float(dispreferred[rate_col]),
                "preferred_minus_dispreferred": float(preferred[rate_col] - dispreferred[rate_col]),
            }
        )
    return records


def _preferred_dispreferred_contrast_delta_frame(contrast_df: pd.DataFrame) -> pd.DataFrame:
    if contrast_df.empty:
        return pd.DataFrame()
    group_cols = ["pc", "pc_index", "fraction", "category", "keyword", "metric_level"]
    records: List[Dict[str, Any]] = []
    comparable = contrast_df[contrast_df["group_side"].isin(["high", "low"])].copy()
    for keys, group in comparable.groupby(group_cols, sort=True, dropna=False):
        by_side = {str(row["group_side"]): row for _, row in group.iterrows()}
        high = by_side.get("high")
        low = by_side.get("low")
        if high is None or low is None:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        records.append(
            {
                **base,
                "high_group": str(high["group"]),
                "low_group": str(low["group"]),
                "preferred_rate_high": float(high["preferred_example_rate"]),
                "dispreferred_rate_high": float(high["dispreferred_example_rate"]),
                "preferred_minus_dispreferred_high": float(high["preferred_minus_dispreferred"]),
                "preferred_rate_low": float(low["preferred_example_rate"]),
                "dispreferred_rate_low": float(low["dispreferred_example_rate"]),
                "preferred_minus_dispreferred_low": float(low["preferred_minus_dispreferred"]),
                "contrast_delta": float(
                    high["preferred_minus_dispreferred"] - low["preferred_minus_dispreferred"]
                ),
            }
        )
    return pd.DataFrame(records)


def _focus_pc_summary_frame(
    *,
    keyword_delta_df: pd.DataFrame,
    contrast_delta_df: pd.DataFrame,
    focus_pcs: Sequence[int],
) -> pd.DataFrame:
    if keyword_delta_df.empty:
        return pd.DataFrame()
    focus_labels = {f"PC{pc}" for pc in focus_pcs}
    keyword_focus = keyword_delta_df[
        (keyword_delta_df["pc"].isin(focus_labels)) & (keyword_delta_df["metric_level"] == "keyword")
    ].copy()
    contrast_focus = contrast_delta_df[
        (contrast_delta_df["pc"].isin(focus_labels)) & (contrast_delta_df["metric_level"] == "keyword")
    ].copy()
    join_cols = ["pc", "pc_index", "fraction", "category", "keyword"]
    merged = keyword_focus.merge(
        contrast_focus[
            [
                *join_cols,
                "preferred_rate_high",
                "dispreferred_rate_high",
                "preferred_minus_dispreferred_high",
                "preferred_rate_low",
                "dispreferred_rate_low",
                "preferred_minus_dispreferred_low",
                "contrast_delta",
            ]
        ],
        on=join_cols,
        how="left",
    )
    columns = [
        "pc",
        "fraction",
        "section",
        "keyword",
        "category",
        "high_example_rate",
        "low_example_rate",
        "high_minus_low",
        "preferred_rate_high",
        "dispreferred_rate_high",
        "preferred_minus_dispreferred_high",
        "preferred_rate_low",
        "dispreferred_rate_low",
        "preferred_minus_dispreferred_low",
        "contrast_delta",
    ]
    return merged[columns].sort_values(["pc", "fraction", "section", "category", "keyword"])


def _expected_output_paths(output_dir: Path, num_pcs: int) -> List[Path]:
    paths = [
        _summary_path(output_dir),
        _keyword_frequency_path(output_dir),
        _keyword_delta_path(output_dir),
        _pc1_quantile_path(output_dir),
        _section_parse_summary_path(output_dir),
        _pc_group_stats_path(output_dir),
        _keyword_group_section_path(output_dir),
        _category_group_section_path(output_dir),
        _keyword_group_section_delta_path(output_dir),
        _category_group_section_delta_path(output_dir),
        _preferred_dispreferred_contrast_path(output_dir),
        _preferred_dispreferred_contrast_delta_path(output_dir),
        _focus_pc_summary_path(output_dir),
        _config_path(output_dir),
    ]
    for pc_num in range(1, num_pcs + 1):
        paths.append(_pc_side_md_path(output_dir, pc_num, "high"))
        paths.append(_pc_side_md_path(output_dir, pc_num, "low"))
    return paths


def _write_outputs(
    *,
    output_dir: Path,
    summary_df: pd.DataFrame,
    keyword_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    quantile_df: pd.DataFrame,
    section_parse_summary_df: pd.DataFrame,
    pc_group_stats_df: pd.DataFrame,
    keyword_group_section_df: pd.DataFrame,
    category_group_section_df: pd.DataFrame,
    keyword_group_section_delta_df: pd.DataFrame,
    category_group_section_delta_df: pd.DataFrame,
    preferred_dispreferred_contrast_df: pd.DataFrame,
    preferred_dispreferred_contrast_delta_df: pd.DataFrame,
    focus_pc_summary_df: pd.DataFrame,
    markdown_outputs: Mapping[str, str],
    config: Mapping[str, Any],
) -> None:
    for path_text, text in markdown_outputs.items():
        _atomic_write_text(Path(path_text), text)
    _atomic_write_csv(_summary_path(output_dir), summary_df)
    _atomic_write_csv(_keyword_frequency_path(output_dir), keyword_df)
    _atomic_write_csv(_keyword_delta_path(output_dir), delta_df)
    _atomic_write_csv(_pc1_quantile_path(output_dir), quantile_df)
    _atomic_write_csv(_section_parse_summary_path(output_dir), section_parse_summary_df)
    _atomic_write_csv(_pc_group_stats_path(output_dir), pc_group_stats_df)
    _atomic_write_csv(_keyword_group_section_path(output_dir), keyword_group_section_df)
    _atomic_write_csv(_category_group_section_path(output_dir), category_group_section_df)
    _atomic_write_csv(_keyword_group_section_delta_path(output_dir), keyword_group_section_delta_df)
    _atomic_write_csv(_category_group_section_delta_path(output_dir), category_group_section_delta_df)
    _atomic_write_csv(_preferred_dispreferred_contrast_path(output_dir), preferred_dispreferred_contrast_df)
    _atomic_write_csv(
        _preferred_dispreferred_contrast_delta_path(output_dir),
        preferred_dispreferred_contrast_delta_df,
    )
    _atomic_write_csv(_focus_pc_summary_path(output_dir), focus_pc_summary_df)
    _atomic_write_json(_config_path(output_dir), config)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    scores_path = _resolve_path(args.scores)
    meta_path = _resolve_path(args.meta)
    output_root = _resolve_path(args.output_root)
    output_dir = _output_dir(output_root, args.output_name)

    started_at = _utc_now_iso()
    scores, meta = _load_inputs(scores_path, meta_path, args.num_pcs, args.top_k)
    expected_paths = _expected_output_paths(output_dir, args.num_pcs)
    _check_no_existing(expected_paths, args.overwrite)

    patterns = _compile_keyword_patterns(DEFAULT_KEYWORDS)
    section_texts, section_parse_df, section_parse_summary_df = _parse_all_sections(meta, args.sections)
    counts_by_section, nonempty_by_section = _precompute_section_keyword_counts(section_texts, patterns)
    summary_df, keyword_df, delta_df, markdown_outputs = _build_top_bottom_outputs(
        scores=scores,
        meta=meta,
        num_pcs=args.num_pcs,
        top_k=args.top_k,
        patterns=patterns,
        text_max_chars=args.text_max_chars,
        output_dir=output_dir,
    )
    quantile_df = _pc1_quantile_summary(scores=scores, meta=meta, patterns=patterns)
    (
        pc_group_stats_df,
        keyword_group_section_df,
        category_group_section_df,
        keyword_group_section_delta_df,
        category_group_section_delta_df,
    ) = _build_fractional_outputs(
        scores=scores,
        meta=meta,
        num_pcs=args.num_pcs,
        stat_fracs=args.stat_fracs,
        sections=args.sections,
        section_texts=section_texts,
        patterns=patterns,
        counts_by_section=counts_by_section,
        nonempty_by_section=nonempty_by_section,
    )
    preferred_dispreferred_contrast_df = _preferred_dispreferred_contrast_frame(
        keyword_group_section_df,
        category_group_section_df,
    )
    preferred_dispreferred_contrast_delta_df = _preferred_dispreferred_contrast_delta_frame(
        preferred_dispreferred_contrast_df
    )
    focus_pc_summary_df = _focus_pc_summary_frame(
        keyword_delta_df=keyword_group_section_delta_df,
        contrast_delta_df=preferred_dispreferred_contrast_delta_df,
        focus_pcs=args.focus_pcs,
    )

    config = {
        "argv": sys.argv,
        "python_executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now_iso(),
        "versions": _package_versions(),
        "args": {
            "scores": str(args.scores),
            "meta": str(args.meta),
            "output_root": str(args.output_root),
            "output_name": args.output_name,
            "num_pcs": int(args.num_pcs),
            "top_k": int(args.top_k),
            "stat_fracs": [float(frac) for frac in args.stat_fracs],
            "sections": list(args.sections),
            "focus_pcs": [int(pc) for pc in args.focus_pcs],
            "text_max_chars": int(args.text_max_chars),
            "overwrite": bool(args.overwrite),
        },
        "resolved_paths": {
            "scores": str(scores_path),
            "meta": str(meta_path),
            "output_root": str(output_root),
            "output_dir": str(output_dir),
        },
        "score_shape": [int(scores.shape[0]), int(scores.shape[1])],
        "metadata_rows": int(len(meta)),
        "required_metadata_columns": REQUIRED_META_COLUMNS,
        "keywords": DEFAULT_KEYWORDS,
        "keyword_categories": KEYWORD_CATEGORIES,
        "duplicated_keywords": _duplicated_keywords(),
        "ambiguous_duplicated_keyword_notes": {
            "soft": (
                "'soft' is counted in both color_lighting and detail_sharpness. "
                "Interpret category-level aggregates involving soft with this ambiguity in mind."
            )
        },
        "contrast_definitions": {
            "preferred_dispreferred_contrast": (
                "preferred section example_rate minus dispreferred section example_rate "
                "inside the same PC group"
            ),
            "contrast_delta": (
                "preferred_dispreferred_contrast(high group) minus "
                "preferred_dispreferred_contrast(low group) for the same PC, fraction, keyword/category"
            ),
        },
        "section_marker_summary": {
            "user_profile_marker_count_zero_rows": int(
                (section_parse_df["user_profile_marker_count"] == 0).sum()
            ),
            "user_profile_marker_count_one_rows": int(
                (section_parse_df["user_profile_marker_count"] == 1).sum()
            ),
            "user_profile_marker_count_gt_one_rows": int(
                (section_parse_df["user_profile_marker_count"] > 1).sum()
            ),
        },
        "outputs": {
            "summary_csv": str(_summary_path(output_dir)),
            "keyword_frequency_csv": str(_keyword_frequency_path(output_dir)),
            "keyword_delta_csv": str(_keyword_delta_path(output_dir)),
            "pc1_quantile_csv": str(_pc1_quantile_path(output_dir)),
            "section_parse_summary_csv": str(_section_parse_summary_path(output_dir)),
            "pc_group_stats_csv": str(_pc_group_stats_path(output_dir)),
            "keyword_group_section_csv": str(_keyword_group_section_path(output_dir)),
            "category_group_section_csv": str(_category_group_section_path(output_dir)),
            "keyword_group_section_delta_csv": str(_keyword_group_section_delta_path(output_dir)),
            "category_group_section_delta_csv": str(_category_group_section_delta_path(output_dir)),
            "preferred_dispreferred_contrast_csv": str(
                _preferred_dispreferred_contrast_path(output_dir)
            ),
            "preferred_dispreferred_contrast_delta_csv": str(
                _preferred_dispreferred_contrast_delta_path(output_dir)
            ),
            "focus_pc_summary_csv": str(_focus_pc_summary_path(output_dir)),
            "markdown_files": sorted(markdown_outputs.keys()),
            "config_json": str(_config_path(output_dir)),
        },
    }

    _write_outputs(
        output_dir=output_dir,
        summary_df=summary_df,
        keyword_df=keyword_df,
        delta_df=delta_df,
        quantile_df=quantile_df,
        section_parse_summary_df=section_parse_summary_df,
        pc_group_stats_df=pc_group_stats_df,
        keyword_group_section_df=keyword_group_section_df,
        category_group_section_df=category_group_section_df,
        keyword_group_section_delta_df=keyword_group_section_delta_df,
        category_group_section_delta_df=category_group_section_delta_df,
        preferred_dispreferred_contrast_df=preferred_dispreferred_contrast_df,
        preferred_dispreferred_contrast_delta_df=preferred_dispreferred_contrast_delta_df,
        focus_pc_summary_df=focus_pc_summary_df,
        markdown_outputs=markdown_outputs,
        config=config,
    )
    print(
        json.dumps(
            {
                "analyze_row_pca_text_summary": {
                    "output_dir": str(output_dir),
                    "score_shape": [int(scores.shape[0]), int(scores.shape[1])],
                    "metadata_rows": int(len(meta)),
                    "num_pcs": int(args.num_pcs),
                    "top_k": int(args.top_k),
                    "stat_fracs": [float(frac) for frac in args.stat_fracs],
                    "sections": list(args.sections),
                    "summary_rows": int(len(summary_df)),
                    "keyword_rows": int(len(keyword_df)),
                    "pc1_quantile_rows": int(len(quantile_df)),
                    "pc_group_stats_rows": int(len(pc_group_stats_df)),
                    "keyword_group_section_rows": int(len(keyword_group_section_df)),
                    "category_group_section_rows": int(len(category_group_section_df)),
                    "preferred_dispreferred_contrast_rows": int(
                        len(preferred_dispreferred_contrast_df)
                    ),
                    "focus_pc_summary_rows": int(len(focus_pc_summary_df)),
                    "config_path": str(_config_path(output_dir)),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
