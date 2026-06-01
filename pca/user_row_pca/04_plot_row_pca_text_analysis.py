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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


INSTALL_COMMAND = (
    "conda activate ppd_pca && "
    "conda install -c conda-forge -y numpy pandas pyarrow matplotlib"
)
REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "matplotlib": "matplotlib",
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402


DEFAULT_SECTIONS = ["user_profile", "preferred"]
DEFAULT_FOCUS_PCS = [4, 5, 8]
CATEGORY_ORDER = ["style", "color_lighting", "composition", "detail_sharpness", "aesthetic"]
GROUP_ORDER = ["bottom_10pct", "middle_45_55pct", "top_10pct"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot section-aware row PCA text analysis CSV outputs."
    )
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--sections", nargs="+", default=DEFAULT_SECTIONS)
    parser.add_argument("--focus-pcs", type=int, nargs="+", default=DEFAULT_FOCUS_PCS)
    parser.add_argument("--top-keywords", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.fraction <= 0.0 or args.fraction >= 0.5:
        raise ValueError(f"--fraction must be in (0, 0.5), got {args.fraction}")
    if not args.sections:
        raise ValueError("--sections must contain at least one section.")
    if not args.focus_pcs:
        raise ValueError("--focus-pcs must contain at least one PC number.")
    if args.top_keywords <= 0:
        raise ValueError(f"--top-keywords must be > 0, got {args.top_keywords}")


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _fraction_label(fraction: float) -> str:
    percent = fraction * 100.0
    if abs(percent - round(percent)) < 1e-9:
        return f"top{int(round(percent))}"
    return "top" + f"{percent:g}".replace(".", "p")


def _pc_sort_key(pc: str) -> int:
    match = re.search(r"(\d+)$", str(pc))
    return int(match.group(1)) if match else 10**9


def _pc_labels(num_pcs: int = 10) -> List[str]:
    return [f"PC{i}" for i in range(1, num_pcs + 1)]


def _focus_labels(focus_pcs: Sequence[int]) -> List[str]:
    return [f"PC{int(pc)}" for pc in focus_pcs]


def _check_required_files(analysis_dir: Path) -> None:
    required = [
        "category_high_low_delta_by_pc_group_section.csv",
        "keyword_high_low_delta_by_pc_group_section.csv",
        "preferred_dispreferred_contrast_delta_by_pc_group.csv",
        "focus_pc_summary.csv",
        "pc_group_stats.csv",
        "section_parse_summary.csv",
    ]
    missing = [str(analysis_dir / name) for name in required if not (analysis_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required analysis CSV file(s): {missing}")


def _atomic_savefig(path: Path, fig: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
        fig.savefig(tmp_path, format="png", dpi=160, bbox_inches="tight")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    finally:
        plt.close(fig)


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


def _check_no_existing(paths: Iterable[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        preview = "\n".join(existing[:30])
        suffix = "" if len(existing) <= 30 else f"\n... and {len(existing) - 30} more"
        raise FileExistsError(f"Output file(s) already exist. Pass --overwrite:\n{preview}{suffix}")


def _package_versions() -> Dict[str, str]:
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _load_frames(analysis_dir: Path) -> Dict[str, pd.DataFrame]:
    return {
        "category_delta": pd.read_csv(analysis_dir / "category_high_low_delta_by_pc_group_section.csv"),
        "keyword_delta": pd.read_csv(analysis_dir / "keyword_high_low_delta_by_pc_group_section.csv"),
        "contrast_delta": pd.read_csv(analysis_dir / "preferred_dispreferred_contrast_delta_by_pc_group.csv"),
        "focus": pd.read_csv(analysis_dir / "focus_pc_summary.csv"),
        "group_stats": pd.read_csv(analysis_dir / "pc_group_stats.csv"),
        "section_parse": pd.read_csv(analysis_dir / "section_parse_summary.csv"),
    }


def _ordered_pivot(
    df: pd.DataFrame,
    *,
    index_col: str,
    column_col: str,
    value_col: str,
    row_order: Sequence[str],
    column_order: Sequence[str],
) -> pd.DataFrame:
    pivot = df.pivot_table(index=index_col, columns=column_col, values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=list(row_order))
    existing_columns = [col for col in column_order if col in pivot.columns]
    remaining_columns = [col for col in pivot.columns if col not in existing_columns]
    pivot = pivot.reindex(columns=[*existing_columns, *remaining_columns])
    return pivot.fillna(0.0)


def _heatmap(
    data: pd.DataFrame,
    *,
    title: str,
    path: Path,
    cmap: str = "coolwarm",
    annotate: bool = True,
) -> None:
    values = data.to_numpy(dtype=np.float64)
    max_abs = float(np.nanmax(np.abs(values))) if values.size else 1.0
    max_abs = max(max_abs, 1e-9)
    fig_width = max(7.0, 0.55 * len(data.columns) + 2.5)
    fig_height = max(4.5, 0.42 * len(data.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs))
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    if annotate and len(data.columns) <= 20 and len(data.index) <= 12:
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if abs(value) < max_abs * 0.6 else "white",
                )
    fig.colorbar(image, ax=ax, label="delta")
    fig.tight_layout()
    _atomic_savefig(path, fig)


def _plot_category_high_low_heatmaps(
    *,
    category_delta: pd.DataFrame,
    output_dir: Path,
    fraction: float,
    sections: Sequence[str],
) -> List[Path]:
    paths: List[Path] = []
    pc_order = _pc_labels()
    frac_label = _fraction_label(fraction)
    for section in sections:
        subset = category_delta[
            (category_delta["fraction"] == fraction) & (category_delta["section"] == section)
        ].copy()
        data = _ordered_pivot(
            subset,
            index_col="pc",
            column_col="category",
            value_col="high_minus_low",
            row_order=pc_order,
            column_order=CATEGORY_ORDER,
        )
        path = output_dir / f"category_high_low_heatmap_{section}_{frac_label}.png"
        _heatmap(
            data,
            title=f"Category high-low delta ({section}, {frac_label})",
            path=path,
        )
        paths.append(path)
    return paths


def _plot_category_contrast_heatmap(
    *,
    contrast_delta: pd.DataFrame,
    output_dir: Path,
    fraction: float,
) -> Path:
    subset = contrast_delta[
        (contrast_delta["fraction"] == fraction) & (contrast_delta["metric_level"] == "category")
    ].copy()
    data = _ordered_pivot(
        subset,
        index_col="pc",
        column_col="category",
        value_col="contrast_delta",
        row_order=_pc_labels(),
        column_order=CATEGORY_ORDER,
    )
    path = output_dir / f"preferred_dispreferred_contrast_delta_category_{_fraction_label(fraction)}.png"
    _heatmap(
        data,
        title=f"Preferred-dispreferred category contrast delta ({_fraction_label(fraction)})",
        path=path,
    )
    return path


def _keyword_label(row: Any) -> str:
    return f"{row.keyword}\n({row.category})"


def _select_focus_keywords(
    df: pd.DataFrame,
    *,
    value_col: str,
    top_keywords: int,
) -> List[str]:
    df = df.copy()
    df["keyword_label"] = df.apply(_keyword_label, axis=1)
    ranked = (
        df.assign(abs_value=df[value_col].abs())
        .groupby("keyword_label", sort=False)["abs_value"]
        .max()
        .sort_values(ascending=False)
    )
    return ranked.head(top_keywords).index.tolist()


def _plot_keyword_focus_heatmap(
    *,
    keyword_delta: pd.DataFrame,
    output_dir: Path,
    fraction: float,
    focus_pcs: Sequence[int],
    section: str,
    top_keywords: int,
) -> Path:
    focus_labels = _focus_labels(focus_pcs)
    subset = keyword_delta[
        (keyword_delta["fraction"] == fraction)
        & (keyword_delta["section"] == section)
        & (keyword_delta["pc"].isin(focus_labels))
    ].copy()
    subset["keyword_label"] = subset.apply(_keyword_label, axis=1)
    keywords = _select_focus_keywords(subset, value_col="high_minus_low", top_keywords=top_keywords)
    subset = subset[subset["keyword_label"].isin(keywords)]
    data = _ordered_pivot(
        subset,
        index_col="pc",
        column_col="keyword_label",
        value_col="high_minus_low",
        row_order=focus_labels,
        column_order=keywords,
    )
    path = output_dir / f"keyword_high_low_heatmap_{section}_{_fraction_label(fraction)}_focus.png"
    _heatmap(
        data,
        title=f"Focus keyword high-low delta ({section}, {_fraction_label(fraction)})",
        path=path,
        annotate=True,
    )
    return path


def _plot_contrast_keyword_focus_heatmap(
    *,
    contrast_delta: pd.DataFrame,
    output_dir: Path,
    fraction: float,
    focus_pcs: Sequence[int],
    top_keywords: int,
) -> Path:
    focus_labels = _focus_labels(focus_pcs)
    subset = contrast_delta[
        (contrast_delta["fraction"] == fraction)
        & (contrast_delta["metric_level"] == "keyword")
        & (contrast_delta["pc"].isin(focus_labels))
    ].copy()
    subset["keyword_label"] = subset.apply(_keyword_label, axis=1)
    keywords = _select_focus_keywords(subset, value_col="contrast_delta", top_keywords=top_keywords)
    subset = subset[subset["keyword_label"].isin(keywords)]
    data = _ordered_pivot(
        subset,
        index_col="pc",
        column_col="keyword_label",
        value_col="contrast_delta",
        row_order=focus_labels,
        column_order=keywords,
    )
    path = output_dir / f"preferred_dispreferred_contrast_delta_keyword_focus_{_fraction_label(fraction)}.png"
    _heatmap(
        data,
        title=f"Focus keyword preferred-dispreferred contrast delta ({_fraction_label(fraction)})",
        path=path,
        annotate=True,
    )
    return path


def _plot_focus_pc_summary(
    *,
    focus: pd.DataFrame,
    output_dir: Path,
    fraction: float,
    focus_pcs: Sequence[int],
    top_keywords: int,
) -> Path:
    focus_labels = _focus_labels(focus_pcs)
    subset = focus[(focus["fraction"] == fraction) & (focus["pc"].isin(focus_labels))].copy()
    path = output_dir / f"focus_pc_summary_PC{'_PC'.join(str(pc) for pc in focus_pcs)}_{_fraction_label(fraction)}.png"
    n_pcs = len(focus_labels)
    fig, axes = plt.subplots(n_pcs, 1, figsize=(10, max(4.0, 3.2 * n_pcs)), squeeze=False)
    axes_flat = axes[:, 0]
    for ax, pc in zip(axes_flat, focus_labels):
        pc_rows = subset[subset["pc"] == pc].copy()
        if pc_rows.empty:
            ax.set_title(f"{pc}: no focus rows")
            ax.axis("off")
            continue
        pc_rows["keyword_label"] = pc_rows["keyword"] + " (" + pc_rows["category"] + ")"
        ranked = (
            pc_rows.assign(abs_delta=pc_rows["contrast_delta"].abs())
            .sort_values("abs_delta", ascending=False)
            .drop_duplicates("keyword_label")
            .head(top_keywords)
            .sort_values("contrast_delta")
        )
        colors = ["#4c78a8" if value >= 0 else "#e45756" for value in ranked["contrast_delta"]]
        ax.barh(ranked["keyword_label"], ranked["contrast_delta"], color=colors, alpha=0.9)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{pc} top keyword contrast deltas")
        ax.set_xlabel("contrast_delta")
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    _atomic_savefig(path, fig)
    return path


def _plot_pc_group_stats(
    *,
    group_stats: pd.DataFrame,
    output_dir: Path,
) -> Path:
    subset = group_stats[group_stats["group"].isin(GROUP_ORDER)].copy()
    subset["pc_num"] = subset["pc"].map(_pc_sort_key)
    subset = subset.sort_values(["pc_num", "group"])
    path = output_dir / "pc_group_stats_norm_textlen.png"
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for group in GROUP_ORDER:
        group_df = subset[subset["group"] == group].sort_values("pc_num")
        axes[0].plot(group_df["pc"], group_df["mean_vec_norm_mean"], marker="o", label=group)
        axes[1].plot(group_df["pc"], group_df["text_word_count_mean"], marker="o", label=group)
    axes[0].set_title("Mean vector norm by PC group")
    axes[0].set_ylabel("mean_vec_norm_mean")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("Text word count by PC group")
    axes[1].set_ylabel("text_word_count_mean")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    _atomic_savefig(path, fig)
    return path


def _plot_section_coverage(section_parse: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "section_coverage.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(section_parse["section"], section_parse["nonempty_rate"], color="#4c78a8")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("nonempty_rate")
    ax.set_title("Parsed section coverage")
    ax.grid(True, axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    _atomic_savefig(path, fig)
    return path


def _expected_output_paths(output_dir: Path, sections: Sequence[str], fraction: float, focus_pcs: Sequence[int]) -> List[Path]:
    frac_label = _fraction_label(fraction)
    paths = [
        output_dir / f"preferred_dispreferred_contrast_delta_category_{frac_label}.png",
        output_dir / f"keyword_high_low_heatmap_user_profile_{frac_label}_focus.png",
        output_dir / f"preferred_dispreferred_contrast_delta_keyword_focus_{frac_label}.png",
        output_dir / f"focus_pc_summary_PC{'_PC'.join(str(pc) for pc in focus_pcs)}_{frac_label}.png",
        output_dir / "pc_group_stats_norm_textlen.png",
        output_dir / "section_coverage.png",
        output_dir / "plot_row_pca_text_analysis_config.json",
    ]
    for section in sections:
        paths.append(output_dir / f"category_high_low_heatmap_{section}_{frac_label}.png")
    return paths


def main() -> None:
    args = parse_args()
    _validate_args(args)
    analysis_dir = _resolve_path(args.analysis_dir)
    output_dir = _resolve_path(args.output_dir) if args.output_dir is not None else analysis_dir / "plots"
    _check_required_files(analysis_dir)
    expected_paths = _expected_output_paths(output_dir, args.sections, args.fraction, args.focus_pcs)
    _check_no_existing(expected_paths, args.overwrite)

    started_at = _utc_now_iso()
    frames = _load_frames(analysis_dir)
    output_paths: List[Path] = []
    output_paths.extend(
        _plot_category_high_low_heatmaps(
            category_delta=frames["category_delta"],
            output_dir=output_dir,
            fraction=args.fraction,
            sections=args.sections,
        )
    )
    output_paths.append(
        _plot_category_contrast_heatmap(
            contrast_delta=frames["contrast_delta"],
            output_dir=output_dir,
            fraction=args.fraction,
        )
    )
    if "user_profile" in args.sections:
        output_paths.append(
            _plot_keyword_focus_heatmap(
                keyword_delta=frames["keyword_delta"],
                output_dir=output_dir,
                fraction=args.fraction,
                focus_pcs=args.focus_pcs,
                section="user_profile",
                top_keywords=args.top_keywords,
            )
        )
    output_paths.append(
        _plot_contrast_keyword_focus_heatmap(
            contrast_delta=frames["contrast_delta"],
            output_dir=output_dir,
            fraction=args.fraction,
            focus_pcs=args.focus_pcs,
            top_keywords=args.top_keywords,
        )
    )
    output_paths.append(
        _plot_focus_pc_summary(
            focus=frames["focus"],
            output_dir=output_dir,
            fraction=args.fraction,
            focus_pcs=args.focus_pcs,
            top_keywords=args.top_keywords,
        )
    )
    output_paths.append(_plot_pc_group_stats(group_stats=frames["group_stats"], output_dir=output_dir))
    output_paths.append(_plot_section_coverage(frames["section_parse"], output_dir=output_dir))

    config_path = output_dir / "plot_row_pca_text_analysis_config.json"
    config = {
        "argv": sys.argv,
        "python_executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now_iso(),
        "versions": _package_versions(),
        "args": {
            "analysis_dir": str(args.analysis_dir),
            "output_dir": str(args.output_dir) if args.output_dir is not None else None,
            "fraction": float(args.fraction),
            "sections": list(args.sections),
            "focus_pcs": [int(pc) for pc in args.focus_pcs],
            "top_keywords": int(args.top_keywords),
            "overwrite": bool(args.overwrite),
        },
        "resolved_paths": {
            "analysis_dir": str(analysis_dir),
            "output_dir": str(output_dir),
        },
        "outputs": [str(path) for path in output_paths],
    }
    _atomic_write_json(config_path, config)
    print(
        json.dumps(
            {
                "plot_row_pca_text_analysis_summary": {
                    "analysis_dir": str(analysis_dir),
                    "output_dir": str(output_dir),
                    "fraction": float(args.fraction),
                    "num_plots": len(output_paths),
                    "config_path": str(config_path),
                    "outputs": [str(path) for path in output_paths],
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
