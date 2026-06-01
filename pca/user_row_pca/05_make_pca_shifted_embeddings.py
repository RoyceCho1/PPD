#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


INSTALL_COMMAND = (
    "conda activate ppd_pca && "
    "conda install -c conda-forge -y numpy pandas pyarrow scikit-learn matplotlib joblib tqdm"
)
REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "joblib": "joblib",
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


EXPECTED_EMB_DIM = 3584
PCA_KIND_CHOICES = ("raw", "l2")
SELECTION_GROUP_CHOICES = ("low", "mid", "high")

# Hypothesis notes from Phase 3 analysis (not fixed truths).
DEFAULT_SHIFT_SPECS: Tuple[Tuple[int, Tuple[float, ...], str], ...] = (
    (5, (1.0, 2.0), "warm/bright/soft/dramatic/colorful/intricate/texture"),
    (8, (-1.0, -2.0), "warm/vibrant/colorful/detailed/intricate/stylized/depth"),
    (4, (1.0,), "exploratory aesthetic axis with mixed pref-dispref contrast"),
)


@dataclass(frozen=True)
class ShiftSpec:
    pc: int
    pc_index_zero_based: int
    alpha: float
    semantic_note: str


@dataclass(frozen=True)
class RealizedShiftSpec:
    pc: int
    pc_index_zero_based: int
    alpha: float
    score_std: float
    delta_l2_norm: float
    pca_kind: str
    semantic_note: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build PCA-shifted row-embedding run scaffolding with strict validation. "
            "Default shift specs (if no PC/alpha override): PC5 +1,+2; PC8 -1,-2; PC4 +1."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--pca-path", type=Path, required=True)
    parser.add_argument("--pc-stats-path", type=Path, required=True)
    parser.add_argument("--scores-path", type=Path, required=True)
    parser.add_argument("--meta-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--pca-kind", choices=PCA_KIND_CHOICES, required=True)

    parser.add_argument("--assignment-jsonl", type=Path, default=None)
    parser.add_argument("--assignment-jsonl-glob", default=None)
    parser.add_argument(
        "--smoke-query-from-support",
        action="store_true",
        help=(
            "Only bypass for missing real assignment input. Marks outputs smoke-only and "
            "not evaluation-valid."
        ),
    )

    parser.add_argument("--pc", type=int, action="append", default=None)
    parser.add_argument("--pcs", type=int, nargs="+", default=None)
    parser.add_argument("--alpha", type=float, action="append", default=None)
    parser.add_argument("--alphas", type=float, nargs="+", default=None)

    parser.add_argument("--selection-pcs", type=int, nargs="+", default=None)
    parser.add_argument(
        "--selection-groups",
        nargs="+",
        choices=SELECTION_GROUP_CHOICES,
        default=None,
    )
    parser.add_argument("--num-rows-per-group", type=int, default=None)
    parser.add_argument("--row-ids-file", type=Path, default=None)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _must_exist_file(path: Path, flag_name: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{flag_name} must be an existing file: {path}")


def _must_exist_dir(path: Path, flag_name: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{flag_name} must be an existing directory: {path}")


def _normalize_ints(first: Optional[List[int]], second: Optional[List[int]]) -> List[int]:
    values = list(first or []) + list(second or [])
    return values


def _normalize_floats(first: Optional[List[float]], second: Optional[List[float]]) -> List[float]:
    values = list(first or []) + list(second or [])
    return values


def _validate_assignment_inputs(args: argparse.Namespace) -> Tuple[str, bool, List[Path]]:
    has_single = args.assignment_jsonl is not None
    has_glob = bool(args.assignment_jsonl_glob)
    smoke_only = False

    if has_single and has_glob:
        raise ValueError(
            "Provide exactly one of --assignment-jsonl or --assignment-jsonl-glob (not both)."
        )

    if not has_single and not has_glob:
        if not args.smoke_query_from_support:
            raise ValueError(
                "Normal mode requires exactly one of --assignment-jsonl or "
                "--assignment-jsonl-glob. The only bypass is explicit "
                "--smoke-query-from-support."
            )
        smoke_only = True
        return "smoke_query_from_support", smoke_only, []

    if has_single:
        assignment_path = _resolve(args.assignment_jsonl)
        _must_exist_file(assignment_path, "--assignment-jsonl")
        return "assignment_jsonl", smoke_only, [assignment_path]

    matched = sorted(Path(path_str) for path_str in glob.glob(args.assignment_jsonl_glob))
    if not matched:
        raise FileNotFoundError(
            f"--assignment-jsonl-glob matched no files: {args.assignment_jsonl_glob}"
        )
    resolved = [_resolve(path) for path in matched]
    for path in resolved:
        _must_exist_file(path, "--assignment-jsonl-glob")
    return "assignment_jsonl_glob", smoke_only, resolved


def _validate_selection_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    if args.row_ids_file is not None:
        row_ids_file = _resolve(args.row_ids_file)
        _must_exist_file(row_ids_file, "--row-ids-file")
        return {
            "mode": "manual",
            "row_ids_file": str(row_ids_file),
            "selection_pcs": [],
            "selection_groups": [],
            "num_rows_per_group": None,
        }

    if not args.selection_pcs:
        raise ValueError("Automatic selection mode requires --selection-pcs.")
    if not args.selection_groups:
        raise ValueError("Automatic selection mode requires --selection-groups.")
    if args.num_rows_per_group is None:
        raise ValueError("Automatic selection mode requires --num-rows-per-group.")
    if args.num_rows_per_group <= 0:
        raise ValueError(
            f"--num-rows-per-group must be > 0, got {args.num_rows_per_group}"
        )

    selection_pcs = sorted(set(int(pc) for pc in args.selection_pcs))
    for pc in selection_pcs:
        if pc <= 0:
            raise ValueError(f"--selection-pcs values must be positive 1-based PCs, got {pc}")

    selection_groups = list(dict.fromkeys(args.selection_groups))
    return {
        "mode": "automatic",
        "row_ids_file": None,
        "selection_pcs": selection_pcs,
        "selection_groups": selection_groups,
        "num_rows_per_group": int(args.num_rows_per_group),
    }


def _detect_kind_in_filename(path: Path) -> Optional[str]:
    lowered = path.name.lower()
    if "_l2" in lowered or "l2_" in lowered:
        return "l2"
    if "_raw" in lowered or "raw_" in lowered:
        return "raw"
    return None


def _validate_pca_kind_scaffolding(args: argparse.Namespace) -> Dict[str, Any]:
    inferred_model_kind = _detect_kind_in_filename(args.pca_path)
    inferred_stats_kind = _detect_kind_in_filename(args.pc_stats_path)
    if inferred_model_kind and inferred_model_kind != args.pca_kind:
        raise ValueError(
            "--pca-kind does not match --pca-path filename hint: "
            f"kind={args.pca_kind}, inferred={inferred_model_kind}, path={args.pca_path}"
        )
    if inferred_stats_kind and inferred_stats_kind != args.pca_kind:
        raise ValueError(
            "--pca-kind does not match --pc-stats-path filename hint: "
            f"kind={args.pca_kind}, inferred={inferred_stats_kind}, path={args.pc_stats_path}"
        )

    with args.pc_stats_path.open("r", encoding="utf-8") as handle:
        stats_payload = json.load(handle)

    payload_kind = stats_payload.get("pca_kind") or stats_payload.get("kind")
    if payload_kind is not None and str(payload_kind) != args.pca_kind:
        raise ValueError(
            "--pca-kind does not match kind declared in stats JSON: "
            f"kind={args.pca_kind}, stats_kind={payload_kind}"
        )

    score_std = stats_payload.get("score_std")
    if not isinstance(score_std, list) or not score_std:
        raise ValueError(
            "PC stats JSON must contain non-empty list field 'score_std' for validation scaffolding."
        )

    score_std_values: List[float] = []
    for idx, value in enumerate(score_std):
        try:
            score_std_values.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"PC stats score_std[{idx}] must be numeric, got value={value!r}"
            ) from exc

    return {
        "stats_declared_kind": payload_kind,
        "stats_num_components": len(score_std_values),
        "score_std": score_std_values,
    }


def _looks_like_pca_object(value: Any) -> bool:
    return hasattr(value, "components_")


def _extract_components_array(candidate: Any, source_label: str) -> Any:
    if _looks_like_pca_object(candidate):
        return candidate.components_
    if isinstance(candidate, Mapping):
        if "pca" in candidate and _looks_like_pca_object(candidate["pca"]):
            return candidate["pca"].components_
        if "components" in candidate:
            return candidate["components"]
        if "components_" in candidate:
            return candidate["components_"]
        pca_like_keys = [key for key, value in candidate.items() if _looks_like_pca_object(value)]
        if len(pca_like_keys) == 1:
            return candidate[pca_like_keys[0]].components_
        if len(pca_like_keys) > 1:
            raise ValueError(
                f"Ambiguous PCA-like objects in payload {source_label}; found keys={pca_like_keys}."
            )
    raise ValueError(
        "Could not extract PCA components from payload. Supported formats: sklearn PCA object, "
        "mapping with key 'pca', mapping with key 'components'/'components_', or mapping with a "
        f"single PCA-like object. source={source_label}"
    )


def _require_numeric_array(name: str, values: Sequence[Any], *, expected_ndim: int, np_mod: Any) -> Any:
    array = np_mod.asarray(values, dtype=np_mod.float64)
    if array.ndim != expected_ndim:
        raise ValueError(
            f"{name} must be {expected_ndim}D numeric array, got shape={tuple(array.shape)}"
        )
    if not np_mod.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/inf values.")
    return array


def _load_pca_components_and_realize_shift_specs(
    *,
    pca_path: Path,
    pca_kind: str,
    shift_specs: Sequence[ShiftSpec],
    score_std_values: Sequence[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    _require_dependencies()
    import joblib  # type: ignore
    import numpy as np  # type: ignore

    payload = joblib.load(pca_path)
    components_values = _extract_components_array(payload, source_label=str(pca_path))
    components = _require_numeric_array(
        "PCA components",
        components_values,
        expected_ndim=2,
        np_mod=np,
    )
    if int(components.shape[1]) != EXPECTED_EMB_DIM:
        raise ValueError(
            "PCA components feature dimension mismatch: "
            f"expected={EXPECTED_EMB_DIM}, actual={components.shape[1]}, pca_path={pca_path}"
        )

    score_std = _require_numeric_array(
        "score_std",
        list(score_std_values),
        expected_ndim=1,
        np_mod=np,
    )

    max_index_from_components = int(components.shape[0]) - 1
    max_index_from_score_std = int(score_std.shape[0]) - 1
    max_usable_index = min(max_index_from_components, max_index_from_score_std)

    realized_specs: List[Dict[str, Any]] = []
    for spec in shift_specs:
        pc_index = int(spec.pc_index_zero_based)
        if pc_index < 0 or pc_index > max_usable_index:
            raise ValueError(
                "Requested PC is out of range for PCA components/score_std: "
                f"pc={spec.pc}, zero_based={pc_index}, components_count={components.shape[0]}, "
                f"score_std_count={score_std.shape[0]}"
            )
        std_k = float(score_std[pc_index])
        delta = float(spec.alpha) * std_k * components[pc_index]
        delta_l2 = float(np.linalg.norm(delta))
        realized = RealizedShiftSpec(
            pc=int(spec.pc),
            pc_index_zero_based=pc_index,
            alpha=float(spec.alpha),
            score_std=std_k,
            delta_l2_norm=delta_l2,
            pca_kind=pca_kind,
            semantic_note=spec.semantic_note,
        )
        realized_specs.append(
            {
                "pc": realized.pc,
                "pc_index_zero_based": realized.pc_index_zero_based,
                "alpha": realized.alpha,
                "score_std": realized.score_std,
                "delta_l2_norm": realized.delta_l2_norm,
                "pca_kind": realized.pca_kind,
                "semantic_note": realized.semantic_note,
                "delta_formula": "delta = alpha * score_std[k] * component[k]",
                "delta_scaling_note": "No additional scaling or normalization applied.",
            }
        )

    return realized_specs, {
        "pca_path": str(pca_path),
        "components_shape": [int(components.shape[0]), int(components.shape[1])],
        "expected_embedding_dim": EXPECTED_EMB_DIM,
        "score_std_length": int(score_std.shape[0]),
    }


def _build_shift_specs(args: argparse.Namespace) -> Tuple[List[ShiftSpec], List[str], bool]:
    pcs = _normalize_ints(args.pc, args.pcs)
    alphas = _normalize_floats(args.alpha, args.alphas)

    used_defaults = False
    raw_specs: List[Tuple[int, float, str]] = []
    if not pcs and not alphas:
        used_defaults = True
        for pc, default_alphas, note in DEFAULT_SHIFT_SPECS:
            for alpha in default_alphas:
                raw_specs.append((pc, float(alpha), note))
    else:
        if not pcs:
            raise ValueError("PC override requires at least one --pc or --pcs value.")
        if not alphas:
            raise ValueError("Alpha override requires at least one --alpha or --alphas value.")
        for pc in pcs:
            for alpha in alphas:
                raw_specs.append((int(pc), float(alpha), "custom"))

    warnings: List[str] = []
    shift_specs: List[ShiftSpec] = []
    for pc, alpha, note in raw_specs:
        if pc <= 0:
            raise ValueError(f"PC values must be positive 1-based integers, got {pc}")
        if abs(alpha) < 1e-12:
            warnings.append(
                f"Skipped PC{pc} alpha=0.0; no-shift behavior is represented by __original baseline rows."
            )
            continue
        shift_specs.append(
            ShiftSpec(pc=pc, pc_index_zero_based=pc - 1, alpha=alpha, semantic_note=note)
        )

    if not shift_specs:
        raise ValueError(
            "No non-zero shift specs remain after validation. Provide at least one non-zero alpha."
        )
    return shift_specs, warnings, used_defaults


def _output_run_dir(output_root: Path, run_name: str) -> Path:
    return output_root / "shifted_embeddings" / run_name


def _validate_output_guard(run_dir: Path, overwrite: bool) -> None:
    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            "Output run directory already exists. Pass --overwrite to replace: "
            f"{run_dir}"
        )


def _validate_args(args: argparse.Namespace) -> Dict[str, Any]:
    args.input_root = _resolve(args.input_root)
    args.pca_path = _resolve(args.pca_path)
    args.pc_stats_path = _resolve(args.pc_stats_path)
    args.scores_path = _resolve(args.scores_path)
    args.meta_path = _resolve(args.meta_path)
    args.output_root = _resolve(args.output_root)

    if not args.run_name.strip():
        raise ValueError("--run-name must be non-empty.")

    _must_exist_dir(args.input_root, "--input-root")
    _must_exist_file(args.pca_path, "--pca-path")
    _must_exist_file(args.pc_stats_path, "--pc-stats-path")
    _must_exist_file(args.scores_path, "--scores-path")
    _must_exist_file(args.meta_path, "--meta-path")

    assignment_mode, smoke_only, assignment_paths = _validate_assignment_inputs(args)
    selection = _validate_selection_inputs(args)
    kind_meta = _validate_pca_kind_scaffolding(args)
    shift_specs, shift_warnings, used_defaults = _build_shift_specs(args)
    realized_shift_specs, pca_payload_meta = _load_pca_components_and_realize_shift_specs(
        pca_path=args.pca_path,
        pca_kind=args.pca_kind,
        shift_specs=shift_specs,
        score_std_values=kind_meta["score_std"],
    )

    run_dir = _output_run_dir(args.output_root, args.run_name)
    _validate_output_guard(run_dir, args.overwrite)

    return {
        "assignment_mode": assignment_mode,
        "smoke_only": smoke_only,
        "assignment_paths": [str(p) for p in assignment_paths],
        "selection": selection,
        "kind_meta": kind_meta,
        "shift_specs": [
            {
                "pc": spec.pc,
                "pc_index_zero_based": spec.pc_index_zero_based,
                "alpha": spec.alpha,
                "semantic_note": spec.semantic_note,
            }
            for spec in shift_specs
        ],
        "realized_shift_specs": realized_shift_specs,
        "pca_payload_meta": pca_payload_meta,
        "warnings": shift_warnings,
        "used_default_shift_specs": used_defaults,
        "expected_embedding_dim": EXPECTED_EMB_DIM,
        "run_dir": str(run_dir),
    }


def _make_summary_scaffold(args: argparse.Namespace, validated: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp_utc": _utc_now_iso(),
        "script": "05_make_pca_shifted_embeddings.py",
        "status": "validated_scaffold_only",
        "pca_kind": args.pca_kind,
        "expected_embedding_dim": EXPECTED_EMB_DIM,
        "smoke_only": bool(validated["smoke_only"]),
        "smoke_query_from_support": bool(args.smoke_query_from_support),
        "used_default_shift_specs": bool(validated["used_default_shift_specs"]),
        "num_shift_specs": len(validated["shift_specs"]),
        "num_realized_shift_specs": len(validated["realized_shift_specs"]),
        "realized_shift_specs": list(validated["realized_shift_specs"]),
        "pca_payload_meta": dict(validated["pca_payload_meta"]),
        "output_run_dir": validated["run_dir"],
        "warnings": list(validated["warnings"]),
        "next_tasks": [
            "T3: row selection and source loading",
            "T4+: artifact generation",
        ],
    }


def main() -> None:
    args = parse_args()
    validated = _validate_args(args)
    summary = _make_summary_scaffold(args, validated)

    print("[OK] CLI validation scaffold complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
