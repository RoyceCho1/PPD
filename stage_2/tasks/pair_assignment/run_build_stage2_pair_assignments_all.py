from __future__ import annotations

"""Run `tasks/pair_assignment/build_stage2_pair_assignments.py` across all embedding shards.

This is a convenience batch runner for the Stage 2 prototype. It scans a
directory of Stage 1 user-embedding shard JSON files, matches each shard to the
corresponding split-level pair pool JSONL, and writes one assignment JSONL per
shard.

By default this runner does not create per-shard summary JSON files. It only
prints a final batch summary to stdout. Summary JSON writing can be enabled
explicitly with `--write-summary-json`.
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


VALID_SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run tasks/pair_assignment/build_stage2_pair_assignments.py over all embedding shards."
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path("data/user_emb_7b_full"),
        help="Directory containing Stage 1 embedding shard JSON files.",
    )
    parser.add_argument(
        "--pair-pool-dir",
        type=Path,
        default=Path("artifacts/pair_pool"),
        help="Directory containing pair_pool_<split>.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pair_assignments"),
        help="Directory to write assignment JSONL files.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=list(VALID_SPLITS),
        help="Subset of splits to process. Defaults to train validation test.",
    )
    parser.add_argument(
        "--max-query-pairs-per-embedding",
        type=int,
        default=1,
        help="Forwarded to tasks/pair_assignment/build_stage2_pair_assignments.py.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Forwarded to tasks/pair_assignment/build_stage2_pair_assignments.py.",
    )
    parser.add_argument(
        "--skip-if-no-query-pairs",
        action="store_true",
        help="Forwarded to tasks/pair_assignment/build_stage2_pair_assignments.py.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Forwarded to tasks/pair_assignment/build_stage2_pair_assignments.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing assignment JSONL files.",
    )
    parser.add_argument(
        "--write-summary-json",
        action="store_true",
        help="Also write per-shard summary JSON files next to each output JSONL.",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used to launch tasks/pair_assignment/build_stage2_pair_assignments.py.",
    )
    return parser.parse_args()


def _validate_splits(splits: Sequence[str]) -> List[str]:
    normalized = [str(split).strip() for split in splits if str(split).strip()]
    invalid = [split for split in normalized if split not in VALID_SPLITS]
    if invalid:
        raise ValueError(f"Unsupported split(s): {invalid}. Expected subset of {list(VALID_SPLITS)}")
    return normalized


def _collect_embedding_shards(embedding_dir: Path, allowed_splits: Sequence[str]) -> List[Path]:
    resolved = embedding_dir.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Embedding directory not found: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Embedding directory is not a directory: {resolved}")

    shards: List[Path] = []
    for split in allowed_splits:
        shards.extend(sorted(resolved.glob(f"{split}_shard*.json")))
    return sorted(path.resolve() for path in shards)


def _infer_split_from_shard(path: Path) -> str:
    stem = path.stem
    for split in VALID_SPLITS:
        prefix = f"{split}_shard"
        if stem.startswith(prefix):
            return split
    raise ValueError(f"Could not infer split from shard filename: {path}")


def _pair_pool_path(pair_pool_dir: Path, split: str) -> Path:
    return (pair_pool_dir.expanduser().resolve() / f"pair_pool_{split}.jsonl").resolve()


def _output_jsonl_path(output_dir: Path, shard_path: Path) -> Path:
    return (output_dir.expanduser().resolve() / f"stage2_pair_assignments_{shard_path.stem}.jsonl").resolve()


def _summary_json_path(output_jsonl_path: Path) -> Path:
    stem = output_jsonl_path.stem
    return output_jsonl_path.with_name(f"{stem}_summary.json")


def _build_command(
    python_executable: str,
    shard_path: Path,
    pair_pool_path: Path,
    output_jsonl_path: Path,
    max_query_pairs_per_embedding: int,
    random_seed: int,
    skip_if_no_query_pairs: bool,
    strict: bool,
    summary_json_path: Optional[Path],
) -> List[str]:
    command = [
        python_executable,
        "stage_2/tasks/pair_assignment/build_stage2_pair_assignments.py",
        "--user-embedding-json",
        str(shard_path),
        "--pair-pool-jsonl",
        str(pair_pool_path),
        "--output-jsonl",
        str(output_jsonl_path),
        "--max-query-pairs-per-embedding",
        str(max_query_pairs_per_embedding),
        "--random-seed",
        str(random_seed),
    ]
    if skip_if_no_query_pairs:
        command.append("--skip-if-no-query-pairs")
    if strict:
        command.append("--strict")
    if summary_json_path is not None:
        command.extend(["--summary-json", str(summary_json_path)])
    return command


def _run_one(command: Sequence[str]) -> Tuple[int, str, str]:
    result = subprocess.run(
        list(command),
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


def main() -> None:
    args = parse_args()
    allowed_splits = _validate_splits(args.splits)

    if args.max_query_pairs_per_embedding <= 0:
        raise ValueError("--max-query-pairs-per-embedding must be >= 1.")

    shards = _collect_embedding_shards(args.embedding_dir, allowed_splits)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_split_counts: Counter[str] = Counter()
    skipped_existing: List[str] = []
    failures: List[Dict[str, Any]] = []
    completed_outputs: List[str] = []

    for shard_path in shards:
        split = _infer_split_from_shard(shard_path)
        pair_pool_path = _pair_pool_path(args.pair_pool_dir, split)
        if not pair_pool_path.exists():
            raise FileNotFoundError(f"Pair pool for split={split} not found: {pair_pool_path}")

        output_jsonl_path = _output_jsonl_path(output_dir, shard_path)
        summary_json_path = _summary_json_path(output_jsonl_path) if args.write_summary_json else None

        if output_jsonl_path.exists() and not args.overwrite:
            skipped_existing.append(str(output_jsonl_path))
            continue

        command = _build_command(
            python_executable=args.python_executable,
            shard_path=shard_path,
            pair_pool_path=pair_pool_path,
            output_jsonl_path=output_jsonl_path,
            max_query_pairs_per_embedding=args.max_query_pairs_per_embedding,
            random_seed=args.random_seed,
            skip_if_no_query_pairs=args.skip_if_no_query_pairs,
            strict=args.strict,
            summary_json_path=summary_json_path,
        )

        print(f"[run_build_stage2_pair_assignments_all] processing {shard_path.name}")
        exit_code, stdout_text, stderr_text = _run_one(command)
        if stdout_text.strip():
            print(stdout_text.strip())
        if exit_code != 0:
            failures.append(
                {
                    "shard": str(shard_path),
                    "exit_code": exit_code,
                    "stderr": stderr_text.strip(),
                }
            )
            print(stderr_text.strip(), file=sys.stderr)
            continue

        per_split_counts[split] += 1
        completed_outputs.append(str(output_jsonl_path))

    summary = {
        "embedding_dir": str(args.embedding_dir.expanduser().resolve()),
        "pair_pool_dir": str(args.pair_pool_dir.expanduser().resolve()),
        "output_dir": str(output_dir),
        "splits": allowed_splits,
        "num_shards_found": len(shards),
        "num_outputs_written": len(completed_outputs),
        "num_skipped_existing": len(skipped_existing),
        "num_failures": len(failures),
        "max_query_pairs_per_embedding": args.max_query_pairs_per_embedding,
        "random_seed": args.random_seed,
        "skip_if_no_query_pairs": bool(args.skip_if_no_query_pairs),
        "strict": bool(args.strict),
        "write_summary_json": bool(args.write_summary_json),
        "per_split_counts": dict(per_split_counts),
        "skipped_existing_outputs": skipped_existing,
        "failures": failures,
    }

    print("[run_build_stage2_pair_assignments_all summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
