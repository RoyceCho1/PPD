from __future__ import annotations

"""Build a tiny Stage 2 assignment JSONL for overfit and label-swap sanity checks."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.expanduser().open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _swap_pair(pair: Mapping[str, Any]) -> Dict[str, Any]:
    swapped = dict(pair)
    swapped["preferred_uid"] = pair.get("dispreferred_uid")
    swapped["dispreferred_uid"] = pair.get("preferred_uid")
    swapped["label_swapped"] = True
    return swapped


def _select_query_pairs(
    query_pairs: List[Mapping[str, Any]],
    *,
    max_query_pairs: int,
    max_captions: int,
    caption: Optional[str],
    swap_labels: bool,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    captions: List[str] = []
    for pair in query_pairs:
        pair_caption = str(pair.get("caption", ""))
        if caption is not None and pair_caption != caption:
            continue
        if max_captions > 0 and pair_caption not in captions:
            if len(captions) >= max_captions:
                continue
            captions.append(pair_caption)
        selected.append(_swap_pair(pair) if swap_labels else dict(pair))
        if len(selected) >= max_query_pairs:
            break
    return selected


def _repeat_query_pairs(query_pairs: List[Mapping[str, Any]], *, repeat: int) -> List[Dict[str, Any]]:
    if repeat <= 1:
        return [dict(pair) for pair in query_pairs]
    repeated: List[Dict[str, Any]] = []
    for repeat_idx in range(repeat):
        for pair_idx, pair in enumerate(query_pairs):
            row = dict(pair)
            original_pair_key = str(row.get("pair_key", f"query_{pair_idx}"))
            row["pair_key"] = f"{original_pair_key}__repeat_{repeat_idx:04d}"
            row["original_pair_key"] = original_pair_key
            row["repeat_idx"] = repeat_idx
            row["repeat_count"] = repeat
            repeated.append(row)
    return repeated


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a tiny Stage 2 assignment JSONL subset.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--user-embedding-id", type=str, default=None)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--max-query-pairs", type=int, default=8)
    parser.add_argument("--max-captions", type=int, default=2)
    parser.add_argument(
        "--repeat-selected-query-pairs",
        type=int,
        default=1,
        help="Repeat each selected query pair this many times inside the output JSONL.",
    )
    parser.add_argument("--caption", type=str, default=None)
    parser.add_argument("--swap-labels", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if int(args.max_query_pairs) < 1:
        raise ValueError("--max-query-pairs must be >= 1.")
    if int(args.repeat_selected_query_pairs) < 1:
        raise ValueError("--repeat-selected-query-pairs must be >= 1.")
    rows = _read_jsonl(args.input_jsonl)
    if not rows:
        raise ValueError(f"No rows in {args.input_jsonl}")

    source_row: Optional[Dict[str, Any]] = None
    if args.user_embedding_id is not None:
        for row in rows:
            if str(row.get("user_embedding_id")) == str(args.user_embedding_id):
                source_row = row
                break
        if source_row is None:
            raise ValueError(f"user_embedding_id not found: {args.user_embedding_id}")
    else:
        source_row = rows[int(args.row_index)]

    query_pairs = _select_query_pairs(
        list(source_row.get("query_pairs") or []),
        max_query_pairs=int(args.max_query_pairs),
        max_captions=int(args.max_captions),
        caption=args.caption,
        swap_labels=bool(args.swap_labels),
    )
    if not query_pairs:
        raise ValueError("No query pairs selected.")
    unique_query_pairs = list(query_pairs)
    query_pairs = _repeat_query_pairs(
        query_pairs,
        repeat=int(args.repeat_selected_query_pairs),
    )

    output_row = dict(source_row)
    output_row["query_pairs"] = query_pairs
    output_row["tiny_subset_source_jsonl"] = str(args.input_jsonl.expanduser().resolve())
    output_row["tiny_subset_max_query_pairs"] = int(args.max_query_pairs)
    output_row["tiny_subset_max_captions"] = int(args.max_captions)
    output_row["tiny_subset_unique_query_pairs"] = len(unique_query_pairs)
    output_row["tiny_subset_repeat_selected_query_pairs"] = int(args.repeat_selected_query_pairs)
    output_row["tiny_subset_caption_filter"] = args.caption
    output_row["label_swapped"] = bool(args.swap_labels)
    _write_jsonl(args.output_jsonl, [output_row])
    print(f"[build_stage2_tiny_assignment_subset] wrote {args.output_jsonl}")
    print(f"  user_embedding_id={output_row.get('user_embedding_id')}")
    print(f"  unique_query_pairs={len(unique_query_pairs)}")
    print(f"  repeated_query_pairs={len(query_pairs)}")
    print(f"  label_swapped={bool(args.swap_labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
