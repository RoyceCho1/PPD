from __future__ import annotations

"""Extract the unique image UID subset actually needed by Stage 2 assignments.

This utility reads one or more `stage2_pair_assignments*.jsonl` files and
collects the image UIDs that appear in assignment pair objects.

By default it reads only `query_pairs`. With `--include-support-pairs`, it also
includes UIDs from `support_pairs`.

It does not generate latents. It does not build manifests. It only extracts the
UID subset that is actually referenced by assignment files.
"""

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract unique preferred/dispreferred image UIDs from Stage 2 assignment JSONL files."
    )
    parser.add_argument(
        "--assignment-jsonl",
        type=Path,
        nargs="+",
        required=True,
        help="One or more stage2_pair_assignments*.jsonl files.",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        required=True,
        help="Write unique UIDs as one UID per line.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path containing the UID list and metadata.",
    )
    parser.add_argument(
        "--include-support-pairs",
        action="store_true",
        help="Also include UIDs from support_pairs.",
    )
    parser.add_argument(
        "--sort-uids",
        action="store_true",
        help="Sort UIDs before writing output files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise immediately on malformed lines, records, or pair objects.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to save the final summary JSON.",
    )
    return parser.parse_args()


def _normalize_uid(value: Any) -> str:
    uid = str(value).strip()
    if not uid:
        raise ValueError("UID is empty.")
    return uid


def iter_jsonl(path: Path) -> Iterator[Tuple[int, str]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Assignment JSONL not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            yield line_no, stripped


def _add_uid(uid: str, uid_order: List[str], uid_set: set[str]) -> None:
    if uid in uid_set:
        return
    uid_set.add(uid)
    uid_order.append(uid)


def extract_uids_from_pair_list(
    pair_list: Any,
    *,
    pair_list_name: str,
    strict: bool,
    counters: Counter[str],
    uid_order: List[str],
    uid_set: set[str],
    file_path: Path,
    line_no: int,
) -> int:
    if not isinstance(pair_list, list):
        message = (
            f"`{pair_list_name}` must be a list at file={file_path}, line={line_no}, "
            f"got type={type(pair_list)}"
        )
        if strict:
            raise ValueError(message)
        counters["num_skipped_malformed_records"] += 1
        return 0

    valid_pairs_seen = 0
    for pair_idx, pair_obj in enumerate(pair_list):
        if not isinstance(pair_obj, Mapping):
            message = (
                f"Pair object must be a dict at file={file_path}, line={line_no}, "
                f"pair_list={pair_list_name}, pair_idx={pair_idx}, got type={type(pair_obj)}"
            )
            if strict:
                raise ValueError(message)
            counters["num_skipped_malformed_pairs"] += 1
            continue

        try:
            preferred_uid = _normalize_uid(pair_obj["preferred_uid"])
            dispreferred_uid = _normalize_uid(pair_obj["dispreferred_uid"])
        except KeyError as exc:
            message = (
                f"Missing required UID field in pair object at file={file_path}, line={line_no}, "
                f"pair_list={pair_list_name}, pair_idx={pair_idx}, missing={exc}"
            )
            if strict:
                raise ValueError(message) from exc
            counters["num_skipped_malformed_pairs"] += 1
            continue
        except ValueError as exc:
            message = (
                f"Invalid UID value in pair object at file={file_path}, line={line_no}, "
                f"pair_list={pair_list_name}, pair_idx={pair_idx}: {exc}"
            )
            if strict:
                raise ValueError(message) from exc
            counters["num_skipped_malformed_pairs"] += 1
            continue

        valid_pairs_seen += 1
        _add_uid(preferred_uid, uid_order, uid_set)
        _add_uid(dispreferred_uid, uid_order, uid_set)

    return valid_pairs_seen


def write_uid_txt(output_path: Path, uids: Sequence[str]) -> None:
    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for uid in uids:
            handle.write(f"{uid}\n")


def write_uid_json(output_path: Path, uids: Sequence[str], num_assignment_files: int) -> None:
    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "uids": list(uids),
        "num_uids": len(uids),
        "num_assignment_files": num_assignment_files,
    }
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_summary_json(output_path: Path, summary: Mapping[str, Any]) -> None:
    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    assignment_files = [path.expanduser().resolve() for path in args.assignment_jsonl]
    uid_set: set[str] = set()
    uid_order: List[str] = []

    counters: Counter[str] = Counter()
    counters["num_records_seen"] = 0
    counters["num_query_pairs_seen"] = 0
    counters["num_support_pairs_seen"] = 0
    counters["num_skipped_malformed_lines"] = 0
    counters["num_skipped_malformed_records"] = 0
    counters["num_skipped_malformed_pairs"] = 0

    for assignment_path in assignment_files:
        for line_no, raw_line in iter_jsonl(assignment_path):
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                message = (
                    f"Failed to parse JSON line at file={assignment_path}, line={line_no}: {exc}"
                )
                if args.strict:
                    raise ValueError(message) from exc
                counters["num_skipped_malformed_lines"] += 1
                continue

            if not isinstance(record, Mapping):
                message = (
                    f"JSONL line must be an object/dict at file={assignment_path}, line={line_no}, "
                    f"got type={type(record)}"
                )
                if args.strict:
                    raise ValueError(message)
                counters["num_skipped_malformed_records"] += 1
                continue

            counters["num_records_seen"] += 1

            query_pairs = record.get("query_pairs")
            query_pairs_seen = extract_uids_from_pair_list(
                query_pairs,
                pair_list_name="query_pairs",
                strict=args.strict,
                counters=counters,
                uid_order=uid_order,
                uid_set=uid_set,
                file_path=assignment_path,
                line_no=line_no,
            )
            counters["num_query_pairs_seen"] += query_pairs_seen

            if args.include_support_pairs:
                support_pairs = record.get("support_pairs")
                support_pairs_seen = extract_uids_from_pair_list(
                    support_pairs,
                    pair_list_name="support_pairs",
                    strict=args.strict,
                    counters=counters,
                    uid_order=uid_order,
                    uid_set=uid_set,
                    file_path=assignment_path,
                    line_no=line_no,
                )
                counters["num_support_pairs_seen"] += support_pairs_seen

    output_uids = sorted(uid_order) if args.sort_uids else list(uid_order)

    write_uid_txt(args.output_txt, output_uids)
    if args.output_json is not None:
        write_uid_json(args.output_json, output_uids, num_assignment_files=len(assignment_files))

    summary: Dict[str, Any] = {
        "assignment_files": [str(path) for path in assignment_files],
        "num_assignment_files": len(assignment_files),
        "num_records_seen": counters["num_records_seen"],
        "num_query_pairs_seen": counters["num_query_pairs_seen"],
        "num_support_pairs_seen": counters["num_support_pairs_seen"],
        "num_unique_uids": len(output_uids),
        "include_support_pairs": bool(args.include_support_pairs),
        "output_txt": str(args.output_txt.expanduser().resolve()),
        "output_json": str(args.output_json.expanduser().resolve()) if args.output_json is not None else None,
        "num_skipped_malformed_lines": counters["num_skipped_malformed_lines"],
        "num_skipped_malformed_records": counters["num_skipped_malformed_records"],
        "num_skipped_malformed_pairs": counters["num_skipped_malformed_pairs"],
    }

    if args.summary_json is not None:
        write_summary_json(args.summary_json, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
