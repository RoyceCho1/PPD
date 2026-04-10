from __future__ import annotations

"""Build Stage 2 held-out query-pair assignments for Stage 1 user embeddings.

This script is a prototype data-assignment generator for PPD Stage 2.
It does not generate latents, does not build latent manifests, and does not run
training. Its job is only to decide which held-out query pair(s) should be
attached to each Stage 1 user-embedding row.

Inputs:
1. Stage 1 user-embedding JSON shard
2. Local preference pair pool JSON/JSONL

Output:
- `stage2_pair_assignments.jsonl` containing, for each user-embedding row,
  the support pairs stored in that row and one or more held-out query pairs
  sampled from the same user's pair pool after excluding the support pairs.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PAIR_FIELD_PATTERN = re.compile(r"^(preferred_image_uid|dispreferred_image_uid|caption)_(\d+)$")
EXPECTED_SUPPORT_PAIR_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage 2 support/query pair assignments from Stage 1 embedding JSON and a local pair pool."
    )
    parser.add_argument(
        "--user-embedding-json",
        type=Path,
        required=True,
        help="Path to a Stage 1 user embedding shard JSON.",
    )

    pair_pool_group = parser.add_mutually_exclusive_group(required=True)
    pair_pool_group.add_argument(
        "--pair-pool-jsonl",
        type=Path,
        default=None,
        help="Local pair pool JSONL path.",
    )
    pair_pool_group.add_argument(
        "--pair-pool-json",
        type=Path,
        default=None,
        help="Local pair pool JSON path.",
    )

    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Path to write stage2_pair_assignments.jsonl.",
    )
    parser.add_argument(
        "--max-query-pairs-per-embedding",
        type=int,
        default=1,
        help="Maximum number of held-out query pairs to attach to each embedding row.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for held-out pair sampling.",
    )
    parser.add_argument(
        "--skip-if-no-query-pairs",
        action="store_true",
        help="Skip embedding rows that have no held-out query pair after support-pair exclusion.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on malformed embedding rows or malformed pair-pool rows instead of skipping them.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to save the final summary JSON.",
    )
    return parser.parse_args()


def _load_json_file(path: Path) -> Any:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"JSON file not found: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON: {resolved} ({exc})") from exc


def _normalize_key(value: Any) -> str:
    return str(value)


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _sort_mixed_keys(keys: Sequence[Any]) -> List[Any]:
    def _sort_key(value: Any) -> Tuple[int, Any]:
        string_value = str(value)
        if string_value.isdigit():
            return (0, int(string_value))
        return (1, string_value)

    return sorted(keys, key=_sort_key)


def _normalize_records(raw: Any) -> List[Any]:
    if isinstance(raw, list):
        return [dict(item) if isinstance(item, Mapping) else item for item in raw]

    if not isinstance(raw, Mapping):
        raise ValueError("Unsupported JSON format: expected list of records or dict-based object.")

    value_types = [isinstance(value, (Mapping, list, tuple)) for value in raw.values()]
    if not any(value_types):
        return [dict(raw)]

    row_keys: set[Any] = set()
    has_columnar_signal = False

    for value in raw.values():
        if isinstance(value, Mapping):
            has_columnar_signal = True
            row_keys.update(value.keys())
        elif isinstance(value, (list, tuple)):
            has_columnar_signal = True
            row_keys.update(range(len(value)))

    if not has_columnar_signal:
        return [dict(raw)]

    if not row_keys:
        return []

    sorted_rows = _sort_mixed_keys(list(row_keys))
    records: List[Dict[str, Any]] = []

    for ridx in sorted_rows:
        record: Dict[str, Any] = {}
        ridx_int = int(ridx) if str(ridx).isdigit() else None
        ridx_str = str(ridx)

        for column, values in raw.items():
            if isinstance(values, Mapping):
                if ridx in values:
                    record[column] = values[ridx]
                elif ridx_str in values:
                    record[column] = values[ridx_str]
                elif ridx_int is not None and ridx_int in values:
                    record[column] = values[ridx_int]
            elif isinstance(values, (list, tuple)):
                if ridx_int is not None and 0 <= ridx_int < len(values):
                    record[column] = values[ridx_int]
            else:
                record[column] = values

        records.append(record)

    return records


def _normalize_pair_pool_json(raw: Any) -> List[Any]:
    if isinstance(raw, Mapping) and raw:
        values = list(raw.values())
        if all(isinstance(value, list) for value in values):
            if all(
                all(isinstance(item, Mapping) for item in value)
                for value in values
            ):
                flattened: List[Dict[str, Any]] = []
                for user_id, records in raw.items():
                    for record in records:
                        item = dict(record)
                        item.setdefault("user_id", user_id)
                        flattened.append(item)
                return flattened

    return _normalize_records(raw)


def _iter_jsonl_records(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"JSONL file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSONL line {line_idx} in {resolved}: {exc}") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"JSONL line {line_idx} in {resolved} is not an object/dict.")
            yield line_idx, dict(record)


def _pair_pool_path(args: argparse.Namespace) -> Path:
    return (args.pair_pool_jsonl or args.pair_pool_json).expanduser().resolve()


def _pair_signature_exact(preferred_uid: str, dispreferred_uid: str, caption: Optional[str]) -> Optional[Tuple[str, str, str]]:
    if caption is None:
        return None
    return (preferred_uid, dispreferred_uid, caption)


def _pair_signature_uids(preferred_uid: str, dispreferred_uid: str) -> Tuple[str, str]:
    return (preferred_uid, dispreferred_uid)


def _canonical_pair_keys(pair: Mapping[str, Any]) -> Tuple[Optional[Tuple[str, str, str]], Tuple[str, str]]:
    preferred_uid = _normalize_key(pair["preferred_uid"])
    dispreferred_uid = _normalize_key(pair["dispreferred_uid"])
    caption_raw = pair.get("caption")
    caption = None if _is_missing_value(caption_raw) else str(caption_raw)
    return (
        _pair_signature_exact(preferred_uid, dispreferred_uid, caption),
        _pair_signature_uids(preferred_uid, dispreferred_uid),
    )


def _pairs_match(candidate: Mapping[str, Any], support: Mapping[str, Any]) -> bool:
    candidate_exact, candidate_uids = _canonical_pair_keys(candidate)
    support_exact, support_uids = _canonical_pair_keys(support)

    if candidate_uids != support_uids:
        return False
    if candidate_exact is not None and support_exact is not None:
        return candidate_exact == support_exact
    return True


def _extract_pair_indices(record: Mapping[str, Any]) -> List[int]:
    pair_indices: set[int] = set()
    for key in record.keys():
        match = PAIR_FIELD_PATTERN.match(str(key))
        if match:
            pair_indices.add(int(match.group(2)))
    return sorted(pair_indices)


def _extract_support_pairs(record: Mapping[str, Any], row_idx: int) -> List[Dict[str, Any]]:
    pair_indices = _extract_pair_indices(record)
    support_pairs: List[Dict[str, Any]] = []

    for pair_idx in pair_indices:
        preferred_key = f"preferred_image_uid_{pair_idx}"
        dispreferred_key = f"dispreferred_image_uid_{pair_idx}"
        caption_key = f"caption_{pair_idx}"

        missing = [
            field_name
            for field_name, key in (
                ("preferred_uid", preferred_key),
                ("dispreferred_uid", dispreferred_key),
                ("caption", caption_key),
            )
            if key not in record or _is_missing_value(record.get(key))
        ]
        if missing:
            raise ValueError(
                f"Malformed support pair in embedding row {row_idx}: pair_idx={pair_idx}, missing={missing}"
            )

        pair: Dict[str, Any] = {
            "preferred_uid": _normalize_key(record[preferred_key]),
            "dispreferred_uid": _normalize_key(record[dispreferred_key]),
            "caption": str(record[caption_key]),
            "pair_idx": pair_idx,
            "pair_key": f"support_{pair_idx}",
        }

        for optional_key in (
            f"source_row_idx_{pair_idx}",
            f"source_row_index_{pair_idx}",
            f"source_idx_{pair_idx}",
            f"pair_source_row_idx_{pair_idx}",
        ):
            if optional_key in record and not _is_missing_value(record.get(optional_key)):
                pair["source_row_idx"] = record[optional_key]
                break

        support_pairs.append(pair)

    if len(support_pairs) != EXPECTED_SUPPORT_PAIR_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_SUPPORT_PAIR_COUNT} support pairs in embedding row {row_idx}, "
            f"but found {len(support_pairs)}."
        )

    return support_pairs


def _resolve_user_embedding_id(record: Mapping[str, Any], row_idx: int, user_id: str) -> str:
    for key in ("user_embedding_id", "embedding_id"):
        value = record.get(key)
        if not _is_missing_value(value):
            return str(value)
    return f"{user_id}_emb_{row_idx:06d}"


def _validate_embedding_row(record: Mapping[str, Any], row_idx: int) -> Tuple[str, str, List[Dict[str, Any]]]:
    if not isinstance(record, Mapping):
        raise ValueError(f"Embedding row {row_idx} is not an object/dict.")

    user_id_value = record.get("user_id")
    if _is_missing_value(user_id_value):
        raise ValueError(f"Embedding row {row_idx} is missing `user_id`.")
    user_id = _normalize_key(user_id_value)

    if "emb" not in record:
        raise ValueError(f"Embedding row {row_idx} is missing `emb`.")

    support_pairs = _extract_support_pairs(record, row_idx=row_idx)
    user_embedding_id = _resolve_user_embedding_id(record, row_idx=row_idx, user_id=user_id)
    return user_embedding_id, user_id, support_pairs


def _build_pool_pair_object(record: Mapping[str, Any], row_idx: int) -> Dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"Malformed pair-pool row {row_idx}: row is not an object/dict.")

    required_fields = ("user_id", "preferred_uid", "dispreferred_uid", "caption")
    missing = [field for field in required_fields if field not in record or _is_missing_value(record.get(field))]
    if missing:
        raise ValueError(f"Malformed pair-pool row {row_idx}: missing fields {missing}.")

    pair: Dict[str, Any] = {
        "user_id": _normalize_key(record["user_id"]),
        "preferred_uid": _normalize_key(record["preferred_uid"]),
        "dispreferred_uid": _normalize_key(record["dispreferred_uid"]),
        "caption": str(record["caption"]),
        "source_row_idx": row_idx,
    }

    if "pair_idx" in record and not _is_missing_value(record.get("pair_idx")):
        pair["pair_idx"] = record["pair_idx"]
    if "pair_key" in record and not _is_missing_value(record.get("pair_key")):
        pair["pair_key"] = str(record["pair_key"])
    elif "pair_idx" in pair:
        pair["pair_key"] = f"pair_{pair['pair_idx']}"

    return pair


def _index_pair_pool(
    args: argparse.Namespace,
) -> Tuple[Dict[str, List[Dict[str, Any]]], int, int, Counter[str]]:
    index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows_seen = 0
    rows_indexed = 0
    skip_reasons: Counter[str] = Counter()

    if args.pair_pool_jsonl is not None:
        resolved = args.pair_pool_jsonl.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"JSONL file not found: {resolved}")

        with resolved.open("r", encoding="utf-8") as handle:
            for row_idx, line in enumerate(handle):
                stripped = line.strip()
                if not stripped:
                    continue
                rows_seen += 1
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    skip_reasons["jsonl_parse_error"] += 1
                    if args.strict:
                        raise ValueError(
                            f"Failed to parse JSONL line {row_idx} in {resolved}: {exc}"
                        ) from exc
                    continue

                try:
                    pair = _build_pool_pair_object(record, row_idx=row_idx)
                except ValueError as exc:
                    skip_reasons[str(exc)] += 1
                    if args.strict:
                        raise
                    continue

                index[pair["user_id"]].append(pair)
                rows_indexed += 1
    else:
        raw = _load_json_file(args.pair_pool_json)
        normalized = _normalize_pair_pool_json(raw)
        for row_idx, record in enumerate(normalized):
            rows_seen += 1
            try:
                pair = _build_pool_pair_object(record, row_idx=row_idx)
            except ValueError as exc:
                skip_reasons[str(exc)] += 1
                if args.strict:
                    raise
                continue

            index[pair["user_id"]].append(pair)
            rows_indexed += 1

    return dict(index), rows_seen, rows_indexed, skip_reasons


def _filter_query_candidates(
    user_pairs: Sequence[Mapping[str, Any]],
    support_pairs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    seen_signatures: set[Tuple[str, str, str]] = set()

    for pair in user_pairs:
        if any(_pairs_match(pair, support_pair) for support_pair in support_pairs):
            continue

        exact_key, uid_key = _canonical_pair_keys(pair)
        dedupe_key = exact_key if exact_key is not None else (uid_key[0], uid_key[1], "")
        if dedupe_key in seen_signatures:
            continue
        seen_signatures.add(dedupe_key)
        filtered.append(dict(pair))

    return filtered


def _select_query_pairs(
    candidates: Sequence[Dict[str, Any]],
    max_query_pairs: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], str]:
    if len(candidates) <= max_query_pairs:
        return list(candidates), "all_candidates"
    return rng.sample(list(candidates), k=max_query_pairs), f"random_sample_without_replacement_k={max_query_pairs}"


def _load_embedding_rows(path: Path) -> List[Any]:
    raw = _load_json_file(path)
    return _normalize_records(raw)


def _write_jsonl(records: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    if args.max_query_pairs_per_embedding <= 0:
        raise ValueError("--max-query-pairs-per-embedding must be >= 1.")

    rng = random.Random(args.random_seed)
    embedding_rows = _load_embedding_rows(args.user_embedding_json)
    pair_pool_index, pair_pool_rows_seen, pair_pool_rows_indexed, pair_pool_skip_reasons = _index_pair_pool(args)

    assignment_records: List[Dict[str, Any]] = []
    embedding_skip_reasons: Counter[str] = Counter()
    num_skipped_no_query_pairs = 0
    num_skipped_malformed_embedding_rows = 0

    for row_idx, record in enumerate(embedding_rows):
        try:
            user_embedding_id, user_id, support_pairs = _validate_embedding_row(record, row_idx=row_idx)
        except ValueError as exc:
            embedding_skip_reasons[str(exc)] += 1
            num_skipped_malformed_embedding_rows += 1
            if args.strict:
                raise
            continue

        user_pairs = pair_pool_index.get(user_id, [])
        filtered_candidates = _filter_query_candidates(user_pairs, support_pairs)

        if len(filtered_candidates) == 0:
            if args.skip_if_no_query_pairs:
                num_skipped_no_query_pairs += 1
                embedding_skip_reasons["no_query_pairs_after_support_exclusion"] += 1
                continue
            raise ValueError(
                f"No held-out query pairs available for user_embedding_id={user_embedding_id}, "
                f"user_id={user_id}, source_embedding_row_idx={row_idx}."
            )

        query_pairs, strategy = _select_query_pairs(
            filtered_candidates,
            max_query_pairs=args.max_query_pairs_per_embedding,
            rng=rng,
        )

        assignment_records.append(
            {
                "user_embedding_id": user_embedding_id,
                "user_id": user_id,
                "support_pairs": support_pairs,
                "query_pairs": query_pairs,
                "num_candidate_pairs_before_filter": len(user_pairs),
                "num_candidate_pairs_after_filter": len(filtered_candidates),
                "query_sampling_strategy": strategy,
                "source_embedding_row_idx": row_idx,
            }
        )

    _write_jsonl(assignment_records, args.output_jsonl)

    summary: Dict[str, Any] = {
        "user_embedding_json": str(args.user_embedding_json.expanduser().resolve()),
        "pair_pool_path": str(_pair_pool_path(args)),
        "output_jsonl": str(args.output_jsonl.expanduser().resolve()),
        "num_embedding_rows_seen": len(embedding_rows),
        "num_assignment_records_written": len(assignment_records),
        "num_skipped_no_query_pairs": num_skipped_no_query_pairs,
        "num_skipped_malformed_embedding_rows": num_skipped_malformed_embedding_rows,
        "num_skipped_malformed_pair_pool_rows": pair_pool_rows_seen - pair_pool_rows_indexed,
        "max_query_pairs_per_embedding": args.max_query_pairs_per_embedding,
        "random_seed": args.random_seed,
        "pair_pool_rows_seen": pair_pool_rows_seen,
        "pair_pool_rows_indexed": pair_pool_rows_indexed,
        "strict": bool(args.strict),
        "skip_if_no_query_pairs": bool(args.skip_if_no_query_pairs),
        "embedding_skip_reason_counts": dict(embedding_skip_reasons),
        "pair_pool_skip_reason_counts": dict(pair_pool_skip_reasons),
    }

    if args.summary_json is not None:
        _write_summary_json(summary, args.summary_json)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
