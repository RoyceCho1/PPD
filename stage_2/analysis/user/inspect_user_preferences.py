from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage_2.stage2_dataset import Stage2PreferenceDataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect one user's preference pairs from a Stage 1 embedding shard."
    )
    parser.add_argument("embedding_json_path", type=Path, help="Path to one Stage 1 embedding shard JSON")
    parser.add_argument(
        "--uid-to-path-json-path",
        type=Path,
        default=Path("data/uid_to_path.json"),
        help="Optional UID->path manifest JSON",
    )
    parser.add_argument(
        "--uid-to-meta-json-path",
        type=Path,
        default=Path("data/uid_to_meta.json"),
        help="Optional UID->meta manifest JSON",
    )
    parser.add_argument("--user-id", type=str, default=None, help="Exact user_id to inspect")
    parser.add_argument(
        "--user-index",
        type=int,
        default=None,
        help="0-based user record index inside the shard JSON",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional maximum number of preference pairs to print",
    )
    return parser


def _summarize_meta(meta: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if meta is None:
        return None
    out: Dict[str, Any] = {
        "num_occurrences": meta.get("num_occurrences"),
        "best_count": meta.get("best_count"),
        "non_best_count": meta.get("non_best_count"),
        "source_paths": meta.get("source_paths"),
    }
    split_counts = meta.get("split_counts")
    if isinstance(split_counts, Mapping):
        out["split_counts"] = dict(split_counts)
    return out


def _select_record(
    dataset: Stage2PreferenceDataset,
    user_id: Optional[str],
    user_index: Optional[int],
) -> tuple[int, Mapping[str, Any]]:
    records = dataset._user_records
    if not records:
        raise ValueError("No user records were found in the embedding JSON.")

    if user_index is not None:
        if user_index < 0 or user_index >= len(records):
            raise IndexError(f"user-index out of range: {user_index} (num_records={len(records)})")
        return user_index, records[user_index]

    if user_id is not None:
        for idx, record in enumerate(records):
            if str(record.get("user_id", idx)) == user_id:
                return idx, record
        raise ValueError(f"user_id '{user_id}' was not found in this shard.")

    return 0, records[0]


def _build_pair_rows(
    dataset: Stage2PreferenceDataset,
    record: Mapping[str, Any],
    user_index: int,
    max_pairs: Optional[int],
) -> Dict[str, Any]:
    parsed = dataset._parse_user_record(record, user_index=user_index)

    pairs: List[Dict[str, Any]] = []
    for pair_idx in parsed.pair_indices:
        pref_key = f"preferred_image_uid_{pair_idx}"
        dispref_key = f"dispreferred_image_uid_{pair_idx}"
        cap_key = f"caption_{pair_idx}"

        if pref_key not in record or dispref_key not in record or cap_key not in record:
            continue

        preferred_uid = str(record[pref_key])
        dispreferred_uid = str(record[dispref_key])

        pair_row: Dict[str, Any] = {
            "pair_idx": pair_idx,
            "caption": str(record[cap_key]),
            "preferred_uid": preferred_uid,
            "preferred_path": dataset.resolve_uid_to_path(preferred_uid),
            "dispreferred_uid": dispreferred_uid,
            "dispreferred_path": dataset.resolve_uid_to_path(dispreferred_uid),
        }

        preferred_meta = dataset._uid_to_meta.get(preferred_uid)
        dispreferred_meta = dataset._uid_to_meta.get(dispreferred_uid)
        if preferred_meta is not None:
            pair_row["preferred_meta"] = _summarize_meta(preferred_meta)
        if dispreferred_meta is not None:
            pair_row["dispreferred_meta"] = _summarize_meta(dispreferred_meta)

        pairs.append(pair_row)
        if max_pairs is not None and len(pairs) >= max_pairs:
            break

    return {
        "user_index": user_index,
        "user_id": parsed.user_id,
        "user_profile_text": parsed.user_profile_text,
        "user_emb_shape": list(parsed.user_emb.shape),
        "num_pairs_detected": len(parsed.pair_indices),
        "pairs_printed": len(pairs),
        "pairs": pairs,
    }


def main() -> None:
    args = _build_parser().parse_args()

    dataset = Stage2PreferenceDataset(
        embedding_json_path=args.embedding_json_path,
        uid_to_path_json_path=args.uid_to_path_json_path if args.uid_to_path_json_path.exists() else None,
        uid_to_meta_json_path=args.uid_to_meta_json_path if args.uid_to_meta_json_path.exists() else None,
        load_images=False,
        skip_malformed_pairs=True,
    )

    user_index, record = _select_record(dataset, user_id=args.user_id, user_index=args.user_index)
    summary = _build_pair_rows(dataset, record=record, user_index=user_index, max_pairs=args.max_pairs)

    print("[inspect_user_preferences]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
