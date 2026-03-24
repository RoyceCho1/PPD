from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset


PAIR_FIELD_PATTERN = re.compile(r"^(preferred_image_uid|dispreferred_image_uid|caption)_(\d+)$")


@dataclass
class _UserParseResult:
    user_id: Any
    user_profile_text: str
    user_emb: torch.FloatTensor
    pair_indices: List[int]


class Stage2PreferenceDataset(Dataset):
    """Dataset for Stage 2 DPO-style personalized diffusion training.

    This dataset treats Stage 1 embedding JSON as the primary source of truth and
    flattens user-level records into pair-level samples. Each sample includes both
    preference pair data and the corresponding user embedding.

    Args:
        embedding_json_path: Path to Stage 1 embedding JSON.
        uid_to_path_json_path: Optional JSON path mapping image UID -> image file path.
        uid_to_meta_json_path: Optional JSON path mapping image UID -> extra metadata.
        load_images: If True, tries to load image objects for preferred/dispreferred
            samples via image paths.
        skip_malformed_pairs: If True, malformed pairs are skipped. If False, raises
            a ValueError on malformed records/pairs.
        strict_emb_dim: If True, enforces user embedding last dim to be 3584.
        expected_emb_dim: Expected embedding dimension for strict checking.
        image_loader: Optional callable(path) -> image object.
    """

    def __init__(
        self,
        embedding_json_path: Union[str, Path],
        uid_to_path_json_path: Optional[Union[str, Path]] = None,
        uid_to_meta_json_path: Optional[Union[str, Path]] = None,
        load_images: bool = False,
        skip_malformed_pairs: bool = True,
        strict_emb_dim: bool = True,
        expected_emb_dim: int = 3584,
        image_loader: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.embedding_json_path = Path(embedding_json_path)
        self.uid_to_path_json_path = Path(uid_to_path_json_path) if uid_to_path_json_path else None
        self.uid_to_meta_json_path = Path(uid_to_meta_json_path) if uid_to_meta_json_path else None
        self.load_images = load_images
        self.skip_malformed_pairs = skip_malformed_pairs
        self.strict_emb_dim = strict_emb_dim
        self.expected_emb_dim = expected_emb_dim
        self.image_loader = image_loader or self._default_image_loader

        self._uid_to_path: Dict[str, str] = self._load_optional_mapping(self.uid_to_path_json_path)
        self._uid_to_meta: Dict[str, Any] = self._load_optional_mapping(self.uid_to_meta_json_path)

        raw = self._load_json_file(self.embedding_json_path)
        self._user_records = self._normalize_records(raw)

        self._stats_cache: Optional[Dict[str, Any]] = None
        self._malformed_pairs_skipped = 0
        self._malformed_users_skipped = 0
        self._pair_count_per_user: Counter[int] = Counter()

        self.samples = self.build_samples()

    @staticmethod
    def _load_json_file(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {path} ({e})") from e

    @staticmethod
    def _normalize_key(key: Any) -> str:
        return str(key)

    def _load_optional_mapping(self, path: Optional[Path]) -> Dict[str, Any]:
        if path is None:
            return {}
        data = self._load_json_file(path)
        if not isinstance(data, Mapping):
            raise ValueError(f"Mapping JSON must be an object/dict: {path}")
        return {self._normalize_key(k): v for k, v in data.items()}

    @staticmethod
    def _sort_mixed_keys(keys: Sequence[Any]) -> List[Any]:
        def _sort_key(x: Any) -> Tuple[int, Any]:
            sx = str(x)
            if sx.isdigit():
                return (0, int(sx))
            return (1, sx)

        return sorted(keys, key=_sort_key)

    def _normalize_records(self, raw: Any) -> List[Dict[str, Any]]:
        """Normalize Stage 1 JSON into a list of user-level records.

        Supports:
        - List[Dict[str, Any]] (record-oriented)
        - Dict[str, Dict[index, value]] / Dict[str, List[value]] (column-oriented)
        - Dict[str, Any] (single-record fallback)
        """
        if isinstance(raw, list):
            if not all(isinstance(x, Mapping) for x in raw):
                raise ValueError("Embedding JSON list must contain only objects/dicts")
            return [dict(x) for x in raw]

        if not isinstance(raw, Mapping):
            raise ValueError(
                "Unsupported embedding JSON format: expected list of records or dict-based object"
            )

        value_types = [isinstance(v, (Mapping, list, tuple)) for v in raw.values()]
        if not any(value_types):
            return [dict(raw)]

        row_keys: set = set()
        has_columnar_signal = False

        for v in raw.values():
            if isinstance(v, Mapping):
                has_columnar_signal = True
                row_keys.update(v.keys())
            elif isinstance(v, (list, tuple)):
                has_columnar_signal = True
                row_keys.update(range(len(v)))

        if not has_columnar_signal:
            return [dict(raw)]

        if not row_keys:
            return []

        sorted_rows = self._sort_mixed_keys(list(row_keys))
        records: List[Dict[str, Any]] = []

        for ridx in sorted_rows:
            rec: Dict[str, Any] = {}
            ridx_int = int(ridx) if str(ridx).isdigit() else None
            ridx_str = str(ridx)

            for col, values in raw.items():
                if isinstance(values, Mapping):
                    if ridx in values:
                        rec[col] = values[ridx]
                    elif ridx_str in values:
                        rec[col] = values[ridx_str]
                    elif ridx_int is not None and ridx_int in values:
                        rec[col] = values[ridx_int]
                elif isinstance(values, (list, tuple)):
                    if ridx_int is not None and 0 <= ridx_int < len(values):
                        rec[col] = values[ridx_int]
                else:
                    rec[col] = values

            records.append(rec)

        return records

    def _parse_user_record(self, record: Mapping[str, Any], user_index: int) -> _UserParseResult:
        user_id = record.get("user_id", user_index)

        user_profile_text = record.get("text")
        if user_profile_text is None:
            user_profile_text = record.get("user_profile_text", "")
        if not isinstance(user_profile_text, str):
            user_profile_text = str(user_profile_text)

        if "emb" not in record:
            raise ValueError(f"Missing 'emb' in user record index={user_index}, user_id={user_id}")

        emb_raw = record["emb"]
        try:
            user_emb = torch.as_tensor(emb_raw, dtype=torch.float32)
        except Exception as e:  # pragma: no cover - defensive
            raise ValueError(
                f"Failed to convert 'emb' to tensor for user_id={user_id}: {e}"
            ) from e

        if user_emb.ndim != 2:
            raise ValueError(
                f"Invalid emb shape for user_id={user_id}: expected 2D [L, D], got {tuple(user_emb.shape)}"
            )
        if self.strict_emb_dim and user_emb.shape[1] != self.expected_emb_dim:
            raise ValueError(
                f"Invalid emb dim for user_id={user_id}: expected D={self.expected_emb_dim}, got D={user_emb.shape[1]}"
            )

        pair_indices: set = set()
        for key in record.keys():
            m = PAIR_FIELD_PATTERN.match(str(key))
            if m:
                pair_indices.add(int(m.group(2)))

        return _UserParseResult(
            user_id=user_id,
            user_profile_text=user_profile_text,
            user_emb=user_emb,
            pair_indices=sorted(pair_indices),
        )

    @staticmethod
    def _is_missing_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        return False

    def build_samples(self) -> List[Dict[str, Any]]:
        """Build flattened pair-level samples from user-level records."""
        samples: List[Dict[str, Any]] = []

        for uidx, record in enumerate(self._user_records):
            if not isinstance(record, Mapping):
                if self.skip_malformed_pairs:
                    self._malformed_users_skipped += 1
                    continue
                raise ValueError(f"User record at index={uidx} is not a JSON object")

            try:
                parsed = self._parse_user_record(record, user_index=uidx)
            except ValueError:
                if self.skip_malformed_pairs:
                    self._malformed_users_skipped += 1
                    continue
                raise

            valid_pairs_for_user = 0
            for k in parsed.pair_indices:
                pref_key = f"preferred_image_uid_{k}"
                dispref_key = f"dispreferred_image_uid_{k}"
                cap_key = f"caption_{k}"

                missing = [
                    name
                    for name, key in (
                        ("preferred", pref_key),
                        ("dispreferred", dispref_key),
                        ("caption", cap_key),
                    )
                    if key not in record or self._is_missing_value(record.get(key))
                ]

                if missing:
                    if self.skip_malformed_pairs:
                        self._malformed_pairs_skipped += 1
                        continue
                    raise ValueError(
                        "Malformed pair detected "
                        f"(user_id={parsed.user_id}, pair_idx={k}). Missing/empty fields: {missing}"
                    )

                preferred_uid = self._normalize_key(record[pref_key])
                dispreferred_uid = self._normalize_key(record[dispref_key])
                caption = str(record[cap_key])

                sample: Dict[str, Any] = {
                    "user_id": parsed.user_id,
                    "user_profile_text": parsed.user_profile_text,
                    "user_emb": parsed.user_emb,
                    "caption": caption,
                    "preferred_uid": preferred_uid,
                    "dispreferred_uid": dispreferred_uid,
                    "pair_idx": k,
                }

                preferred_path = self.resolve_uid_to_path(preferred_uid)
                dispreferred_path = self.resolve_uid_to_path(dispreferred_uid)

                if preferred_path is not None:
                    sample["preferred_path"] = preferred_path
                if dispreferred_path is not None:
                    sample["dispreferred_path"] = dispreferred_path

                preferred_meta = self._uid_to_meta.get(preferred_uid)
                dispreferred_meta = self._uid_to_meta.get(dispreferred_uid)
                if preferred_meta is not None:
                    sample["preferred_meta"] = preferred_meta
                if dispreferred_meta is not None:
                    sample["dispreferred_meta"] = dispreferred_meta

                if self.load_images:
                    if preferred_path is None or dispreferred_path is None:
                        msg = (
                            "Image loading requested but path missing "
                            f"(user_id={parsed.user_id}, pair_idx={k}, "
                            f"preferred_uid={preferred_uid}, dispreferred_uid={dispreferred_uid})"
                        )
                        if self.skip_malformed_pairs:
                            self._malformed_pairs_skipped += 1
                            continue
                        raise ValueError(msg)
                    sample["preferred_image"] = self.image_loader(preferred_path)
                    sample["dispreferred_image"] = self.image_loader(dispreferred_path)

                samples.append(sample)
                valid_pairs_for_user += 1

            self._pair_count_per_user[valid_pairs_for_user] += 1

        return samples

    def resolve_uid_to_path(self, uid: Union[str, int]) -> Optional[str]:
        """Resolve image UID into file path if uid->path mapping is available."""
        return self._uid_to_path.get(self._normalize_key(uid))

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset-level statistics for debugging and monitoring."""
        if self._stats_cache is not None:
            return self._stats_cache

        emb_lengths: List[int] = [int(s["user_emb"].shape[0]) for s in self.samples]
        unique_users = {s["user_id"] for s in self.samples}

        stats: Dict[str, Any] = {
            "num_samples": len(self.samples),
            "num_users": len(unique_users),
            "embedding_length_min": min(emb_lengths) if emb_lengths else None,
            "embedding_length_max": max(emb_lengths) if emb_lengths else None,
            "pair_count_distribution": dict(sorted(self._pair_count_per_user.items())),
            "malformed_pairs_skipped": self._malformed_pairs_skipped,
            "malformed_users_skipped": self._malformed_users_skipped,
        }
        self._stats_cache = stats
        return stats

    @staticmethod
    def collate_fn(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Pad variable-length user embeddings and collate batch-level fields.

        Returns:
            dict with at least:
            - user_emb: FloatTensor [B, L_max, D]
            - user_emb_attention_mask: LongTensor [B, L_max]
            - captions: list[str]
            - UID/path/meta fields as lists
        """
        if len(batch) == 0:
            raise ValueError("Empty batch passed to collate_fn")

        user_emb_list = [torch.as_tensor(item["user_emb"], dtype=torch.float32) for item in batch]
        for i, emb in enumerate(user_emb_list):
            if emb.ndim != 2:
                raise ValueError(f"Batch item {i} has invalid user_emb shape: {tuple(emb.shape)}")

        batch_size = len(user_emb_list)
        max_len = max(emb.shape[0] for emb in user_emb_list)
        emb_dim = user_emb_list[0].shape[1]

        padded = torch.zeros((batch_size, max_len, emb_dim), dtype=torch.float32)
        attn_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, emb in enumerate(user_emb_list):
            seq_len = emb.shape[0]
            if emb.shape[1] != emb_dim:
                raise ValueError(
                    f"Inconsistent user_emb dimension in batch: item0 D={emb_dim}, item{i} D={emb.shape[1]}"
                )
            padded[i, :seq_len] = emb
            attn_mask[i, :seq_len] = 1

        out: Dict[str, Any] = {
            "user_id": [item["user_id"] for item in batch],
            "caption": [str(item["caption"]) for item in batch],
            "user_profile_text": [str(item.get("user_profile_text", "")) for item in batch],
            "user_emb": padded,
            "user_emb_attention_mask": attn_mask,
            "preferred_uid": [item["preferred_uid"] for item in batch],
            "dispreferred_uid": [item["dispreferred_uid"] for item in batch],
        }

        optional_keys = [
            "preferred_path",
            "dispreferred_path",
            "preferred_meta",
            "dispreferred_meta",
            "pair_idx",
            "preferred_image",
            "dispreferred_image",
        ]
        for key in optional_keys:
            if any(key in item for item in batch):
                out[key] = [item.get(key) for item in batch]

        return out

    @staticmethod
    def _default_image_loader(path: str) -> Any:
        try:
            from PIL import Image
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "Pillow is required for load_images=True. Install via `pip install Pillow`."
            ) from e
        return Image.open(path).convert("RGB")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        # Return shallow copy so downstream code can modify fields safely.
        return dict(sample)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage2PreferenceDataset quick sanity test")
    parser.add_argument("embedding_json_path", type=str, help="Path to Stage1 embedding JSON")
    parser.add_argument("--uid-to-path-json-path", type=str, default=None)
    parser.add_argument("--uid-to-meta-json-path", type=str, default=None)
    parser.add_argument("--load-images", action="store_true")
    parser.add_argument(
        "--skip-malformed-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to skip malformed pairs (default: True)",
    )
    parser.add_argument("--index", type=int, default=0, help="Sample index to print")
    return parser.parse_args()


def _summarize_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())[:8]})"
    return value


if __name__ == "__main__":
    args = _parse_args()

    dataset = Stage2PreferenceDataset(
        embedding_json_path=args.embedding_json_path,
        uid_to_path_json_path=args.uid_to_path_json_path,
        uid_to_meta_json_path=args.uid_to_meta_json_path,
        load_images=args.load_images,
        skip_malformed_pairs=args.skip_malformed_pairs,
    )

    print("[Stage2PreferenceDataset stats]")
    print(json.dumps(dataset.get_stats(), indent=2, ensure_ascii=False))

    if len(dataset) == 0:
        print("\nNo valid samples found.")
    else:
        index = max(0, min(args.index, len(dataset) - 1))
        sample = dataset[index]
        summarized = {k: _summarize_value(v) for k, v in sample.items()}

        print(f"\n[Sample #{index}]")
        print(json.dumps(summarized, indent=2, ensure_ascii=False))
