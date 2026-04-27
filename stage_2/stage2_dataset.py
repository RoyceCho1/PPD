from __future__ import annotations

import argparse
import json
import re
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset


PAIR_FIELD_PATTERN = re.compile(r"^(preferred_image_uid|dispreferred_image_uid|caption)_(\d+)$")
LATENT_TENSOR_KEYS = ("latent", "image_embeds", "tensor")


@dataclass
class _UserParseResult:
    user_id: Any
    user_profile_text: str
    user_emb: torch.FloatTensor
    pair_indices: List[int]


class Stage2PreferenceDataset(Dataset):

    def __init__(
        self,
        embedding_json_path: Union[str, Path],
        assignment_jsonl_path: Optional[Union[str, Path]] = None,
        latent_manifest_jsonl_path: Optional[Union[str, Path]] = None,
        uid_to_path_json_path: Optional[Union[str, Path]] = None,
        uid_to_meta_json_path: Optional[Union[str, Path]] = None,
        load_images: bool = False,
        load_latents: bool = False,
        skip_malformed_pairs: bool = True, # 정보가 누락된 pair의 경우 skip한다
        skip_missing_latents: bool = False,
        validate_assignment_support_pairs: bool = True,
        strict_emb_dim: bool = True, # user embedding의 마지막 차원이 3584가 아니면 에러를 발생시킨다
        expected_emb_dim: int = 3584,
        image_loader: Optional[Callable[[str], Any]] = None,
        preloaded_uid_to_path: Optional[Mapping[str, Any]] = None,
        preloaded_uid_to_meta: Optional[Mapping[str, Any]] = None,
        preloaded_latent_manifest: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        self.embedding_json_path = Path(embedding_json_path)
        self.assignment_jsonl_path = Path(assignment_jsonl_path) if assignment_jsonl_path else None
        self.latent_manifest_jsonl_path = (
            Path(latent_manifest_jsonl_path) if latent_manifest_jsonl_path else None
        )
        self.uid_to_path_json_path = Path(uid_to_path_json_path) if uid_to_path_json_path else None
        self.uid_to_meta_json_path = Path(uid_to_meta_json_path) if uid_to_meta_json_path else None
        self.load_images = load_images
        self.load_latents = load_latents
        self.skip_malformed_pairs = skip_malformed_pairs
        self.skip_missing_latents = skip_missing_latents
        self.validate_assignment_support_pairs = validate_assignment_support_pairs
        self.strict_emb_dim = strict_emb_dim
        self.expected_emb_dim = expected_emb_dim
        self.image_loader = image_loader or self._default_image_loader

        self._uid_to_path: Dict[str, str] = (
            preloaded_uid_to_path  # type: ignore[assignment]
            if preloaded_uid_to_path is not None
            else self._load_optional_mapping(self.uid_to_path_json_path)
        ) # dictionary 형태로 저장
        self._uid_to_meta: Dict[str, Any] = (
            preloaded_uid_to_meta  # type: ignore[assignment]
            if preloaded_uid_to_meta is not None
            else self._load_optional_mapping(self.uid_to_meta_json_path)
        ) # dictionary 형태로 저장
        self._latent_manifest: Dict[str, Dict[str, Any]] = (
            preloaded_latent_manifest  # type: ignore[assignment]
            if preloaded_latent_manifest is not None
            else self._load_optional_latent_manifest(self.latent_manifest_jsonl_path)
        )

        raw = self._load_json_file(self.embedding_json_path) # json 파일 로드
        self._user_records = self._normalize_records(raw) 

        self._stats_cache: Optional[Dict[str, Any]] = None # for get_stats()
        self._malformed_pairs_skipped = 0 # malformed pair의 개수
        self._malformed_users_skipped = 0 # malformed user의 개수
        self._missing_latents_skipped = 0
        self._assignment_validation_failures_skipped = 0
        self._pair_count_per_user: Counter[int] = Counter() # user별 pair의 개수
        self._num_assignment_records = 0
        self._num_query_samples = 0

        if self.load_latents and self.latent_manifest_jsonl_path is None:
            raise ValueError("load_latents=True requires latent_manifest_jsonl_path.")

        self.samples = self.build_samples() # sample 생성

    @staticmethod # self를 건드리지 않고, 독립적으로 작동할 수 있게 하는 데코레이터
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
        return str(key) # key값 string으로 변환

    def _load_optional_mapping(self, path: Optional[Path]) -> Dict[str, Any]:
        if path is None:
            return {}
        data = self._load_json_file(path)
        if not isinstance(data, Mapping):
            raise ValueError(f"Mapping JSON must be an object/dict: {path}")
        return {self._normalize_key(k): v for k, v in data.items()}

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Mapping[str, Any]]]:
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
                yield line_idx, record

    def _load_optional_latent_manifest(self, path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
        if path is None:
            return {}

        manifest: Dict[str, Dict[str, Any]] = {}
        for line_idx, record in self._iter_jsonl(path):
            uid_raw = record.get("uid")
            if self._is_missing_value(uid_raw):
                raise ValueError(f"latent manifest line {line_idx} is missing `uid`.")
            uid = self._normalize_key(uid_raw)
            if uid in manifest:
                raise ValueError(f"Duplicate uid in latent manifest: {uid}")
            manifest[uid] = dict(record)
        return manifest

    @staticmethod
    def _sort_mixed_keys(keys: Sequence[Any]) -> List[Any]:
        def _sort_key(x: Any) -> Tuple[int, Any]:
            sx = str(x)
            if sx.isdigit():
                return (0, int(sx))
            return (1, sx)

        return sorted(keys, key=_sort_key)

    # json 파일의 형태를 정제하는 함수
    def _normalize_records(self, raw: Any) -> List[Dict[str, Any]]:
        
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

    @staticmethod
    def _expected_user_embedding_id(user_id: Any, row_idx: int) -> str:
        return f"{user_id}_emb_{row_idx:06d}"

    def _extract_support_pairs_from_record(
        self,
        record: Mapping[str, Any],
        parsed: _UserParseResult,
    ) -> List[Dict[str, Any]]:
        support_pairs: List[Dict[str, Any]] = []

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
                raise ValueError(
                    "Malformed support pair detected "
                    f"(user_id={parsed.user_id}, pair_idx={k}). Missing/empty fields: {missing}"
                )

            support_pairs.append(
                {
                    "preferred_uid": self._normalize_key(record[pref_key]),
                    "dispreferred_uid": self._normalize_key(record[dispref_key]),
                    "caption": str(record[cap_key]),
                    "pair_idx": k,
                    "pair_key": f"support_{k}",
                }
            )

        return support_pairs

    def _normalize_pair_like(
        self,
        pair: Mapping[str, Any],
        *,
        default_pair_idx: Optional[int] = None,
        default_pair_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        required = ("preferred_uid", "dispreferred_uid", "caption")
        missing = [key for key in required if key not in pair or self._is_missing_value(pair.get(key))]
        if missing:
            raise ValueError(f"Pair object is missing required fields: {missing}")

        normalized: Dict[str, Any] = {
            "preferred_uid": self._normalize_key(pair["preferred_uid"]),
            "dispreferred_uid": self._normalize_key(pair["dispreferred_uid"]),
            "caption": str(pair["caption"]),
        }

        if "pair_idx" in pair and not self._is_missing_value(pair.get("pair_idx")):
            normalized["pair_idx"] = pair["pair_idx"]
        elif default_pair_idx is not None:
            normalized["pair_idx"] = default_pair_idx

        if "pair_key" in pair and not self._is_missing_value(pair.get("pair_key")):
            normalized["pair_key"] = str(pair["pair_key"])
        elif default_pair_key is not None:
            normalized["pair_key"] = default_pair_key

        if "source_row_idx" in pair and not self._is_missing_value(pair.get("source_row_idx")):
            normalized["source_row_idx"] = pair["source_row_idx"]
        if "user_id" in pair and not self._is_missing_value(pair.get("user_id")):
            normalized["user_id"] = pair["user_id"]

        return normalized

    @staticmethod
    def _pair_signature(pair: Mapping[str, Any]) -> Tuple[str, str, str]:
        return (
            str(pair["preferred_uid"]),
            str(pair["dispreferred_uid"]),
            str(pair["caption"]),
        )

    def _load_latent_tensor(self, latent_path: str) -> torch.FloatTensor:
        try:
            loaded = torch.load(latent_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load latent tensor: {latent_path} ({e})") from e

        tensor: Optional[torch.Tensor] = None
        if torch.is_tensor(loaded):
            tensor = loaded
        elif isinstance(loaded, Mapping):
            for key in LATENT_TENSOR_KEYS:
                candidate = loaded.get(key)
                if torch.is_tensor(candidate):
                    tensor = candidate
                    break

        if tensor is None:
            raise TypeError(
                f"Unsupported latent payload at {latent_path}: expected Tensor or mapping with one of {LATENT_TENSOR_KEYS}"
            )

        tensor = tensor.detach().float().cpu()
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor

    def _apply_optional_query_sidecars(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        preferred_uid = sample["preferred_uid"]
        dispreferred_uid = sample["dispreferred_uid"]

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
                raise ValueError(
                    "Image loading requested but path missing "
                    f"(preferred_uid={preferred_uid}, dispreferred_uid={dispreferred_uid})"
                )
            sample["preferred_image"] = self.image_loader(preferred_path)
            sample["dispreferred_image"] = self.image_loader(dispreferred_path)

        return sample

    def _apply_optional_latent_sidecars(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._latent_manifest:
            return sample

        preferred_uid = sample["preferred_uid"]
        dispreferred_uid = sample["dispreferred_uid"]
        preferred_manifest = self._latent_manifest.get(preferred_uid)
        dispreferred_manifest = self._latent_manifest.get(dispreferred_uid)

        if preferred_manifest is None or dispreferred_manifest is None:
            if self.skip_missing_latents:
                self._missing_latents_skipped += 1
                return None
            missing = []
            if preferred_manifest is None:
                missing.append(f"preferred_uid={preferred_uid}")
            if dispreferred_manifest is None:
                missing.append(f"dispreferred_uid={dispreferred_uid}")
            raise KeyError("Missing latent manifest entry for " + ", ".join(missing))

        preferred_latent_path = preferred_manifest.get("latent_path")
        dispreferred_latent_path = dispreferred_manifest.get("latent_path")
        if self._is_missing_value(preferred_latent_path) or self._is_missing_value(dispreferred_latent_path):
            if self.skip_missing_latents:
                self._missing_latents_skipped += 1
                return None
            raise ValueError(
                f"latent manifest record is missing latent_path for preferred_uid={preferred_uid} or "
                f"dispreferred_uid={dispreferred_uid}"
            )

        sample["preferred_latent_path"] = str(preferred_latent_path)
        sample["dispreferred_latent_path"] = str(dispreferred_latent_path)
        sample["preferred_latent_meta"] = preferred_manifest
        sample["dispreferred_latent_meta"] = dispreferred_manifest

        if self.load_latents:
            sample["preferred_latent"] = self._load_latent_tensor(str(preferred_latent_path))
            sample["dispreferred_latent"] = self._load_latent_tensor(str(dispreferred_latent_path))

        return sample

    def build_samples(self) -> List[Dict[str, Any]]:
        if self.assignment_jsonl_path is not None:
            return self._build_assignment_samples()
        return self._build_legacy_samples()

    def _build_legacy_samples(self) -> List[Dict[str, Any]]:
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

    def _build_assignment_samples(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        for _, assignment_record in self._iter_jsonl(self.assignment_jsonl_path):
            self._num_assignment_records += 1

            try:
                source_embedding_row_idx_raw = assignment_record.get("source_embedding_row_idx")
                if self._is_missing_value(source_embedding_row_idx_raw):
                    raise ValueError("Assignment record is missing `source_embedding_row_idx`.")
                source_embedding_row_idx = int(source_embedding_row_idx_raw)
                if not (0 <= source_embedding_row_idx < len(self._user_records)):
                    raise IndexError(
                        f"source_embedding_row_idx out of range: {source_embedding_row_idx} "
                        f"(num_embedding_rows={len(self._user_records)})"
                    )

                embedding_record = self._user_records[source_embedding_row_idx]
                if not isinstance(embedding_record, Mapping):
                    raise ValueError(f"Embedding row {source_embedding_row_idx} is not a mapping-like object.")

                parsed = self._parse_user_record(embedding_record, user_index=source_embedding_row_idx)
                assignment_user_id = assignment_record.get("user_id")
                if self._is_missing_value(assignment_user_id):
                    raise ValueError("Assignment record is missing `user_id`.")
                if self._normalize_key(assignment_user_id) != self._normalize_key(parsed.user_id):
                    raise ValueError(
                        f"user_id mismatch between assignment ({assignment_user_id}) and embedding row ({parsed.user_id}) "
                        f"for source_embedding_row_idx={source_embedding_row_idx}"
                    )

                expected_user_embedding_id = self._normalize_key(
                    embedding_record.get(
                        "user_embedding_id",
                        self._expected_user_embedding_id(parsed.user_id, source_embedding_row_idx),
                    )
                )
                assignment_user_embedding_id = self._normalize_key(
                    assignment_record.get("user_embedding_id", expected_user_embedding_id)
                )
                if assignment_user_embedding_id != expected_user_embedding_id:
                    raise ValueError(
                        f"user_embedding_id mismatch: assignment={assignment_user_embedding_id}, "
                        f"expected={expected_user_embedding_id}"
                    )

                expected_support_pairs = self._extract_support_pairs_from_record(embedding_record, parsed)
                assignment_support_pairs_raw = assignment_record.get("support_pairs")
                if not isinstance(assignment_support_pairs_raw, list):
                    raise ValueError("Assignment record `support_pairs` must be a list.")
                assignment_support_pairs = [
                    self._normalize_pair_like(
                        pair,
                        default_pair_idx=idx,
                        default_pair_key=f"support_{idx}",
                    )
                    for idx, pair in enumerate(assignment_support_pairs_raw)
                ]

                if self.validate_assignment_support_pairs:
                    expected_signatures = [self._pair_signature(pair) for pair in expected_support_pairs]
                    assignment_signatures = [self._pair_signature(pair) for pair in assignment_support_pairs]
                    if assignment_signatures != expected_signatures:
                        raise ValueError(
                            f"Support pair mismatch for user_embedding_id={assignment_user_embedding_id}"
                        )

                query_pairs_raw = assignment_record.get("query_pairs")
                if not isinstance(query_pairs_raw, list):
                    raise ValueError("Assignment record `query_pairs` must be a list.")

                valid_query_pairs_for_assignment = 0
                for query_idx, query_pair_raw in enumerate(query_pairs_raw):
                    if not isinstance(query_pair_raw, Mapping):
                        raise ValueError(f"query_pairs[{query_idx}] is not a mapping-like object.")
                    query_pair = self._normalize_pair_like(
                        query_pair_raw,
                        default_pair_idx=query_idx,
                        default_pair_key=f"query_{query_idx}",
                    )

                    sample: Dict[str, Any] = {
                        "user_embedding_id": assignment_user_embedding_id,
                        "user_id": parsed.user_id,
                        "source_embedding_row_idx": source_embedding_row_idx,
                        "user_profile_text": parsed.user_profile_text,
                        "user_emb": parsed.user_emb,
                        "caption": str(query_pair["caption"]),
                        "preferred_uid": query_pair["preferred_uid"],
                        "dispreferred_uid": query_pair["dispreferred_uid"],
                        "query_pair_idx": query_pair.get("pair_idx", query_idx),
                        "query_pair_key": query_pair.get("pair_key"),
                        "query_source_row_idx": query_pair.get("source_row_idx"),
                        "support_pairs": assignment_support_pairs,
                        "query_pair": query_pair,
                        "num_candidate_pairs_before_filter": assignment_record.get(
                            "num_candidate_pairs_before_filter"
                        ),
                        "num_candidate_pairs_after_filter": assignment_record.get(
                            "num_candidate_pairs_after_filter"
                        ),
                        "query_sampling_strategy": assignment_record.get("query_sampling_strategy"),
                    }

                    try:
                        sample = self._apply_optional_query_sidecars(sample)
                        sample = self._apply_optional_latent_sidecars(sample)
                    except Exception:
                        if self.skip_malformed_pairs:
                            if self._latent_manifest:
                                self._missing_latents_skipped += 1
                            else:
                                self._assignment_validation_failures_skipped += 1
                            continue
                        raise

                    if sample is None:
                        continue

                    samples.append(sample)
                    valid_query_pairs_for_assignment += 1
                    self._num_query_samples += 1

                self._pair_count_per_user[valid_query_pairs_for_assignment] += 1
            except Exception:
                if self.skip_malformed_pairs:
                    self._assignment_validation_failures_skipped += 1
                    continue
                raise

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
        if self.assignment_jsonl_path is not None:
            stats.update(
                {
                    "num_assignment_records": self._num_assignment_records,
                    "num_query_samples": self._num_query_samples,
                    "missing_latents_skipped": self._missing_latents_skipped,
                    "assignment_validation_failures_skipped": self._assignment_validation_failures_skipped,
                }
            )
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
            "user_embedding_id",
            "source_embedding_row_idx",
            "query_pair_idx",
            "query_pair_key",
            "query_source_row_idx",
            "support_pairs",
            "query_pair",
            "num_candidate_pairs_before_filter",
            "num_candidate_pairs_after_filter",
            "query_sampling_strategy",
            "preferred_latent_path",
            "dispreferred_latent_path",
            "preferred_latent_meta",
            "dispreferred_latent_meta",
        ]
        for key in optional_keys:
            if any(key in item for item in batch):
                out[key] = [item.get(key) for item in batch]

        for key in ("preferred_latent", "dispreferred_latent"):
            if not any(key in item for item in batch):
                continue

            values = [item.get(key) for item in batch]
            if all(torch.is_tensor(value) for value in values):
                first_shape = tuple(values[0].shape)  # type: ignore[index]
                for idx, value in enumerate(values[1:], start=1):
                    if tuple(value.shape) != first_shape:  # type: ignore[union-attr]
                        raise ValueError(
                            f"Inconsistent latent tensor shapes in batch for {key}: "
                            f"item0={first_shape}, item{idx}={tuple(value.shape)}"
                        )
                out[key] = torch.stack(values, dim=0)  # type: ignore[arg-type]
            else:
                out[key] = values

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
    parser.add_argument("--assignment-jsonl-path", type=str, default=None)
    parser.add_argument("--latent-manifest-jsonl-path", type=str, default=None)
    parser.add_argument("--uid-to-path-json-path", type=str, default=None)
    parser.add_argument("--uid-to-meta-json-path", type=str, default=None)
    parser.add_argument("--load-images", action="store_true")
    parser.add_argument("--load-latents", action="store_true")
    parser.add_argument(
        "--skip-missing-latents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to skip samples whose preferred/dispreferred latent lookup fails (default: False)",
    )
    parser.add_argument(
        "--validate-assignment-support-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to validate support_pairs from assignment JSONL against the Stage 1 embedding row (default: True)",
    )
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
        assignment_jsonl_path=args.assignment_jsonl_path,
        latent_manifest_jsonl_path=args.latent_manifest_jsonl_path,
        uid_to_path_json_path=args.uid_to_path_json_path,
        uid_to_meta_json_path=args.uid_to_meta_json_path,
        load_images=args.load_images,
        load_latents=args.load_latents,
        skip_missing_latents=args.skip_missing_latents,
        validate_assignment_support_pairs=args.validate_assignment_support_pairs,
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
