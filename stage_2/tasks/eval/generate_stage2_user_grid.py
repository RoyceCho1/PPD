from __future__ import annotations

"""Generate support/query/generation grids from a Stage 2 checkpoint.

This script is intentionally separate from the training loop. It loads a saved
Stage 2 user-conditioning checkpoint, samples the Stable Cascade prior for a
small fixed set of validation users, decodes the resulting image embeddings, and
saves visual grids that show the support images used for the user embedding next
to the generated result.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE2_DIR = REPO_ROOT / "stage_2"
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from infer_stage2 import (  # noqa: E402
    CONDITIONS,
    _decode_one,
    _finite_flags,
    _load_pipeline,
    _prepare_prior_pipeline,
    _pipeline_dtype,
    _resolve_device,
    _run_prior_condition,
    _tensor_diagnostics,
)
from stage2_dataset import Stage2PreferenceDataset  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("artifacts/stage2_generation_grids")
DEFAULT_EMBEDDING_JSON_PATH = Path("data/user_emb_7b_full/validation_shard0.json")
DEFAULT_ASSIGNMENT_JSONL_PATH = Path(
    "artifacts/pair_assignments/validation/stage2_pair_assignments_validation_shard0.jsonl"
)
DEFAULT_UID_TO_PATH_JSON_PATH = Path("data/validation_uid_to_path.json")
DEFAULT_PRIOR_MODEL_ID = "stabilityai/stable-cascade-prior"
DEFAULT_DECODER_MODEL_ID = "stabilityai/stable-cascade"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if torch.is_tensor(value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _scale_label(scale: float) -> str:
    return f"{float(scale):.6g}".replace("-", "m").replace(".", "p")


def _load_json_mapping(path: Path) -> Dict[str, str]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping JSON at {resolved}")
    return {str(key): str(value) for key, value in payload.items()}


def _torch_load_checkpoint(path: Path) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        loaded = torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(resolved, map_location="cpu")
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Checkpoint must be a mapping: {resolved}")
    return loaded


def _apply_checkpoint_arg_defaults(args: argparse.Namespace) -> None:
    checkpoint = _torch_load_checkpoint(args.checkpoint_path)
    critical = checkpoint.get("critical_config")
    if not isinstance(critical, Mapping):
        return
    if args.user_scale is None and critical.get("user_scale") is not None:
        args.user_scale = float(critical["user_scale"])
        print(f"[generate_stage2_user_grid] using checkpoint user_scale={args.user_scale}")


def _make_run_dir(output_root: Path, run_name: Optional[str]) -> Path:
    stamp = run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root.expanduser().resolve() / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_dataset(args: argparse.Namespace) -> Stage2PreferenceDataset:
    return Stage2PreferenceDataset(
        embedding_json_path=args.embedding_json_path,
        assignment_jsonl_path=args.assignment_jsonl_path,
        uid_to_path_json_path=args.uid_to_path_json_path,
        load_images=False,
        load_latents=False,
        skip_malformed_pairs=False,
        validate_assignment_support_pairs=bool(args.validate_assignment_support_pairs),
    )


def _select_user_query_samples(
    dataset: Stage2PreferenceDataset,
    *,
    num_users: int,
    queries_per_user: int,
    user_embedding_ids: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    wanted = {str(item) for item in user_embedding_ids} if user_embedding_ids else None
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for sample in dataset.samples:
        user_embedding_id = str(sample.get("user_embedding_id", ""))
        if not user_embedding_id:
            continue
        if wanted is not None and user_embedding_id not in wanted:
            continue
        if user_embedding_id not in grouped:
            grouped[user_embedding_id] = []
            order.append(user_embedding_id)
        if len(grouped[user_embedding_id]) < queries_per_user:
            grouped[user_embedding_id].append(dict(sample))

    selected: List[Dict[str, Any]] = []
    if wanted is None:
        chosen_order = _evenly_spaced_items(order, max_items=num_users)
    else:
        chosen_order = [user_embedding_id for user_embedding_id in order if user_embedding_id in wanted]

    for user_embedding_id in chosen_order:
        selected.extend(grouped[user_embedding_id][:queries_per_user])

    if not selected:
        raise ValueError("No user/query samples selected.")
    return selected


def _evenly_spaced_items(items: Sequence[str], *, max_items: int) -> List[str]:
    if max_items <= 0:
        return []
    if len(items) <= max_items:
        return list(items)
    # For len=100 and max_items=4 this gives indices 0, 25, 50, 75.
    indices = [min(len(items) - 1, int(idx * len(items) / max_items)) for idx in range(max_items)]
    chosen: List[str] = []
    seen = set()
    for index in indices:
        item = items[index]
        if item in seen:
            continue
        chosen.append(item)
        seen.add(item)
    return chosen


def _image_path_for_uid(uid_to_path: Mapping[str, str], uid: Any) -> Optional[Path]:
    raw = uid_to_path.get(str(uid))
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_image_or_placeholder(path: Optional[Path], label: str, size: int) -> Any:
    from PIL import Image, ImageDraw

    if path is not None and path.exists():
        return Image.open(path).convert("RGB")
    image = Image.new("RGB", (size, size), color=(238, 238, 238))
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), "missing", fill=(80, 80, 80))
    draw.text((12, 32), label[:42], fill=(80, 80, 80))
    return image


def _fit_image(image: Any, *, cell_size: int, label: str, label_height: int) -> Any:
    from PIL import Image, ImageDraw, ImageOps

    canvas = Image.new("RGB", (cell_size, cell_size + label_height), color=(255, 255, 255))
    fitted = ImageOps.contain(image.convert("RGB"), (cell_size, cell_size))
    x = (cell_size - fitted.width) // 2
    y = label_height + (cell_size - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    draw = ImageDraw.Draw(canvas)
    text = str(label)
    if len(text) > 38:
        text = text[:35] + "..."
    draw.text((8, 8), text, fill=(0, 0, 0))
    return canvas


def _wrap_text(text: str, width: int) -> List[str]:
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _save_visual_grid(
    *,
    output_path: Path,
    sample: Mapping[str, Any],
    uid_to_path: Mapping[str, str],
    generated: Mapping[str, Path],
    support_pairs_limit: int,
    cell_size: int,
) -> None:
    from PIL import Image, ImageDraw

    rows: List[List[Tuple[Any, str]]] = []
    support_pairs = list(sample.get("support_pairs") or [])[:support_pairs_limit]
    for idx, pair in enumerate(support_pairs):
        pref_path = _image_path_for_uid(uid_to_path, pair.get("preferred_uid"))
        dispref_path = _image_path_for_uid(uid_to_path, pair.get("dispreferred_uid"))
        rows.append(
            [
                (_load_image_or_placeholder(pref_path, f"support {idx} preferred", cell_size), f"support {idx} preferred"),
                (_load_image_or_placeholder(dispref_path, f"support {idx} dispreferred", cell_size), f"support {idx} dispreferred"),
            ]
        )

    query_row: List[Tuple[Any, str]] = []
    query_row.append(
        (
            _load_image_or_placeholder(_image_path_for_uid(uid_to_path, sample.get("preferred_uid")), "query preferred", cell_size),
            "query preferred",
        )
    )
    query_row.append(
        (
            _load_image_or_placeholder(
                _image_path_for_uid(uid_to_path, sample.get("dispreferred_uid")),
                "query dispreferred",
                cell_size,
            ),
            "query dispreferred",
        )
    )
    for condition, path in generated.items():
        query_row.append((_load_image_or_placeholder(path, f"generated {condition}", cell_size), f"generated {condition}"))
    rows.append(query_row)

    num_cols = max(len(row) for row in rows)
    label_height = 34
    caption_lines = _wrap_text(str(sample.get("caption", "")), width=max(42, num_cols * 34))
    caption_height = 24 + 18 * min(len(caption_lines), 4)
    grid_width = num_cols * cell_size
    row_height = cell_size + label_height
    grid_height = caption_height + len(rows) * row_height
    grid = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    header = f"user={sample.get('user_embedding_id')} query={sample.get('query_pair_key')}"
    draw.text((8, 6), header[:120], fill=(0, 0, 0))
    for line_idx, line in enumerate(caption_lines[:4]):
        draw.text((8, 26 + line_idx * 18), line, fill=(40, 40, 40))

    y = caption_height
    for row in rows:
        for col_idx, (image, label) in enumerate(row):
            cell = _fit_image(image, cell_size=cell_size, label=str(label), label_height=label_height)
            grid.paste(cell, (col_idx * cell_size, y))
        y += row_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def _sample_user_tensors(sample: Mapping[str, Any], device: torch.device) -> Tuple[Tensor, Tensor]:
    user_emb = torch.as_tensor(sample["user_emb"], dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.ones((1, int(user_emb.shape[1])), dtype=torch.long, device=device)
    return user_emb, mask


def _condition_scale_plan(args: argparse.Namespace) -> List[Tuple[str, Optional[float], str]]:
    scales = [float(item) for item in (args.inference_user_scale_sweep or [args.inference_user_scale])]
    plan: List[Tuple[str, Optional[float], str]] = []
    for condition in args.condition:
        if condition in ("base", "branch_off"):
            plan.append((str(condition), None, str(condition)))
            continue
        for scale in scales:
            key = str(condition)
            if args.inference_user_scale_sweep is not None:
                key = f"{condition}_scale_{_scale_label(scale)}"
            plan.append((str(condition), float(scale), key))
    return plan


def _collect_user_scale_stats(prior: Any) -> Dict[str, Any]:
    values: List[float] = []
    rows: List[Dict[str, Any]] = []
    for name, param in prior.named_parameters():
        if not (name == "user_scale" or name.endswith(".user_scale")):
            continue
        value = float(param.detach().float().cpu().item())
        values.append(value)
        rows.append({"name": name, "value": value})
    if not values:
        return {"count": 0, "rows": []}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": float(sum(values) / len(values)),
        "mean_abs": float(sum(abs(item) for item in values) / len(values)),
        "rows": rows,
    }


def _run_prior_generation(
    *,
    args: argparse.Namespace,
    samples: Sequence[Mapping[str, Any]],
    run_dir: Path,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prior_pipe, compatibility = _prepare_prior_pipeline(args, device)
    compatibility["checkpoint_user_scale_stats"] = _collect_user_scale_stats(prior_pipe.prior)
    embeddings_dir = run_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    condition_plan = _condition_scale_plan(args)

    for sample_idx, sample in enumerate(samples):
        user_emb, user_mask = _sample_user_tensors(sample, device)
        sample_id = f"sample_{sample_idx:04d}_{sample.get('user_embedding_id')}_{sample.get('query_pair_key')}"
        sample_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in sample_id)
        condition_records: Dict[str, Dict[str, Any]] = {}
        for condition, scale, condition_key in condition_plan:
            if scale is not None:
                args._active_inference_user_scale = float(scale)
            elif hasattr(args, "_active_inference_user_scale"):
                delattr(args, "_active_inference_user_scale")
            embeddings, residual_summary = _run_prior_condition(
                pipe=prior_pipe,
                prompt=str(sample["caption"]),
                condition=str(condition),
                user_emb=user_emb,
                user_mask=user_mask,
                args=args,
                device=device,
                seed=int(args.seed),
            )
            has_nan, has_inf, all_finite = _finite_flags(embeddings)
            if not all_finite:
                raise RuntimeError(
                    f"Non-finite prior embeddings for {sample_id}/{condition_key}: nan={has_nan}, inf={has_inf}"
                )
            tensor_path = embeddings_dir / f"{sample_id}_{condition_key}.pt"
            torch.save(embeddings.detach().cpu(), tensor_path)
            condition_records[str(condition_key)] = {
                "condition": str(condition),
                "inference_user_scale": None if scale is None else float(scale),
                "embedding_path": str(tensor_path.relative_to(run_dir)),
                "embedding_diagnostics": _tensor_diagnostics(embeddings),
                "residual_summary": residual_summary,
            }
        records.append(
            {
                "sample_id": sample_id,
                "user_embedding_id": sample.get("user_embedding_id"),
                "user_id": sample.get("user_id"),
                "query_pair_key": sample.get("query_pair_key"),
                "caption": sample.get("caption"),
                "preferred_uid": sample.get("preferred_uid"),
                "dispreferred_uid": sample.get("dispreferred_uid"),
                "support_pairs": sample.get("support_pairs"),
                "conditions": condition_records,
            }
        )

    del prior_pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records, compatibility


def _decode_generated_images(
    *,
    args: argparse.Namespace,
    records: Sequence[Mapping[str, Any]],
    samples_by_id: Mapping[str, Mapping[str, Any]],
    uid_to_path: Mapping[str, str],
    run_dir: Path,
    device: torch.device,
) -> List[Dict[str, Any]]:
    if args.decode_mode == "none":
        return [dict(record) for record in records]

    decoder_pipe = _load_pipeline("StableCascadeDecoderPipeline", args.decoder_model_id, args, device)
    decoder_dtype = _pipeline_dtype(decoder_pipe)
    images_dir = run_dir / "images"
    grids_dir = run_dir / "grids"
    images_dir.mkdir(parents=True, exist_ok=True)
    grids_dir.mkdir(parents=True, exist_ok=True)

    decoded_records: List[Dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        sample = samples_by_id[sample_id]
        generated_paths: Dict[str, Path] = {}
        condition_records = {str(key): dict(value) for key, value in dict(record["conditions"]).items()}
        for condition, condition_record in condition_records.items():
            embedding_path = run_dir / str(condition_record["embedding_path"])
            image_embeddings = torch.load(embedding_path, map_location="cpu")
            if not torch.is_tensor(image_embeddings):
                raise TypeError(f"Expected tensor at {embedding_path}")
            image_embeddings = image_embeddings.to(device=device, dtype=decoder_dtype)
            image = _decode_one(
                decoder_pipe=decoder_pipe,
                image_embeddings=image_embeddings,
                prompt=str(record["caption"]),
                seed=int(args.seed),
                args=args,
                device=device,
                num_inference_steps=int(args.decoder_steps),
            )
            image_path = images_dir / f"{sample_id}_{condition}.png"
            image.save(image_path)
            condition_record["decoded_image_path"] = str(image_path.relative_to(run_dir))
            generated_paths[condition] = image_path
            condition_records[condition] = condition_record

        grid_path: Optional[Path] = None
        if bool(args.save_grid):
            grid_path = grids_dir / f"{sample_id}_grid.png"
            _save_visual_grid(
                output_path=grid_path,
                sample=sample,
                uid_to_path=uid_to_path,
                generated=generated_paths,
                support_pairs_limit=int(args.max_support_pairs),
                cell_size=int(args.grid_cell_size),
            )

        decoded = dict(record)
        decoded["conditions"] = condition_records
        if grid_path is not None:
            decoded["grid_path"] = str(grid_path.relative_to(run_dir))
        decoded_records.append(decoded)

    del decoder_pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return decoded_records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Stage 2 user support/query/output grids from a checkpoint.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--embedding-json-path", type=Path, default=DEFAULT_EMBEDDING_JSON_PATH)
    parser.add_argument("--assignment-jsonl-path", type=Path, default=DEFAULT_ASSIGNMENT_JSONL_PATH)
    parser.add_argument("--uid-to-path-json-path", type=Path, default=DEFAULT_UID_TO_PATH_JSON_PATH)
    parser.add_argument("--prior-model-id", type=str, default=DEFAULT_PRIOR_MODEL_ID)
    parser.add_argument("--decoder-model-id", type=str, default=DEFAULT_DECODER_MODEL_ID)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--patch-path", action="append", default=None)
    parser.add_argument(
        "--user-scale",
        type=float,
        default=None,
        help="Patch-time user_scale. Defaults to checkpoint critical_config.user_scale.",
    )
    parser.add_argument("--user-projection-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--user-projection-norm-affine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--user-adapter-projection-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--user-adapter-zero-init-out", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inference-user-scale", type=float, default=1.0)
    parser.add_argument(
        "--inference-user-scale-sweep",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Run user-branch conditions across multiple inference-only scales. "
            "Base/branch_off are generated once because they do not use the user branch."
        ),
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--prior-steps", type=int, default=20)
    parser.add_argument("--prior-guidance-scale", type=float, default=4.0)
    parser.add_argument("--decoder-steps", type=int, default=20)
    parser.add_argument("--decoder-guidance-scale", type=float, default=0.0)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--queries-per-user", type=int, default=1)
    parser.add_argument("--user-embedding-id", action="append", default=None)
    parser.add_argument("--condition", action="append", choices=CONDITIONS, default=None)
    parser.add_argument("--decode-mode", choices=("decoder", "none"), default="decoder")
    parser.add_argument("--save-grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-support-pairs", type=int, default=5)
    parser.add_argument("--grid-cell-size", type=int, default=256)
    parser.add_argument("--validate-assignment-support-pairs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _apply_checkpoint_arg_defaults(args)
    if args.user_scale is None:
        args.user_scale = 1.0
    if args.condition is None:
        args.condition = ["real_user"]
    if int(args.num_users) < 1:
        raise ValueError("--num-users must be >= 1")
    if int(args.queries_per_user) < 1:
        raise ValueError("--queries-per-user must be >= 1")

    run_dir = _make_run_dir(args.output_dir, args.run_name)
    device = _resolve_device(args.device)
    dataset = _build_dataset(args)
    uid_to_path = _load_json_mapping(args.uid_to_path_json_path)
    selected_samples = _select_user_query_samples(
        dataset,
        num_users=int(args.num_users),
        queries_per_user=int(args.queries_per_user),
        user_embedding_ids=args.user_embedding_id,
    )

    print(f"[generate_stage2_user_grid] selected samples: {len(selected_samples)}")
    print(f"[generate_stage2_user_grid] run dir: {run_dir}")
    records, compatibility = _run_prior_generation(args=args, samples=selected_samples, run_dir=run_dir, device=device)
    samples_by_id = {str(record["sample_id"]): sample for record, sample in zip(records, selected_samples)}
    decoded_records = _decode_generated_images(
        args=args,
        records=records,
        samples_by_id=samples_by_id,
        uid_to_path=uid_to_path,
        run_dir=run_dir,
        device=device,
    )

    summary = {
        "mode": "stage2_user_generation_grid",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime": {
            "device": str(device),
            "torch_dtype": args.torch_dtype,
            "height": int(args.height),
            "width": int(args.width),
            "prior_steps": int(args.prior_steps),
            "prior_guidance_scale": float(args.prior_guidance_scale),
            "decoder_steps": int(args.decoder_steps),
            "decoder_guidance_scale": float(args.decoder_guidance_scale),
            "decode_mode": args.decode_mode,
            "conditions": list(args.condition),
            "inference_user_scale": float(args.inference_user_scale),
            "inference_user_scale_sweep": list(args.inference_user_scale_sweep or []),
            "seed": int(args.seed),
        },
        "inputs": {
            "checkpoint_path": str(args.checkpoint_path.expanduser().resolve()),
            "embedding_json_path": str(args.embedding_json_path.expanduser().resolve()),
            "assignment_jsonl_path": str(args.assignment_jsonl_path.expanduser().resolve()),
            "uid_to_path_json_path": str(args.uid_to_path_json_path.expanduser().resolve()),
            "prior_model_id": args.prior_model_id,
            "decoder_model_id": args.decoder_model_id,
        },
        "dataset_stats": dataset.get_stats(),
        "checkpoint_compatibility": compatibility,
        "records": decoded_records,
    }
    _write_json(run_dir / "summary.json", summary)
    print(f"[generate_stage2_user_grid] summary: {run_dir / 'summary.json'}")
    for record in decoded_records:
        if "grid_path" in record:
            print(f"[generate_stage2_user_grid] grid: {run_dir / str(record['grid_path'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
