from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _rewrite_json_metadata(json_path: Path, dry_run: bool) -> Dict[str, Any]:
    pt_path = json_path.with_suffix(".pt")
    if not pt_path.exists():
        return {
            "json_path": str(json_path),
            "status": "missing_pt",
            "expected_latent_path": str(pt_path.resolve()),
        }

    with json_path.open("r", encoding="utf-8") as handle:
        data: Dict[str, Any] = json.load(handle)

    previous_latent_path = data.get("latent_path")
    next_latent_path = str(pt_path.resolve())
    if previous_latent_path == next_latent_path:
        return {
            "json_path": str(json_path),
            "status": "unchanged",
            "latent_path": next_latent_path,
        }

    if not dry_run:
        data["latent_path"] = next_latent_path
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    return {
        "json_path": str(json_path),
        "status": "updated",
        "previous_latent_path": previous_latent_path,
        "latent_path": next_latent_path,
    }


def update_split_sidecars(split_dir: Path, dry_run: bool, progress_every: int) -> Dict[str, Any]:
    split_root = split_dir.expanduser().resolve()
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_root}")
    if not split_root.is_dir():
        raise NotADirectoryError(f"Split path is not a directory: {split_root}")

    json_paths = sorted(path.resolve() for path in split_root.rglob("*.json") if path.is_file())
    updated = 0
    unchanged = 0
    missing_pt: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []

    for idx, json_path in enumerate(json_paths, start=1):
        result = _rewrite_json_metadata(json_path, dry_run=dry_run)
        status = str(result["status"])
        if status == "updated":
            updated += 1
            if len(examples) < 5:
                examples.append(result)
        elif status == "unchanged":
            unchanged += 1
        elif status == "missing_pt":
            missing_pt.append(result)
        else:
            raise RuntimeError(f"Unexpected sidecar update status: {status}")

        if progress_every > 0 and (idx == 1 or idx == len(json_paths) or idx % progress_every == 0):
            print(
                json.dumps(
                    {
                        "scanned": idx,
                        "total": len(json_paths),
                        "updated": updated,
                        "unchanged": unchanged,
                        "missing_pt": len(missing_pt),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    return {
        "split_dir": str(split_root),
        "dry_run": bool(dry_run),
        "total_json_files": len(json_paths),
        "updated": updated,
        "unchanged": unchanged,
        "missing_pt": missing_pt,
        "updated_examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite latent sidecar JSON `latent_path` fields to match colocated .pt files."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        help="Split directory containing sharded .pt/.json latent bundles.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without rewriting JSON files.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N JSON files. Set <=0 to disable periodic progress logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = update_split_sidecars(
        split_dir=args.split_dir,
        dry_run=bool(args.dry_run),
        progress_every=int(args.progress_every),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
