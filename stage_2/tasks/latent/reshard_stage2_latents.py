from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _shard_for_name(name: str) -> str:
    stem = Path(name).stem
    return stem[:2] if len(stem) >= 2 else "xx"


def _move_file(source: Path, target: Path, dry_run: bool) -> bool:
    if source == target:
        return False
    if target.exists():
        return False
    if dry_run:
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    return True


def _rewrite_json_metadata(json_path: Path, pt_target_path: Path, dry_run: bool) -> bool:
    with json_path.open("r", encoding="utf-8") as handle:
        data: Dict[str, Any] = json.load(handle)

    next_latent_path = str(pt_target_path.resolve())
    if data.get("latent_path") == next_latent_path:
        return False

    data["latent_path"] = next_latent_path
    if dry_run:
        return True

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return True


def reshard_split(split_dir: Path, dry_run: bool) -> Dict[str, int]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    if not split_dir.is_dir():
        raise NotADirectoryError(f"Split path is not a directory: {split_dir}")

    moved_files = 0
    updated_json = 0
    skipped_existing = 0
    direct_files = sorted(path for path in split_dir.iterdir() if path.is_file())

    for path in direct_files:
        if path.suffix not in {".pt", ".json"}:
            continue
        shard = _shard_for_name(path.name)
        target = split_dir / shard / path.name
        moved = _move_file(path, target, dry_run=dry_run)
        if moved:
            moved_files += 1
        elif target.exists():
            skipped_existing += 1

        if path.suffix == ".json":
            json_target = target if moved or target.exists() else path
            pt_target = json_target.with_suffix(".pt")
            if _rewrite_json_metadata(json_target, pt_target, dry_run=dry_run):
                updated_json += 1

        if moved_files and moved_files % 10000 == 0:
            print(
                json.dumps(
                    {
                        "moved_files": moved_files,
                        "updated_json": updated_json,
                        "skipped_existing": skipped_existing,
                        "last_target": str(target),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    return {
        "moved_files": moved_files,
        "updated_json": updated_json,
        "skipped_existing": skipped_existing,
        "direct_files_scanned": len(direct_files),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move flat Stage 2 latent bundles into 2-character shard subdirectories."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        help="Directory containing flat latent .pt/.json files for one split.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without moving or rewriting files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = reshard_split(args.split_dir.expanduser().resolve(), dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
