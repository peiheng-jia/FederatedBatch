#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "datasets.json"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, metadata: dict) -> list[str]:
    errors = []
    if not path.exists():
        return [f"missing file: {path}"]
    actual_size = path.stat().st_size
    if actual_size != metadata["size_bytes"]:
        errors.append(f"size mismatch: expected {metadata['size_bytes']}, got {actual_size}")
        return errors
    actual_hash = sha256(path)
    if actual_hash != metadata["sha256"]:
        errors.append(f"SHA-256 mismatch: expected {metadata['sha256']}, got {actual_hash}")
    return errors


def stage(source: Path, destination: Path, metadata: dict, *, force: bool = False) -> None:
    """Verify before copying; publish only a verified, atomically staged file."""
    errors = verify(source, metadata)
    if errors:
        raise ValueError(f"invalid source {source}: {'; '.join(errors)}")
    if source.resolve() == destination.resolve():
        return
    if destination.exists():
        errors = verify(destination, metadata)
        if not errors:
            return
        if not force:
            raise ValueError(f"destination is invalid; inspect it before using --force: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", suffix=".part", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        errors = verify(temporary, metadata)
        if errors:
            raise ValueError(f"invalid staged copy: {'; '.join(errors)}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Stage and verify the exact processed datasets used by the manuscript.')
    parser.add_argument("--dataset", default="all", help="pancreas, immune, lung, or all")
    parser.add_argument("--source-dir", type=Path, help="directory containing downloaded files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace an invalid destination only after source verification")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    datasets = manifest["datasets"]
    selected = list(datasets) if args.dataset == "all" else [args.dataset]
    unknown = [name for name in selected if name not in datasets]
    if unknown:
        raise SystemExit(f"unknown dataset: {', '.join(unknown)}")
    if not args.verify_only and args.source_dir is None:
        raise SystemExit("--source-dir is required unless --verify-only is used")

    failed = False
    for name in selected:
        metadata = datasets[name]
        destination = args.output_dir / metadata["filename"]
        if not args.verify_only:
            source = args.source_dir / metadata["filename"]
            try:
                stage(source, destination, metadata, force=args.force)
            except (OSError, ValueError) as exc:
                print(f"[FAIL] {name}: {exc}")
                failed = True
                continue

        errors = verify(destination, metadata)
        if errors:
            failed = True
            for error in errors:
                print(f"[FAIL] {name}: {error}")
        else:
            print(f"[OK] {name}: {destination}")

    if failed:
        print("Dataset verification failed. Do not run manuscript experiments with these files.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
