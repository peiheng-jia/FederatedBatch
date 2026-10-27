#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "datasets.json"
USER_AGENT = "FederatedBatch-reproducibility/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Figshare API returned HTTP {exc.code} for {url}. "
            "If an interactive WAF challenge is active, download through a browser "
            "and stage files with scripts/prepare/prepare_data.py."
        ) from exc


def resolve_files(article_id: int) -> tuple[dict, dict[str, dict]]:
    payload = request_json(f"https://api.figshare.com/v2/articles/{article_id}")
    result = {}
    for record in payload.get("files", []):
        name = record.get("name")
        if name:
            if str(name) in result:
                raise RuntimeError(f"ambiguous duplicate provider filename: {name}")
            result[str(name)] = record
    return payload, result


def check_provider_record(record: dict, metadata: dict) -> None:
    """Reject an updated/mismatched provider artifact before downloading it."""
    if record.get("name") != metadata["filename"]:
        raise ValueError("provider filename does not match the manuscript manifest")
    if record.get("size") != metadata["size_bytes"]:
        raise ValueError(f"provider size mismatch for {metadata['filename']}")
    provider_md5 = record.get("computed_md5") or record.get("supplied_md5")
    if not provider_md5 or provider_md5.lower() != metadata["md5"].lower():
        raise ValueError(f"provider MD5 absent or mismatched for {metadata['filename']}")
    if not str(record.get("download_url", "")).startswith("https://"):
        raise ValueError(f"missing HTTPS provider download URL for {metadata['filename']}")


def download(url: str, destination: Path, metadata: dict) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", suffix=".part", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
        verify(temporary, metadata)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify(path: Path, metadata: dict) -> None:
    if path.stat().st_size != metadata["size_bytes"]:
        raise ValueError(
            f"size mismatch for {path.name}: expected {metadata['size_bytes']}, "
            f"got {path.stat().st_size}"
        )
    observed = sha256(path)
    if observed != metadata["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {metadata['sha256']}, got {observed}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Download the registered Figshare datasets and verify immutable checksums.')
    parser.add_argument("--dataset", choices=("all", "pancreas", "immune", "lung"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resolve-only", action="store_true", help="query and check small provider metadata only; never download dataset payloads")
    args = parser.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = list(manifest["datasets"]) if args.dataset == "all" else [args.dataset]
    needed = []
    if not args.resolve_only:
        for name in selected:
            metadata = manifest["datasets"][name]
            destination = args.output_dir / metadata["filename"]
            if destination.exists() and not args.force:
                verify(destination, metadata)
                print(f"[OK] {name}: existing verified file {destination}")
            else:
                needed.append(name)
        if not needed:
            return 0
    else:
        needed = selected

    payload, files = resolve_files(int(manifest["collection"]["article_id"]))
    checked_records = []
    # Check all selected provider records before any dataset transfer.
    for name in needed:
        metadata = manifest["datasets"][name]
        filename = metadata["filename"]
        if filename not in files:
            raise RuntimeError(f"Figshare article does not list required file {filename!r}")
        check_provider_record(files[filename], metadata)
        checked_records.append({"dataset": name, **files[filename]})
    if args.resolve_only:
        print(json.dumps({
            "article_id": manifest["collection"]["article_id"],
            "provider_current_version": payload.get("version"),
            "provider_current_doi": payload.get("doi"),
            "historical_download_version": "MISSING",
            "files": checked_records,
            "dataset_payloads_downloaded": False,
        }, indent=2))
        return 0
    for name in needed:
        metadata = manifest["datasets"][name]
        filename = metadata["filename"]
        destination = args.output_dir / filename
        url = files[filename]["download_url"]
        print(f"[DOWNLOAD] {name}: provider file id={files[filename].get('id', 'unknown')}")
        download(str(url), destination, metadata)
        print(f"[OK] {name}: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
