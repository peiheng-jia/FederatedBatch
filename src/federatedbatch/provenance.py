"""Run provenance helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import numbers
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

EXPECTED_PACKAGE_VERSIONS = {
    "anndata": "0.10.5.post1",
    "harmonypy": "0.0.9",
    "h5py": "3.10.0",
    "leidenalg": "0.10.2",
    "matplotlib": "3.8.2",
    "nbformat": "5.9.2",
    "numpy": "1.26.4",
    "pandas": "2.1.4",
    "plotly": "5.18.0",
    "python-igraph": "0.11.4",
    "PyYAML": "6.0.1",
    "scanpy": "1.9.8",
    "scikit-learn": "1.3.2",
    "scikit-misc": "0.3.1",
    "scipy": "1.11.4",
    "seaborn": "0.13.1",
    "torch": "2.1.2",
    "tqdm": "4.66.1",
    "umap-learn": "0.5.5",
}


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(
    names: Iterable[str] = EXPECTED_PACKAGE_VERSIONS,
) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def git_revision(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_metadata(repo_root: Path, data_path: Path | None = None) -> dict:
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "git_revision": git_revision(repo_root),
    }
    if data_path is not None:
        metadata["data"] = {
            "path": str(data_path.resolve()),
            "size_bytes": data_path.stat().st_size,
            "sha256": file_sha256(data_path),
        }
    return metadata


def write_json(path: Path, value: object) -> None:
    value = _json_safe(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str, allow_nan=False)
        handle.write("\n")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, numbers.Real) and not math.isfinite(float(value)):
        return None
    return value
