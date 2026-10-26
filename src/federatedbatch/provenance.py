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

RELEASE_PROVENANCE = "RELEASE_PROVENANCE.json"
FIGURE5_VERSIONED_PATHS = (
    "configs/figure2_legacy_freshqueue.yaml",
    "configs/figure5_privacy.yaml",
    "experiments/figure5_privacy.py",
    "experiments/moderate_skew_freshqueue_confirmation.py",
    "scripts/prepare/prepare_figure5_explicit_baseline.py",
    "src/federatedbatch/federated.py",
    "src/federatedbatch/provenance.py",
    "src/federatedbatch/queue.py",
    "src/concord",
)

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
        top_level = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if Path(top_level).resolve() != repo_root.resolve():
            return None
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tree_sha256(path: Path) -> str:
    """Hash a source tree by relative paths and individual file hashes."""

    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and "__pycache__" not in candidate.parts
        and candidate.suffix not in {".pyc", ".pyo"}
    )
    for candidate in files:
        relative = candidate.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(candidate).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_identity(
    repo_root: Path,
    required_paths: Iterable[str] = (),
) -> dict[str, str | None]:
    """Return a verified Git or release-archive source identity.

    A normal clone is identified by its Git commit. A source archive without
    ``.git`` is accepted only when RELEASE_PROVENANCE.json supplies a release
    version and matching hashes for every declared file/tree.
    """

    repo_root = repo_root.resolve()
    commit = git_revision(repo_root)
    if commit is not None:
        return {"kind": "git", "revision": commit, "git_commit": commit}

    metadata_path = repo_root / RELEASE_PROVENANCE
    if not metadata_path.is_file():
        raise RuntimeError(
            "source version cannot be verified: this is not a Git checkout and "
            f"{RELEASE_PROVENANCE} is missing"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {RELEASE_PROVENANCE}: {exc}") from exc
    if metadata.get("schema_version") != 1 or not str(metadata.get("release_version", "")).strip():
        raise RuntimeError(
            f"invalid {RELEASE_PROVENANCE}: schema_version=1 and release_version are required"
        )
    files = metadata.get("files_sha256")
    trees = metadata.get("trees_sha256")
    if not isinstance(files, dict) or not isinstance(trees, dict):
        raise RuntimeError(
            f"invalid {RELEASE_PROVENANCE}: files_sha256 and trees_sha256 mappings are required"
        )

    for relative, expected in files.items():
        candidate = repo_root / relative
        if not candidate.is_file() or file_sha256(candidate) != expected:
            raise RuntimeError(f"release source verification failed for file: {relative}")
    for relative, expected in trees.items():
        candidate = repo_root / relative
        if not candidate.is_dir() or _tree_sha256(candidate) != expected:
            raise RuntimeError(f"release source verification failed for tree: {relative}")

    uncovered = []
    for required in required_paths:
        required_path = Path(required).as_posix().rstrip("/")
        covered = required_path in files or required_path in trees
        covered |= any(required_path.startswith(tree.rstrip("/") + "/") for tree in trees)
        if not covered:
            uncovered.append(required_path)
    if uncovered:
        raise RuntimeError(
            f"{RELEASE_PROVENANCE} does not cover required source paths: "
            + ", ".join(uncovered)
        )

    metadata_hash = file_sha256(metadata_path)
    version = str(metadata["release_version"])
    return {
        "kind": "release_archive",
        "revision": f"release:{version}:{metadata_hash}",
        "git_commit": metadata.get("source_commit"),
    }


def git_paths_clean(repo_root: Path, paths: Iterable[str]) -> bool:
    """Return whether versioned paths match HEAD, including untracked files."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal", "--", *paths],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0 and not result.stdout.strip()


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
