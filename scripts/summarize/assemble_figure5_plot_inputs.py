#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description='Assemble Figure 5 renderer inputs from freshly generated stage outputs.')
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "membership_attack.csv": args.membership.resolve(),
        "reconstruction_per_cell.csv": args.reconstruction.resolve(),
        "privacy_accounting.csv": args.accounting.resolve(),
        "runtime_benchmark.csv": args.runtime.resolve(),
        "correctness.csv": args.correctness.resolve(),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        frame = pd.read_csv(source)
        if frame.empty:
            raise ValueError(f"empty Figure 5 stage output: {source}")
        destination = output / name
        frame.to_csv(destination, index=False)
        records[name] = {
            "source": str(source),
            "source_sha256": sha256(source),
            "rows": int(len(frame)),
            "output_sha256": sha256(destination),
        }
    (output / "assembly_provenance.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "operation": "schema-preserving CSV assembly; no metric recomputation",
                "files": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Figure 5 plot inputs assembled in {output}")


if __name__ == "__main__":
    main()
