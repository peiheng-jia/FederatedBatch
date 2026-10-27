#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml
from scipy.optimize import brentq
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[2]


def gaussian_delta(epsilon: float, sigma: float) -> float:
    return max(
        0.0,
        float(
            norm.cdf(0.5 / sigma - epsilon * sigma)
            - math.exp(epsilon) * norm.cdf(-0.5 / sigma - epsilon * sigma)
        ),
    )


def epsilon_for(sigma: float, delta: float) -> float:
    upper = 1.0
    while gaussian_delta(upper, sigma) > delta:
        upper *= 2.0
    return float(brentq(lambda value: gaussian_delta(value, sigma) - delta, 0.0, upper))


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate Figure 5 analytic Gaussian accounting from the locked config.')
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "figure5.yaml")
    parser.add_argument("--figure2-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["privacy_accounting"]
    provenance = json.loads(args.figure2_provenance.read_text(encoding="utf-8"))
    reference = provenance.get("reference_protocol", {})
    max_participation = provenance.get(
        "max_remote_participation_required", reference.get("max_remote_participation")
    )
    if int(max_participation or 0) != 1:
        raise ValueError("privacy accounting requires one-time remote participation")
    candidate = provenance.get("candidates", {}).get("Joint-C50-sigma2")
    observed_c = candidate["C"] if candidate else reference.get("joint_C")
    if not math.isclose(float(observed_c), float(config["clipping_C"]), abs_tol=1e-9):
        raise ValueError("Figure 2 clipping threshold differs from Figure 5 accounting")
    if candidate and float(candidate["sigma"]) != float(config["selected_sigma"]):
        raise ValueError("Figure 2 selected sigma differs from Figure 5 accounting")

    rows = []
    for sigma in config["sigmas"]:
        for delta in config["deltas"]:
            rows.append(
                {
                    "sigma": float(sigma),
                    "delta": float(delta),
                    "epsilon": epsilon_for(float(sigma), float(delta)),
                    "accountant": "exact analytic Gaussian privacy profile",
                    "formula": "delta(eps)=Phi(1/(2sigma)-eps*sigma)-exp(eps)*Phi(-1/(2sigma)-eps*sigma)",
                    "privacy_unit": "one remotely participating cell",
                    "adjacency": config["adjacency"],
                    "l2_sensitivity": "C",
                    "noise_std": "sigma*C",
                    "mechanisms_per_participating_cell": 1,
                    "round_composition_per_cell": 1,
                    "subsampling_amplification": False,
                    "scope": config["scope"],
                }
            )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Figure 5 privacy accounting written to {output}")


if __name__ == "__main__":
    main()
