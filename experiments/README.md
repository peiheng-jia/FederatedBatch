# Manuscript experiments

This directory contains the final experiment entry points. Full commands,
environments and inputs are documented in the
[paper reproduction guide](../docs/reproduction.md).

| Figure | Entry point | Purpose |
|---|---|---|
| 1 | `run_figure1_canonical7.py` | Seven-seed Pancreas benchmark |
| 2 | `run_figure2_from_scratch.py` | NoQueue, clipped unnoised A/B and protected A/B ablation |
| 3–4 | `run_final_figure3_4_canonical7.py` | Multi-dataset benchmark and corrected LOBO analysis |
| 5 | `figure5_privacy.py` | Explicit-embedding membership-attack comparator |
| 5 | `run_figure5_explicit_full_data.py` | Full-data explicit-embedding reconstruction comparator |
| 5 | `final_private_ab_attack_validation.py` | Attacks on unnoised and protected A/B releases |
| 5 | `final_expression_reconstruction_attack.py` | Matched expression reconstruction analysis |
| 5 | `secure_AB_spu_poc.py` | Local three-party secure A/B computation benchmark |

Shared partition, schedule, evaluation and statistics functions are internal
modules under `src/federatedbatch/experiment_support/`; they are not standalone
experiments. Plotting entry points are under `scripts/render/`, including
`render_figure1_bcd.R` for Figure 1B–D.

Figure 5 retains the explicit-embedding comparator to evaluate the information
exposed by that interface. It is an active attack control, not the final
FederatedBatch training protocol. Final remote interaction uses RoundOnce,
joint cell-level clipping and Gaussian noise. Its DP accounting covers the
specified remote A/B release channel; ABY3 is a separate local feasibility
benchmark.
