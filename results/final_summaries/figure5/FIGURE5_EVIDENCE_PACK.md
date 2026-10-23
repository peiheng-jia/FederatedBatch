# Figure 5 Evidence Pack

## Figure scope

**Figure 5. Privacy protection and secure computation of cross-client contrastive interaction.** This figure uses only completed experiments. No model, privacy attack, DP accountant, or secure-computation benchmark was rerun.

## Panel A: Membership inference attack

Source files:

- `results/final_private_AB_attack_validation/membership_attack.csv`
- `results/final_private_AB_attack_validation/attack_summary.csv`
- `results/final_private_AB_attack_validation/PRIVACY_ATTACK_REPORT.md`

The plotted unit is the prespecified training seed (`n=7`). Points show seed-level AUROC; diamonds and error bars show the mean and unadjusted normal-approximation 95% CI across seeds.

| Interface | Mean AUROC | SD | 95% CI | Candidates / seed | Attack definition |
|---|---:|---:|---:|---:|---|
| Original explicit embeddings | 0.999945 | 0.000101 | 0.999851–1.000038 | 299–313 | Reused direct-exposure membership classifier |
| Remote A/B (`sigma=0`) | 1.000000 | 0 | 1.000000–1.000000 | 8,670 | White-box adjacent-case oracle; deterministic release |
| Final FederatedBatch | 0.626134 | 0.001340 | 0.624894–0.627373 | 8,670; 64 noise draws per hypothesis/candidate | Exact Gaussian likelihood-ratio oracle |

These AUROCs describe different observable interfaces and attack constructions. They should not be interpreted as a controlled ranking under an identical threat model. In particular, direct individual-embedding exposure no longer exists in the final method; the final AUROC quantifies a deliberately strong oracle attack on the noisy aggregate A/B release. An AUROC of 0.626 remains above random guessing and does not imply elimination of leakage.

## Panel B: Gene-expression reconstruction

Source files:

- `results/final_expression_reconstruction_attack/reconstruction_per_cell.csv`
- `results/final_expression_reconstruction_attack/reconstruction_summary.csv`
- `results/final_expression_reconstruction_attack/decoder_config.json`
- `results/final_expression_reconstruction_attack/EXPRESSION_RECONSTRUCTION_REPORT.md`

Panel B uses the matched MLP-decoder analysis of the same 210 target cells across all three interfaces. Boxplots show the cell-level distribution, horizontal white lines show medians, and white diamonds show arithmetic means. The target is normalized-plus-log1p expression over the model's 2,000 HVGs. The decoders were newly trained for the relevant latent spaces with cell-disjoint auxiliary and target sets.

| Interface | Pearson mean | Pearson median | RMSE mean | RMSE median | n cells |
|---|---:|---:|---:|---:|---:|
| Original explicit embeddings | 0.656354 | 0.656075 | 0.241704 | 0.240722 | 210 |
| Remote A/B (`sigma=0`) | 0.658127 | 0.656455 | 0.240953 | 0.244107 | 210 |
| Final FederatedBatch | 0.355259 | 0.358163 | 0.390258 | 0.370884 | 210 |

Lower Pearson correlation and higher RMSE mean weaker reconstruction by the attacker. The aggregate `sigma=0` interface remains vulnerable and is not claimed to be private. The final noisy release materially suppresses expression reconstruction relative to direct embedding exposure, but this experiment concerns the cross-client representation/release channel rather than model-update leakage.

## Panel C: Differential privacy accounting

Source files:

- `results/round_once_jointDP_seed218/privacy_accounting.csv`
- Confirmation of the finalized setting: `results/round_once_jointDP_multiseed_102/privacy_accounting.csv`
- `results/round_once_jointDP_multiseed_102/MULTISEED_102_REPORT.md`

The exact analytic Gaussian privacy profile is

`delta(epsilon) = Phi(1/(2 sigma) - epsilon sigma) - exp(epsilon) Phi(-1/(2 sigma) - epsilon sigma)`.

At `delta=10^-5`, the recorded values are:

| sigma | epsilon |
|---:|---:|
| 0.5 | 9.997256 |
| 1 | 4.377178 |
| 2 | 1.993091 |
| 4 | 0.926342 |

The privacy unit is one remotely participating cell, with add/remove-one-cell adjacency. One remote cell's contributions to all target clients are concatenated and clipped once with L2 sensitivity `C`; one joint Gaussian vector with standard deviation `sigma*C` is released. One-time participation gives one mechanism per participating cell and no across-round composition for that cell. The highlighted finalized setting is Joint-C50 (`C=1206.3391650512762`, `sigma=2`, `epsilon=1.993091`, `delta=10^-5`). The other sigma values are accounting points and do not imply full canonical-seven-seed validation of those alternatives.

**Formal cell-level DP applies only to the remote A/B release channel, not the entire federated training process.** FedAvg updates, local training, and the final global model are outside this guarantee.

## Panel D: Secure computation feasibility

Source files:

- `results/secure_AB_spu_poc/runtime_benchmark.csv`
- `results/secure_AB_spu_poc/correctness.csv`
- `results/secure_AB_spu_poc/communication_benchmark.csv`
- `results/secure_AB_spu_poc/SECURE_AB_SPU_POC_REPORT.md`

The real-scale benchmark comprises five clients, 128 anchors per target, 102 remote cells per source, and 100 latent dimensions. It uses SecretFlow-SPU 0.9.5 with three-party ABY3 replicated arithmetic sharing over FM64 and 18 fixed-point fractional bits. After one warm-up, three measured repetitions were run.

| Backend | Mean time (s) | SD (s) | Successful runs |
|---|---:|---:|---:|
| Plaintext | 0.108667 | 0.012174 | 3/3 |
| SPU / ABY3 | 2.641581 | 0.054306 | 3/3 |

The recorded paired slowdown is `24.5528 +/- 3.28x`. Mean relative errors were `0.03410%` for A and `0.006795%` for B; mean B cosine similarity was `0.999999998`. The reconstructed-gradient cosine averaged `0.9879`, with one repetition affected by a threshold-sensitive noisy-A clamp disagreement documented in the original report.

The PoC assumes passive/semi-honest security with an honest majority: at most one of the three computation parties may be corrupted, and two parties must not collude. It used a local in-memory simulator and trusted input loader. Therefore, the measured runtime is for one protected A/B interaction, not full FederatedBatch training, and it is not a cross-institution WAN benchmark. Communication bytes were unavailable from the simulator API.

## Statistical units and interpretation boundaries

- Panel A: seven training seeds are the independent summary replicates; candidate cells and noise draws are nested attack observations.
- Panel B: 210 matched target cells are displayed. The figure does not treat genes as independent replicates.
- Panel C: deterministic analytic accounting, not an empirical replicate analysis.
- Panel D: three measured repetitions after warm-up at the full interaction scale.
- The privacy attack and formal DP statements cover the cross-client A/B release channel only.
- Neither the attack AUROC nor the secure-computation PoC establishes end-to-end privacy of federated model updates or the released global model.

## Output and vector QC

The combined figure and panels A-D were exported directly through Matplotlib's PDF and SVG vector backends with `pdf.fonttype=42`, `ps.fonttype=42`, and `svg.fonttype='none'`. Automated inspection found zero PDF `/Image` objects and zero SVG `<image>` elements in every output; SVG files retain `<text>` elements and PDF files contain embedded Type 0/TrueType fonts. No panel or plot element was rasterized. The 600-dpi PNG files are preview exports only and were not used to construct the PDF or SVG.

**Adobe-editable vector QC: PASS.**

## Caption draft

**Figure 5. Privacy protection and secure computation of cross-client contrastive interaction.** **(A)** Membership-inference AUROC and attack accuracy for original explicit embedding sharing, the matched clipped Remote A/B interface without Gaussian noise (`sigma=0`), and FederatedBatch (`sigma=2`). The explicit-embedding condition reports balanced accuracy, whereas the A/B oracle conditions report ordinary accuracy on balanced H0/H1 observations. Points denote the seven training-seed results, diamonds denote means, and error bars denote unadjusted t-based 95% confidence intervals across seeds, bounded to the valid [0, 1] metric range; dashed lines indicate the 0.5 chance level. Member and nonmember observations are balanced within every evaluated condition. The interfaces support different attacker observations and attack constructions and should not be interpreted as an identical-threat-model ranking. **(B)** Matched expression reconstruction over 210 target cells using newly trained cell-disjoint MLP decoders. Lower Pearson correlation and higher RMSE indicate weaker attacker reconstruction; boxes show interquartile ranges, center lines show medians, whiskers extend to the most extreme observed point within 1.5 times the interquartile range, and diamonds show means. **(C)** Exact analytic Gaussian privacy accounting at `delta=10^-5` under add/remove-one-remote-cell adjacency, one-time cell participation, and one joint clipped Gaussian A/B release. The selected setting (`sigma=2`) gives `epsilon=1.993091`; other points show accounting values rather than fully validated model settings. Formal cell-level differential privacy applies only to the cross-client A/B release channel, not to end-to-end federated training, model updates, or the final global model. **(D)** Runtime and peak resident set size (RSS) for one real-scale protected interaction using SecretFlow-SPU with three-party ABY3 (`5` clients, `128` anchors per target, `102` remote cells per source, and `100` latent dimensions). Runtime is shown on a logarithmic y-axis. Points show three measured runs after warm-up and diamonds/error bars show mean +/- SD. The local semi-honest PoC completed 3/3 runs with a mean runtime of `2.642 +/- 0.054 s` versus `0.109 +/- 0.012 s` for plaintext, and mean peak RSS of `931.9 +/- 5.5 MiB` versus `602.1 +/- 5.9 MiB`; the runtime slowdown was approximately `24.6x`, with relative errors of `0.0341%` for A and `0.0068%` for B. This is a local secure-computation PoC, not a WAN deployment or full-training benchmark.
