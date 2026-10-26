# Figure 4 preprocessing fix report

## Scope and status

**Status: PASS.** Only FedscGen held-out inference preprocessing was changed. No model was retrained, no checkpoint was modified, and the original Figure 4 directory was not overwritten.

The corrected order is: full-gene `normalize_total(target_sum=10000)` -> `log1p` -> subset to the 2,000 HVGs selected from the corresponding training split. Figure 3 and all non-FedscGen Figure 4 results remain unchanged.

## Preflight

Seed `218`, held-out batch `celseq` passed cell-order, gene-order, and input-matrix checks. The maximum absolute difference versus the native expected matrix was `0`.

The saved checkpoint `/mnt/h/code/FedCONCORD/FederatedBatch-FreshQueueDoseResponse/results/final_figure4_canonical7/training/fedscgen/seed_218/celseq/trained_model/model_params.pt` loaded successfully; both without-correction inference and the official correction workflow completed. Training called: `False`.

## Updated results

Means and SDs use the manuscript aggregation: average nine held-out batches within each seed, then summarize across seven seeds.

| Method | Metric | Before mean +/- SD | After mean +/- SD | Change |
|---|---|---:|---:|---:|
| Centralized CONCORD | ARI | 0.650392 +/- 0.027733 | 0.650392 +/- 0.027733 | +0.000000 |
| Centralized CONCORD | NMI | 0.790594 +/- 0.009126 | 0.790594 +/- 0.009126 | +0.000000 |
| Centralized CONCORD | ASW_celltype | 0.158045 +/- 0.003157 | 0.158045 +/- 0.003157 | +0.000000 |
| Centralized CONCORD | Graph_connectivity | 0.493499 +/- 0.002630 | 0.493499 +/- 0.002630 | +0.000000 |
| FedscGen (without correction) | ARI | 0.536027 +/- 0.011722 | 0.525030 +/- 0.018667 | -0.010997 |
| FedscGen (without correction) | NMI | 0.765113 +/- 0.004105 | 0.762529 +/- 0.006653 | -0.002584 |
| FedscGen (without correction) | ASW_celltype | 0.285684 +/- 0.007404 | 0.351225 +/- 0.005898 | +0.065541 |
| FedscGen (without correction) | Graph_connectivity | 0.465611 +/- 0.005216 | 0.465901 +/- 0.008645 | +0.000290 |
| FedscGen (correction workflow) | ARI | 0.400334 +/- 0.008454 | 0.547851 +/- 0.021200 | +0.147517 |
| FedscGen (correction workflow) | NMI | 0.623959 +/- 0.007164 | 0.776348 +/- 0.009061 | +0.152389 |
| FedscGen (correction workflow) | ASW_celltype | 0.113247 +/- 0.005824 | 0.369951 +/- 0.010547 | +0.256704 |
| FedscGen (correction workflow) | Graph_connectivity | 0.381801 +/- 0.008500 | 0.492373 +/- 0.005562 | +0.110572 |
| FederatedBatch (zero-shot) | ARI | 0.658742 +/- 0.015171 | 0.658742 +/- 0.015171 | +0.000000 |
| FederatedBatch (zero-shot) | NMI | 0.790438 +/- 0.005645 | 0.790438 +/- 0.005645 | +0.000000 |
| FederatedBatch (zero-shot) | ASW_celltype | 0.148651 +/- 0.003509 | 0.148651 +/- 0.003509 | +0.000000 |
| FederatedBatch (zero-shot) | Graph_connectivity | 0.493723 +/- 0.006313 | 0.493723 +/- 0.006313 | +0.000000 |

## Updated paired tests

Differences are FederatedBatch minus comparator. Confidence intervals are unadjusted t-based 95% intervals; paired t-test p values are Holm-adjusted jointly across 12 tests (three comparators x four metrics).

| Comparator | Metric | Mean difference | 95% CI | Raw p | Holm p | Positive seeds |
|---|---|---:|---:|---:|---:|---:|
| FedscGen (without correction) | ARI | +0.133712 | [+0.110754, +0.156670] | 7.4641e-06 | 7.4641e-05 | 7/7 |
| FedscGen (without correction) | NMI | +0.027909 | [+0.019302, +0.036517] | 0.000213022 | 0.00170417 | 7/7 |
| FedscGen (without correction) | ASW_celltype | -0.202574 | [-0.210168, -0.194981] | 8.69253e-10 | 1.0431e-08 | 0/7 |
| FedscGen (without correction) | Graph_connectivity | +0.027822 | [+0.017831, +0.037813] | 0.000489988 | 0.00342991 | 7/7 |
| FedscGen (correction workflow) | ARI | +0.110891 | [+0.090822, +0.130960] | 1.01525e-05 | 9.13728e-05 | 7/7 |
| FedscGen (correction workflow) | NMI | +0.014090 | [+0.006248, +0.021931] | 0.004585 | 0.02751 | 7/7 |
| FedscGen (correction workflow) | ASW_celltype | -0.221300 | [-0.232327, -0.210273] | 4.78214e-09 | 5.26035e-08 | 0/7 |
| FedscGen (correction workflow) | Graph_connectivity | +0.001350 | [-0.006870, +0.009570] | 0.701643 | 1 | 5/7 |
| Centralized CONCORD | ARI | +0.008350 | [-0.023301, +0.040001] | 0.542457 | 1 | 5/7 |
| Centralized CONCORD | NMI | -0.000156 | [-0.011611, +0.011300] | 0.974538 | 1 | 3/7 |
| Centralized CONCORD | ASW_celltype | -0.009393 | [-0.014667, -0.004120] | 0.00477937 | 0.02751 | 0/7 |
| Centralized CONCORD | Graph_connectivity | +0.000224 | [-0.005620, +0.006069] | 0.928254 | 1 | 3/7 |

## Provenance

- Original results: `/mnt/h/code/FedCONCORD/FederatedBatch-FreshQueueDoseResponse/results/final_figure4_canonical7`
- Corrected results: `/mnt/h/code/FedCONCORD/FederatedBatch-FreshQueueDoseResponse/results/figure4_preprocessing_fixed`
- Checkpoints: `results/final_figure4_canonical7/training/fedscgen/seed_<seed>/<held-out>/trained_model/model_params.pt`
- Evaluation: the existing Figure 4 Leiden configuration and legacy induced-subgraph graph-connectivity implementation.
- The two FedscGen correction communication phases and all model/training hyperparameters were unchanged.

## 中文结论

- FedscGen (without correction)：ARI 变化 `-0.010997`，NMI 变化 `-0.002584`。
- FedscGen (correction workflow)：ARI 变化 `+0.147517`，NMI 变化 `+0.152389`。
- 修复成功；63 个 seed x held-out 场景均使用旧 checkpoint 完成，无重新训练。
- 主要方法比较的方向未改变：FederatedBatch 的 ARI/NMI 仍高于两种 FedscGen 条件，且四项差异均在 12-test Holm 校正后显著。
- 但 correction workflow 的解释发生实质变化：修复后它不再降低 held-out ARI/NMI，而是相对旧错误结果分别提高 0.147517 和 0.152389。
- 需要同步修改 Results 4 中全部 FedscGen 数值，以及任何声称 correction workflow 降低 held-out 性能的表述；FederatedBatch 与 centralized CONCORD 的原数值不变。
