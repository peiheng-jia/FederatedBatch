# Results

## FederatedBatch enables privacy-preserving batch integration while retaining competitive performance on the Pancreas dataset

We first tested whether federated contrastive training can preserve single-cell
integration quality without sharing raw data across institutions (Figure 1). In
the three-client Pancreas benchmark, FederatedBatch achieved `ARI = 0.637` and
`NMI = 0.799`, substantially outperforming PCA (`ARI = 0.314`, `NMI = 0.670`)
and remaining close to centralized CONCORD (`ARI = 0.616`, `NMI = 0.806`).
Although Harmony showed the strongest overall benchmark performance, FederatedBatch
maintained competitive batch correction (`ASW(batch) = -0.018`) together with
favorable structure preservation, including the highest graph connectivity
among the compared methods (`0.104`) and a high `iLISI` value (`3.374`)
(Figure 1B).

The UMAP visualizations supported these quantitative results. When colored by
batch, FederatedBatch markedly reduced batch-driven separation relative to PCA and
showed substantially improved cross-batch mixing (Figure 1C). When colored by
cell type, major pancreatic populations remained well organized after
integration (Figure 1D). Together, these results indicate that FederatedBatch
achieves a practical balance between privacy-preserving federated training,
batch correction and biological structure preservation.

## Ablation studies define the operating regime and robustness of FederatedBatch

We next used ablation experiments to define the operating regime of FederatedBatch
on the Pancreas dataset (Figure 2). Temperature tuning showed that clustering
performance was strongest at low-to-moderate values, with `tau = 0.07`
yielding `ARI = 0.581` and `NMI = 0.784`, whereas higher temperatures reduced
clustering accuracy despite modest gains in separation-related `ASW` metrics
(Figure 2A). Queue-size analysis further showed that the global queue improved
performance relative to the no-queue baseline (`ARI = 0.570` at `queue size =
0` versus `0.581` at `queue size = 1000`) and that the gain saturated once the
queue reached a moderate size (Figure 2B). These results support the use of
`tau = 0.07` and `queue size = 1000` in the main experiments.

Local training depth mainly affected the trade-off between accuracy and
efficiency (Figure 2C). Increasing local epochs improved absolute performance,
with `5` local epochs reaching the best `ARI` (`0.581`), but also increased
runtime substantially relative to `1` epoch (`262.9 s` versus `113.0 s`). We
also tested robustness under non-IID client heterogeneity. FederatedBatch remained
stable under both label skew and quantity skew, with only modest decreases in
clustering accuracy (`ARI = 0.561` and `0.558` versus `0.581` under IID) and
similar `NMI`, `iLISI` and graph-connectivity values across all three settings
(Figure 2D). UMAPs of the selected configuration further showed clear
cell-type organization (`ASW = 0.133`) together with low batch-driven
separation (`ASW = -0.014`) (Figure 2E), supporting the robustness of the
chosen parameter regime.

## FederatedBatch generalizes across multiple tissues and remains competitive with centralized integration

To test whether the method generalizes beyond Pancreas, we evaluated
FederatedBatch on three tissues: Pancreas, Immune and Lung (Figure 3). Across all
datasets, FederatedBatch consistently outperformed PCA and generally exceeded the
federated baseline FedscGen, while remaining close to centralized CONCORD. On
Pancreas, FederatedBatch achieved `ARI = 0.496` and `NMI = 0.750`, substantially
higher than FedscGen (`ARI = 0.256`, `NMI = 0.561`) and with stronger
cell-type preservation (`ASW(cell) = 0.187` versus `-0.017`). On Immune,
FederatedBatch remained nearly identical to centralized CONCORD in clustering
accuracy (`ARI = 0.502` versus `0.503`) while showing stronger biological
structure preservation (`ASW(cell) = 0.065` versus `0.034`) and the highest
batch-mixing score (`iLISI = 4.261`). On Lung, FederatedBatch achieved the best
`NMI` among all compared methods (`0.702`) and improved `ASW(cell)` relative
to both centralized CONCORD and FedscGen (`0.107` versus `0.038` and `0.020`,
respectively).

The UMAP comparisons were consistent with these trends (Figure 3A-F).
FederatedBatch preserved clearer cell-type organization than FedscGen and showed
substantially better batch mixing than PCA across all three datasets. The
summary bubble plot in Figure 3G further showed that FederatedBatch remained close
to centralized CONCORD across `ARI`, `NMI`, `iLISI` and graph connectivity,
while providing a stronger overall federated alternative than FedscGen. These
results indicate that FederatedBatch is not limited to a single tissue context but
provides a transferable federated integration strategy across distinct
biological systems.

## FederatedBatch enables true zero-shot inference on unseen batches

We finally examined whether FederatedBatch can generalize to an unseen batch
without requiring the new site to participate in additional correction rounds.
Using leave-one-batch-out evaluation on the Pancreas dataset, the model was
trained on 8 of 9 batches and then directly applied to the held-out batch in
each scenario (Figure 4A). Across all nine held-out settings, FederatedBatch
achieved competitive overall performance with `ARI = 0.579 +- 0.133` and
`NMI = 0.779 +- 0.084`, while requiring `0` additional communication rounds
(Figure 4B). FedscGen reached slightly higher clustering accuracy
(`ARI = 0.630 +- 0.130`, `NMI = 0.781 +- 0.052`), but only after the held-out
batch participated in `2` extra correction rounds.

Under this stricter deployment setting, FederatedBatch better preserved biological
structure, achieving higher average `ASW(celltype)` than FedscGen
(`0.211 +- 0.043` versus `0.179 +- 0.087`). Performance varied across held-out
batches, but FederatedBatch remained robust across the full LOBO benchmark,
outperforming FedscGen in `7/9` scenarios for `ASW(celltype)` and in `2/9`
scenarios for `ARI` (Figure 4C). The paired UMAP visualizations further showed
that FederatedBatch maintained coherent cell-type organization without requiring
the held-out batch to rejoin the federation (Figure 4D). These results
highlight a practical advantage of FederatedBatch: a new site can download a
trained model and perform local inference directly, without any additional
data exchange.
