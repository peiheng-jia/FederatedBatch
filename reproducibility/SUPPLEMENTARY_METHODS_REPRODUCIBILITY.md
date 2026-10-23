# Supplementary Methods: Reproducibility (release draft)

FederatedBatch experiments use the canonical training seeds `218, 394, 542, 639,
788, 814, 880`. Processed Pancreas, Immune, and Lung inputs are obtained from
Figshare DOI `10.6084/m9.figshare.12420968`; exact filenames, dimensions,
annotation keys, file sizes, and SHA-256 checksums are recorded in
`data/datasets.json`. The raw `.h5ad` files are intentionally external to this
compact release.

The final FederatedBatch protocol uses 20 communication rounds, five local
epochs, batch size 128, learning rate 0.001, 2,000 Seurat-v3 HVGs,
RoundOnce remote interaction in rounds 4--20, 102 unique remote cells per source
and active round, one-time participation, Joint-C50 clipping with
`C=1206.3391650512762`, `sigma=2`, `tau_q=0.20`, and `alpha=0.10`. Formal
cell-level DP applies only to the cross-client A/B release channel.

Figure 2 compares NoQueue, matched Remote A/B (`sigma=0`), and FederatedBatch.
Figure 3 compares PCA, Harmony, centralized CONCORD, FedscGen, and FederatedBatch
on Pancreas, Immune, and Lung. Figure 4 is a nine-batch Pancreas LOBO analysis
using the corrected held-out preprocessing results; panel A is author-managed and
automated outputs contain panels B/C. Figure 5 contains membership attack,
expression reconstruction, analytic DP accounting, and a local SecretFlow-SPU /
ABY3 PoC.

The compact tables and plot inputs are bundled, but a clean-checkout end-to-end
reproduction is **not verified**. Fresh runs require the external datasets,
checkpoints, environment-specific dependencies, and in some cases saved UMAP
coordinates. FedscGen is included as a vendored source snapshot with a local
patch; its configured upstream commit is recorded but cannot be verified from
Git metadata in this tree. No final Git commit or Zenodo DOI exists yet.
