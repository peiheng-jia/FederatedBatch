# Abstract

Single-cell RNA sequencing (scRNA-seq) studies are increasingly performed
across institutions, but privacy, governance and data-sharing constraints often
prevent centralized integration of raw expression data. Federated learning
offers a natural alternative for single-cell batch correction. Here we present
FederatedBatch, a federated contrastive learning framework that combines
synchronous federated training with a server-maintained global queue of latent
representations, providing additional negatives for contrastive learning
without sharing raw data. On the Pancreas benchmark, FederatedBatch achieved
`ARI = 0.637` and `NMI = 0.799`, outperforming PCA and approaching
centralized CONCORD. Across the Pancreas, Immune and Lung datasets,
FederatedBatch consistently outperformed the federated baseline FedscGen. In
leave-one-batch-out evaluation, FederatedBatch enabled true zero-shot inference on
unseen batches with no additional communication (`ARI = 0.579 +- 0.133`,
`NMI = 0.779 +- 0.084`), whereas FedscGen required extra correction rounds.
These results support FederatedBatch as a practical framework for privacy-
preserving collaborative single-cell integration.

## Keywords

Federated learning; single-cell RNA sequencing; batch effect correction;
contrastive learning; privacy-preserving machine learning; zero-shot inference
