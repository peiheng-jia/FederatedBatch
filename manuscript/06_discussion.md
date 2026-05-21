# Discussion

FederatedBatch shows that federated contrastive learning can achieve strong
single-cell batch correction performance without sharing raw data. Across the
Pancreas benchmark, cross-dataset evaluation and LOBO analysis, the method
remained competitive with centralized CONCORD, consistently outperformed PCA,
and showed favorable biological structure preservation relative to the
federated baseline FedscGen. These findings suggest that privacy-preserving
training does not necessarily come at the cost of useful single-cell
integration quality.

Our results also indicate that the global queue is central to this behavior.
In federated settings, each client is otherwise limited to local mini-batches
and therefore to a restricted pool of negative samples. The queue partly
alleviates this problem by expanding contrastive supervision across clients
without exposing raw expression data. The ablation results further suggest that
federated contrastive integration depends on a balance among queue size,
temperature and local training depth, rather than on increasing any one
component in isolation.

An especially important result is that FederatedBatch supports true zero-shot
inference on unseen batches. In the LOBO analysis, the model was trained on 8
batches and directly applied to the held-out batch with no additional
communication, while maintaining competitive performance across all nine
scenarios. This distinguishes FederatedBatch from FedscGen, which requires
additional correction rounds involving the held-out batch. In practice, this
makes FederatedBatch better aligned with deployment scenarios in which a new site
should be able to download a trained model and use it locally without rejoining
the federation.

This study has several limitations. We evaluated only three datasets and a
modest number of clients, so broader validation across larger federations and
more diverse tissues will be important. In addition, the current framework does
not yet incorporate stronger system-level privacy protections such as secure
aggregation or differential privacy. Future work should also examine more
extreme non-IID settings, asynchronous participation and extensions to
multimodal or spatial single-cell data. Despite these limitations, FederatedBatch
provides a practical foundation for privacy-preserving collaborative
single-cell integration.
