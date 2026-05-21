# Methods

## 1 Problem formulation

We consider a federated single-cell integration setting with $K$ clients, where
client $k$ holds a local scRNA-seq dataset
$\mathcal{D}_k = \{X_k, B_k, Y_k\}$. Here, $X_k \in \mathbb{R}^{n_k \times d}$
denotes the gene-expression matrix, $B_k$ denotes batch labels, and $Y_k$
denotes cell-type annotations used only for evaluation. Raw expression data are
not exchanged across clients. The goal is to learn a shared encoder
$f_{\theta}: \mathbb{R}^d \rightarrow \mathbb{R}^p$ that maps cells into a
latent space in which batch-specific variation is reduced while biological
structure is preserved. Formally, the federated objective can be written as

$$
\min_{\theta} \sum_{k=1}^{K}\frac{n_k}{N}\,\mathcal{L}_k(\theta),
$$

where $N=\sum_{k=1}^{K} n_k$ and $\mathcal{L}_k$ denotes the client-specific
contrastive training objective.

## 2 FederatedBatch framework

FederatedBatch combines local contrastive representation learning with synchronous
federated aggregation and a server-maintained global queue. Each client
instantiates a CONCORD encoder with latent dimension 100, a hidden layer of 128
units and dropout probability 0.1. Local models are trained on client-specific
mini-batches, and after each communication round the server aggregates model
parameters using sample-size-weighted FedAvg. The resulting global weights are
then broadcast back to all clients for the next round.

The main design element of FederatedBatch is the global queue. In a conventional
federated setting, each client is limited to negative samples drawn from local
data only. FederatedBatch mitigates this constraint by storing latent
representations collected from all clients in a fixed-size first-in-first-out
queue on the server. At each round, clients contribute a subset of locally
encoded representations, the server updates the queue, and the queue is used as
an additional source of negatives during subsequent client-side contrastive
training. Let $z_i=f_{\theta}(x_i)$ be the embedding of cell $x_i$ and let
$z_i^{+}$ denote its positive counterpart under the local contrastive
construction. The client-side loss is written in standard InfoNCE form as

$$
\mathcal{L}_k =
-\frac{1}{|\mathcal{B}_k|}\sum_{i \in \mathcal{B}_k}
\log
\frac{\exp(\mathrm{sim}(z_i,z_i^{+})/\tau)}
{\exp(\mathrm{sim}(z_i,z_i^{+})/\tau)+
\sum_{j \in \mathcal{N}_k \cup \mathcal{Q}}
\exp(\mathrm{sim}(z_i,z_j^{-})/\tau)},
$$

where $\mathcal{B}_k$ is a local mini-batch, $\mathcal{N}_k$ is the set of
client-local negatives, $\mathcal{Q}$ denotes the global queue, and $\tau$ is
the temperature parameter. This design increases negative-sample diversity
without exposing raw expression matrices.

To make the queue update explicit, let $\mathcal{S}_k^{(t)}$ denote the set of
latent representations contributed by client $k$ after round $t$. The server
updates the queue as

$$
\mathcal{Q}^{(t+1)}=
\mathrm{FIFO}_{M}\Bigl(\mathcal{Q}^{(t)} \cup \bigcup_{k=1}^{K}\mathcal{S}_k^{(t)}\Bigr),
$$

where $\mathrm{FIFO}_{M}(\cdot)$ truncates the queue to the most recent $M$
representations. In our main experiments, $M=1000$.

## 3 Federated training and inference

The FederatedBatch training procedure consists of four steps. First, the server
initializes the global model and an empty queue. Second, each client performs
local contrastive training on its own data. Third, locally updated parameters
are sent to the server and aggregated with FedAvg. Fourth, the server updates
the queue with newly collected latent representations and redistributes the
aggregated model for the next round. Unless otherwise noted, experiments used
20 communication rounds, 5 local epochs per round, batch size 128, learning
rate 0.001, temperature 0.07 and queue size 1000. Figure 2 evaluates the
sensitivity of the method to temperature, queue size, local epochs and client
heterogeneity. Parameter aggregation at round $t$ follows

$$
\theta^{(t+1)}=\sum_{k=1}^{K}\frac{n_k}{N}\theta_k^{(t+1)},
$$

where $\theta_k^{(t+1)}$ denotes the locally updated model from client $k$.
Equivalently, the local and global steps can be summarized as

$$
\theta_k^{(t+1)}=\mathrm{LocalUpdate}\bigl(\theta^{(t)},\mathcal{D}_k,\mathcal{Q}^{(t)}\bigr),
$$

followed by FedAvg aggregation on the server.

For zero-shot deployment, a trained global encoder is downloaded once and
applied directly to new local data with no additional communication. In the
leave-one-batch-out (LOBO) experiments, one batch was completely excluded from
training, the model was trained on the remaining batches, and embeddings for
the held-out batch were generated directly using the final global encoder. This
setting was contrasted with FedscGen, which required 2 additional correction
rounds involving the held-out batch. Zero-shot inference is therefore simply

$$
z_{\mathrm{new}} = f_{\theta^{(T)}}(x_{\mathrm{new}}),
$$

where $\theta^{(T)}$ is the final global model after federated training.

## 4 Datasets and study design

We evaluated FederatedBatch on three public scRNA-seq integration benchmarks
downloaded from the scIB dataset collection: Pancreas, Immune and Lung. The
Pancreas dataset contained 16,382 cells from 9 batches and 14 annotated cell
types, the Immune dataset contained 33,506 cells from 10 batches and 16 cell
types, and the Lung dataset contained 32,472 cells from 16 batches and 17 cell
types. For the main federated experiments, each dataset was partitioned into 3
clients by assigning non-overlapping subsets of batches to different clients,
thereby mimicking a multi-institution setting with distributed batch ownership.
Additional robustness experiments considered IID, label-skew and
quantity-skew client partitions, and the LOBO evaluation iteratively held out
each Pancreas batch as an unseen test set. For a held-out batch $b_h$, the
LOBO training and inference protocol can be written as

$$
\mathcal{D}_{\mathrm{train}}^{(h)}=\mathcal{D}\setminus b_h, \qquad
\mathcal{D}_{\mathrm{test}}^{(h)}=b_h,
$$

where the model is trained only on $\mathcal{D}_{\mathrm{train}}^{(h)}$ and
then directly applied to $\mathcal{D}_{\mathrm{test}}^{(h)}$ without any extra
communication.

## 5 Data preprocessing

All datasets were processed with Scanpy-based workflows. Batch and cell-type
annotation fields were harmonized across datasets when necessary. For each
dataset, the top 2,000 highly variable genes were selected using a Seurat
v3-style feature-selection strategy as implemented in Concord/Scanpy.
Expression counts were library-size normalized and log-transformed before model
training. For all integration methods, the same preprocessed inputs were used
within each experiment.

## 6 Baseline methods

The methods compared depended on the experiment. In the Pancreas benchmark
(Figure 1), we compared FederatedBatch against centralized CONCORD, Harmony and
PCA. In the multi-dataset and LOBO experiments (Figures 3 and 4), we compared
FederatedBatch with centralized CONCORD, FedscGen and PCA where applicable.
Centralized CONCORD served as the centralized reference, PCA as a simple linear
baseline, and FedscGen as the main federated comparator.

## 7 Evaluation metrics

Integration quality was assessed using six metrics. Clustering performance was
measured by adjusted Rand index (ARI) and normalized mutual information (NMI)
after applying Leiden clustering to the learned embeddings. Batch mixing was
assessed by silhouette width with respect to batch labels (`ASW(batch)`), where
values closer to zero or below zero indicate better mixing. Biological
conservation was assessed by silhouette width with respect to cell-type labels
(`ASW(cell type)`), graph connectivity, and `iLISI`. For LOBO experiments,
`ASW(batch)` is not defined for the held-out batch because each test set
contains a single batch only. Structure metrics were computed from the learned
embedding using shared neighborhood-graph settings across methods.

## 8 Implementation details

FederatedBatch was implemented in Python using PyTorch for model training and
custom synchronous federated-training loops built around the CONCORD
implementation. Data preprocessing, neighborhood-graph construction, Leiden
clustering and several evaluation steps were performed with Scanpy. All runs
used fixed random seeds within each script, and experiment outputs were written
to versioned result directories containing summary tables and, where relevant,
per-batch embeddings and metadata. Code and processed data paths are organized
within the project repository to support reproduction of the main benchmark,
ablation, multi-dataset and LOBO experiments.
