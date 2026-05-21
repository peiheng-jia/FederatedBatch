# Introduction

Single-cell RNA sequencing (scRNA-seq) has reshaped the study of cellular
heterogeneity by enabling transcriptome profiling at single-cell resolution and
by supporting atlas-scale efforts such as the Human Cell Atlas[1] . As
scRNA-seq datasets have expanded across studies, tissues and platforms,
however, technical variation introduced by protocol differences, sequencing
chemistry and site-specific processing has become a major obstacle to joint
analysis [2]. These batch effects can dominate low-dimensional embeddings,
distort clustering and obscure biological signals if they are not properly
corrected [3]. Accordingly, batch integration has become a core step in
single-cell analysis pipelines.

Several powerful integration methods have been proposed, including Seurat-based
anchor mapping [4], Harmony [5], deep generative approaches such as scVI [6],
and more recent contrastive methods such as CONCORD [7]. Benchmarking studies
show that these methods can substantially improve integration quality, although
their relative strengths differ across biological conservation, batch removal
and scalability criteria [8]. A common practical assumption, however, is that
datasets can be pooled or jointly processed in a centralized environment. In
many biomedical settings this is unrealistic, because data are distributed
across hospitals or institutes and are subject to privacy, governance and
data-sharing constraints. Genomic data sharing remains particularly sensitive,
even after de-identification, making centralized single-cell integration
difficult in real-world multi-institutional studies [9].

Federated learning offers a natural alternative by allowing institutions to
train a shared model without moving raw data off site [10]. This framework has
shown growing value in medical and digital health applications, where
distributed ownership and privacy protection are central requirements
[11-13]. Applying federated learning to scRNA-seq integration is nevertheless
challenging. The model must learn embeddings that are robust to technical batch
variation while preserving fine-grained biological structure, despite client
heterogeneity and non-IID data partitioning. Existing work such as FedscGen,
which extends the scGen framework to federated batch correction [14,15],
represents an important first step, but it still requires held-out data to
participate in an additional communication workflow for correction. This limits
its usefulness in deployment scenarios where a new site should be able to use a
trained model directly.

Here we present FederatedBatch, a federated contrastive learning framework for
privacy-preserving scRNA-seq batch correction. FederatedBatch is motivated by the
observation that contrastive representation learning can be highly effective
for single-cell integration, but in federated settings each client has access
only to local mini-batches and therefore to a limited set of negative examples
[7,16]. To address this, FederatedBatch introduces a server-maintained global queue
that stores latent representations collected during federated training,
enabling richer contrastive supervision without sharing raw expression
profiles. We show that this design yields robust integration performance across
ablation, non-IID, cross-dataset and leave-one-batch-out evaluations, while
supporting true zero-shot inference on unseen batches. Together, these results
position FederatedBatch as a practical framework for privacy-preserving single-cell
integration in distributed biomedical settings.

## References

[1] Regev A, Teichmann SA, Lander ES et al. The Human Cell Atlas. *eLife*,
2017, 6:e27041.

[2] Hwang B, Lee JH and Bang D. Single-cell RNA sequencing technologies and
bioinformatics pipelines. *Exp Mol Med*, 2018, 50:1-14.

[3] Haghverdi L, Lun ATL, Morgan MD and Marioni JC. Batch effects in
single-cell RNA-sequencing data are corrected by matching mutual nearest
neighbors. *Nat Biotechnol*, 2018, 36:421-427.

[4] Stuart T, Butler A, Hoffman P et al. Comprehensive integration of
single-cell data. *Cell*, 2019, 177:1888-1902.e21.

[5] Korsunsky I, Millard N, Fan J et al. Fast, sensitive and accurate
integration of single-cell data with Harmony. *Nat Methods*, 2019,
16:1289-1296.

[6] Lopez R, Regier J, Cole MB, Jordan MI and Yosef N. Deep generative
modeling for single-cell transcriptomics. *Nat Methods*, 2018, 15:1053-1058.

[7] Zhu Q, Liu Y, Zhang Y et al. CONCORD enables unsupervised batch correction
via contrastive learning with optimal transport in single-cell genomics.
*Nat Biotechnol*, 2026.

[8] Luecken MD, Buttner M, Chaichoompu K et al. Benchmarking atlas-level data
integration in single-cell genomics. *Nat Methods*, 2022, 19:41-50.

[9] Wan Z, Hazel JW, Clayton EW et al. Sociotechnical safeguards for genomic
data privacy. *Nat Rev Genet*, 2022, 23:429-445.

[10] McMahan B, Moore E, Ramage D, Hampson S and y Arcas BA.
Communication-efficient learning of deep networks from decentralized data. In:
Singh A and Zhu J (eds). *Proceedings of the 20th International Conference on
Artificial Intelligence and Statistics*. PMLR, 2017, 54:1273-1282.

[11] Kaissis G, Makowski M, Ruckert D and Braren RF. Secure,
privacy-preserving and federated machine learning in medical imaging.
*Nat Mach Intell*, 2020, 2:305-311.

[12] Rieke N, Hancox J, Li W et al. The future of digital health with
federated learning. *npj Digit Med*, 2020, 3:119.

[13] Xu J, Glicksberg BS, Su C et al. Federated learning for healthcare
informatics. *J Healthc Inform Res*, 2021, 5:1-19.

[14] Lotfollahi M, Wolf FA and Theis FJ. scGen predicts single-cell
perturbation responses. *Nat Methods*, 2019, 16:715-721.

[15] Bakhtiari M, Redwan T, Alhussein M et al. Batch effect correction in
single-cell RNA-seq data using federated learning. *Genome Biol*, 2025, 26:204.

[16] Chen T, Kornblith S, Norouzi M and Hinton G. A simple framework for
contrastive learning of visual representations. In: Daume H III and Singh A
(eds). *Proceedings of the 37th International Conference on Machine Learning*.
PMLR, 2020, 119:1597-1607.
