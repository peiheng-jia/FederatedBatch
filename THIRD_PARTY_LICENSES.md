# Third-Party Software

## CONCORD

`src/concord/` is derived from CONCORD 1.0.13 and includes the local
queue-aware changes used by FederatedBatch. The upstream project is available
at <https://github.com/Gartner-Lab/Concord> and is distributed under the MIT
License. A copy of that license is retained at `src/concord/LICENSE.md`.

Local changes add:

- `global_queue` and `queue_weight` configuration values;
- `NTXent_with_queue` in `src/concord/model/loss.py`;
- queue propagation through the CONCORD trainer; and
- dynamic queue updates on the `Concord` object.

## FedscGen

The authoritative implementation is the checksummed vendored snapshot under
`baselines/fedscgen/upstream/`, pinned to upstream revision
`b266f744fcc0d97ac0db25f00316b6a080929d0a`. Unmodified files were matched to
the GitHub blobs for that revision; the only Python differences are the two
files covered exactly by `baselines/fedscgen/patches/single_machine.patch`.
The source manifest records upstream blob identifiers and vendored SHA-256
values. A byte-identical duplicate Python tree remains locally under
`third_party/FedscGen/` for audit history but is excluded from the public Git
inventory.

FedscGen is distributed primarily under Apache-2.0 (`upstream/LICENSE`) with
`scgen_utils.py` subject to BSD-3-Clause; the upstream component license is
retained at `upstream/licenses/BSD-3-Clause-scgen_utils`. Copyright headers and
the upstream `NOTICE` are retained. The local compatibility patch disables
single-process CrypTen initialization and fixes CPU/GPU device selection; it
does not replace the VAE or federated algorithm.

Upstream project: <https://github.com/Mohammad-Bakhtiari/FedscGen>. Related
article: Bakhtiari et al., *FedscGen: privacy-preserving federated batch effect
correction of single-cell RNA sequencing data*, Genome Biology (2025),
<https://doi.org/10.1186/s13059-025-03684-6>.
