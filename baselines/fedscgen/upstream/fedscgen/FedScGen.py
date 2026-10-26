"""Copyright 2024 Mohammad Bakhtiari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import anndata
import scanpy as sc
import os
from anndata import AnnData
from scipy import sparse
import numpy as np
import torch
from functools import partial
from copy import deepcopy
import scarches as sca
from fedscgen.scgen_utils import CustomScGen, TORCH_DTYPE
from fedscgen.utils import set_seed, check_weights_nan
from fedscgen.plots import single_plot
import warnings
import crypten
from numba.core.errors import NumbaDeprecationWarning  # Adjust based on actual availability

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)
warnings.filterwarnings('ignore', message="No data for colormapping provided via 'c'")


class ScGen(CustomScGen):
    """ This class is a wrapper around CustomScGen to use the implemented functionalities for local training of the model.
    """

    def __init__(self, init_model_path: str, ref_model_path: str, adata: AnnData, hidden_layer_sizes: list,
                 z_dimension: int, batch_key, cell_key, lr, epoch, batch_size, stopping, overwrite, device):
        super().__init__(adata, hidden_layer_sizes, z_dimension, device=device)
        self.stopping = stopping
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.ref_model_path = ref_model_path
        self.batch_key = batch_key
        self.cell_key = cell_key
        self.source = None
        self.target = None
        # The model is automatically loaded
        if init_model_path is not None:
            if not overwrite and os.path.exists(init_model_path):
                self.load_model(init_model_path)
            else:
                self.save(init_model_path, overwrite=True)
        self.model.to(self.device)

    def train_centralized(self, output_path: str):
        """ Trains the model on the source dataset

        """
        plot = partial(single_plot, batch_key=self.batch_key, cell_key=self.cell_key, umap_directory=output_path)
        remove_batch_effect = partial(self.batch_removal, batch_key=self.batch_key, cell_label_key=self.cell_key)
        if self.source is not None:
            all_data = deepcopy(self.adata)
            self.adata = self.source
        self.train(n_epochs=self.epoch, early_stopping_kwargs=self.stopping, lr=self.lr, batch_size=self.batch_size)
        if self.source is not None:
            corrected = remove_batch_effect(self.source)
            plot(corrected, plot_name="_source_corrected.png")
            corrected = sca.models.scgen.map_query_data(corrected_reference=corrected,
                                                        query=self.target,
                                                        reference_model=self,
                                                        batch_key=self.batch_key)
            plot(corrected, plot_name="_integrated.png")
            self.adata = all_data
        corrected = remove_batch_effect(self.adata)
        corrected.write(os.path.join(output_path, "corrected.h5ad"))
        plot(corrected, plot_name="_data_corrected.png")
        self.save(self.ref_model_path, overwrite=True)

    def split(self, target_batches):
        """ Splits the source and target datasets based on the target batches
        """
        self.target = self.adata[self.adata.obs[self.batch_key].isin(target_batches)]
        self.source = self.adata[~self.adata.obs[self.batch_key].isin(target_batches)]

    def load_model(self, model_path):
        """ Loads init or a reference model into self.model

        Parameters
        ----------
        model_path
        kwargs

        Returns
        -------

        """
        print(f"Loading model from {model_path}")
        if os.path.exists(model_path):
            saved_state_dict = torch.load(f"{model_path}/model_params.pt")
            self.model.load_state_dict(saved_state_dict)


class FedScGen(ScGen):
    """ Train and correct the batch effect in a federated learning setting using ScGen model
    Methods
    -------
    local_update(global_weights)
        Update the local model with global weights
    find_batch_size()
        Find the number of cells in each batch
    avg_local_cells(max_batch_cells)
        Find the average of cells in each batch
    apply_delta(adata_latent, max_batch_avg)
        Apply the average of cells in each batch to the cells in the same batch
    remove_batch_effect(adata, max_batch_avg, return_latent)
        Remove the batch effect from the dataset
    """

    def __init__(self, n_total_samples, aggregation="str", init_model_path=None, smpc=True, debug=False, **kwargs):
        super().__init__(init_model_path, **kwargs)
        self.unique_cell_types = np.unique(self.adata.obs[self.cell_key])
        self.adata_latent = None
        self.round = 0
        self.n_samples = self.adata.X.shape[0]
        self.sample_ration = self.n_samples / n_total_samples
        self.aggregation = aggregation
        self.smpc = smpc
        self.debug = debug


    def local_update(self, global_weights):
        """ Update the local model with global weights
        Parameters
        ----------
        global_weights: dict
            The weights of the global model
        Returns
        -------
        dict, int
            The weights of the local model and the number of samples in the local dataset
        """
        self.round += 1
        check_weights_nan(global_weights, "before training", self.debug)
        self.set_weights(global_weights)
        self.train(n_epochs=self.epoch, early_stopping_kwargs=self.stopping, lr=self.lr, batch_size=self.batch_size)
        return self.get_local_updates()

    def find_batch_size(self):
        """ Find the number of cells in each batch
            Assuming there is only one batch for each client

        Returns
        -------
        dict
            The number of cells in each batch
        """
        self.adata_latent = self.get_latent_adata(self.adata)
        cell_batch_size = {cell_type: self.adata[self.adata_latent.obs[self.cell_key] == cell_type].shape[0]
                           for cell_type in self.unique_cell_types
                           }
        return cell_batch_size

    def get_latent_adata(self, adata):
        """ Get the latent representation of the dataset
        Parameters
        ----------
        adata: `~anndata.AnnData`
            Annotated data matrix. adata must have `batch_key` and `cell_label_key` which you pass to the function in its obs.
        """
        x = adata.X.A if sparse.issparse(adata.X) else adata.X
        latent_all = self.get_latent(x)
        adata_latent = anndata.AnnData(latent_all)
        adata_latent.obs = adata.obs.copy(deep=True)
        return adata_latent

    def avg_local_cells(self, max_batch_cells: list):
        """ Find the average of cells for each cell type where the local batch is dominant

        Parameters
        ----------
        max_batch_cells: list
            The list of cell types where the local batch is dominant
        Returns
        -------
        dict
            The average of cells for each cell type where the local batch is dominant
        """
        max_batch_avg = {}
        for cell_type in max_batch_cells:
            if cell_type in self.unique_cell_types:
                target_cell = self.adata_latent[self.adata_latent.obs[self.cell_key] == cell_type].X
                max_batch_avg[cell_type] = np.average(target_cell, axis=0)
            else:
                raise Exception(f"{cell_type} is not present among available cell types")
        return max_batch_avg

    def apply_delta(self, adata_latent, max_batch_avg: dict):
        """Applying delta on cells in each cell type, except for standalone cell types, based on the average of cells
         in the dominant batch.

        Parameters
        ----------
        adata_latent: `~anndata.AnnData`
            Annotated data matrix. adata must have `batch_key` and `cell_label_key` which you pass to the function in its obs.
        max_batch_avg: dict
            The average of cells for each cell type where the local batch is dominant
        Returns
        -------
        list, list
            The list of cells where the local batch is dominant and the list of cells where the local batch is not dominant
        """
        unique_cell_types = np.unique(adata_latent.obs[self.cell_key])
        shared_ct, not_shared_ct = [], []
        for batch_lbl in adata_latent.obs[self.batch_key].unique():
            target_batch = adata_latent[adata_latent.obs[self.batch_key] == batch_lbl]
            for cell_type in unique_cell_types:
                temp_cell = target_batch[target_batch.obs[self.cell_key] == cell_type]
                if cell_type in max_batch_avg:
                    temp_cell.X = max_batch_avg[cell_type] - np.average(temp_cell.X, axis=0) + temp_cell.X
                    shared_ct.append(temp_cell)
                else:
                    not_shared_ct.append(temp_cell)
        return shared_ct, not_shared_ct

    def remove_batch_effect(self, max_batch_avg, return_latent=True):
        """ Remove the batch effect in local dataset
        Parameters
        ----------
        adata: `~anndata.AnnData`
            Annotated data matrix. adata must have `batch_key` and `cell_label_key` which you pass to the function in its obs.
        max_batch_avg: dict
            The average of cells for each cell type where the local batch is dominant
        return_latent: `bool`
            if `True` returns corrected latent representation
        Returns
        -------
        `~anndata.AnnData`
            adata of corrected gene expression in adata.X and corrected latent space in adata.obsm["latent_corrected"].
        """
        if self.adata_latent is None:
            print("No latent representation found. Getting latent representation")
            self.get_latent_adata(self.adata)
        shared_ct, not_shared_ct = self.apply_delta(self.adata_latent, max_batch_avg)
        all_shared_ann = anndata.AnnData.concatenate(*shared_ct, batch_key="concat_batch", index_unique=None)
        if "concat_batch" in all_shared_ann.obs.columns:
            del all_shared_ann.obs["concat_batch"]
        if len(not_shared_ct) < 1:
            corrected = sc.AnnData(self.reconstruct(all_shared_ann.X, use_data=True), obs=all_shared_ann.obs)
        else:
            all_not_shared_ann = anndata.AnnData.concatenate(*not_shared_ct, batch_key="concat_batch",
                                                             index_unique=None)
            all_corrected_data = anndata.AnnData.concatenate(all_shared_ann, all_not_shared_ann,
                                                             batch_key="concat_batch",
                                                             index_unique=None)
            if "concat_batch" in all_shared_ann.obs.columns:
                del all_corrected_data.obs["concat_batch"]

            corrected = sc.AnnData(self.reconstruct(all_corrected_data.X, use_data=True), all_corrected_data.obs)
        return self.finalize_batch_correction(self.adata, corrected, return_latent)

    def finalize_batch_correction(self, adata, corrected, return_latent):
        """ Finalize the batch correction
        Parameters
        ----------
        adata: `~anndata.AnnData`
            Annotated data matrix. adata must have `batch_key` and `cell_label_key` which you pass to the function in its obs.
        corrected: `~anndata.AnnData`
            adata of corrected gene expression in adata.X and corrected latent space in adata.obsm["latent_corrected"].
        return_latent: `bool`
            if `True` returns corrected latent representation
        Returns
        -------
        `~anndata.AnnData`
            adata of corrected gene expression in adata.X and corrected latent space in adata.obsm["latent_corrected"].
        """
        corrected.var_names = adata.var_names.tolist()
        corrected = corrected[adata.obs_names]
        corrected.layers["original_data"] = adata.X
        if adata.raw is not None:
            adata_raw = anndata.AnnData(X=adata.raw.X, var=adata.raw.var)
            adata_raw.obs_names = adata.obs_names
            corrected.raw = adata_raw
            corrected.obsm["original_data"] = adata.raw.X
        if return_latent:
            corrected.obsm["latent_corrected"] = self.get_latent(corrected.X)
        return corrected

    def get_local_updates(self):
        """Get the local updates.

        Returns
        -------
        dict, int : Without SMPC
        list of crypten.cryptensor, crypten.cryptensor : With SMPC
        """
        if self.smpc:
            weights = self.get_weights().values()
            if self.aggregation == "weighted_fedavg":
               weights = [param * self.sample_ration for param in weights]
            check_weights_nan(weights, "after training", self.debug)
            encrypted_weights = [crypten.cryptensor(param) for param in weights]
            return encrypted_weights

        return self.get_weights(), self.n_samples

    def get_weights(self):
        """ Get the weights of the model
        Returns
        -------
        dict
            The weights of the model
        """
        return self.model.state_dict()

    def set_weights(self, state_dict):
        """ Set the weights of the model
        Parameters
        ----------
        state_dict: dict
            The weights of the model
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state_dict:
                    param.data.copy_(state_dict[name].to(param.device))
