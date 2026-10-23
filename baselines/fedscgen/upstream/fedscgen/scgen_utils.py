"""

Copyright (c) 2022, Mohammad Lotfollahi, Theislab
Copyright (c) Mohammad Bakhtiari 2024

Licensed under the BSD 3-Clause License (see LICENSE-BSD for original terms)
Modifications licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications made to the original work are licensed under the Apache License, Version 2.0.
Original work is licensed under the BSD 3-Clause License.

"""
import __init__
from fedscgen.utils import seed_worker
from anndata import AnnData
from scarches.models.scgen.vaearith import vaeArith
from scarches.trainers.scgen.trainer import vaeArithTrainer
from scarches.trainers.scgen._utils import print_progress
from scarches.utils.monitor import EarlyStopping
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from scipy import sparse
import scarches as sca

TORCH_DTYPE = torch.float32


class AnndataDataset(Dataset):
    def __init__(self, adata):
        if sparse.issparse(adata.X):
            self.data = adata.X.A
        else:
            self.data = adata.X

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=TORCH_DTYPE)


class CustomTrainer(vaeArithTrainer):
    """ Adopted from ScArches
    Author: Mohammad Lotfollahi, Sergei Rybakov, Marco Wagenstetter, Mohsen Naghipourfar
    Date: 20.11.2023
    Version: 0.5.9
    Availability: https://github.com/theislab/scarches
    Link to the used code: https://github.com/theislab/scarches/blob/master/scarches/trainers/scgen/trainer.py

    This class contains the implementation of the VAEARITH Trainer with custom train and validate methods.
    This class utilizes PyTorch Dataloaders to feed data to the network.
    Parameters
    ----------
    model: vaeArith
    adata: : `~anndata.AnnData`
        Annotated Data Matrix for training VAE network.
    n_epochs: int
        Number of epochs to iterate and optimize network weights
    train_frac: Float
        Defines the fraction of data that is used for training and data that is used for validation.
    batch_size: integer

    Methods
    -------
    train(n_epochs, lr, eps, **extras_kwargs)
        Train the ScGen model using the given parameters.
    validate(train_loss_end_epoch, valid_loss)
        Validate the ScGen model using the given parameters.

    """

    def __init__(self, model, adata, train_frac: float = 0.9, batch_size=32, shuffle=False,
                 early_stopping_kwargs: dict = {
                     "early_stopping_metric": "val_loss",
                     "threshold": 0,
                     "patience": 20,
                     "reduce_lr": True,
                     "lr_patience": 13,
                     "lr_factor": 0.1}, device=None, **kwargs):
        self.model = model

        self.seed = kwargs.get("seed", 2021)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            # Only move to CUDA if device is not CPU
            if device is not None and device.type != 'cpu':
                self.model.cuda(device)  # put model to cuda(gpu)
            else:
                self.model.to(device if device is not None else torch.device('cpu'))
        else:
            self.model.to(torch.device('cpu'))
        self.device = device if device is not None else torch.device('cpu')

        self.adata = adata

        self.train_frac = train_frac
        self.shuffle = shuffle
        self.batch_size = batch_size

        early_stopping_kwargs = (early_stopping_kwargs if early_stopping_kwargs else dict())
        self.early_stopping = EarlyStopping(**early_stopping_kwargs)
        self.monitor = kwargs.pop("monitor", True)

        # Optimization attributes
        self.optim = None
        self.epoch = -1  # epoch = self.epoch + 1 in compute metrics
        self.best_epoch = None
        self.best_state_dict = None

        self.logs = defaultdict(list)
        print(self.adata.obs.batch.value_counts())
        train_data, valid_data = self.train_valid_split(self.adata)
        train_dataset = AnndataDataset(train_data)
        valid_dataset = AnndataDataset(valid_data)

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                       worker_init_fn=seed_worker,
                                       num_workers=4, drop_last=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4,
                                       worker_init_fn=seed_worker)

    def train(self, n_epochs=100, lr=0.001, eps=1e-8, **extras_kwargs):
        """
        Train the ScGen model using PyToch Dataloaders.
        Parameters
        ----------
        n_epochs
        lr
        eps
        extras_kwargs

        Returns
        -------

        """
        self.n_epochs = n_epochs
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.optim = torch.optim.Adam(
            params, lr=lr, eps=eps)  # consider changing the param. like weight_decay, eps, etc.
        loss_hist = []

        for self.epoch in range(self.n_epochs):
            self.model.train()
            self.iter_logs = defaultdict(list)
            train_loss = 0
            loss_hist.append(0)

            for x_mb in self.train_loader:
                x_mb = x_mb.to(self.device)  # to cuda or cpu
                reconstructions, mu, logvar = self.model(x_mb)

                loss = self.model._loss_function(x_mb, reconstructions, mu, logvar)

                self.optim.zero_grad()

                loss.backward()
                self.optim.step()

                self.iter_logs["loss"].append(loss.item())
                train_loss += loss.item()
            # self.model.eval()
            self.on_epoch_end()
            self.validate()

            if not self.check_early_stop():
                break

        if self.best_state_dict is not None:
            print("Saving best state of network...")
            print("Best State was in Epoch", self.best_epoch)
            self.model.load_state_dict(self.best_state_dict)

    def validate(self):
        """
        Validate the ScGen model using PyToch Dataloaders.
        Parameters
        ----------
        train_loss_end_epoch
        valid_loss

        Returns
        -------

        """
        self.on_epoch_end()

        valid_loss = 0
        train_loss_end_epoch = 0
        self.iter_logs = defaultdict(list)
        with torch.no_grad():  # disables the gradient calculation
            self.model.eval()

            # Loop through the train data using train_loader
            for x_mb in self.train_loader:
                x_mb = x_mb.to(self.device)
                reconstructions, mu, logvar = self.model(x_mb)
                loss = self.model._loss_function(x_mb, reconstructions, mu, logvar)
                train_loss_end_epoch += loss.item()

            # Loop through the validation data using valid_loader
            for x_mb in self.valid_loader:
                x_mb = x_mb.to(self.device)
                reconstructions, mu, logvar = self.model(x_mb)
                loss = self.model._loss_function(x_mb, reconstructions, mu, logvar)
                self.iter_logs["loss"].append(loss.item())
                valid_loss += loss.item()
                # Get Validation Logs
            for key in self.iter_logs:
                if "loss" in key:
                    self.logs["val_" + key].append(
                        sum(self.iter_logs[key][:]) / len(self.iter_logs[key][:]))

            # Monitor Logs
            if self.monitor:
                print_progress(self.epoch, self.logs, self.n_epochs)


class CustomVAEArith(vaeArith):
    """ Adopted from ScArches
    Author: Mohammad Lotfollahi, Sergei Rybakov, Marco Wagenstetter, Mohsen Naghipourfar
    Date: 20.11.2023
    Version: 0.5.9
    Availability: https://github.com/theislab/scarches
    Link to the used code: https://github.com/theislab/scarches/blob/master/scarches/models/scgen/vaearith.py

    This class is a wrapper around scarches.models.scgen.vaearith.vaeArith to customize the get_latent and reconstruct methods.

    Parameters
    ----------
    x_dim: int
        Number of input features.

    Methods
    -------
    get_latent(data: torch.Tensor)
        Map `data` into the latent space.
    reconstruct(data, use_data: bool)
        Map back the latent space encoding via the decoder.
    """

    def __init__(self, x_dim: int, **kwargs):
        super().__init__(x_dim, **kwargs)

    def _sample_z(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
            Samples from standard Normal distribution with shape [size, z_dim] and
            applies re-parametrization trick. It is actually sampling from latent
            space distributions with N(mu, var) computed by the Encoder.

        Parameters
            ----------
        mean:
        Mean of the latent Gaussian
            log_var:
        Standard deviation of the latent Gaussian
            Returns
            -------
        Returns Torch Tensor containing latent space encoding of 'x'.
        The computed Tensor of samples with shape [size, z_dim].
        """
        std = torch.exp(0.5 * log_var)
        if not self.training:
            torch.manual_seed(42)
        eps = torch.randn_like(std)
        return mu + std * eps

    def get_latent(self, data: torch.Tensor) -> torch.Tensor:
        """ Map `data` in to the latent space. It uses PyTorch Dataloader to
         feed data in encoder part of VAE and compute the latent space coordinates for each sample in data.
        """
        print("Getting latent space coordinates...")
        if not torch.is_tensor(data):
            data = torch.tensor(data)
        if len(data) < 100000:
            data = data.to(next(self.encoder.parameters()).device)
            with torch.no_grad():
                mu, logvar = self.encoder(data)
                latent = self._sample_z(mu, logvar)
            return latent
        for i in range(0, len(data), 1000):
            data_ = data[i:i + 1000]
            data_ = data_.to(next(self.encoder.parameters()).device)
            with torch.no_grad():
                mu, logvar = self.encoder(data_)
                latent_ = self._sample_z(mu, logvar)
            if i == 0:
                latent = latent_
            else:
                latent = torch.cat((latent, latent_), 0)
        return latent

    def reconstruct(self, data, use_data=False) -> torch.Tensor:
        """ Map back the latent space encoding via the decoder using PyTorch Dataloader.

        """
        print("Reconstructing data...")
        if not torch.is_tensor(data):
            data = torch.tensor(data)

        if use_data:
            latent = data
        else:
            latent = self.get_latent(data)
        if len(latent) < 100000:
            latent = latent.to(next(self.decoder.parameters()).device)
            with torch.no_grad():
                return self.decoder(latent)
        for i in range(0, len(latent), 100):
            latent_ = latent[i:i + 100]
            latent_ = latent_.to(next(self.decoder.parameters()).device)
            with torch.no_grad():
                rec = self.decoder(latent_)
                reconstructions = rec if i == 0 else torch.cat((reconstructions, rec), 0)
        return reconstructions


class CustomScGen(sca.models.scgen):
    """ Adopted from ScArches
    Author: Mohammad Lotfollahi, Sergei Rybakov, Marco Wagenstetter, Mohsen Naghipourfar
    Date: 20.11.2023
    Version: 0.5.9
    Availability: https://github.com/theislab/scarches
    Link to the used code: https://github.com/theislab/scarches/blob/master/scarches/models/scgen/vaearith_model.py

    This class is a wrapper around scarches.models.scgen.vaearith.vaeArith and scarches.trainers.scgen.trainer.vaeArithTrainer
    It is used to train a model and perform batch correction on adata object.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    hidden_layer_sizes: list
        List of layer sizes for the hidden layers.
    z_dimension: int
        Dimension of the latent space.
    dr_rate: float
        Dropout rate for the model.
    """

    def __init__(self, adata: AnnData, hidden_layer_sizes: list = [128, 128], z_dimension: int = 10,
                 dr_rate: float = 0.05, device=None):
        self.device = device
        self.adata = adata

        self.x_dim_ = adata.n_vars

        self.z_dim_ = z_dimension
        self.hidden_layer_sizes_ = hidden_layer_sizes
        self.dr_rate_ = dr_rate
        self.model = CustomVAEArith(self.x_dim_, hidden_layer_sizes=self.hidden_layer_sizes_, z_dimension=self.z_dim_,
                                    dr_rate=self.dr_rate_)
        self.is_trained_ = False
        self.trainer = None

    def train(self, n_epochs: int = 100, lr: float = 0.001, eps: float = 1e-8, batch_size=32, **kwargs):
        """
        Instantiate the custom trainer and train the model.
        Parameters
        ----------
        n_epochs
        lr
        eps
        batch_size
        kwargs

        Returns
        -------

        """
        if self.trainer is None:
            self.trainer = CustomTrainer(self.model, self.adata, batch_size=batch_size, device=self.device, **kwargs)
        self.trainer.train(n_epochs, lr, eps)
        self.is_trained_ = True
