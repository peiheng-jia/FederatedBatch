import random

SEED = 42
import numpy as np

import os
from collections import Counter
import shutil
import anndata
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch
import torch.nn as nn
import torchmetrics
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import ast
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans
from itertools import combinations
import copy
from collections import Counter
from typing import List, Dict, Union
import crypten
from crypten.config import cfg
from collections import OrderedDict

def set_seed(seed=None):
    """Sets the seed for reproducibility. Checks for an environment variable first, then defaults to SEED."""
    global SEED
    if seed is None:
        seed = SEED
    else:
        SEED = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    # crypten.init()  # Disabled for single-machine setup
    # cfg.debug.debug_mode = True  # Disabled for single-machine setup
    # crypten.manual_seed(seed, seed, seed)  # Disabled for single-machine setup
    # cfg.encoder.precision_bits = 32  # Disabled for single-machine setup
    print(f"✅ Seed set to {seed} (from `set_seed()`)")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed + SEED)
    random.seed(worker_seed + SEED)
    torch.manual_seed(worker_seed + SEED)


def get_cuda_device(device_index: int):
    # Support CPU mode when device_index is -1
    if device_index == -1 or not torch.cuda.is_available():
        return torch.device("cpu")
    else:
        torch.cuda.set_device(device_index)  # Set the device globally
        return torch.device(f"cuda:{device_index}")


def combine_test_loaders(loaders):
    combined_dataset = ConcatDataset([dl.dataset for dl in loaders])
    return DataLoader(combined_dataset, batch_size=32, shuffle=False, worker_init_fn=seed_worker, num_workers=4)


class Data:
    def __init__(self, model_path, adata, output_path, n_clients, epoch, stopping, batch_key,
                 cell_key, source_name="source",
                 target_name="target"):
        self.init_model = model_path
        self.ref_model = ""  # os.path.join(model_path, "ref_model")
        # self.folder_path = data_path
        self.output_path = output_path
        self.epoch = epoch
        self.stopping = stopping
        self.n_clients = n_clients
        self.batch_key = batch_key
        self.cell_label_key = cell_key
        self.umap_directory = os.path.join(output_path, "UMAPs")

        # create output directory if it doesn't exist
        os.makedirs(self.umap_directory, exist_ok=True)

        # reading main dataset, source and target data
        self.dataset = adata  # self.read_data(data_path)

        self.source = adata  # self.read_data(data_path)

        self.target = self.read_data(f'{target_name}.h5ad') if target_name else None

        # initializing containers for corrected and integrated data
        self.source_corrected = None
        self.target_corrected = None
        self.integrated = None
        self.network = None
        if n_clients:
            # adding clients
            self.clients = self.add_clients()

    def read_data(self, file_name):
        path = os.path.join(self.folder_path, file_name)
        if os.path.exists(path):
            return anndata.read_h5ad(path)
        raise FileNotFoundError

    def add_clients(self):
        clients = {}
        for i in range(1, self.n_clients + 1):
            client = {}
            file_path = f'c{i}.h5ad'
            client['data'] = self.read_data(file_path)
            client['corrected'] = None
            client['integrated'] = None
            client['umap_path'] = os.path.join(self.output_path, f'c{i}_umap.png')
            clients[f'c{i}'] = client
        return clients

    def get_client(self, id):
        return self.clients.get(id)

    def next_client(self):
        if not self.clients:
            raise StopIteration("No more clients.")
        client_id, client_data = self.clients.popitem()
        return client_id, client_data

    def save_as_zip(self):
        shutil.make_archive(self.output_path, 'zip', self.output_path)

    def write(self):
        if self.source_corrected:
            self.source_corrected.write(os.path.join(self.output_path, "source_corrected.h5ad"))
        else:
            raise Warning("Source corrected data is None!!!")
        if self.target_corrected:
            self.target_corrected.write(os.path.join(self.output_path, "target_corrected.h5ad"))
        else:
            print("Target corrected data is None!!!")

        if self.integrated:
            self.integrated.write(os.path.join(self.output_path, "integrated.h5ad"))
        else:
            print("Integrated data is None!!!")


def encode_labels(y):
    # Convert string labels to numeric labels
    le = LabelEncoder()
    return le.fit_transform(y)


def evaluate(model, val, criterion, num_classes, device):
    # Validation loop
    model.eval()
    val_loss = 0.0
    total = 0

    # Initialize metrics on CPU (to avoid `torch.cumsum` GPU issue)
    auc = torchmetrics.AUROC(num_classes=num_classes, average="macro", task="multiclass").cpu()
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).cpu()

    with torch.no_grad():
        for batch_x, batch_y in val:
            d_size = batch_y.size(0)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * d_size

            # Move outputs & labels to CPU before computing metrics
            acc.update(outputs.argmax(dim=1).cpu(), batch_y.cpu())
            auc.update(F.softmax(outputs, dim=1).cpu(), batch_y.cpu())

            total += d_size

        # Compute final metric values
        acc_score = acc.compute().item()
        auc_score = auc.compute().item()
        acc.reset()
        auc.reset()

        # Ensure correct dataset size handling
        size = total if total > 0 else len(val.dataset)
        val_loss /= size

    return val_loss, acc_score, auc_score


def classify_celltypes(x_train, y_train, x_test, y_test, epochs, lr, batch_size, n_classes, init_model, model_name,
                       hidden_size, device):
    if model_name.lower() == "knn":
        return train_knn(x_train, y_train, x_test, y_test)
    elif model_name.lower() == "kmeans":
        return train_kmeans(x_train, y_train, x_test, y_test, n_classes)
    test_loader, train_loader = load_dataloaders(x_train, y_train, x_test, y_test, batch_size)
    criterion, model, optimizer = instantiate_model(x_test.shape[1], n_classes, lr, init_model, model_name, hidden_size,
                                                    device)

    return train_classifier(model, train_loader, test_loader, optimizer, criterion, epochs, n_classes, device)


def train_knn(x_train, y_train, x_test, y_test):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_binarized = lb.transform(y_test)
    y_pred_binarized = lb.transform(y_pred)
    try:
        auc = roc_auc_score(y_test_binarized, y_pred_binarized, multi_class='ovr')
    except ValueError:
        auc = None  # or any default value
    return [0], [0], [accuracy], [auc]


def train_kmeans(x_train, y_train, x_test, y_test, n_clusters):
    # Train a KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    train_preds = kmeans.fit_predict(x_train)

    # Assign a label to each cluster based on the most frequent true label in that cluster
    cluster_to_label_mapping = {}
    for i in range(n_clusters):
        cluster_points = y_train[train_preds == i]
        most_common = Counter(cluster_points).most_common(1)[0][0]
        cluster_to_label_mapping[i] = most_common
    test_preds = kmeans.predict(x_test)
    y_pred = np.array([cluster_to_label_mapping[cluster] for cluster in test_preds])
    accuracy = accuracy_score(y_test, y_pred)
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_binarized = lb.transform(y_test)
    y_pred_binarized = lb.transform(y_pred)
    try:
        auc = roc_auc_score(y_test_binarized, y_pred_binarized, multi_class='ovr')
    except ValueError:
        auc = None  # or any default value

    return [0], [0], [accuracy], [auc]


def custom_kfold(batches, n_batch_out):
    train_test_splits = []
    combinations_list = testset_combination(batches, n_batch_out)

    for combination in combinations_list:
        test_indices = []
        for batch in combination:
            test_indices += np.where(batches == batch)[0].tolist()
        train_indices = np.setdiff1d(np.arange(len(batches)), test_indices)
        train_test_splits.append((train_indices, test_indices))
    return train_test_splits


def rnd_client_batch(batches, test_batches, n_clients_batch):
    train_batches = copy.deepcopy(batches)
    for batch in test_batches:
        train_batches.remove(batch)
    client_batches = np.random.choice(batches, n_clients_batch)
    for batch in client_batches:
        batches.remove(batch)
    return client_batches


def testset_combination(batches, n_batch_out):
    unique_batches = np.unique(batches)
    combinations_list = list(combinations(unique_batches, n_batch_out))
    return combinations_list


def train_classifier(model, train, test, optimizer, criterion, epochs, n_classes, device):
    # Train the model
    acc_scores = []
    auc_scores = []
    train_loss = []
    test_loss = []

    for epoch in range(epochs):
        # Training loop
        model.train()
        loss = 0.0
        for batch_x, batch_y in train:
            loss = train_on_batch(batch_x.to(device), batch_y.to(device), criterion, model, optimizer, loss)
        loss /= len(train.dataset)
        train_loss.append(loss)
        if test:
            val_loss, val_acc, val_auc = evaluate(model, test, criterion, n_classes, device)
            test_loss.append(val_loss)
            acc_scores.append(val_acc)
            auc_scores.append(val_auc)
            print(
                f"Epoch [{epoch + 1}/{epochs}],"
                f" Train Loss: {loss:.4f},"
                f" Val Loss: {val_loss:.4f},"
                f" Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.04}")
    return train_loss, test_loss, acc_scores, auc_scores


def train_on_batch(batch_x, batch_y, criterion, model, optimizer, train_loss):
    optimizer.zero_grad()
    outputs = model(batch_x.float())
    loss = criterion(outputs, batch_y.long())
    loss.backward()
    optimizer.step()
    train_loss += loss.item() * batch_x.size(0)
    return train_loss


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        # out = self.softmax(out)
        return out


class MLP_Norm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_Norm, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)

    def forward(self, x):
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)
        return out


def cpm_normalization(data):
    # Compute counts per million
    counts_per_million = (data / np.sum(data, axis=1).reshape(-1, 1)) * 1e6
    return counts_per_million


def normalize_data(data, method):
    if isinstance(data, pd.Series):
        data = data.to_numpy().reshape(-1, 1)
    if method == "log":
        # Add 1 to avoid taking log of zero
        return np.log1p(data)
    elif method == "quantile":
        transformer = QuantileTransformer(output_distribution='uniform')
        return transformer.fit_transform(data)
    elif method == "z_score":
        scaler = StandardScaler()
        return scaler.fit_transform(data)
    elif method == "min_max":
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)
    elif method == "cpm":
        # Normalize to 'counts per million'
        return cpm_normalization(data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def instantiate_model(input_size, n_classes, lr, init_model, model_name, hidden_size, device):
    # Instantiate the model, loss function, and optimizer
    if model_name.lower() == "mlp":
        model = MLP(input_size=input_size, hidden_size=64, output_size=n_classes)
    elif model_name.lower() == "mlp-norm":
        model = MLP_Norm(input_size=input_size, hidden_size=hidden_size, output_size=n_classes)
    else:
        raise ModuleNotFoundError(f"There is no implemented model for {model_name}")
    if os.path.exists(init_model):
        model.load_state_dict(torch.load(init_model))
    else:
        torch.save(model.state_dict(), init_model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return criterion, model.to(device), optimizer


def load_dataloaders(x_train, y_train, x_test, y_test, batch_size=32):
    # Convert the data to PyTorch tensors and create DataLoader objects
    num_workers=4
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              worker_init_fn=seed_worker, num_workers=num_workers)
    test_loader = None
    if len(x_test) > 0:
        val_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
        test_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, worker_init_fn=seed_worker, shuffle=False,
                                 num_workers=num_workers)
    return test_loader, train_loader


def plot_centralized():
    f = open("integrated", "r")
    integrated = [float(l.strip().split("Acc:")[-1].strip()) for l in f if len(l.strip()) > 0]
    f = open("all", "r")
    all = [float(l.strip().split("Acc:")[-1].strip()) for l in f if len(l.strip()) > 0]
    f = open("integ-lowdim", "r")
    lowdim = [float(l.strip().split("Acc:")[-1].strip()) for l in f if len(l.strip()) > 0]
    # Create a Pandas DataFrame with the accuracy values
    df = pd.DataFrame({'Integrated': integrated, 'Original': all, 'Integrated-Low-dim': lowdim})

    # Create a line plot using Seaborn's `lineplot` function
    sns.lineplot(data=df)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.savefig("Cent-Acc.png")
    return lowdim


def plot_federated(lowdim):
    f = open("fed-org", "r")
    client_acc = [[], []]
    for l in f:
        if "Round" in l:
            continue
        if "Client" in l:
            c_ind = int(l.strip().split(" ")[1].strip())
            continue
        if "Aggregation" in l:
            continue
        if len(l.strip()) > 0:
            client_acc[c_ind].append(float(l.strip().split("Acc:")[-1].strip()))

    f = open("fed-low", "r")
    low_client_acc = [[], []]
    for l in f:
        if "Round" in l:
            continue
        if "Client" in l:
            c_ind = int(l.strip().split(" ")[1].strip())
            continue
        if "Aggregation" in l:
            continue
        if len(l.strip()) > 0:
            low_client_acc[c_ind].append(float(l.strip().split("Acc:")[-1].strip()))

    acc = [np.mean(client_acc, axis=0), np.mean(low_client_acc, axis=0), [max(lowdim)] * 100]
    df = pd.DataFrame({'Original': acc[0], 'Low-dim': acc[1], "best centralized": acc[2]})

    # Create a line plot using Seaborn's `lineplot` function
    sns.lineplot(data=df)
    plt.xlabel('Communication rounds')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over communication rounds')
    plt.savefig("Fed-Acc.png")


def plot_metrics(df, plt_name):
    df_melt = df.melt('Epoch', var_name='Metric', value_name='Score')
    plt.figure(figsize=(10, 7))
    sns.lineplot(data=df_melt, x='Epoch', y='Score', hue='Metric')
    plt.title('Model Metrics over epochs')
    plt.savefig(plt_name)
    plt.close()


def to_csv(train_loss, test_loss, acc_scores, auc_scores, file_name):
    # Create a dictionary with accuracy and AUC
    data = {'Epoch': list(range(1, len(acc_scores) + 1)),
            'Train Loss': train_loss,
            'Test Loss': test_loss,
            'Accuracy': acc_scores,
            'AUC': auc_scores}

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)
    if file_name:
        df.to_csv(file_name, index=False)
    return df


def kfold_generator(x, n_folds):
    kf = KFold(n_splits=n_folds)
    for _, (train_ind, test_ind) in enumerate(kf.split(x), 1):
        yield train_ind, test_ind


def get_latent(adata, latent):
    if latent:
        return adata.obsm['latent_corrected']
    return adata.X


def calc_obsm_pca(adata_file_paths, n_components=50, common_space=False):
    adata_files = {}
    if common_space:
        pca = PCA(n_components=n_components, svd_solver='full')
    for counter, (key, path) in enumerate(adata_file_paths.items()):
        adata = anndata.read_h5ad(path)
        if not common_space:
            pca = PCA(n_components=n_components, svd_solver='full')
            pca.fit(adata.X)
        elif counter == 0:
            pca.fit(adata.X)

        adata.obsm[f'pca_{n_components}'] = pca.transform(adata.X)
        adata_files[key] = adata

    return adata_files



def aggregate(
    state_dicts: Union[List[Dict[str, torch.Tensor]], List[List[crypten.cryptensor]]],
    n_samples: Union[List[int], crypten.cryptensor] = None,
    smpc: bool = False,
    param_keys: List[str] = None,
    aggregation: str = "fedavg"

) -> Dict[str, torch.Tensor]:
    """
    Aggregates model weights from multiple clients using either standard federated averaging
    or Secure Multi-Party Computation (SMPC), ensuring the output format is consistent.

    Parameters
    ----------
    state_dicts : List[Dict[str, torch.Tensor]] (if smpc=False) or List[List[crypten.cryptensor]] (if smpc=True)
        - If `smpc=False`: A list of state dictionaries from each client.
        - If `smpc=True`: A list of encrypted weight lists where each index corresponds to a specific layer.
    n_samples : List[int] (if smpc=False) or crypten.cryptensor (if smpc=True)
        - The number of training samples per client.
    smpc : bool
        Whether to use Secure Multi-Party Computation (SMPC) for aggregation.
    param_keys : List[str], optional
        The names of model parameters (used in SMPC case).

    Returns
    -------
    Dict[str, torch.Tensor]
        The aggregated global model weights.
    """

    if smpc:
        # Sum encrypted weights across clients layer-wise
        aggregated_weights = [sum(weights) for weights in zip(*state_dicts)]
        if aggregation == "fedavg":
            n_clients = len(state_dicts)
            aggregated_weights = [w / n_clients for w in aggregated_weights]
        # fractions applied on weights in the clients side
        encrypted_global_weights = {
            param_keys[i]: aggregated_weights[i] for i in range(len(param_keys))
        }

        # Compute weighted average in encrypted form
        # encrypted_global_weights = {
        #     param_keys[i]: aggregated_weights[i] / total_samples for i in range(len(param_keys))
        # }


        # **Ensure format consistency**: Convert encrypted tensors to decrypted torch.Tensor
        global_weights = {
            param: encrypted_global_weights[param].get_plain_text().clone().detach().to(torch.float32)
            for param in encrypted_global_weights
        }
    else:
        # Compute sample ratios for federated averaging
        if aggregation == "fedavg":
            sample_ratios = [1/len(state_dicts)] * len(state_dicts)
        else:
            sample_ratios = [n / sum(n_samples) for n in n_samples]
        global_weights = {}

        for param in state_dicts[0].keys():
            global_weights[param] = torch.stack(
                [state_dicts[i][param] * sample_ratios[i] for i in range(len(state_dicts))]
            ).sum(0)

    return global_weights



def average_weights(state_dicts, sample_ratios):
    keys = state_dicts[0].keys()
    for state_dict in state_dicts:
        assert keys == state_dict.keys()
    state_dict_avg = {
        key: torch.stack([sample_ratios[i] * state_dict[key] for i, state_dict in enumerate(state_dicts)]).sum(0)
        for key in keys}
    return state_dict_avg


def stratify_kfold(adata, n_splits):
    # Get the study labels
    studies = adata.obs['study'].values

    # Create the StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_splits)

    # Split the data
    for train_index, test_index in skf.split(np.zeros(len(studies)), studies):
        yield train_index, test_index


def drop(data, cell_key, drop_cell_values):
    if drop_cell_values:
        drop_cell_values = ast.literal_eval(drop_cell_values)
        print(drop_cell_values)
        data = data[~data.obs[cell_key].isin(drop_cell_values)]

    return data


def aggregate_batch_sizes(batch_sizes: dict):
    shared = Counter(celltype for client in batch_sizes.values() for celltype in client)
    shared = {celltype for celltype, count in shared.items() if count > 1}
    max_client = {}
    for client_id, batch in batch_sizes.items():
        for cell_type, size in batch.items():
            if cell_type in shared:
                if cell_type in max_client:
                    dominant_batch_size = batch_sizes[max_client[cell_type]][cell_type]
                    if size > dominant_batch_size:
                        max_client[cell_type] = client_id
                else:
                    max_client[cell_type] = client_id
    clients_majority_cell_batch = {}
    for cell_type, client_id in max_client.items():
        if client_id in clients_majority_cell_batch:
            clients_majority_cell_batch[client_id].append(cell_type)
        else:
            clients_majority_cell_batch[client_id] = [cell_type]
    return clients_majority_cell_batch


def remove_cell_types(adata, cell_types: list, cell_key):
    if len(cell_types) > 0:
        print(f"Removing {cell_types} cell types from adata.obs['{cell_key}']")
        return adata[~adata.obs[cell_key].isin(cell_types), :]
    return adata


def combine_cell_types(adata, cell_types: list, cell_key, combined_label="others"):
    if len(cell_types) > 0:
        print(f"Combining {cell_types} cell types into '{combined_label}' in adata.obs['{cell_key}']")
        if pd.api.types.is_categorical_dtype(adata.obs[cell_key]):
            if combined_label not in adata.obs[cell_key].cat.categories:
                adata.obs[cell_key].cat.add_categories(combined_label, inplace=True)

        mask = adata.obs[cell_key].isin(cell_types)
        adata.obs.loc[mask, cell_key] = combined_label

        cell_types_present = [cell for cell in cell_types if cell in adata.obs[cell_key].cat.categories]

        if len(cell_types_present) > 0:
            adata.obs[cell_key] = adata.obs[cell_key].cat.remove_categories(cell_types_present)
    return adata


def translate(s):
    chars_to_remove = "[]()'',"
    trans = str.maketrans(' ', '-', chars_to_remove)
    return s.strip().translate(trans)


def get_w(model):
    """ Get the weights of the model

    Parameters
    ----------
    model
        The model

    Returns
    -------
    dict
        The weights of the model
    """
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.data.clone().cpu()
    return weights


def set_w(model, weights):
    """ Set the weights of the model
    """
    for name, param in model.named_parameters():
        if name in weights:
            param.data.copy_(weights[name].to(param.device))


def abs_diff_centrally_corrected(centrally_corrected, fed_corrected, fed_corrected_with_new_studies):
    if fed_corrected_with_new_studies is None:
        abs_diff = np.abs(centrally_corrected.X - fed_corrected.X)
        indices = ["Without new studies"]
    else:
        abs_diff = np.abs(centrally_corrected.X - fed_corrected_with_new_studies.X)
        indices = ["With new studies"]
    rows = [{"Mean": np.mean(abs_diff), "Standard deviation": np.std(abs_diff), "Maximum": np.max(abs_diff),
             "Minimum": np.min(abs_diff), "Sum": np.sum(abs_diff)}]

    df = pd.DataFrame(rows, index=indices)
    return df


def check_adata_nan(adata):
    n_nans = np.isnan(adata.X).sum()
    if n_nans > 0:
        print(f"🚨 Checking for NaNs in data: {n_nans} NaNs found")


def check_weights_nan(weights, when, debug):
    if debug:
        # Convert odict_values to a list
        if isinstance(weights, (dict, OrderedDict)):  # Ensure it's a dictionary-like object
            weights = weights if isinstance(weights, dict) else dict(weights)  # Convert OrderedDict to dict

            for name, param in weights.items():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"⚠️ NaN or Inf found in {name} {when}!")

        elif isinstance(weights, list):  # Handle list of tensors
            for param in weights:
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"⚠️ NaN or Inf found {when}!")

        elif isinstance(weights, torch.Tensor):  # Handle single tensor case
            if torch.isnan(weights).any() or torch.isinf(weights).any():
                print(f"⚠️ NaN or Inf found {when}!")

        else:
            print(f"⚠️ Unexpected type {type(weights)} in check_weights_nan {when}!")