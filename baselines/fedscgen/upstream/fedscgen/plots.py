import copy
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
import matplotlib.lines as mlines


def umap_plot(data, wspace, plt_name, batch_key, cell_label_key, use_rep=None):
    batch_color_dict, cell_color_dict = gen_color_dict(batch_key, cell_label_key, data)
    if use_rep:
        sc.pp.neighbors(data, use_rep=use_rep)
    else:
        sc.pp.neighbors(data)
    sc.tl.umap(data)
    combined_color_dict = {**batch_color_dict, **cell_color_dict}
    sc.pl.umap(data, color=[batch_key, cell_label_key], frameon=False, wspace=wspace, save=plt_name,
               palette=combined_color_dict)


def tsne_plot(data, wspace, plt_name, batch_key, cell_label_key, use_rep=None):
    # ... (the color setup remains the same)
    batch_color_dict, cell_color_dict = gen_color_dict(batch_key, cell_label_key, data)
    if use_rep:
        sc.pp.neighbors(data, use_rep=use_rep)
    else:
        sc.pp.neighbors(data)
    sc.tl.tsne(data)  # Run t-SNE instead of UMAP
    combined_color_dict = {**batch_color_dict, **cell_color_dict}
    sc.pl.tsne(data, color=[batch_key, cell_label_key], frameon=False, wspace=wspace, save=plt_name,
               palette=combined_color_dict)


def gen_color_dict(batch_key, cell_label_key, data):
    # Define your color map using the specified colors
    hex_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1a55FF', '#55a868',
        '#c44e52', '#8172b3', '#ccb974', '#64B5CD', '#FFB447', '#82C341',
        '#D17049', '#705696'
    ]
    batch_labels = data.obs[batch_key].unique().tolist()
    cell_labels = data.obs[cell_label_key].unique().tolist()
    assert len(cell_labels) <= len(hex_colors), "Not enough colors for labels"
    if len(hex_colors) >= len(batch_labels):
        batch_color_dict = {label: hex_colors[i] for i, label in enumerate(batch_labels)}
    else:
        batch_color_dict = {label: hex_colors[i % len(hex_colors)] for i, label in enumerate(batch_labels)}
    if len(hex_colors) >= len(cell_labels):
        cell_color_dict = {label: hex_colors[i] for i, label in enumerate(cell_labels)}
    else:
        cell_color_dict = {label: hex_colors[i % len(hex_colors)] for i, label in enumerate(cell_labels)}
    return batch_color_dict, cell_color_dict


def plot_all_umaps(uncorrected, corrected, batch_key, cell_key, umap_directory):
    plot = partial(umap_plot, batch_key=batch_key, cell_label_key=cell_key)
    set_uamp_conf(umap_directory)
    plot(uncorrected, wspace=0.6, plt_name="dataset.png")
    plot(corrected, wspace=0.6, plt_name="source_corrected.png")


def set_uamp_conf(umap_directory):
    sc.settings.figdir = umap_directory
    sc.settings.set_figure_params(dpi=200, frameon=False)
    sc.set_figure_params(dpi=200)
    sc.set_figure_params(figsize=(4, 4))


def single_plot(adata, batch_key, cell_key, umap_directory, plot_name):
    set_uamp_conf(umap_directory)
    umap_plot(adata, batch_key=batch_key, cell_label_key=cell_key, plt_name=plot_name, wspace=0.6)


def plot_umap_fed_scgen(model, adata, test_adata, test_batches, batches, batch_key, cell_key, output_dir):
    train_batches = copy.deepcopy(batches)
    for batch in test_batches:
        train_batches.remove(batch)
    trns_test_batches = translate(str(test_batches))
    if len(trns_test_batches) == 0:
        trns_test_batches = "BO0"
    # UMAP entire dataset
    corrected_dataset = model.batch_removal(adata, batch_key=batch_key, cell_label_key=cell_key,
                                            return_latent=True)
    plot_all_umaps(adata, corrected_dataset, batch_key, cell_key,
                   umap_directory=f"{output_dir}/{trns_test_batches}/dataset")

    # UMAP entire train adata
    train_adata = adata[adata.obs[batch_key].isin(train_batches)].copy()
    corrected_train_adata = model.batch_removal(train_adata,
                                                batch_key=batch_key,
                                                cell_label_key=cell_key,
                                                return_latent=True)
    plot_all_umaps(train_adata, corrected_train_adata, batch_key, cell_key,
                   umap_directory=f"{output_dir}/{trns_test_batches}/train")

    # UMAP entire test adata

    if len(test_batches) > 0:
        corrected_test_adata = model.batch_removal(test_adata,
                                                   batch_key=batch_key,
                                                   cell_label_key=cell_key,
                                                   return_latent=True)
        plot_all_umaps(test_adata, corrected_test_adata, batch_key, cell_key,
                       umap_directory=f"{output_dir}/{trns_test_batches}/test")


def ss_agx_kbet_plot(silhouette_scores, average_gene_expression, kbet_scores, plt_name):
    # Define the metrics and their corresponding data
    metrics_data = {
        'Silhouette Score': silhouette_scores,
        'Average Gene Expression': average_gene_expression,
        'kBET Score': kbet_scores
    }

    # Create subplots
    fig, axs = plt.subplots(nrows=len(metrics_data), figsize=(7, 15))

    # Iterate over the metrics and their corresponding data
    for i, (metric, data) in enumerate(metrics_data.items()):
        # print(i, metric)
        data = data.transpose()
        if metric == 'Average Gene Expression':
            data_source = pd.DataFrame({metric: data.loc['Uncorrected'].tolist(),
                                        'Method': ['Uncorrected'] * len(data.loc['Uncorrected'].tolist())})
            data_corrected = pd.DataFrame({metric: data.loc['Corrected'].tolist(),
                                           'Method': ['Corrected'] * len(data.loc['Corrected'].tolist())})
            data_combined = pd.concat([data_source, data_corrected])
            sns.violinplot(x='Method', y=metric, data=data_combined, ax=axs[i])
        else:
            sns.barplot(x=data.index, y=metric, data=data, ax=axs[i])

        axs[i].set_title(metric)

    plt.tight_layout()
    plt.savefig(plt_name)


def subplot(results, output_dir):
    # Convert the results to a DataFrame
    data = []
    for fold, rounds in results.items():
        for round, scores in rounds.items():
            data.append({
                'Fold': fold,
                'Round': round,
                'Loss': scores['loss'],
                'Accuracy': scores['acc'],
                'AUC': scores['auc'],
            })
    df = pd.DataFrame(data)
    df.to_csv(f"{output_dir}/fed_metrics.csv")
    # Convert the round to an integer for easier plotting
    df['Round'] = df['Round'].str.slice(start=6).astype(int)
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Define the handles and labels list
    handles, labels = [], []

    # Plot the loss, accuracy, and AUC for each fold
    for fold, group in df.groupby('Fold'):
        sns.lineplot(ax=axes[0], x='Round', y='Loss', data=group, linestyle="dashed", legend=False)
        sns.lineplot(ax=axes[1], x='Round', y='Accuracy', data=group, linestyle="dashed", legend=False)
        sns.lineplot(ax=axes[2], x='Round', y='AUC', data=group, linestyle="dashed", legend=False)
        handles.append(mlines.Line2D([], [], color=plt.gca().lines[-1].get_color(), linestyle="dashed"))
        # handles.append(line1.lines[0])
        # Add the handles and labels

    avg = df.groupby('Round').mean()
    sns.lineplot(ax=axes[0], x='Round', y='Loss', data=avg, linestyle="solid", legend=False)
    sns.lineplot(ax=axes[1], x='Round', y='Accuracy', data=avg, linestyle="solid", legend=False)
    sns.lineplot(ax=axes[2], x='Round', y='AUC', data=avg, linestyle="solid", legend=False)
    handles.append(mlines.Line2D([], [], color=plt.gca().lines[-1].get_color(), linestyle="solid"))
    # chars_to_remove = "[]()''"
    # trans = str.maketrans('', '', chars_to_remove)
    labels = [translate(fold) for fold in df.Fold.unique()]
    labels.append("Avg")

    axes[0].set_title('Loss')
    axes[1].set_title('Accuracy')
    axes[2].set_title('AUC by Fold and Round')

    # Define the legend for the whole figure, outside the plots
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=len(handles) // 2)
    plt.savefig(f"{output_dir}/fed_metrics.png")


def translate(s):
    chars_to_remove = "[]()'',"
    trans = str.maketrans(' ', '-', chars_to_remove)
    return s.strip().translate(trans)
