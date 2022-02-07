from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchmetrics.functional
from _solution.tasks.conditions.dataset import _SamplewiseUnpackedConditionsDataset


def plot_histogram():
    ds = _SamplewiseUnpackedConditionsDataset()
    list_n_nodes = [ds.__getitem__(idx)['n_nodes'][0] for idx in range(ds.__len__())]

    fig, ax = plt.subplots()
    ax.hist(list_n_nodes, bins=30)
    ax.set_xlabel("n_nodes")
    ax.set_ylabel("count")


def _get_plotted_data():
    path_outputs = Path(__file__).parent / "outputs"

    y_hat =torch.load(path_outputs / "y_hat.pickle")
    y =torch.load(path_outputs / "y.pickle")
    n_nodes =torch.load(path_outputs / "n_nodes.pickle")

    n_nodes_ascending = sorted(set(n_nodes.tolist()))
    for n in sorted(n_nodes_ascending, reverse=True):
        if (y[n_nodes >= n] == 1).int().sum() >= 30:
            max_significant_n = n
            break
    list_significant_ns = [n for n in n_nodes_ascending if n <= max_significant_n]
    list_aucs = [torchmetrics.functional.auroc(y_hat[n_nodes >= n], y.int()[n_nodes >= n]).item() for n in list_significant_ns]

    ds = _SamplewiseUnpackedConditionsDataset()
    all_n_nodes = torch.Tensor([ds.__getitem__(idx)['n_nodes'][0] for idx in range(ds.__len__())])
    all_ys = torch.Tensor([ds.__getitem__(idx)['y'] for idx in range(ds.__len__())])
    list_ratios = [all_ys[all_n_nodes > n].mean().item() for n in list_significant_ns]

    return list_significant_ns, list_aucs, list_ratios


def plot_auc_over_size():
    list_significant_ns, list_aucs, list_ratios = _get_plotted_data()
    fig, ax = plt.subplots()
    ax.scatter(list_significant_ns, list_aucs, label='AUC')
    ax.scatter(list_significant_ns, list_ratios, label='ratio of positives among ground truth')
    ax.set_xlabel("n_nodes >=")
    fig.legend(loc='center')


def main():
    plot_histogram()
    plot_auc_over_size()
    pass

if __name__ == '__main__':
    main()