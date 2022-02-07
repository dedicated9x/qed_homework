import torch
import torch.utils.data
from src.data.conditions_prediction_dataset import ConditionsPredictionToyTask
from src.featurization.gat_featurizer import GatGraphFeaturizer
from _solution.common.utils import pprint_sample


class SparseConditionsDataset(torch.utils.data.Dataset):
    def __init__(self, min_idx=0, max_idx=4403):
        self.min_idx = min_idx
        self.max_idx = max_idx
        self.featurizer = GatGraphFeaturizer(n_jobs=1)

        dataset_loader = ConditionsPredictionToyTask()
        self.wrapped_samples_sparse = self.featurizer.load(dataset_loader.feat_dir)
        self.wrapped_targets = dataset_loader.load_metadata()['ortho_lithiation'].values.astype(int)

    def __len__(self):
        return self.max_idx - self.min_idx

    def __getitem__(self, idx):
        global_idx = idx + self.min_idx

        return {
            "n_nodes": self.wrapped_samples_sparse['n_nodes'][global_idx],
            "atom": self.wrapped_samples_sparse['atom'][global_idx],
            "bond": self.wrapped_samples_sparse['bond'][global_idx],
            "y": self.wrapped_targets[global_idx],
        }

class DenseConditionsDataset(SparseConditionsDataset):
    def __init__(self, *args, **kwargs):
        super(DenseConditionsDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        sample = super(DenseConditionsDataset, self).__getitem__(idx)
        atom_sparse = sample['atom']
        bond_sparse = sample['bond']

        atom_dense = atom_sparse.todense()
        bond_dense = bond_sparse.todense()

        return {
            "n_nodes": torch.tensor(sample["n_nodes"]),
            "atom": torch.Tensor(atom_dense).squeeze(),
            "bond": torch.Tensor(bond_dense).squeeze(),
            "y": torch.tensor(sample["y"]).type(torch.float32)
        }

class _SamplewiseUnpackedConditionsDataset(SparseConditionsDataset):
    def __init__(self, *args, **kwargs):
        super(_SamplewiseUnpackedConditionsDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        global_idx = idx + self.min_idx

        sample_sparse = {
            col: self.wrapped_samples_sparse[col][global_idx:(global_idx + 1)]
            for col in self.wrapped_samples_sparse.keys()
        }
        sample_dense = self.featurizer.unpack(sample_sparse)
        return {
            "n_nodes": sample_dense['n_nodes'],
            "atom": sample_dense['atom'].squeeze(),
            "bond": sample_dense['bond'].squeeze(),
            "y": self.wrapped_targets[global_idx],
        }

def main():
    def _show_dataset_differences():
        list_representative_idxs = [0, 23]
        for ds_cls in [
            SparseConditionsDataset,
            DenseConditionsDataset,
            _SamplewiseUnpackedConditionsDataset
        ]:
            print("\nClass =", ds_cls)
            ds = ds_cls(max_idx=3522)
            for idx in list_representative_idxs:
                print(f"  IDX = {idx}")
                sample = ds.__getitem__(idx)
                pprint_sample(sample)

    _show_dataset_differences()

if __name__ == '__main__':
    main()
