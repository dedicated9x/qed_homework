import torch.nn as nn
from vit_pytorch.cct import cct_14
from torch.hub import load_state_dict_from_url


class CctFlower17(nn.Module):
    def __init__(self):
        super(CctFlower17, self).__init__()
        cct = cct_14(
            num_classes=1000,
            img_size=384,
            positional_embedding="learnable",
            n_conv_layers=2,
            kernel_size=7,
        )
        cct = self._load_imagenet_weights(cct)
        self.cct = self._replace_last_layer(cct)

    def forward(self, x):
        x = self.cct(x)
        return x

    def _load_imagenet_weights(self, cct):
        state_dict_image_net = load_state_dict_from_url(
            url='http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/finetuned/cct_14_7x2_384_imagenet.pth',
            progress=False
        )
        cct.load_state_dict(state_dict_image_net)
        return cct

    def _replace_last_layer(self, cct):
        cct.classifier.fc = nn.Linear(in_features=384, out_features=17, bias=True)
        return cct


def main():
    import torch
    import timeit

    def _measure_times():
        model = CctFlower17()
        model.to("cuda:0")
        n_repeats = 50

        for idx, batch_size in enumerate([1, 1, 2, 4, 8, 16]):
            list_batches = [torch.rand(batch_size, 3, 384, 384) for i in range(n_repeats)]
            iter_idxs = iter(range(n_repeats))
            time_all = timeit.timeit(
                lambda: model(list_batches[next(iter_idxs)].to("cuda:0")),
                number=n_repeats
            )
            time_single = time_all / n_repeats
            if idx > 0:
                print(f"Batch size: {batch_size}, then average time={time_single:.3f}s")

    _measure_times()


if __name__ == '__main__':
    main()
