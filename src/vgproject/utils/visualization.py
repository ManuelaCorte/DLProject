from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder  # type: ignore
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchviz import make_dot  # type: ignore

from vgproject.data.dataset import VGDataset
from vgproject.metrics.loss import Loss
from vgproject.models.vg_model.vg_model import VGModel
from vgproject.utils.misc import custom_collate

from .config import Config
from .data_types import BatchSample, BboxType, Sample, Split


def visualize(samples: List[Sample], predictions: Tensor) -> None:
    ncols = 3
    nrows = int(len(samples) / ncols)
    print(nrows, ncols)
    _, ax = plt.subplots(nrows, ncols, figsize=(24, 24))
    for i, sample in enumerate(samples):
        img: Tensor = read_image(sample.image_path)
        # bboxes: Tensor = torch.stack(
        #     [
        #         unnormalize_bbox(img, sample.bounding_box),
        #         unnormalize_bbox(img, predictions[i]),
        #     ]
        # )
        bboxes: Tensor = torch.stack(
            [
                sample.bounding_box,
                predictions[i],
            ]
        ).squeeze(1)
        result: Tensor = draw_bounding_boxes(img, bboxes, width=2, colors=(255, 0, 0))
        ax[i // ncols, i % ncols].imshow(result.permute(1, 2, 0))
        ax[i // ncols, i % ncols].set_title(sample.caption)
        ax[i // ncols, i % ncols].axis("off")
    plt.axis("off")
    plt.show()


def unnormalize_bbox(image: Tensor, bbox: Tensor) -> Tensor:
    x: int
    y: int
    y, x = image.shape[1], image.shape[2]
    xmin, ymin, xmax, ymax = bbox.squeeze(0)
    xmin_unnorm: float = xmin.item() * x
    ymin_unnorm: float = ymin.item() * y
    xmax_unnorm: float = xmax.item() * x
    ymax_unnorm: float = ymax.item() * y
    return torch.tensor([xmin_unnorm, ymin_unnorm, xmax_unnorm, ymax_unnorm])


def visualize_network(model: torch.nn.Module, batch: List[BatchSample]) -> None:
    output: Tensor = model(batch)
    make_dot(
        output.mean(), params=dict(model.named_parameters()), show_attrs=True
    ).render("model_graph", directory="../runs", format="png")


def find_lr() -> None:
    config = Config()
    model = VGModel(config).train()

    loss_func = Loss(config.train.l1, config.train.l2)

    # Separate parameters to train
    params: List[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(
        [
            {"params": params, "lr": config.train.lr},
        ]
    )
    train_dataset: VGDataset = VGDataset(
        dir_path=config.dataset_path,
        split=Split.TRAIN,
        output_bbox_type=BboxType.XYWH,
        augment=True,
        preprocessed=True,
    )
    train_dataloader: DataLoader[Tuple[BatchSample, Tensor]] = DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        collate_fn=custom_collate,
        num_workers=2,
        shuffle=True,
        drop_last=True,
    )
    lr_finder = LRFinder(
        model,
        optimizer,
        loss_func,
        device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    )
    lr_finder.range_test(train_dataloader, end_lr=100, num_iter=100, diverge_th=20)
    f, ax = plt.subplots(1)
    lr_finder.plot(suggest_lr=True, ax=ax)  # to inspect the loss-learning rate graph
    plt.show()


def plot_grad_flow(named_parameters: Iterator[Tuple[str, nn.Parameter]]) -> None:
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                raise RuntimeError(f"Gradient of {n} is None")
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )


if __name__ == "__main__":
    config = Config()

    model = VGModel(config).train()
    train_dataset: VGDataset = VGDataset(
        dir_path=config.dataset_path,
        split=Split.TRAIN,
        output_bbox_type=BboxType.XYWH,
        augment=True,
        preprocessed=True,
    )
    train_dataloader: DataLoader[Tuple[BatchSample, Tensor]] = DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        collate_fn=custom_collate,
        num_workers=2,
        shuffle=True,
        drop_last=True,
    )
    for batch, bbox in train_dataloader:
        visualize_network(model, batch)
        break
