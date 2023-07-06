from typing import List
from matplotlib import pyplot as plt

from vgproject.data.dataset import VGDataset
from .data_types import BboxType, Sample, Split
from .config import Config
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from torch import Tensor
import torch

# import albumentations as A


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


if __name__ == "__main__":
    cfg = Config.get_instance()  # type: ignore
    dataset = VGDataset(
        dir_path=cfg.dataset_path,
        split=Split.TEST,
        output_bbox_type=BboxType.XYXY,
        augment=False,
        preprocessed=True,
    )
    samples = dataset.samples[:6]
    print(samples[0].bounding_box, samples[0].bounding_box.shape)
    predictions = torch.stack(
        [sample.bounding_box for sample in dataset.samples[14:20]]
    )
    visualize(samples, predictions)
