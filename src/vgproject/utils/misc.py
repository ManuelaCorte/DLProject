from typing import List, Tuple
import torch
from vgproject.utils.data_types import BatchSample
from torch import Tensor, device
import numpy as np
import torchvision.transforms as T
from PIL import Image


def custom_collate(
    batch: List[Tuple[BatchSample, torch.Tensor]]
) -> Tuple[List[BatchSample], torch.Tensor]:
    bboxes: List[torch.Tensor] = []
    samples: List[BatchSample] = []
    for sample, bbox in batch:
        samples.append(BatchSample(sample.image, sample.caption))
        bboxes.append(bbox)
    return samples, torch.stack(bboxes)


# Bounding box already in the correct format
def transform_sample(
    image: Image.Image,
    box: Tensor,
    target_size: int = 224,
    device: device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor]:
    x: int
    y: int
    x, y = image.size[0], image.size[1]

    x_scale: float = target_size / x
    y_scale: float = target_size / y

    trans = T.Compose(
        transforms=[
            T.Resize((target_size, target_size)),
            T.CenterCrop(target_size),
            T.PILToTensor(),
        ]
    )
    image_tensor: Tensor = trans(image).to(device)  # type: ignore
    if image_tensor.shape[0] == 1:
        image_tensor = image_tensor.repeat(3, 1, 1)

    xmin, ymin, xmax, ymax = box.squeeze(0)

    xmin = np.round(xmin * x_scale)
    ymin = np.round(ymin * y_scale)
    xmax = np.round(xmax * x_scale)
    ymax = np.round(ymax * y_scale)

    bbox_tensor: Tensor = torch.tensor([xmin, ymin, xmax, ymax], device=device)
    return image_tensor, bbox_tensor
