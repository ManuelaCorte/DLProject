from typing import List, Tuple
import torch
from vgproject.utils.data_types import BatchSample
from torch import Tensor, device
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


# Transform image according to CLIP preprocess function
# Normalize bounding box coordinates to be independent of image size
def transform_sample(
    image: Image.Image,
    box: Tensor,
    target_size: int = 224,
    device: device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor]:
    x: int
    y: int
    x, y = image.size[0], image.size[1]

    # Same transformation as in the CLIP preprocess function
    trans = T.Compose(
        transforms=[
            T.Resize(
                size=(target_size, target_size),
                interpolation=T.InterpolationMode.BICUBIC,
                max_size=None,
                antialias="warn",
            ),
            T.CenterCrop(target_size),
            T.ToTensor(),
        ]
    )

    image_tensor: Tensor = trans(image).to(device)  # type: ignore
    if image_tensor.shape[0] == 1:
        image_tensor = image_tensor.repeat(3, 1, 1)

    image_tensor = T.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )(image_tensor)

    xmin, ymin, xmax, ymax = box.squeeze(0)

    xmin_norm: float = xmin.item() / x
    ymin_norm: float = ymin.item() / y
    xmax_norm: float = xmax.item() / x
    ymax_norm: float = ymax.item() / y
    bbox_tensor: Tensor = torch.tensor(
        [xmin_norm, ymin_norm, xmax_norm, ymax_norm], device=device
    )
    return image_tensor, bbox_tensor
