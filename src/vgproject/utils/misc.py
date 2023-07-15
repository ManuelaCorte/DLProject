import json
import random
from typing import Any, Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import Tensor

from .data_types import BatchSample


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
    augment: bool,
    target_size: int = 224,
) -> Tuple[Tensor, Tensor]:
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Same transformation as in the CLIP preprocess function
    if augment:
        trans = A.Compose(
            transforms=[
                A.Resize(target_size, target_size, interpolation=cv2.INTER_CUBIC, p=1),
                A.CenterCrop(
                    target_size,
                    target_size,
                    always_apply=True,
                ),
                A.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                    max_pixel_value=255.0,
                    always_apply=True,
                ),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
                A.GaussianBlur(p=1),
                A.PixelDropout(dropout_prob=0.02),
                A.Rotate(limit=20),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=[]),
        )
    else:
        trans = A.Compose(
            transforms=[
                A.Resize(target_size, target_size, interpolation=cv2.INTER_CUBIC, p=1),
                A.CenterCrop(
                    target_size,
                    target_size,
                    always_apply=True,
                ),
                A.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=[]),
        )

    transformed_sample: Dict[str, Any] = trans(
        image=np.array(image), bboxes=box.tolist()
    )

    bbox_tensor: Tensor = torch.tensor(transformed_sample["bboxes"][0]) / target_size
    # print(bbox_tensor)
    return transformed_sample["image"], bbox_tensor.to(torch.float32)


def init_torch(seed: int = 41) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    train_samples = json.load(open("../data/processed/train_samples.json", "r"))[0]

    print(
        transform_sample(
            Image.open(train_samples["image_path"]),
            torch.tensor(train_samples["bounding_box"]),
            True,
        )
    )
