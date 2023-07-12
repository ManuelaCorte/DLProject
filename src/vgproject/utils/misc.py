import random
from typing import Any, Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import Tensor, device

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
    device: device,
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
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=[]),
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
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=[]),
        )

    transformed_sample: Dict[str, Any] = trans(
        image=np.array(image), bboxes=box.tolist()
    )

    bbox_tensor: Tensor = (
        torch.tensor(transformed_sample["bboxes"][0], requires_grad=True) / target_size
    )
    # print(bbox_tensor)
    return transformed_sample["image"], bbox_tensor.to(torch.float32)


# # Save top 5 checkpoints either locally or on wandb
# class Checkpoint:
#     def __init__(self, save_locally: bool, wandb: bool) -> None:
#         self.save_locally = save_locally
#         self.wandb = wandb
#         self.top_5_checkpoints: List[Tuple[int, float]] = []

#     def save_checkpoint(
#         self, model: nn.Module, optimizer: optim.Optimizer, epoch: int, loss: float
#     ) -> None:
#         checkpoint: Dict[str, Any] = {
#             "epoch": epoch,
#             "model_state_dict": model.state_dict(),
#             "optimizer_state_dict": optimizer.state_dict(),
#             "loss": loss,
#         }

#         if self.save_locally:
#             torch.save(checkpoint, f"checkpoints/checkpoint_{epoch}.pt")
#         if self.wandb:
#             wandb.log_artifact(f"checkpoints/checkpoint_{epoch}.pt")

#         self.top_5_checkpoints.append((epoch, loss))
#         self.top_5_checkpoints.sort(key=lambda x: x[1])
#         if len(self.top_5_checkpoints) > 5:
#             self.top_5_checkpoints.pop(0)


def init_torch(seed: int = 41) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
