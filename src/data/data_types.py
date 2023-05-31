from dataclasses import dataclass
from pathlib import Path
import torch
from enum import Enum


@dataclass(frozen=True)
class Sample:
    image_path: Path
    caption: str
    bounding_box: torch.Tensor


# @dataclass(frozen=True)
# class BatchSample:
#     image: torch.Tensor
#     caption: torch.Tensor


@dataclass(frozen=True)
class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
