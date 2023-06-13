from dataclasses import dataclass
from enum import Enum
from torch import Tensor


@dataclass(frozen=True)
class Sample:
    image_path: str
    caption: str
    bounding_box: Tensor


@dataclass(frozen=True)
class BatchSample:
    image: Tensor
    caption: Tensor


@dataclass(frozen=True)
class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class Result:
    bounding_box: Tensor
    score: Tensor
