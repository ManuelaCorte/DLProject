from dataclasses import dataclass
from enum import Enum
from torch import Tensor, device


@dataclass(frozen=True)
class Sample:
    image_path: str
    caption: str
    bounding_box: Tensor


class BatchSample:
    def __init__(self, image: Tensor, caption: Tensor) -> None:
        self.image: Tensor = image
        self.caption: Tensor = caption

    def to(self, device: device | str) -> None:
        self.image = self.image.to(device)
        self.caption = self.caption.to(device)


@dataclass(frozen=True)
class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class Result:
    bounding_box: Tensor
    score: Tensor
