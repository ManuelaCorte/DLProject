from dataclasses import dataclass
from enum import Enum
from typing import Any

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

    def to(self, device: device | str) -> Any:
        return self.__class__(self.image.to(device), self.caption.to(device))

    def __str__(self) -> str:
        return f"BatchSample(image={self.image.shape}, caption={self.caption.shape})"


@dataclass(frozen=True)
class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class Result:
    bounding_box: Tensor
    score: Tensor


# XYXY: top left and bottom right corners
# XYWH: top left corner, width and height
# CXCWH: center coordinates, width and height
@dataclass(frozen=True)
class BboxType(Enum):
    XYXY = "xyxy"
    XYWH = "xywh"
    CXCWH = "cxcwh"

    def __str__(self) -> str:
        return super().__str__()
