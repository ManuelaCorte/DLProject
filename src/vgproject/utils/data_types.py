from dataclasses import dataclass
from enum import Enum
from typing import Any

from torch import Tensor, device


class Sample:
    def __init__(self, image_path: str, caption: str, bounding_box: Tensor) -> None:
        self.image_path = image_path
        self.caption = caption
        self.bounding_box = bounding_box

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "caption": self.caption,
            "bounding_box": self.bounding_box.tolist(),
        }

    @staticmethod
    def fromJSON(json: dict[str, Any]) -> Any:
        return Sample(json["image_path"], json["caption"], Tensor(json["bounding_box"]))


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
