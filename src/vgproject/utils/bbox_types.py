from enum import Enum
from dataclasses import dataclass

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
