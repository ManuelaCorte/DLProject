from torch import Tensor
import torch.nn as nn
from torchvision.ops import generalized_box_iou_loss


class Loss:
    def __init__(self, l: float) -> None:
        self.l1_loss = nn.SmoothL1Loss(reduction="mean")
        self.giou_loss = generalized_box_iou_loss
        self.l: float = l
        self.loss: Tensor

    def compute(self, out: Tensor, bbox: Tensor) -> Tensor:
        # self.loss = self.giou_loss(out, bbox, reduction="mean") + self.l * self.l1_loss(
        #     out, bbox
        # )
        self.loss = self.l1_loss(out, bbox)
        return self.loss

    def to_float(self) -> float:
        return self.loss.item()
