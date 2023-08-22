import torch.nn as nn
from torch import Tensor
from torchvision.ops import distance_box_iou_loss


class Loss:
    def __init__(self, l1: float, l2: float) -> None:
        self.l1_loss = nn.SmoothL1Loss(reduction="mean")
        self.iou_loss = distance_box_iou_loss
        self.l1: float = l1
        self.l2: float = l2
        self.loss: Tensor

    # Both bounding boxex tensors are in xyxy format
    def compute(self, prediction: Tensor, gt_bbox: Tensor) -> Tensor:
        self.loss = self.l1 * self.l1_loss(
            gt_bbox, prediction
        ) + self.l2 * self.iou_loss(gt_bbox, prediction, reduction="mean")
        return self.loss

    def to_float(self) -> float:
        return self.loss.item()
