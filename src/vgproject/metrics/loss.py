import torch.nn as nn
from torch import Tensor
from torchvision.ops import box_convert, generalized_box_iou_loss


class Loss:
    def __init__(self, l1: float, l2: float) -> None:
        self.l1_loss = nn.SmoothL1Loss(reduction="mean")
        self.giou_loss = generalized_box_iou_loss
        self.l1: float = l1
        self.l2: float = l2
        self.loss: Tensor

    def compute(self, prediction: Tensor, gt_bbox: Tensor) -> Tensor:
        bbox = box_convert(prediction, in_fmt="xywh", out_fmt="xyxy")
        self.loss = self.l1 * self.l1_loss(gt_bbox, bbox) + self.l2 * self.giou_loss(
            gt_bbox, bbox, reduction="mean"
        )
        return self.loss

    def to_float(self) -> float:
        return self.loss.item()
