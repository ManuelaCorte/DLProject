from pprint import pprint
from typing import Dict, List, Tuple

import torch
from torch import Tensor, device, optim
from torch.utils.data import DataLoader
from torchvision.ops import box_convert, box_iou
from tqdm import tqdm

from vgproject.metrics.loss import Loss
from vgproject.metrics.metrics import Metric
from vgproject.models.vg_model.vg_model import VGModel
from vgproject.utils.config import Config
from vgproject.utils.data_types import BatchSample
from vgproject.utils.visualization import plot_grad_flow


def train_one_epoch(
    dataloader: DataLoader[Tuple[BatchSample, Tensor]],
    model: VGModel,
    loss: Loss,
    optimizer: optim.Optimizer,
    img_size: int,
    device: device,
) -> Dict[str, float]:
    model.train()
    loss_list: List[Tensor] = []
    iou_list: List[Tensor] = []
    acc_25: List[Tensor] = []
    acc_50: List[Tensor] = []
    acc_75: List[Tensor] = []
    acc_90: List[Tensor] = []

    for idx, (batch, bbox) in enumerate(tqdm(dataloader, desc="Batches")):
        optimizer.zero_grad()
        # Move to gpu
        for sample in batch:
            sample = sample.to(device)
        bbox = bbox.to(device)

        # Forward pass
        out: Tensor = model(batch)

        # Loss and metrics
        out_xyxy = box_convert(out * img_size, in_fmt="xywh", out_fmt="xyxy")
        bbox_xyxy = box_convert(bbox * img_size, in_fmt="xywh", out_fmt="xyxy")
        batch_loss: Tensor = loss.compute(out_xyxy, bbox_xyxy)

        # Backward pass
        batch_loss.backward()
        # plot_grad_flow(model.named_parameters())

        optimizer.step()

        out_xyxy_det: Tensor = out_xyxy.detach()
        bbox_xyxy_det: Tensor = bbox_xyxy.detach()
        batch_iou: Tensor = torch.diagonal(box_iou(out_xyxy_det, bbox_xyxy_det))

        loss_list.append(batch_loss.detach())
        iou_list.append(batch_iou.mean())
        acc_25.append(accuracy(batch_iou, 0.25))
        acc_50.append(accuracy(batch_iou, 0.5))
        acc_75.append(accuracy(batch_iou, 0.75))
        acc_90.append(accuracy(batch_iou, 0.9))

        if (idx * len(batch)) % 4096 == 0:
            report: Dict[str, float] = {
                "Train loss": batch_loss.detach().item(),
                "Train avg iou": batch_iou.mean().item(),
            }
            pprint(f"Batches: {idx}, {report}")

    return {
        Metric.LOSS.value: torch.stack(loss_list).mean().item(),
        Metric.IOU.value: torch.stack(iou_list).mean().item(),
        Metric.ACCURACY_25.value: torch.stack(acc_25).mean().item(),
        Metric.ACCURACY_50.value: torch.stack(acc_50).mean().item(),
        Metric.ACCURACY_75.value: torch.stack(acc_75).mean().item(),
        Metric.ACCURACY_90.value: torch.stack(acc_90).mean().item(),
    }


@torch.no_grad()
def validate(
    dataloader: DataLoader[Tuple[BatchSample, Tensor]],
    model: VGModel,
    loss: Loss,
    img_size: int,
    device: torch.device,
) -> Dict[str, float]:
    # As accuracy we take the average IoU
    model.eval()
    loss_list: List[Tensor] = []
    iou_list: List[Tensor] = []
    acc_25: List[Tensor] = []
    acc_50: List[Tensor] = []
    acc_75: List[Tensor] = []
    acc_90: List[Tensor] = []

    for batch, bbox in tqdm(dataloader, desc="Batches"):
        # Move to gpu
        for sample in batch:
            sample.to(device)
        bbox = bbox.to(device)

        # Forward pass
        out: Tensor = model(batch)

        out = box_convert(out * img_size, in_fmt="xywh", out_fmt="xyxy").detach()
        bbox = box_convert(bbox * img_size, in_fmt="xywh", out_fmt="xyxy").detach()

        batch_loss: Tensor = loss.compute(out, bbox).detach()
        batch_iou: Tensor = torch.diagonal(box_iou(out, bbox)).detach()

        loss_list.append(batch_loss)
        iou_list.append(batch_iou.mean())
        acc_25.append(accuracy(batch_iou, 0.25))
        acc_50.append(accuracy(batch_iou, 0.5))
        acc_75.append(accuracy(batch_iou, 0.75))
        acc_90.append(accuracy(batch_iou, 0.9))

    return {
        Metric.LOSS.value: torch.stack(loss_list).mean().item(),
        Metric.IOU.value: torch.stack(iou_list).mean().item(),
        Metric.ACCURACY_25.value: torch.stack(acc_25).mean().item(),
        Metric.ACCURACY_50.value: torch.stack(acc_50).mean().item(),
        Metric.ACCURACY_75.value: torch.stack(acc_75).mean().item(),
        Metric.ACCURACY_90.value: torch.stack(acc_90).mean().item(),
    }


def accuracy(iou: Tensor, threshold: float) -> Tensor:
    return torch.tensor(len(iou[iou >= threshold]) / len(iou))


if __name__ == "__main__":
    box = torch.tensor(
        [[0.4, 0.4, 0.6, 0.6], [0.5, 0.5, 0.7, 0.7], [0.1, 0.1, 0.2, 0.2]]
    )
    box_gt = torch.rand(3, 4)
    t = torch.diagonal(box_iou(box, box_gt))
    print(t, t.shape)
    print(accuracy(t, 0.5))
