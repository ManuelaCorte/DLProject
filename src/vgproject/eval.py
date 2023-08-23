import json
from typing import Any, Dict, List, Tuple

import torch
import torchvision.transforms.functional as T
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import box_iou, box_convert
from tqdm import tqdm

from vgproject.data.dataset import VGDataset
from vgproject.metrics.loss import Loss
from vgproject.metrics.metrics import Metric, MetricsLogger
from vgproject.models.vg_model.vg_model import VGModel
from vgproject.utils.config import Config
from vgproject.utils.data_types import BatchSample, BboxType, Split
from vgproject.utils.misc import custom_collate
import clip


@torch.no_grad()
def eval() -> None:
    cfg = Config()
    dataset = VGDataset(
        dir_path=cfg.dataset_path,
        split=Split.TEST,
        output_bbox_type=BboxType.XYXY,
        augment=False,
        preprocessed=True,
    )

    dataloader: DataLoader[Tuple[BatchSample, Tensor]] = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=custom_collate,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint: Dict[str, Any] = torch.load("../model0.pth", map_location=device)
    model: VGModel = VGModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loss = Loss(cfg.train.l1, cfg.train.l2)

    metrics: MetricsLogger = MetricsLogger()
    loss_list: List[Tensor] = []
    iou_list: List[Tensor] = []
    acc_25: List[Tensor] = []
    acc_50: List[Tensor] = []
    acc_75: List[Tensor] = []
    acc_90: List[Tensor] = []
    cos_sim_pred: List[Tensor] = []
    cos_sim_gt: List[Tensor] = []

    for idx, (batch, bboxes) in enumerate(tqdm(dataloader, desc="Batches")):
        # Move to gpu
        for sample in batch:
            sample = sample.to(device)
        bboxes = bboxes.to(device)

        # Forward pass
        out: Tensor = model(batch)

        # Loss and metrics
        out_xyxy: Tensor = box_convert(
            out * cfg.model.img_size, in_fmt="xywh", out_fmt="xyxy"
        )
        bbox_xyxy: Tensor = box_convert(
            bboxes * cfg.model.img_size, in_fmt="xywh", out_fmt="xyxy"
        )
        batch_loss: Tensor = loss.compute(out_xyxy, bbox_xyxy)

        out_xyxy_det: Tensor = out_xyxy.detach()
        bbox_xyxy_det: Tensor = bbox_xyxy.detach()
        batch_iou: Tensor = torch.diagonal(box_iou(out_xyxy_det, bbox_xyxy_det))

        loss_list.append(batch_loss.detach())
        iou_list.append(batch_iou.mean())

        acc_25.append(accuracy(batch_iou, 0.25))
        acc_50.append(accuracy(batch_iou, 0.5))
        acc_75.append(accuracy(batch_iou, 0.75))
        acc_90.append(accuracy(batch_iou, 0.9))

        image_features_gt = torch.stack(
            [
                T.crop(sample.image, bbox[1], bbox[0], bbox[3], bbox[2])
                for sample, bbox in zip(batch, bboxes * cfg.model.img_size)
            ]
        )
        image_features_pred = torch.stack(
            [
                T.crop(sample.image, bbox[1], bbox[0], bbox[3], bbox[2])
                for sample, bbox in zip(batch, out * cfg.model.img_size)
            ]
        )
        text_features = torch.stack([sample.text for sample in batch])
        cos_sim_pred.append(
            compute_cosine_similarity(image_features_pred, text_features)
        )
        cos_sim_gt.append(compute_cosine_similarity(image_features_gt, text_features))

    json.dump(
        {
            Metric.LOSS.value: torch.stack(loss_list).mean().item(),
            Metric.IOU.value: torch.stack(iou_list).mean().item(),
            Metric.ACCURACY_25.value: torch.stack(acc_25).mean().item(),
            Metric.ACCURACY_50.value: torch.stack(acc_50).mean().item(),
            Metric.ACCURACY_75.value: torch.stack(acc_75).mean().item(),
            Metric.ACCURACY_90.value: torch.stack(acc_90).mean().item(),
            Metric.COSINE_SIMILARITY.value
            + " prediction": torch.stack(cos_sim_pred).mean().item(),
            Metric.COSINE_SIMILARITY.value
            + " ground truth": torch.stack(cos_sim_gt).mean().item(),
        },
        open("test_metrics.json", "w"),
    )

    print(metrics)


def compute_cosine_similarity(image_features: Tensor, text_features: Tensor) -> Tensor:
    clip_model, preprocess = clip.load("RN_50")
    image_features = clip_model.encode_image(preprocess(image_features))
    text_features = clip_model.encode_text(text_features)

    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    return torch.nn.functional.cosine_similarity(image_norm, text_norm, dim=-1)


def accuracy(iou: Tensor, threshold: float) -> Tensor:
    return torch.tensor(len(iou[iou >= threshold]) / len(iou))


if __name__ == "__main__":
    eval()
