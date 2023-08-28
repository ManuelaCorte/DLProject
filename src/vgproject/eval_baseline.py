import json
from typing import Any, Dict, List, Tuple

import clip
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as FT
from pprint import pprint
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import box_convert, box_iou
from tqdm import tqdm

from vgproject.data.dataset import VGDataset
from vgproject.metrics.loss import Loss
from vgproject.metrics.metrics import Metric, MetricsLogger
from vgproject.models.baseline import Baseline
from vgproject.utils.config import Config
from vgproject.utils.data_types import BatchSample, BboxType, Split, Result
from vgproject.utils.misc import custom_collate


def compute_cosine_similarity(
    image_features: Tensor, text_features: Tensor, transform_func: T.Compose | None
) -> Tensor:
    if transform_func is not None:
        image_features = clip_model.encode_image(transform_func(image_features))
    else:
        image_features = clip_model.encode_image(image_features)
    text_features = clip_model.encode_text(text_features.squeeze_(1))

    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    return torch.nn.functional.cosine_similarity(image_norm, text_norm, dim=-1).detach()


# Compute the fraction of samples st IoU > threshold
def accuracy(iou: Tensor, threshold: float) -> Tensor:
    return torch.tensor(len(iou[iou >= threshold]) / len(iou))


@torch.no_grad()
def eval_baseline() -> None:
    cfg = Config()
    test_data = VGDataset(
        dir_path=cfg.dataset_path,
        split=Split.TEST,
        output_bbox_type=BboxType.XYXY,
        transform=False,
        augment=False,
        preprocessed=True,
    )

    dataloader: DataLoader[Any] = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate,
        drop_last=True,
    )

    baseline = Baseline()

    metrics: MetricsLogger = MetricsLogger()
    iou_list: List[Tensor] = []
    acc_25: List[Tensor] = []
    acc_50: List[Tensor] = []
    acc_75: List[Tensor] = []
    acc_90: List[Tensor] = []
    cos_sim_pred: List[Tensor] = []
    cos_sim_gt: List[Tensor] = []

    transformation = T.Compose(
        [
            T.Resize(cfg.model.img_size),
            T.CenterCrop(cfg.model.img_size),
            lambda x: x / 255.0,
            lambda x: x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x,
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    for idx, (batch, bboxes) in enumerate(tqdm(dataloader, desc="Batches")):
        for sample in batch:
            sample = sample.to(baseline.device)
        bboxes = bboxes.to(baseline.device)

        prediction: List[Result] = baseline.predict(batch)
        bbox_pred: Tensor = torch.stack([p.bounding_box for p in prediction]).to(
            baseline.device
        )
        bbox_gt: Tensor = bboxes.clone().detach().squeeze(1).to(baseline.device)

        batch_iou: Tensor = torch.diagonal(box_iou(bbox_gt, bbox_pred))

        iou_list.append(batch_iou.mean())
        acc_25.append(accuracy(batch_iou, 0.25))
        acc_50.append(accuracy(batch_iou, 0.5))
        acc_75.append(accuracy(batch_iou, 0.75))
        acc_90.append(accuracy(batch_iou, 0.9))

        box_gt_xyxy = box_convert(bbox_gt, in_fmt="xywh", out_fmt="xyxy")
        box_pred_xywh = box_convert(bbox_pred, in_fmt="xyxy", out_fmt="xywh")
        image_features_gt = torch.stack(
            [
                FT.crop(
                    sample.image,
                    int(bbox[1].item()),
                    int(bbox[0].item()),
                    int(bbox[3].item()),
                    int(bbox[2].item()),
                )
                for sample, bbox in zip(batch, bbox_gt)
            ]
        ).to(baseline.device)

        image_features_pred = torch.stack(
            [
                FT.crop(
                    sample.image,
                    int(bbox[1].item()),
                    int(bbox[0].item()),
                    int(bbox[3].item()),
                    int(bbox[2].item()),
                )
                for sample, bbox in zip(batch, box_pred_xywh)
            ]
        ).to(baseline.device)

        text_features = torch.stack([sample.caption for sample in batch]).to(
            baseline.device
        )
        cos_sim_pred.append(
            compute_cosine_similarity(
                image_features_pred, text_features, transformation
            )
        )
        cos_sim_gt.append(
            compute_cosine_similarity(image_features_gt, text_features, transformation)
        )

    json.dump(
        {
            Metric.IOU.value: torch.stack(iou_list).cpu().numpy().tolist(),
            Metric.ACCURACY_25.value: torch.stack(acc_25).cpu().numpy().tolist(),
            Metric.ACCURACY_50.value: torch.stack(acc_50).cpu().numpy().tolist(),
            Metric.ACCURACY_75.value: torch.stack(acc_75).cpu().numpy().tolist(),
            Metric.ACCURACY_90.value: torch.stack(acc_90).cpu().numpy().tolist(),
            Metric.COSINE_SIMILARITY.value
            + " prediction": torch.stack(cos_sim_pred).cpu().numpy().tolist(),
            Metric.COSINE_SIMILARITY.value
            + " ground truth": torch.stack(cos_sim_gt).cpu().numpy().tolist(),
        },
        open("../test_metrics_baseline.json", "w"),
    )

    pprint(
        object={
            Metric.IOU.value: torch.stack(iou_list).mean().item(),
            Metric.ACCURACY_25.value: torch.stack(acc_25).mean().item(),
            Metric.ACCURACY_50.value: torch.stack(acc_50).mean().item(),
            Metric.ACCURACY_75.value: torch.stack(acc_75).mean().item(),
            Metric.ACCURACY_90.value: torch.stack(acc_90).mean().item(),
            Metric.COSINE_SIMILARITY.value
            + " prediction": torch.stack(cos_sim_pred).mean().item(),
            Metric.COSINE_SIMILARITY.value
            + " ground truth": torch.stack(cos_sim_gt).mean().item(),
        }
    )


if __name__ == "__main__":
    clip_model, _ = clip.load("RN50")
    eval_baseline()
