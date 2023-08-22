from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from vgproject.data.dataset import VGDataset
from vgproject.metrics.loss import Loss
from vgproject.models.vg_model.vg_model import VGModel
from vgproject.utils.config import Config
from vgproject.utils.data_types import BatchSample, BboxType, Split
from vgproject.utils.misc import custom_collate


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

    checkpoint: Dict[str, Any] = torch.load("../model9.pth", map_location=device)
    model: VGModel = VGModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loss = Loss(cfg.train.l1, cfg.train.l2)
    losses: List[float] = []
    for batch, gt_bboxes in tqdm(dataloader):
        for sample in batch:
            sample.to(device)

        predictions: Tensor = model(batch)
        batch_loss = loss.compute(predictions, gt_bboxes)
        losses.append(batch_loss.item())
    print(sum(losses) / len(losses))


if __name__ == "__main__":
    eval()
