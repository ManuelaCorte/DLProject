from typing import Any, Dict, List, Tuple
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from vgproject.data.dataset import VGDataset
from vgproject.models.vg_model.vg_model import VGModel
from vgproject.utils.data_types import Split, BatchSample, BboxType
from vgproject.metrics.loss import Loss
from vgproject.utils.misc import custom_collate
from vgproject.utils.config import Config


@torch.no_grad()
def eval() -> None:
    cfg = Config.get_instance()  # type: ignore
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

    checkpoint: Dict[str, Any] = torch.load(cfg.logging["path"])
    model: VGModel = checkpoint["model"]
    model.eval()
    loss = Loss(0.9)
    losses: List[float] = []
    for batch, gt_bboxes in tqdm(dataloader):
        for sample in batch:
            sample.to(device)

        predictions: Tensor = model(batch)
        batch_loss = loss.compute(predictions, gt_bboxes)
        losses.append(batch_loss.item())
    print(sum(losses) / len(losses))
