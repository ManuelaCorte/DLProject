from typing import Any, List
from tqdm import tqdm
from vgproject.utils.misc import custom_collate
from vgproject.data.dataset import VGDataset
from vgproject.utils.data_types import Result, Split, BboxType
from vgproject.models.baseline import Baseline
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from torch import Tensor
from vgproject.utils.config import Config

cfg = Config.get_instance()  # type: ignore

baseline = Baseline()

test_data = VGDataset(
    dir_path=cfg.dataset_path,
    split=Split.TEST,
    output_bbox_type=BboxType.XYXY,
)


dataloader: DataLoader[Any] = DataLoader(
    dataset=test_data,
    batch_size=cfg.batch_size,
    shuffle=False,
    collate_fn=custom_collate,
    drop_last=True,
)


batches_acc: List[Tensor] = []
for batch, bboxes in tqdm(dataloader):
    prediction: List[Result] = baseline.predict(batch)
    bbox_pred: Tensor = torch.stack([p.bounding_box for p in prediction]).to(
        baseline.device
    )
    bbox_gt: Tensor = bboxes.clone().detach().squeeze(1).to(baseline.device)
    iou = box_iou(bbox_pred, bbox_gt)
    acc = torch.mean(torch.diagonal(iou))
    batches_acc.append(acc)
    # print('Accuracy: ', acc)

accuracy = torch.mean(torch.stack(batches_acc))
print("IoU: ", accuracy)
