from typing import Any, List, Tuple
from tqdm import tqdm
from vgproject.utils.bbox_types import BboxType
from vgproject.utils.misc import custom_collate
from vgproject.data.dataset import VGDataset
from vgproject.data.data_types import BatchSample, Result, Split
from vgproject.models.baseline import Baseline
from clip import clip
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from torch import Tensor

baseline = Baseline()

test_data = VGDataset(
    dir_path="../data/raw/refcocog/",
    split=Split.TEST,
    output_bbox_type=BboxType.XYXY,
    transform_image=T.Compose([T.Resize(640), T.ToTensor()]),
    transform_text=clip.tokenize,
    dependencies=False,
)

batch_size = 16
dataloader: DataLoader[Any] = DataLoader(
    dataset=test_data,
    batch_size=batch_size,
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
    print(bbox_pred.shape, bbox_gt.shape)
    iou = box_iou(bbox_pred, bbox_gt)
    acc = torch.mean(torch.diagonal(iou))
    batches_acc.append(acc)
    # print('Accuracy: ', acc)

accuracy = torch.mean(torch.stack(batches_acc))
print("Accuracy: ", accuracy)

# 0.5263 Overall accuracy 0.5340
