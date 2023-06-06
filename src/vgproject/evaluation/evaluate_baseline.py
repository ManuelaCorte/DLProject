from vgproject.utils.bbox_types import BboxType
from vgproject.utils.misc import custom_collate
from vgproject.data.dataset import VGDataset
from vgproject.data.data_types import Split
from vgproject.models.baseline import Baseline
from clip import clip
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

baseline = Baseline()

test_data = VGDataset(
    "../data/raw/refcocog/",
    split=Split.TEST,
    output_bbox_type=BboxType.XYXY,
    transform_image= T.Compose([T.Resize(640), T.ToTensor()]),
    transform_text=clip.tokenize,
)

dataloader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=custom_collate, drop_last=True)


batches_acc = []
for batch, bboxes in dataloader:
    prediction = baseline.predict(batch)
    bbox_pred = torch.stack([p.bounding_box for p in prediction])
    bbox_gt = bboxes.clone().detach().squeeze(1)
    # print(bbox_pred.shape, bbox_gt.shape)
    iou = box_iou(bbox_pred, bbox_gt)
    acc = torch.mean(torch.diagonal(iou))
    batches_acc.append(acc)
    print('Accuracy: ', acc)

accuracy = torch.mean(torch.stack(batches_acc))
print('Accuracy: ', accuracy)


# tensor([0.5292]) 0.5263 Overall accuracy
