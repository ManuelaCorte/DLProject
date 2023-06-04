from typing import List
from vgproject.utils.bbox_types import BboxType
from vgproject.data.dataset import VGDataset
from vgproject.data.data_types import Split
from vgproject.models.baseline import Baseline
from clip import clip
import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.ops import box_iou

baseline = Baseline()

dataset = VGDataset(
    "../data/raw/refcocog/",
    split=Split.TEST,
    output_bbox_type=BboxType.XYXY,
    transform_image=baseline.clip_preprocessor,
    transform_text=clip.tokenize,
)


iou: List[torch.Tensor] = []
for sample in dataset:
    prediction = baseline.predict(sample)
    bbox_pred = prediction["bbox"]
    # bbox_gt = torch.tensor(sample.bounding_box)
    bbox_gt = sample.bounding_box.clone().detach()
    iou.append(box_iou(bbox_pred, bbox_gt))

    # bboxes = torch.cat([prediction['bbox'], bbox_gt])
    # image = read_image(sample.image_path)
    # res = draw_bounding_boxes(image, boxes=bboxes, colors=['red', 'blue'])
    # T.ToPILImage()(res).show()
    # plt.imshow(res.permute(1, 2, 0)) # type: ignore
    # plt.title(sample.caption)
    # plt.show()

accuracy = 0.0
for score in iou:
    accuracy += score.data[0]
print(accuracy / len(iou))

# tensor([0.5292]) Overall accuracy
