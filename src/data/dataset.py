from torch.utils.data import Dataset
from torchvision.ops import box_convert
import torch
import pickle
import json
from .data_types import Sample, Split
from utils.bbox_types import BboxType
from pathlib import Path
from typing import List


# The Dataset contains samples with an image with a bounding box and a caption associated with the bounding box.
class VGDataset(Dataset):
    def __init__(
        self, dir_path, split, output_bbox_type, transform_image, transform_text
    ) -> None:
        super().__init__()
        self.dir_path: str = dir_path
        self.split: Split = split
        self.output_bbox_type: BboxType = output_bbox_type
        self.samples: List[Sample] = self.get_samples()
        self.transform_image = transform_image
        self.transform_text = transform_text

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, ref_id) -> Sample:
        return self.samples[ref_id]

    def get_samples(self) -> List[Sample]:
        with open(self.dir_path + "annotations/instances.json", "r") as inst, open(
            self.dir_path + "annotations/refs(umd).p", "rb"
        ) as refs:
            instances = json.load(inst)
            references = pickle.load(refs)

        samples: List[Sample] = []
        for ref in references:
            if self.split.value == ref["split"]:
                image_path = self.get_image_path(ref["image_id"], instances)
                caption = self.get_longest_caption(ref["sentences"])
                bbox = self.get_bounding_box(ref["ann_id"], instances)
                samples.append(Sample(image_path, caption, bbox))
        return samples

    def get_image_path(self, img_id, instances) -> Path:
        image_name = next(
            image["file_name"] for image in instances["images"] if image["id"] == img_id
        )
        path = self.dir_path + "images/" + image_name
        return path

    def get_longest_caption(self, captions) -> str:
        longest_caption = captions[0]
        for caption in captions:
            if len(caption["sent"]) > len(longest_caption["sent"]):
                longest_caption = caption
        return longest_caption["sent"]

    # Bounding boxed converted to format compatible with yolo or torchvision
    def get_bounding_box(self, ann_id, instances) -> torch.Tensor:
        bbox = next(
            ann["bbox"] for ann in instances["annotations"] if ann["id"] == ann_id
        )
        bounding_box = torch.tensor([])
        match self.output_bbox_type:
            case BboxType.XYXY:
                bounding_box = box_convert(
                    torch.tensor([bbox]), in_fmt="xywh", out_fmt=BboxType.XYXY.value
                )
            case BboxType.XYWH:
                bounding_box = box_convert(
                    torch.tensor([bbox]), in_fmt="xywh", out_fmt=BboxType.XYWH.value
                )
            case BboxType.CXCWH:
                bounding_box = box_convert(
                    torch.tensor([bbox]), in_fmt="xywh", out_fmt=BboxType.CXCWH.value
                )
            case _:
                raise ValueError(
                    f"Invalid output bounding box type: {self.output_bbox_type}"
                )
        return bounding_box
