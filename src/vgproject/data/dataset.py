import json
import pickle
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch import Tensor, device, tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.ops import box_convert

from vgproject.data.process import preprocess
from clip import tokenize
from vgproject.utils.data_types import BatchSample, BboxType, Sample, Split
from vgproject.utils.misc import transform_sample


# The Dataset contains samples with an image with a bounding box and a caption associated with the bounding box.
class VGDataset(Dataset[Tuple[BatchSample, Tensor]]):
    def __init__(
        self,
        dir_path: str,
        split: Split,
        output_bbox_type: BboxType,
        augment: bool,
        transform: bool = True,
        preprocessed: bool = False,
        preprocessed_path: str = "../data/processed/",
    ) -> None:
        super().__init__()
        self.dir_path: str = dir_path
        self.split: Split = split
        self.output_bbox_type: BboxType = output_bbox_type
        self.augment: bool = augment
        self.transform: bool = transform
        self.device: device = torch.device(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        if preprocessed:
            preprocess(dir_path, preprocessed_path, output_bbox_type)
            with open(
                preprocessed_path + f"{self.split.value}_samples.json", "rb"
            ) as samples:
                self.samples: List[Sample] = json.load(
                    samples, object_hook=Sample.fromJSON
                )
        else:
            self.samples: List[Sample] = self.get_samples()  # type: ignore

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, ref_id: int) -> Tuple[BatchSample, Tensor]:
        if self.transform:
            extended_caption: str = f"find the region that corresponds to the description {self.samples[ref_id].caption}"
            caption: Tensor = tokenize(extended_caption, truncate=True)  # type: ignore
            image, bbox = transform_sample(
                Image.open(self.samples[ref_id].image_path),
                self.samples[ref_id].bounding_box,
                self.augment,
            )
        else:
            caption: Tensor = tokenize(self.samples[ref_id].caption, truncate=True)  # type: ignore
            image = read_image(self.samples[ref_id].image_path)
            bbox = self.samples[ref_id].bounding_box
        return BatchSample(image, caption), bbox

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
                caption = self.get_caption(ref["sentences"])
                bbox = self.get_bounding_box(ref["ann_id"], instances)
                samples.append(Sample(image_path, caption, bbox))
        return samples

    def get_image_path(self, img_id: int, instances: Dict[str, Any]) -> str:
        image_name = next(
            image["file_name"] for image in instances["images"] if image["id"] == img_id
        )
        path = self.dir_path + "images/" + image_name
        return path

    def get_caption(self, captions: List[Dict[str, Any]]) -> str:
        longest_caption = captions[0]
        for caption in captions:
            if len(caption["sent"]) > len(longest_caption["sent"]):
                longest_caption = caption
        return longest_caption["sent"]

    # Bounding boxed converted to format compatible with yolo or torchvision
    def get_bounding_box(self, ann_id: int, instances: Dict[str, Any]) -> Tensor:
        bbox = next(
            ann["bbox"] for ann in instances["annotations"] if ann["id"] == ann_id
        )
        bounding_box: Tensor = tensor([])
        match self.output_bbox_type.name:
            case BboxType.XYXY.name:
                bounding_box = box_convert(
                    tensor([bbox]), in_fmt="xywh", out_fmt=BboxType.XYXY.value
                )
            case BboxType.XYWH.name:
                bounding_box = box_convert(
                    tensor([bbox]), in_fmt="xywh", out_fmt=BboxType.XYWH.value
                )
            case BboxType.CXCWH.name:
                bounding_box = box_convert(
                    tensor([bbox]), in_fmt="xywh", out_fmt=BboxType.CXCWH.value
                )

        return bounding_box
