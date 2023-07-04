from torch.utils.data import Dataset
from torchvision.ops import box_convert
import torch
import pickle
import json
from vgproject.data.process import preprocess
from vgproject.utils.data_types import BboxType, Sample, Split, BatchSample
from typing import Any, Dict, List, Tuple
from torchvision.io import read_image
from torch import Tensor, tensor, device
from PIL import Image
from clip import clip


# The Dataset contains samples with an image with a bounding box and a caption associated with the bounding box.
class VGDataset(Dataset[Tuple[BatchSample, Tensor]]):
    def __init__(
        self,
        dir_path: str,
        split: Split,
        output_bbox_type: BboxType,
        transform_image: Any = None,
        preprocessed: bool = False,
        preprocessed_path: str = "../data/processed/",
    ) -> None:
        super().__init__()
        self.dir_path: str = dir_path
        self.split: Split = split
        self.output_bbox_type: BboxType = output_bbox_type
        self.transform_image: Any = transform_image
        self.device: device = torch.device(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        if preprocessed:
            preprocess(dir_path, preprocessed_path)
            with open(
                preprocessed_path + f"{self.split.value}_samples.p", "rb"
            ) as samples:
                self.samples: List[Sample] = pickle.load(samples)
        else:
            self.samples: List[Sample] = self.get_samples()  # type: ignore

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, ref_id: int) -> Tuple[BatchSample, Tensor]:
        caption: Tensor = clip.tokenize(self.samples[ref_id].caption)

        if self.transform_image is not None:
            image_trans, bbox_trans = self.transform_image(
                Image.open(self.samples[ref_id].image_path),
                self.samples[ref_id].bounding_box,
                device=self.device,
            )
            sample_trans: BatchSample = BatchSample(image_trans, caption).to(
                self.device
            )
            return sample_trans, bbox_trans
        else:
            image: Tensor = read_image(self.samples[ref_id].image_path)
            bbox: Tensor = self.samples[ref_id].bounding_box.to(device=self.device)
            sample: BatchSample = BatchSample(image, caption).to(self.device)
            return sample, bbox

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
        return f"find the region that corresponds to the description {longest_caption['sent']}"

    # Bounding boxed converted to format compatible with yolo or torchvision
    def get_bounding_box(self, ann_id: int, instances: Dict[str, Any]) -> Tensor:
        bbox = next(
            ann["bbox"] for ann in instances["annotations"] if ann["id"] == ann_id
        )
        bounding_box: Tensor = tensor([])
        match self.output_bbox_type:
            case BboxType.XYXY:
                bounding_box = box_convert(
                    tensor([bbox]), in_fmt="xywh", out_fmt=BboxType.XYXY.value
                )
            case BboxType.XYWH:
                bounding_box = box_convert(
                    tensor([bbox]), in_fmt="xywh", out_fmt=BboxType.XYWH.value
                )
            case BboxType.CXCWH:
                bounding_box = box_convert(
                    tensor([bbox]), in_fmt="xywh", out_fmt=BboxType.CXCWH.value
                )

        return bounding_box
