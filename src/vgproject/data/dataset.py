from torch.utils.data import Dataset
from torchvision.ops import box_convert
import torch
import pickle
import json
from .data_types import Sample, Split, BatchSample
from vgproject.utils.bbox_types import BboxType
from typing import Any, Dict, List, Tuple
from torchvision.io import read_image
import spacy
from spacy.tokens import Doc, Span
from torch import Tensor, tensor
from PIL import Image


# The Dataset contains samples with an image with a bounding box and a caption associated with the bounding box.
class VGDataset(Dataset[Tuple[BatchSample, Tensor]]):
    def __init__(
        self,
        dir_path: str,
        split: Split,
        output_bbox_type: BboxType,
        transform_image: Any,
        transform_text: Any,
        dependencies: bool = False,
    ) -> None:
        super().__init__()
        self.dir_path: str = dir_path
        self.split: Split = split
        self.output_bbox_type: BboxType = output_bbox_type
        self.transform_image = transform_image
        self.transform_text = transform_text
        self.text_processor = spacy.load(name="en_core_web_lg")
        self.device = torch.device(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.samples: List[Sample] = self.get_samples(dependencies=dependencies)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, ref_id: int) -> Tuple[BatchSample, Tensor]:
        caption = self.transform_text(self.samples[ref_id].caption)

        if self.transform_image is not None:
            image_trans, bbox_trans = self.transform_image(
                Image.open(self.samples[ref_id].image_path),
                self.samples[ref_id].bounding_box,
                device=self.device,
            )
            sample = BatchSample(image_trans, caption).to(self.device)
            return sample, bbox_trans
        else:
            image = read_image(self.samples[ref_id].image_path)
            bbox: Tensor = self.samples[ref_id].bounding_box.to(device=self.device)
            sample = BatchSample(image, caption).to(self.device)
            return sample, bbox

    def get_samples(self, dependencies: bool = False) -> List[Sample]:
        with open(self.dir_path + "annotations/instances.json", "r") as inst, open(
            self.dir_path + "annotations/refs(umd).p", "rb"
        ) as refs:
            instances = json.load(inst)
            references = pickle.load(refs)

        samples: List[Sample] = []
        for ref in references:
            if self.split.value == ref["split"]:
                image_path = self.get_image_path(ref["image_id"], instances)
                caption = self.get_caption(ref["sentences"], dependencies)
                bbox = self.get_bounding_box(ref["ann_id"], instances)
                samples.append(Sample(image_path, caption, bbox))
        return samples

    def get_image_path(self, img_id: int, instances: Dict[str, Any]) -> str:
        image_name = next(
            image["file_name"] for image in instances["images"] if image["id"] == img_id
        )
        path = self.dir_path + "images/" + image_name
        return path

    def get_caption(self, captions: List[Dict[str, Any]], dependencies: bool) -> str:
        longest_caption = captions[0]
        for caption in captions:
            if len(caption["sent"]) > len(longest_caption["sent"]):
                longest_caption = caption
        if dependencies:
            return self.get_relevant_caption(
                doc=self.text_processor(longest_caption["sent"])
            )
        return longest_caption["sent"]

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

    def get_relevant_caption(self, doc: Doc) -> str:
        # for chunck in doc.noun_chunks:
        #         return chunck.text
        # for token in doc:
        #     # Mainly -ing verbs
        #     if("acl" in token.dep_):
        #         subtree = list(token.subtree)
        #         end = subtree[0].i
        #         sent = doc[0:end]
        #         if len(sent) > 1:
        #             return sent

        # subject which/that something
        for token in doc:
            if "relcl" in token.dep_:
                subtree = list(token.subtree)
                end: int = subtree[0].i
                sent: Span = doc[0:end]
                if len(sent) > 1:
                    return str(sent)

        # Subjects
        for token in doc:
            if "subj" in token.dep_:
                subtree = list(token.subtree)
                start: int = subtree[0].i
                end = subtree[-1].i + 1
                sent_subj: Span = doc[start:end]
                if len(sent_subj) > 1:
                    return str(sent_subj)
        return str(doc)
