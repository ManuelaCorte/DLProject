import json
import os
import pickle
from typing import Any, Dict, List, Tuple

from torch import Tensor, tensor
from torchvision.ops import box_convert
from tqdm import tqdm

from vgproject.utils.data_types import BboxType, Sample, Split


def get_samples(
    dir_path: str, bbox_type: BboxType
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    with open(dir_path + "annotations/instances.json", "r") as inst, open(
        dir_path + "annotations/refs(umd).p", "rb"
    ) as refs:
        instances = json.load(inst)
        references = pickle.load(refs)
    train_samples: List[Sample] = []
    val_samples: List[Sample] = []
    test_samples: List[Sample] = []
    for ref in tqdm(references, desc=f"Processing dataset"):
        image_path = get_image_path(dir_path, ref["image_id"], instances)
        caption = get_caption(ref["sentences"])
        bbox = get_bounding_box(ref["ann_id"], instances, bbox_type)
        split = ref["split"]
        # print(split)
        match split:
            case Split.TRAIN.value:
                train_samples.append(Sample(image_path, caption, bbox))
            case Split.VAL.value:
                val_samples.append(Sample(image_path, caption, bbox))
            case Split.TEST.value:
                test_samples.append(Sample(image_path, caption, bbox))
            case _:
                raise ValueError(f"Invalid split: {split}")
    return train_samples, val_samples, test_samples


def get_image_path(dir_path: str, img_id: int, instances: Dict[str, Any]) -> str:
    image_name = next(
        image["file_name"] for image in instances["images"] if image["id"] == img_id
    )
    path = dir_path + "images/" + image_name
    return path


def get_caption(captions: List[Dict[str, Any]]) -> str:
    longest_caption = captions[0]
    for caption in captions:
        if len(caption["sent"]) > len(longest_caption["sent"]):
            longest_caption = caption
    return longest_caption["sent"]


# Bounding boxed converted to format compatible with yolo or torchvision
def get_bounding_box(
    ann_id: int, instances: Dict[str, Any], bbox_type: BboxType
) -> Tensor:
    bbox = next(ann["bbox"] for ann in instances["annotations"] if ann["id"] == ann_id)
    bounding_box: Tensor = tensor([])
    match bbox_type.name:
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


# If the files already exist, don't preprocess again
def preprocess(in_path: str, out_path: str, bbox_type: BboxType) -> None:
    if (
        os.path.exists(f"{out_path}train_samples.json")
        and os.path.exists(f"{out_path}val_samples.json")
        and os.path.exists(f"{out_path}test_samples.json")
    ):
        return
    train_samples, val_samples, test_samples = get_samples(in_path, bbox_type)

    json.dump(
        train_samples,
        open(f"{out_path}train_samples.json", "w"),
        default=Sample.as_dict,
    )

    json.dump(
        val_samples,
        open(f"{out_path}val_samples.json", "w"),
        default=Sample.as_dict,
    )

    json.dump(
        test_samples,
        open(f"{out_path}test_samples.json", "w"),
        default=Sample.as_dict,
    )


if __name__ == "__main__":
    preprocess("../data/raw/refcocog/", "../data/processed/", BboxType.XYWH)

    train: List[Sample] = json.load(
        open("../data/processed/train_samples.json", "r"), object_hook=Sample.fromJSON
    )
    val: List[Sample] = json.load(
        open("../data/processed/val_samples.json", "r"), object_hook=Sample.fromJSON
    )
    test: List[Sample] = json.load(
        open("../data/processed/test_samples.json", "r"), object_hook=Sample.fromJSON
    )
    print(len(train), len(val), len(test))
    print(train[0].image_path, train[0].caption, train[0].bounding_box.shape)
