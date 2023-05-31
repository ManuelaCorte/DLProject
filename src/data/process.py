import pickle
import json
from pathlib import Path
from typing import List, TypedDict

raw_data_path = "data/raw/refcocog/"
processed_data_path = "data/processed/refcocog/"
processed_data_path = "/media/manuela/SSK Storage/processed/"


def get_image_path(img_id: int) -> Path:
    # find image in instances with id euqal to img_id
    image_name = next(
        image["file_name"] for image in instances["images"] if image["id"] == img_id
    )
    image_path = Path(raw_data_path) / "images" / Path(image_name)
    return image_path


def get_longest_caption(captions: List) -> str:
    longest_caption = captions[0]
    for caption in captions:
        if len(caption["sent"]) > len(longest_caption["sent"]):
            longest_caption = caption
    return longest_caption["sent"]


# Bounding boxed converted to format compatible with yolo or torchvision
def get_bounding_box(ann_id) -> List[int]:
    bbox = next(ann["bbox"] for ann in instances["annotations"] if ann["id"] == ann_id)
    return bbox


References = TypedDict(
    "References",
    {
        "image_id": int,
        "split": str,
        "sentences": List,
        "file_name": str,
        "category_id": int,
        "ann_id": int,
        "sent_ids": List,
        "ref_id": int,
    },
)
with open(raw_data_path + "annotations/instances.json", "r") as inst, open(
    raw_data_path + "annotations/refs(umd).p", "rb"
) as ref:
    instances = json.load(inst)
    references = pickle.load(ref)

processed_data_train = []
processed_data_val = []
processed_data_test = []

for ref in references:
    image = get_image_path(ref["image_id"])  # type: ignore
    caption = get_longest_caption(ref["sentences"])  # type: ignore
    bbox = get_bounding_box(ref["ann_id"])  # type: ignore

    match ref["split"]:  # type: ignore
        case "train":
            processed_data_train.append(
                {
                    "image_id": image,
                    "caption": caption,
                    "bounding_box": bbox,
                }
            )
        case "val":
            processed_data_val.append(
                {
                    "image_id": image,
                    "caption": caption,
                    "bounding_box": bbox,
                }
            )
        case "test":
            processed_data_test.append(
                {
                    "image_id": image,
                    "caption": caption,
                    "bounding_box": bbox,
                }
            )
        case _:
            raise ValueError("Invalid split value. Must be one of: train, val, test")


pickle.dump(
    processed_data_train, open(processed_data_path + "processed_data_train.p", "xb")
)
pickle.dump(
    processed_data_val, open(processed_data_path + "processed_data_val.p", "xb")
)
pickle.dump(
    processed_data_test, open(processed_data_path + "processed_data_test.p", "xb")
)
