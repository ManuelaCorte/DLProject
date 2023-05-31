import torch
from clip import clip
from ultralytics import YOLO
from torchvision.transforms import ToPILImage
from typing import List, TypedDict, Any
from torchvision.io import read_image
from data.data_types import Sample


class Baseline:
    def __init__(self) -> None:
        self.clip_model, self.clip_preprocessor = clip.load("RN50")
        self.yolo = YOLO()

    def predict(self, sample: Sample) -> dict[str, Any]:
        bbox_predictions = self.yolo(sample.image_path, max_det=50, verbose=False)
        image = read_image(sample.image_path)
        objects: List[torch.Tensor] = []
        boxes: List[torch.Tensor] = []
        if len(bbox_predictions[0].boxes) == 0:
            return {
                "bbox": torch.tensor([0, 0, 0, 0]).unsqueeze(0),
                "score": torch.tensor([0]),
            }
        for xmin, ymin, xmax, ymax in bbox_predictions[0].boxes.xyxy.int():
            boxes.append(torch.tensor([xmin, ymin, xmax, ymax]))
            objects.append(image[:, ymin:ymax, xmin:xmax])
        scores = self.compute_clip_similarity(objects, sample.caption)
        max_score = torch.argmax(scores)
        return {
            "bbox": boxes[max_score].clone().detach().unsqueeze(0),
            "score": scores[max_score],
        }

    @torch.no_grad()
    def compute_clip_similarity(self, crops, caption) -> torch.Tensor:
        text = clip.tokenize(caption)
        crops = [self.clip_preprocessor(ToPILImage()(crop)) for crop in crops]
        images = torch.stack(crops)
        # print(images.shape)
        logits_per_image, logits_per_text = self.clip_model(images, text)
        return logits_per_image
