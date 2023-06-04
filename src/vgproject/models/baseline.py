import torch
from clip import clip
from ultralytics import YOLO
from torchvision.transforms import ToPILImage
from typing import List, Any
from vgproject.data.data_types import BatchSample, Result


class Baseline:
    def __init__(self) -> None:
        self.clip_model, self.clip_preprocessor = clip.load("RN50")
        self.yolo = YOLO()

    def predict(self, batch: List[BatchSample]) -> dict[str, Any]:
        images = [ToPILImage()(sample.image) for sample in batch]
        batch_bbox_predictions = self.yolo(images, max_det=50, verbose=False)
        results: List[Result] = []
        for sample, image_bboxes in zip(batch, batch_bbox_predictions, strict=True):
            crops: List[torch.Tensor] = []
            boxes: List[torch.Tensor] = []
            for bbox in image_bboxes.boxes:
                xmin, ymin, xmax, ymax = bbox.xyxy.int()[0]
                boxes.append(torch.tensor([xmin, ymin, xmax, ymax]))
                crops.append(sample.image[:, ymin:ymax, xmin:xmax])
            if len(crops) == 0:
                results.append(Result(torch.tensor([0, 0, 0, 0]), torch.tensor([0]))) 
                continue
            scores = self.compute_clip_similarity(crops, sample.caption)
            max_score = torch.argmax(scores)
            results.append(Result( boxes[max_score], scores[max_score]))
            # print(results)
        return results

    @torch.no_grad()
    def compute_clip_similarity(self, crops, caption) -> torch.Tensor:
        # text = clip.tokenize(caption)
        crops = [self.clip_preprocessor(ToPILImage()(crop)) for crop in crops]
        images = torch.stack(crops)
        # print(images.shape)
        logits_per_image, logits_per_text = self.clip_model(images, caption)
        return logits_per_image
    
    
