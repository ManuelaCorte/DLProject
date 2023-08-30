from typing import List

import torch
from clip import clip
from clip.model import CLIP
from PIL import Image
from torch import Tensor
from torchvision.transforms import ToPILImage
from torchvision.transforms.transforms import Compose
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results

from vgproject.utils.data_types import BatchSample, Result


class Baseline:
    def __init__(self) -> None:
        self.device = torch.device(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")
        self.clip_model: CLIP
        self.clip_preprocessor: Compose
        self.clip_model, self.clip_preprocessor = clip.load(
            name="RN50", device=self.device
        )
        self.yolo: YOLO = YOLO()

    def predict(self, batch: List[BatchSample]) -> List[Result]:
        images: List[Image.Image] = [ToPILImage()(sample.image) for sample in batch]
        batch_bbox_predictions: List[Results] = self.yolo(
            images, max_det=50, verbose=False, device=self.device
        )  # type: ignore
        results: List[Result] = []

        for sample, image_bboxes in zip(batch, batch_bbox_predictions, strict=True):
            image: Tensor = sample.image
            crops: List[torch.Tensor] = []
            boxes: List[torch.Tensor] = []

            for bbox in image_bboxes.boxes.xyxy:
                bbox = bbox.to(self.device)
                xmin, ymin, xmax, ymax = bbox.int()

                crops.append(image[:, ymin:ymax, xmin:xmax])
                boxes.append(bbox)

            if len(crops) == 0:
                results.append(
                    Result(
                        torch.tensor([0, 0, 0, 0], device=self.device),
                        torch.tensor([0], device=self.device),
                    )
                )
                continue
            scores: Tensor = self.compute_clip_similarity(
                crops, sample.caption.to(self.device)
            )
            max_score: Tensor = torch.argmax(scores)
            results.append(Result(boxes[max_score], scores[max_score]))
            # print(results)
        return results

    # Combination of cropping and blurring for the object proposals
    @torch.no_grad()
    def compute_clip_similarity(
        self,
        crops: List[torch.Tensor],
        caption: torch.Tensor,
    ) -> torch.Tensor:
        images = torch.stack(
            [self.clip_preprocessor(ToPILImage()(crop)) for crop in crops]
        ).to(self.device)
        # print(images.shape)
        logits_per_image, logits_per_text = self.clip_model(images, caption)
        return logits_per_image
