from typing import List

import torch
from clip import clip
from clip.model import CLIP
from PIL import Image
from torch import Tensor
from torchvision.transforms import GaussianBlur, ToPILImage
from torchvision.transforms.transforms import Compose
from tqdm import tqdm
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
        # for sample in batch:
        #     print(f"image: {sample.image.is_cuda}, caption: {sample.caption.is_cuda}")
        images: List[Image.Image] = [ToPILImage()(sample.image) for sample in batch]
        batch_bbox_predictions: List[Results] = self.yolo(
            images, max_det=50, verbose=False, device=self.device
        )  # type: ignore
        results: List[Result] = []

        for sample, image_bboxes in zip(batch, batch_bbox_predictions, strict=True):
            image: Tensor = sample.image
            crops: List[torch.Tensor] = []
            blurs: List[torch.Tensor] = []
            boxes: List[torch.Tensor] = []

            for bbox in image_bboxes.boxes:
                bbox = bbox.to(self.device)
                xmin, ymin, xmax, ymax = bbox.xyxy.int()[0]
                boxes.append(torch.tensor([xmin, ymin, xmax, ymax], device=self.device))
                blurred: Tensor = GaussianBlur(25, 50)(sample.image).to(self.device)
                blurred[:, ymin:ymax, xmin:xmax] = image[:, ymin:ymax, xmin:xmax]
                blurs.append(blurred)
                crops.append(image[:, ymin:ymax, xmin:xmax])
            if len(crops) == 0:
                results.append(
                    Result(
                        torch.tensor([0, 0, 1, 1], device=self.device),
                        torch.tensor([0], device=self.device),
                    )
                )
                continue
            scores: Tensor = self.compute_clip_similarity(
                crops, blurs, sample.caption.to(self.device)
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
        blurs: List[torch.Tensor],
        caption: torch.Tensor,
    ) -> torch.Tensor:
        # text = clip.tokenize(caption)
        crops = [self.clip_preprocessor(ToPILImage()(crop)) for crop in crops]
        blurs = [self.clip_preprocessor(ToPILImage()(blur)) for blur in blurs]
        images_crops: Tensor = torch.stack(tensors=crops).to(device=self.device)
        # print(images.shape)
        logits_per_image_crops, _ = self.clip_model(images_crops, caption)
        images_blurs: Tensor = torch.stack(
            tensors=blurs,
        ).to(device=self.device)
        # print(images.shape)
        logits_per_image_blurs, _ = self.clip_model(images_blurs, caption)

        return logits_per_image_crops + logits_per_image_blurs
