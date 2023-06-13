from clip.model import CLIP
import torch
from clip import clip
from torchvision.transforms.transforms import Compose
from ultralytics import YOLO
from torchvision.transforms import ToPILImage, GaussianBlur
from typing import List
from ultralytics.yolo.engine.results import Results
from vgproject.data.data_types import BatchSample, Result
from tqdm import tqdm
from PIL import Image
from torch import Tensor


class Baseline:
    def __init__(self) -> None:
        self.device = torch.device(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")
        self.clip_model: CLIP
        self.clip_preprocessor: Compose
        self.clip_model, self.clip_preprocessor = clip.load("RN50", device=self.device)
        self.yolo: YOLO = YOLO()

    def predict(self, batch: List[BatchSample]) -> List[Result]:
        images: List[Image.Image] = [ToPILImage()(sample.image) for sample in batch]
        batch_bbox_predictions: List[Results] | None = self.yolo(
            images, max_det=50, verbose=False, device=self.device
        )
        results: List[Result] = []
        if batch_bbox_predictions is None:
            return results
        for sample, image_bboxes in tqdm(
            zip(batch, batch_bbox_predictions, strict=True)
        ):
            image: Tensor = sample.image
            crops: List[torch.Tensor] = []
            blurs: List[torch.Tensor] = []
            boxes: List[torch.Tensor] = []
            for bbox in image_bboxes.boxes:  # type: ignore
                xmin, ymin, xmax, ymax = bbox.xyxy.int()[0]  # type: ignore
                boxes.append(torch.tensor([xmin, ymin, xmax, ymax]))
                blurred: Tensor = GaussianBlur(25, 50)(sample.image)
                blurred[:, ymin:ymax, xmin:xmax] = image[:, ymin:ymax, xmin:xmax]
                blurs.append(blurred)
                crops.append(image[:, ymin:ymax, xmin:xmax])
            if len(crops) == 0:
                results.append(Result(torch.tensor([0, 0, 0, 0]), torch.tensor([0])))
                continue
            scores: Tensor = self.compute_clip_similarity(crops, blurs, sample.caption)
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
        images_blurs: Tensor = torch.stack(tensors=blurs)
        # print(images.shape)
        logits_per_image_blurs, _ = self.clip_model(images_blurs, caption)
        return logits_per_image_crops + logits_per_image_blurs
