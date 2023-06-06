from typing import List, Tuple
import torch
from vgproject.data.data_types import BatchSample


def custom_collate(batch) -> Tuple[List[BatchSample], torch.Tensor]:
    bboxes = []
    samples = []
    for sample, bbox in batch:
        samples.append(BatchSample(sample.image, sample.caption))
        bboxes.append(bbox)
    return samples, torch.stack(bboxes)