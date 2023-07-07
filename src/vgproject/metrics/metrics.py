from typing import Dict, List
from enum import Enum
from dataclasses import dataclass


@dataclass(frozen=True)
class Metric(Enum):
    ACCURACY = "accuracy"  # IoU > 0.5 -> 1 else 0
    IOU = "iou"
    COSINE_SIMILARITY = "cosine_similarity"
    CLIP_SCORE = "clip_score"  # max(100 * cos(E_I, E_C), 0)


@dataclass(frozen=True)
class Reduction(Enum):
    MEAN = "mean"
    SUM = "sum"
    NONE = "none"


class MetricsLogger:
    def __init__(self) -> None:
        self.metrics: Dict[Metric, List[float]] = {}

    def update_metric(self, metric: Metric, value: float) -> None:
        if metric in self.metrics:
            self.metrics[metric].append(value)
        else:
            self.metrics[metric] = [value]

    def get_metric(
        self, metric: Metric, reduction: Reduction = Reduction.MEAN
    ) -> float:
        raise NotImplementedError

    def __str__(self) -> str:
        res = "Metrics:\n"
        for metric, values in self.metrics.items():
            res += f"{metric.value}: {sum(values) / len(values)}\n"
        return res
