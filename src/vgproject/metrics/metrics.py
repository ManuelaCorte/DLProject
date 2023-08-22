from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


@dataclass(frozen=True)
class Metric(Enum):
    LOSS = "loss"
    ACCURACY_25 = "accuracy25"  # IoU > 0.25 -> 1 else 0
    ACCURACY_50 = "accuracy50"  # IoU > 0.5 -> 1 else 0
    ACCURACY_75 = "accuracy75"  # IoU > 0.75 -> 1 else 0
    ACCURACY_90 = "accuracy90"  # IoU > 0.9 -> 1 else 0
    IOU = "avg_iou"
    COSINE_SIMILARITY = "cosine_similarity"


@dataclass(frozen=True)
class Reduction(Enum):
    MEAN = "mean"
    SUM = "sum"
    NONE = "none"


class MetricsLogger:
    def __init__(self, metrics: Dict[str, List[float]] | None = None) -> None:
        self.metrics: Dict[str, List[float]] = {}
        if metrics is None:
            for metric in Metric:
                self.metrics[metric.value] = []
        else:
            self.metrics = metrics

    def update_metric(self, metrics: Dict[str, float]) -> None:
        for metric, value in metrics.items():
            self.metrics[metric].append(value)

    def get_metric(
        self, metric: Metric, red: Reduction = Reduction.NONE
    ) -> float | List[float]:
        values: List[float] = self.metrics[metric.value]
        match red.name:
            case Reduction.MEAN.name:
                return sum(values) / len(values)
            case Reduction.SUM.name:
                return sum(values)
            case Reduction.NONE.name:
                return values
            case _:
                raise ValueError(f"Reduction {red.name} doesn't exists")

    def __str__(self) -> str:
        res = "Metrics:\n"
        for metric, values in self.metrics.items():
            res += f"{metric}: {sum(values) / len(values)}\n"
        return res
