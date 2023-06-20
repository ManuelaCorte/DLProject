from typing import List, OrderedDict
import torch
import torch.nn as nn
from torch import Tensor

from vgproject.data.data_types import BatchSample
from .visual_encoder import VisualEncoder
from .text_encoder import TextEncoder


# TODO: maybe add LN and REG token
class VGModel(nn.Module):
    def __init__(
        self,
        emb_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_backbone: VisualEncoder = VisualEncoder().to(self.device)
        self.text_encoder: TextEncoder = TextEncoder().to(self.device)
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=4,
            batch_first=True,
            device=self.device,
        )
        self.reg_head = MLP(emb_dim * 5, 4, 256).to(self.device)

    def forward(self, sample: BatchSample) -> Tensor:
        visual_features_dict: OrderedDict[str, Tensor] = self.visual_backbone(
            sample.image
        )
        text_features: Tensor = self.text_encoder(sample.caption)
        attended_features: List[Tensor] = []
        for visual_features in visual_features_dict.values():
            attention: Tensor = self.attention_layer(
                visual_features, text_features, text_features
            )
            attended_features.append(attention[0])

        aggregated_features: Tensor = torch.cat(attended_features, dim=1)
        return self.reg_head(aggregated_features)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


if __name__ == "__main__":
    test = VGModel()
    sample = BatchSample(
        image=torch.rand(1, 3, 224, 224),
        caption=torch.rand(1, 77),
    )
    out = test(sample)
    print(out)
    print(out.shape)
