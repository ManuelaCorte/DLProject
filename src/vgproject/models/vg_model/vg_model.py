from typing import List, OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
from vgproject.data.dataset import VGDataset
from vgproject.utils.misc import custom_collate, transform_sample
from vgproject.utils.config import Config
from vgproject.utils.bbox_types import BboxType
from clip import clip
from vgproject.data.data_types import BatchSample, Split
from .visual_encoder import VisualEncoder
from .text_encoder import TextEncoder


class VGModel(nn.Module):
    def __init__(
        self,
        emb_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.emb_dim: int = emb_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_backbone: VisualEncoder = VisualEncoder().to(self.device)
        self.text_encoder: TextEncoder = TextEncoder().to(self.device)
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=4,
            batch_first=True,
            device=self.device,
        )
        self.pooling: nn.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(emb_dim)
        self.reg_head: MLP = MLP(emb_dim, 4, 256).to(self.device)

    def forward(self, batch: List[BatchSample]) -> Tensor:
        captions: Tensor = torch.stack([sample.caption for sample in batch]).squeeze(1)
        text_features: Tensor = self.text_encoder(captions)

        images: Tensor = torch.stack([sample.image for sample in batch])
        visual_features_dict: OrderedDict[str, Tensor] = self.visual_backbone(images)

        attended_features: List[Tensor] = []
        for visual_features in visual_features_dict.values():
            attention: Tensor = self.attention_layer(
                visual_features, text_features, text_features
            )
            attended_features.append(attention[0])

        aggregated_features: Tensor = torch.cat(attended_features, dim=1)
        pooled_features: Tensor = self.pooling(aggregated_features)
        return self.reg_head(pooled_features)


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
    cfg = Config.get_instance()  # type: ignore
    dataset = VGDataset(
        dir_path=cfg.dataset["path"],
        split=Split.TEST,
        output_bbox_type=BboxType.XYXY,
        transform_image=transform_sample,
        transform_text=clip.tokenize,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.model["batch_size"],
        collate_fn=custom_collate,
        drop_last=True,
    )
    test = VGModel()
    for batch, bbox in dataloader:
        out = test(batch)
        print(out)
        print(out.shape)
