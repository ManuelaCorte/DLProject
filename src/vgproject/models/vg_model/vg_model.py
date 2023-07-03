from typing import List, OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
from vgproject.data.dataset import VGDataset
from vgproject.utils.misc import custom_collate, transform_sample
from vgproject.utils.config import Config
from vgproject.utils.bbox_types import BboxType
from vgproject.data.data_types import BatchSample, Split
from .visual_encoder import VisualEncoder
from .text_encoder import TextEncoder


class VGModel(nn.Module):
    def __init__(
        self,
        mlp_hidden_dim_1: int,
        mlp_hidden_dim_2: int,
    ) -> None:
        super().__init__()
        cfg = Config.get_instance().model  # type: ignore
        emb_dim: int = cfg["emb_dim"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_backbone: VisualEncoder = VisualEncoder().to(self.device)
        self.text_encoder: TextEncoder = TextEncoder().to(self.device)
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=4,
            batch_first=True,
            device=self.device,
        )
        self.reg_head: MLP = MLP(
            emb_dim * 5, 4, hidden_dim_1=mlp_hidden_dim_1, hidden_dim_2=mlp_hidden_dim_2
        ).to(self.device)

    def forward(self, batch: List[BatchSample]) -> Tensor:
        captions: Tensor = torch.stack([sample.caption for sample in batch]).squeeze(1)
        text_features: Tensor = self.text_encoder(captions)

        images: Tensor = torch.stack([sample.image for sample in batch])
        visual_features: OrderedDict[str, Tensor] = self.visual_backbone(images)

        attended_features: List[Tensor] = []
        for feature in visual_features.values():
            attention: Tensor = self.attention_layer(
                feature, text_features, text_features
            )
            attended_features.append(attention[0])

        aggregated_features: Tensor = torch.cat(attended_features, dim=1)
        return self.reg_head(aggregated_features)


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim_1: int, hidden_dim_2: int
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, output_dim),
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
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.model["batch_size"],
        collate_fn=custom_collate,
        drop_last=True,
    )
    test = VGModel(1024, 256)
    for batch, bbox in dataloader:
        out = test(batch)
        print(out)
        print(out.shape, bbox.shape)
