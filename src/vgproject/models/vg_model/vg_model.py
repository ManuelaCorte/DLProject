from typing import List, OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
from vgproject.data.dataset import VGDataset
from vgproject.utils.misc import custom_collate
from vgproject.utils.config import Config
from vgproject.utils.data_types import BatchSample, Split, BboxType
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
        self.attention_layers: nn.ModuleList = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=emb_dim,
                    num_heads=4,
                    batch_first=True,
                    device=self.device,
                )
                for _ in range(5)
            ]
        )
        self.reg_head: MLP = MLP(
            emb_dim * 5, 4, hidden_dim_1=mlp_hidden_dim_1, hidden_dim_2=mlp_hidden_dim_2
        ).to(self.device)

    def forward(self, batch: List[BatchSample]) -> Tensor:
        captions: Tensor = torch.stack([sample.caption for sample in batch]).squeeze(1)
        text_features: Tensor = self.text_encoder(captions).unsqueeze(1)

        images: Tensor = torch.stack([sample.image for sample in batch])
        visual_features: OrderedDict[str, Tensor] = self.visual_backbone(images)

        attended_features: List[Tensor] = []
        for i, visual_feature in enumerate(visual_features.values()):
            vis_feature: Tensor = visual_feature.unsqueeze(1)
            # print(vis_feature.shape, text_features.shape)
            attended_feature: Tensor = self.attention_layers[i](
                query=text_features, key=vis_feature, value=vis_feature
            )[0].squeeze(1)
            attended_features.append(attended_feature)

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
        dir_path=cfg.dataset_path,
        split=Split.VAL,
        output_bbox_type=BboxType.XYXY,
        augment=True,
        preprocessed=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=3,
        collate_fn=custom_collate,
        shuffle=True,
        drop_last=True,
    )
    test = VGModel(1024, 256)
    for batch, bbox in dataloader:
        out = test(batch)
        print(out, bbox)
        print(out.shape, bbox.shape)
        break
