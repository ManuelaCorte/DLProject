from typing import List, OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from vgproject.data.dataset import VGDataset
from vgproject.models.vg_model.text_encoder import TextEncoder
from vgproject.models.vg_model.visual_encoder import VisualEncoder
from vgproject.utils.config import Config
from vgproject.utils.data_types import BatchSample, BboxType, Split
from vgproject.utils.misc import count_parameters, custom_collate

from .decoder import Decoder
from .fusion_module import FusionModule


class VGModel(nn.Module):
    def __init__(
        self,
        cfg: Config,
    ) -> None:
        super().__init__()
        self.cfg: Config = cfg
        embed_dim: int = cfg.model.embed_dim
        mlp_hidden_dim: int = cfg.model.mlp_hidden_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.visual_encoder = VisualEncoder()

        self.text_encoder = TextEncoder(
            cfg.train.batch_size, cfg.model.clip_ctx_length, embed_dim
        )

        self.fusion_module: FusionModule = FusionModule(
            embed_dim, cfg.model.clip_embed_dim, cfg.model.proj_img_size
        ).to(self.device)

        self.decoder: Decoder = Decoder(
            embed_dim,
            cfg.model.proj_img_size,
            cfg.model.clip_ctx_length,
            cfg.model.decoder_heads,
            cfg.model.decoder_layers,
            cfg.model.decoder_dim_feedforward,
        ).to(self.device)

        self.reg_head: MLP = MLP(
            input_dim=embed_dim, output_dim=4, hidden_dim_1=mlp_hidden_dim
        ).to(self.device)

    def forward(self, batch: List[BatchSample]) -> Tensor:
        # Get text features
        text_sequence, global_text_features = self.text_encoder(
            torch.stack([sample.caption for sample in batch]).squeeze(1).to(self.device)
        )

        # Get image features
        visual_features: OrderedDict[str, Tensor] = self.visual_encoder(
            torch.stack([sample.image for sample in batch]).to(self.device)
        )

        # Fuse features
        fused_visual_features: Tensor = self.fusion_module(
            visual_features, global_text_features
        )

        # Transformer decoder
        reg_token: Tensor = self.decoder(fused_visual_features, text_sequence)

        # Regression head
        out: Tensor = self.reg_head(reg_token)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim_1: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


if __name__ == "__main__":
    cfg = Config()
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
        shuffle=False,
        drop_last=True,
    )
    test = VGModel(cfg)
    print(count_parameters(test))
    for batch, bbox in dataloader:
        out = test(batch)
        print(out, bbox)
        print(out.shape, bbox.shape)
        break
