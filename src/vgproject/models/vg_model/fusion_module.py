from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor, device


class FusionModule(nn.Module):
    def __init__(self, emb_dim: int, clip_emb_dim: int, proj_img_size: int) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_projection = nn.Sequential(
            nn.Linear(in_features=clip_emb_dim, out_features=clip_emb_dim),
            nn.BatchNorm1d(clip_emb_dim, device=self.device),
        ).to(self.device)

        self.vis_l4_projection = _conv_layer(
            input_dim=clip_emb_dim * 2,
            output_dim=clip_emb_dim,
            kernel_size=3,
            padding=1,
            device=self.device,
        )[
            :2
        ]  # Remove ReLU
        self.norm_layer = nn.Sequential(
            nn.BatchNorm2d(
                clip_emb_dim,
                device=self.device,
            ),
            nn.ReLU(),
        ).to(self.device)
        self.vis_l3_projection = _conv_layer(
            input_dim=clip_emb_dim + clip_emb_dim,
            output_dim=emb_dim,
            kernel_size=3,
            padding=1,
            device=self.device,
        )
        self.vis_l2_projection = _conv_layer(
            input_dim=emb_dim + emb_dim,
            output_dim=emb_dim,
            kernel_size=3,
            padding=1,
            device=self.device,
        )
        self.aggregation = _conv_layer(
            input_dim=clip_emb_dim + emb_dim + emb_dim,
            output_dim=emb_dim,
            kernel_size=1,
            padding=0,
            device=self.device,
        )

        self.coord_conv = nn.Sequential(
            CoordConv(emb_dim + 2, emb_dim),
            _conv_layer(
                input_dim=emb_dim,
                output_dim=emb_dim,
                kernel_size=3,
                padding=1,
                device=self.device,
            ),
        )

    def forward(
        self, visual_features: Tuple[Tensor, Tensor, Tensor], text_features: Tensor
    ) -> Tensor:
        visual_l2_features, visual_l3_features, visual_l4_features = visual_features
        # Visual and text features projection
        text_features_proj: Tensor = (
            self.text_projection(text_features).unsqueeze(-1).unsqueeze(-1)
        )  # B 1024 1 1
        visual_l4_features_proj: Tensor = self.vis_l4_projection(
            visual_l4_features
        )  # B 1024 7 7

        # First fusion l4 (B 1024 7 7) and text (B 1024)
        fused_l4: Tensor = self.norm_layer(
            visual_l4_features_proj * text_features_proj
        )  # B 1024 7 7

        # Second fusion l3 (B 512 14 14) and l4 (B 1024 7 7)
        fused_l4_upsample: Tensor = nn.Upsample(scale_factor=2, mode="nearest")(
            fused_l4
        )  # B 1024 14 14
        cat_features: Tensor = torch.cat([visual_l3_features, fused_l4_upsample], dim=1)
        fused_l3: Tensor = self.vis_l3_projection(cat_features)  # B 512 14 14

        # Third fusion l2 (B 512 28 28) and l3 (B 512 14 14)
        visual_l2_pooling = nn.MaxPool2d(kernel_size=2, stride=2)(
            visual_l2_features
        )  # B 512 14 14
        fused_l2: Tensor = self.vis_l2_projection(
            torch.cat([fused_l3, visual_l2_pooling], dim=1)
        )  # B 512 14 14

        # Aggregate features
        cat_visual_features: Tensor = torch.cat(
            [fused_l2, fused_l3, fused_l4_upsample], dim=1
        )  # B 2048 14 14
        aggregated_features: Tensor = self.aggregation(
            cat_visual_features
        )  # B 512 14 14

        # Add coordinate features
        final_features: Tensor = self.coord_conv(aggregated_features)  # B 512 14 14
        return final_features


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv: nn.Sequential = _conv_layer(
            input_dim=in_channels,
            output_dim=out_channels,
            kernel_size=3,
            padding=1,
            device=self.device,
        )

    def add_coord(self, input: Tensor) -> Tensor:
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=self.device)
        y_range = torch.linspace(-1, 1, h, device=self.device)

        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x: Tensor) -> Tensor:
        x = self.add_coord(x)
        x = self.conv(x)
        return x


def _conv_layer(
    input_dim: int,
    output_dim: int,
    kernel_size: int,
    padding: int,
    device: device,
) -> nn.Sequential:
    module = nn.Sequential(
        nn.Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            padding=padding,
            device=device,
        ),
        nn.BatchNorm2d(output_dim, device=device),
        nn.ReLU(),
    )
    nn.init.xavier_uniform_(module[0].weight)
    return module
