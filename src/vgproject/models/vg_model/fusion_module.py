from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor


class FusionModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_l4 = _conv_layer(1024, 1024, 1)
        self.text_projection = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU())
        self.vis_l4_projection = _conv_layer(1024, 1024, 1, 0)
        self.norm_layer = nn.Sequential(nn.LayerNorm([1024, 7, 7]), nn.ReLU())
        self.vis_l3_projection = _conv_layer(1024 + 1024, 512, 3, 1)
        self.vis_l2_projection = _conv_layer(512 + 512, 512, 3, 1)
        self.aggregation = _conv_layer(1024 + 512 + 512, 512, 1, 0)

    def forward(
        self, visual_features: Tuple[Tensor, Tensor, Tensor], text_features: Tensor
    ) -> Tensor:
        visual_l2_features, visual_l3_features, visual_l4_features = visual_features
        # Visual and text features projection
        text_features_proj = (
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
        cat_features = torch.cat([visual_l3_features, fused_l4_upsample], dim=1)
        fused_l3: Tensor = self.vis_l3_projection(cat_features)  # B 512 14 14

        # Third fusion l2 (B 512 28 28) and l3 (B 512 14 14)
        visual_l2_pooling = nn.MaxPool2d(kernel_size=2, stride=2)(
            visual_l2_features
        )  # B 512 14 14
        fused_l2 = self.vis_l2_projection(
            torch.cat([fused_l3, visual_l2_pooling], dim=1)
        )  # B 512 14 14

        # Aggregate features
        cat_visual_features = torch.cat(
            [fused_l2, fused_l3, fused_l4_upsample], dim=1
        )  # B 2048 14 14
        aggregated_features = self.aggregation(cat_visual_features)  # B 512 14 14
        # TODO: Add spatial coords?
        return aggregated_features


def _conv_layer(
    input_dim: int, output_dim: int, kernel_size: int, padding: int = 0
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            padding=padding,
        ),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(),
    )
