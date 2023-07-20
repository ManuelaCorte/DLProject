import math

import torch
import torch.nn as nn
from torch import Tensor


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        img_size: int,
        clip_ctx_length: int,
        nheads: int,
        nlayers: int,
        dim_feedforward: int,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.pos_embedding_1d = PositionalEncoding1D(d_model, clip_ctx_length).to(
            self.device
        )
        self.pos_embeddinf_2d = PositionalEncoding2D(d_model, img_size, img_size).to(
            self.device
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True,  # Less prone to vanishing gradients??
                device=self.device,
            ),
            num_layers=nlayers,
            norm=nn.LayerNorm(d_model, device=self.device),
        )
        self.reg_token = nn.Parameter(torch.randn(1, 1, d_model)).to(self.device)
        nn.init.kaiming_normal_(self.reg_token, nonlinearity="relu", mode="fan_out")

    def forward(self, vis: Tensor, text: Tensor) -> Tensor:
        text_features: Tensor = self.pos_embedding_1d(text)

        visual_features: Tensor = self.pos_embeddinf_2d(vis)

        visual_features = visual_features.flatten(2).permute(0, 2, 1)  # B HW D

        visual_features = torch.cat(
            [self.reg_token.expand((vis.shape[0], -1, -1)), visual_features], dim=1
        )
        x: Tensor = self.decoder(visual_features, text_features)
        reg_token: Tensor = x[:, 0, :]
        return reg_token


# Positional encodings implemented in separate classes if we want to change them and use learnable positional encodings instead
# Dropout added following the original transformer implementation
# https://github.com/wzlxjtu/PositionalEncoding2D
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, window_len: int) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dropout = nn.Dropout(0.1).to(self.device)
        self.pos_encoding = torch.zeros(window_len, d_model, device=self.device)
        position = torch.arange(0, window_len, device=self.device).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, d_model, 2, dtype=torch.float, device=self.device)
                * -(math.log(10000.0) / d_model)
            )
        )
        self.pos_encoding[:, 0::2] = torch.sin(position.float() * div_term)
        self.pos_encoding[:, 1::2] = torch.cos(position.float() * div_term)

        self.register_buffer("text_pos_encoding", self.pos_encoding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        out = self.dropout(
            token_embedding + self.pos_encoding[: token_embedding.size(1), :]
        )
        return out


# First half of the encodings are used for the height and the second half for the width
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, width: int, height: int) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dropout = nn.Dropout(0.1).to(self.device)
        self.pe = torch.zeros(d_model, height, width, device=self.device)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2, device=self.device)
            * -(math.log(10000.0) / d_model)
        )
        pos_w = torch.arange(0.0, width, device=self.device).unsqueeze(1)
        pos_h = torch.arange(0.0, height, device=self.device).unsqueeze(1)
        self.pe[0:d_model:2, :, :] = (
            torch.sin(pos_w * div_term)  # H d_model/4
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, height, 1)
        )  # d_model/4 H H
        self.pe[1:d_model:2, :, :] = (
            torch.cos(pos_w * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, height, 1)
        )
        self.pe[d_model::2, :, :] = (
            torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )  # d_model/4 W W
        self.pe[d_model + 1 :: 2, :, :] = (
            torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )

        self.register_buffer("visual_pos_encoding", self.pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
