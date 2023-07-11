import math

import torch
import torch.nn as nn
from torch import Tensor


# TODO: Possibly add learnable positional encoding
class Decoder(nn.Module):
    def __init__(self, d_model: int, img_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.pos_embedding_1d = PositionalEncoding1D(d_model, 0.1, 77)
        self.pos_embeddinf_2d = PositionalEncoding2D(d_model, img_size, img_size)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=8,
                batch_first=True,
            ),
            num_layers=6,
            norm=nn.LayerNorm(d_model),
        )
        self.reg_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, vis: Tensor, text: Tensor) -> Tensor:
        text_pe: Tensor = self.pos_embedding_1d(text)
        text = text + text_pe
        visual_pe: Tensor = self.pos_embeddinf_2d(vis)
        visual_features = vis + visual_pe

        visual_features = visual_features.flatten(2).permute(0, 2, 1)  # B HW D

        visual_features = torch.cat(
            [self.reg_token.expand((vis.shape[0], -1, -1)), visual_features], dim=1
        )
        x = self.decoder(visual_features, text)
        return x[:, 0, :]


# https://github.com/wzlxjtu/PositionalEncoding2D
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, dropout_p: float, window_len: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.pos_encoding = torch.zeros(window_len, d_model)
        position = torch.arange(0, window_len).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, d_model, 2, dtype=torch.float)
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


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, width: int, height: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pos_w = torch.arange(0.0, width).unsqueeze(1)
        pos_h = torch.arange(0.0, height).unsqueeze(1)
        self.pe[0:d_model:2, :, :] = (
            torch.sin(pos_w * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, height, 1)
        )
        self.pe[1:d_model:2, :, :] = (
            torch.cos(pos_w * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, height, 1)
        )
        self.pe[d_model::2, :, :] = (
            torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )
        self.pe[d_model + 1 :: 2, :, :] = (
            torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
