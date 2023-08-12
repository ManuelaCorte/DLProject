from typing import Callable, Tuple

import clip
import torch
import torch.nn as nn
from clip.model import CLIP
from torch import Tensor

from vgproject.utils.config import Config


# CLIP transformer encoder
class TextEncoder(nn.Module):
    def __init__(self, batch_size: int, clip_ctx_length, embed_dim) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model: CLIP = clip.load("RN50", device=self.device)[0]
        self.pretrained_model.float()

        # Freeze the backbone
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.pretrained_model.transformer.register_forward_hook(self.hook_fn())
        self.transformer_output: Tensor = torch.empty(
            (batch_size, clip_ctx_length, embed_dim),
            requires_grad=True,
            device=self.device,
        )

    # @torch.no_grad()
    def forward(self, tokenized_caption: Tensor) -> Tuple[Tensor, Tensor]:
        out: Tensor = self.pretrained_model.encode_text(tokenized_caption).to(
            self.device
        )
        # .unsqueeze(1)
        return (
            self.transformer_output,
            out,
        )

    def hook_fn(self) -> Callable[[nn.Module, Tensor, Tensor], None]:
        def hook(module: nn.Module, input: Tensor, output: Tensor) -> None:
            self.transformer_output = output.permute(1, 0, 2)  # L B D -> B L D

        return hook


# Test
if __name__ == "__main__":
    tokenizer = clip.tokenize("Test caption.")
    # print(tokenizer)
    rand = torch.randint(0, 100, size=(3, 77), dtype=torch.int32)
    cfg: Config = Config()
    output = TextEncoder(
        cfg.train.batch_size, cfg.model.clip_ctx_length, cfg.model.embed_dim
    )(rand)
    print(output[0])
    print(output[0].shape, output[1].shape)
