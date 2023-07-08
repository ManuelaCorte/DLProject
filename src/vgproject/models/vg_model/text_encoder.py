from clip.model import CLIP
import clip
import torch
import torch.nn as nn
from torch import Tensor


# CLIP transformer encoder
class TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model: CLIP = clip.load("RN50", device=self.device)[0]
        self.pretrained_model.float()

        # Freeze the backbone
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, tokenized_caption: Tensor) -> Tensor:
        out: Tensor = self.pretrained_model.encode_text(tokenized_caption).to(
            self.device
        )
        # .unsqueeze(1)
        return out.requires_grad_(True)


# Test
if __name__ == "__main__":
    tokenizer = clip.tokenize("Test caption.")
    print(tokenizer)
    rand = torch.randint(0, 100, size=(3, 77), dtype=torch.int32)
    output = TextEncoder()(rand)
    print(output)
    print(output.shape)
