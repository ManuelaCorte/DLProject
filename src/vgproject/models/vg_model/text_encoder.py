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

    @torch.no_grad()
    def forward(self, tokenized_caption: Tensor) -> Tensor:
        tokenized_caption = tokenized_caption.int()
        out: Tensor = self.pretrained_model.encode_text(
            tokenized_caption.to(torch.IntTensor())
        )
        return out


# Test
if __name__ == "__main__":
    test = TextEncoder()
    output = test("Test phrase")
    print(output)
    print(output.shape)
