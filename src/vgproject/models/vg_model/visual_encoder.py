from typing import Any, Callable, OrderedDict
from clip.model import ModifiedResNet
import clip
import torch
import torch.nn as nn
from torch import Tensor


# Class that gets output for all layes of the backbone
# CLIP backbone is a modified ResNet with an attention layer for global pooling
class VisualEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model: ModifiedResNet = clip.load("RN50", device=self.device)[
            0
        ].visual  # type: ignore
        assert isinstance(self.pretrained_model, ModifiedResNet)
        self.layers_outputs: OrderedDict[str, Tensor] = OrderedDict()

        self.pretrained_model.layer1.register_forward_hook(self.hook_fn("layer1"))  # type: ignore
        self.pretrained_model.layer2.register_forward_hook(self.hook_fn("layer2"))  # type: ignore
        self.pretrained_model.layer3.register_forward_hook(self.hook_fn("layer3"))  # type: ignore
        self.pretrained_model.layer4.register_forward_hook(self.hook_fn("layer4"))  # type: ignore

    def hook_fn(self, layer: str) -> Callable[[nn.Module, Tensor, Tensor], None]:
        def hook(module: nn.Module, input: Tensor, output: Tensor) -> None:
            self.layers_outputs[layer] = output

        return hook

    def forward(self, x: Tensor) -> OrderedDict[str, Any]:
        out: Tensor = self.pretrained_model(x)

        for layer in self.layers_outputs.keys():
            layer_features = self.layers_outputs[layer]
            pooling_size: int = layer_features.shape[-1] // 2
            pooling_layer = nn.AdaptiveAvgPool2d((pooling_size, pooling_size))
            pooling_features: Tensor = pooling_layer(layer_features)

            flat_features = torch.flatten(pooling_features, start_dim=1)
            ll1 = nn.Linear(in_features=flat_features.shape[1], out_features=1024)
            relu = nn.ReLU()
            out_layer = ll1(flat_features)
            out_layer = relu(out_layer)
            self.layers_outputs[layer] = out_layer

        self.layers_outputs["output"] = out

        return self.layers_outputs


# Test
if __name__ == "__main__":
    test = VisualEncoder()
    layers = test(torch.rand(1, 3, 224, 224))
    for layer in layers:
        print(f"{layer} with shape: {layers[layer].shape}")
