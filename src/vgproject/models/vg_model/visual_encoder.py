from typing import Any, Callable, OrderedDict
from clip.model import ModifiedResNet
import clip
import torch
import torch.nn as nn
from torch import Tensor

from vgproject.utils.config import Config


# Class that gets output for all layes of the backbone
# CLIP backbone is a modified ResNet with an attention layer for global pooling
class VisualEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model: ModifiedResNet = clip.load("RN50", device=self.device)[
            0
        ].visual  # type: ignore
        self.pretrained_model.float()
        assert isinstance(self.pretrained_model, ModifiedResNet)

        # Freeze the backbone
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Register hooks to get the output of all layers
        self.layers_outputs: OrderedDict[str, Tensor] = OrderedDict()
        self.pretrained_model.layer1.register_forward_hook(self.hook_fn("layer1"))  # type: ignore
        self.pretrained_model.layer2.register_forward_hook(self.hook_fn("layer2"))  # type: ignore
        self.pretrained_model.layer3.register_forward_hook(self.hook_fn("layer3"))  # type: ignore
        self.pretrained_model.layer4.register_forward_hook(self.hook_fn("layer4"))  # type: ignore

        # Project the output of each layer to the same dimensionality as the text features
        cfg = Config.get_instance().visual_encoder  # type: ignore
        resnet_resolution: int = cfg["resnet_resolution"]
        resnet_channels: int = cfg["resnet_channels"]

        self.layers_projections = nn.ModuleList()
        for _ in range(4):
            resnet_resolution //= 2
            in_features: int = resnet_channels * resnet_resolution * resnet_resolution
            resnet_channels *= 2
            layer_projection: nn.Sequential = nn.Sequential(
                nn.AdaptiveAvgPool2d(resnet_resolution),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, cfg["output_dim"], device=self.device),
                # nn.LayerNorm(
                #     cfg["output_dim"], eps=1e-3, device=self.device
                # ),
                # nn.ReLU(),
            )
            self.layers_projections.append(layer_projection)

    @torch.no_grad()
    def forward(self, batch: Tensor) -> OrderedDict[str, Tensor]:
        # Reset the dictionary
        self.layers_outputs = OrderedDict()

        out: Tensor = self.pretrained_model(batch)
        # .unsqueeze(1)

        for idx, (layer_name, layer_output) in enumerate(self.layers_outputs.items()):
            self.layers_outputs[layer_name] = self.layers_projections[idx](layer_output)
            # .unsqueeze(1)
        self.layers_outputs["output"] = out

        return self.layers_outputs

    def hook_fn(self, layer: str) -> Callable[[nn.Module, Tensor, Tensor], None]:
        def hook(module: nn.Module, input: Tensor, output: Tensor) -> None:
            # print(f"Module: {[module for  module in module.modules()]}")
            self.layers_outputs[layer] = output.requires_grad_(True)

        return hook


# Test
if __name__ == "__main__":
    test = VisualEncoder()
    layers: OrderedDict[str, Any] = test(torch.rand(3, 3, 224, 224))
    for layer in layers:
        print(f"{layer}, {layers[layer]} with shape: {layers[layer].shape}")
