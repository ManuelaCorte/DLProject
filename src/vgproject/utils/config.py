import json
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Model:
    clip_embed_dim: int
    clip_ctx_length: int
    embed_dim: int
    mlp_hidden_dim: int
    img_size: int
    proj_img_size: int
    decoder_layers: int
    decoder_heads: int
    decoder_dim_feedforward: int


@dataclass
class Train:
    batch_size: int
    lr: float
    lr_backbone: float
    step_size: int
    l1: float
    l2: float
    sweep: bool


@dataclass
class Logging:
    path: str
    save: bool
    resume: bool
    wandb: bool


class Config:
    def __init__(self) -> None:
        cfg: Dict[str, Any] = json.load(open("../config.json", "r"))
        self.dataset_path: str = cfg["dataset_path"]
        self.epochs: int = cfg["epochs"]
        self.model = Model(**cfg["model"])
        self.train = Train(**cfg["train"])
        self.logging = Logging(**cfg["logging"])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "epochs": self.epochs,
            "model": self.model.__dict__,
            "train": self.train.__dict__,
        }

    # if in other dict there are keys equal to the keys in self, update them
    def update(self, other: Dict[str, Any]):
        for k, v in other.items():
            if k in self.__dict__:
                self.__dict__[k] = v
            if k in self.model.__dict__:
                self.model.__dict__[k] = v
            if k in self.train.__dict__:
                self.train.__dict__[k] = v
            if k in self.logging.__dict__:
                self.logging.__dict__[k] = v


if __name__ == "__main__":
    config = Config()
    print(config.model.clip_embed_dim)
    print(config.as_dict())
    config.update({"epochs": 10, "clip_embed_dim": 100})
    print(config.as_dict())
