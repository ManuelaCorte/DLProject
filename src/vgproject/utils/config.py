import json
from dataclasses import dataclass
from typing import Any, Dict


class Singleton:
    def __init__(self, decorated_class: Any) -> None:
        self._decorated = decorated_class

    def get_instance(self) -> Any:
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.
        """
        try:
            return self._instance  # type: ignore
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self) -> None:
        raise TypeError("Singletons must be accessed through get_instance() method.")

    def __instancecheck__(self, inst: Any) -> bool:
        return isinstance(inst, self._decorated)


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


@dataclass
class Train:
    batch_size: int
    lr: float
    gamma: float


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


if __name__ == "__main__":
    config = Config()
    print(config.model.clip_embed_dim)
    print(config.as_dict())
