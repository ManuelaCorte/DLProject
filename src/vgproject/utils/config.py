# https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons?noredirect=1&lq=1
from typing import Any, Dict
import json


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


@Singleton
class Config:
    def __init__(self) -> None:
        with open(file="../config.json", mode="r") as fp:
            cfg: Dict[str, Any] = json.load(fp=fp)
        for k, v in cfg.items():
            setattr(self, k, v)
        # self.__dict__.update(cfg)
