# -*- coding: utf-8 -*-
"""
Defines the abstract base class for all data loaders.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, Type, Callable


class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders in the framework.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the data loader with a configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration, typically including 'path'
                                     and 'formatter_name'.
        """
        self.config = config

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Yields data samples in the standard framework format.
        This method must be implemented by all subclasses.
        """
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


DATA_LOADERS: Dict[str, Type["BaseDataLoader"]] = {}


def register_dataloader(name: str) -> Callable:
    def decorator(cls: Type["BaseDataLoader"]) -> Type["BaseDataLoader"]:
        if name in DATA_LOADERS:
            raise ValueError(f"Dataloader with name '{name}' is already registered.")
        DATA_LOADERS[name] = cls
        return cls
    return decorator


def get_dataloader(name: str) -> Type["BaseDataLoader"]:
    if name not in DATA_LOADERS:
        raise ValueError(
            f"Dataloader '{name}' not found. "
            f"Available formatters: {list(DATA_LOADERS.keys())}"
        )
    return DATA_LOADERS[name]