# -*- coding: utf-8 -*-
"""
Defines the abstract base class for all data loaders.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, Type, Callable


class BaseDataLoader(ABC):
    """
    Base class for data loaders
    
    All data loaders should inherit from this base class and implement the necessary methods
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Return data iterator
        
        Returns:
            Iterator[Dict[str, Any]]: Data iterator
        """
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Return dataset size
        
        Returns:
            int: Dataset size
        """
        raise NotImplementedError


# Data loader registry
DATA_LOADERS: Dict[str, Type["BaseDataLoader"]] = {}


def register_dataloader(name: str) -> Callable:
    """
    Decorator for registering data loaders
    
    Args:
        name: Data loader name
    """
    def decorator(cls: Type["BaseDataLoader"]) -> Type["BaseDataLoader"]:
        if name in DATA_LOADERS:
            raise ValueError(f"Dataloader with name '{name}' is already registered.")
        DATA_LOADERS[name] = cls
        return cls
    return decorator


def get_dataloader(name: str) -> Type["BaseDataLoader"]:
    """
    Get data loader class by name
    
    Args:
        name: Data loader name
        
    Returns:
        Data loader class
        
    Raises:
        ValueError: If the specified data loader is not found
    """
    if name not in DATA_LOADERS:
        raise ValueError(
            f"Dataloader '{name}' not found. "
            f"Available formatters: {list(DATA_LOADERS.keys())}"
        )
    return DATA_LOADERS[name]