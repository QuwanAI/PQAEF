# -*- coding: utf-8 -*-
"""
Implements the registry for data formatters.

This module is the core of the framework's data adaptability. It allows you to
define transformation logic for any raw data structure and register it under a
unique name. The DataLoader then uses this name (from the config) to pick the
correct transformation logic at runtime.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Callable

# Global registry to store formatter classes
DATA_FORMATTER_REGISTRY: Dict[str, Type['BaseFormatter']] = {}


class BaseFormatter(ABC):
    """Abstract base class for all data formatters."""
    @abstractmethod
    def format(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes a raw data dictionary and transforms it into the standard
        PQAEF format, including calculating and adding the hash.
        """
        raise NotImplementedError
    

def register_formatter(name: str) -> Callable:
    """A decorator to register a data formatter class under a specific name."""
    def decorator(cls: Type['BaseFormatter']) -> Type['BaseFormatter']:
        if name in DATA_FORMATTER_REGISTRY:
            raise ValueError(f"Formatter with name '{name}' is already registered.")
        DATA_FORMATTER_REGISTRY[name] = cls
        return cls
    return decorator

def get_formatter(name: str) -> Type['BaseFormatter']:
    """Retrieves a registered formatter class by its name."""
    if name not in DATA_FORMATTER_REGISTRY:
        raise ValueError(
            f"Formatter '{name}' not found. "
            f"Available formatters: {list(DATA_FORMATTER_REGISTRY.keys())}"
        )
    return DATA_FORMATTER_REGISTRY[name]

