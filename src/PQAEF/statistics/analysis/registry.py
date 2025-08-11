# -*- coding: utf-8 -*-
from typing import Dict, Type, Any, TYPE_CHECKING


if TYPE_CHECKING:
    from PQAEF.statistics.analysis.base_analysis import BaseAnalysis

# Global registry for storing all available analyzer classes
# Keys are names used in configuration, values are the analyzer classes themselves
ANALYSIS_REGISTRY: Dict[str, Type['BaseAnalysis']] = {}

def register_analyzer(name: str):
    """
    A class decorator for registering analyzers to the global registry.

    Args:
        name (str): Unique name used to reference this analyzer in configuration files.
    """
    def decorator(cls: Type['BaseAnalysis']) -> Type['BaseAnalysis']:
        if name in ANALYSIS_REGISTRY:
            raise ValueError(f"Error: Analyzer with name '{name}' is already registered.")
        ANALYSIS_REGISTRY[name] = cls
        print(f"INFO: Analyzer '{cls.__name__}' registered as '{name}'.")
        return cls
    return decorator
