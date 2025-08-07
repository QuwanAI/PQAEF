# -*- coding: utf-8 -*-
from typing import Dict, Type, Any, TYPE_CHECKING


if TYPE_CHECKING:
    from PQAEF.statistics.analysis.base_analysis import BaseAnalysis

# 全局注册表，用于存储所有可用的分析器类
# 键是配置中使用的名称，值是分析器类本身
ANALYSIS_REGISTRY: Dict[str, Type['BaseAnalysis']] = {}

def register_analyzer(name: str):
    """
    一个类装饰器，用于将分析器注册到全局注册表中。

    Args:
        name (str): 在配置文件中用于引用此分析器的唯一名称。
    """
    def decorator(cls: Type['BaseAnalysis']) -> Type['BaseAnalysis']:
        if name in ANALYSIS_REGISTRY:
            raise ValueError(f"Error: Analyzer with name '{name}' is already registered.")
        ANALYSIS_REGISTRY[name] = cls
        print(f"INFO: Analyzer '{cls.__name__}' registered as '{name}'.")
        return cls
    return decorator
