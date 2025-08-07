# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Literal

from ..constant import constant

class BaseTask(ABC):
    """
    Abstract base class for all data annotation tasks.
    Includes logic for checking if a sample or its sentences have already been processed.
    """
    def __init__(self, task_config: Dict[str, Any]):
        self.task_config = task_config

    @abstractmethod
    def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a batch of samples. Subclasses must implement this.
        It's recommended to call `_get_samples_to_process` at the beginning.
        """
        raise NotImplementedError
    
    async def aprocess_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Asynchronously processes a batch of samples.
        """
        raise NotImplementedError(f"Asynchronous processing not implemented for {self.__class__.__name__}")