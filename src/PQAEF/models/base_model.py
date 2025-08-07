# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union

class BaseModel(ABC):
    """
    Abstract base class for all models in the PQAEF framework.
    """
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes the base model.

        Args:
            model_name (str): The unique name for this model instance (e.g., 'gpt-4-judge').
            config (Dict[str, Any]): The configuration dictionary for this model.
        """
        self.model_name = model_name
        self.config = config

    @abstractmethod
    def process(self, inputs: Union[str, List[str], Dict[str, Any]]) -> Any:
        """
        The main processing method for the model.

        Args:
            inputs: The input data, which can vary depending on the model type.
                    (e.g., list of texts, a single prompt, etc.)

        Returns:
            The model's output. The structure will vary by model.
        """
        raise NotImplementedError
    
    async def aprocess(self, inputs: Union[str, List[str], Dict[str, Any]]) -> Any:
        """
        The main asynchronous processing method for the model.
        By default, it raises an error. Subclasses should override this
        for native async support.
        """
        raise NotImplementedError(f"Asynchronous processing not implemented for {self.__class__.__name__}")