# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseDataDumper(ABC):
    """Abstract base class for all data dumpers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the dumper with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration for the dumper, e.g., output path.
        """
        self.config = config

    @abstractmethod
    def dump(self, data: List[Dict[str, Any]], run_metadata: Dict[str, Any]):
        """
        The main method to dump the processed data and generate reports.
        
        Args:
            data (List[Dict[str, Any]]): The list of all processed data samples.
            run_metadata (Dict[str, Any]): A dictionary containing metadata about the run,
                                            such as the list of tasks performed.
        """
        raise NotImplementedError