from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd
import os
from matplotlib.font_manager import FontProperties

class BaseAnalysis(ABC):
    """
    Abstract base class for all analyzer classes.
    Defines the common interface that analyzers must implement.
    """
    def __init__(self, config: Dict[str, Any], output_dir: str, font_props: FontProperties, file_prefix: str):
        self.config = config
        self.output_dir = output_dir
        self.font_props = font_props
        self.file_prefix = file_prefix

    @abstractmethod
    def analyze(self, df: pd.DataFrame, raw_data: List[Dict[str, Any]]) -> Dict[str, Any]: # <--- MODIFIED
        """
        Core method for executing analysis.

        Args:
            df (pd.DataFrame): Preprocessed DataFrame containing all data.
                               Suitable for general statistical analysis.
            raw_data (List[Dict[str, Any]]): Raw, unprocessed data list.
                                             Suitable for specific evaluation analysis requiring original nested structure.

        Returns:
            Dict[str, Any]: A dictionary containing analysis results.
        """
        pass

    def _get_safe_filename(self, base_name: str) -> str:
        import re
        return re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)