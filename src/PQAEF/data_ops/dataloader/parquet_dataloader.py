import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Iterator, List, Union
import random
import pandas as pd
import os

from .base_dataloader import BaseDataLoader, register_dataloader
from PQAEF.utils.template_registry import get_formatter, BaseFormatter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@register_dataloader("ParquetDataLoader")
class ParquetDataLoader(BaseDataLoader):
    """
    Parquet Data Loader
    
    Supports loading Parquet files and converting them to standard format through registered formatters
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Parquet data loader
        
        Args:
            config: Configuration dictionary containing:
                - path: Parquet file path (string) or list of paths
                - formatter_name: Formatter name (optional)
        """
        super().__init__(config)
        # Paths can be a single string or a list of strings
        paths_config = self.config.get('paths')
        self.suffix = self.config.get("suffix", "parquet")
        if isinstance(paths_config, str):
            self.paths: List[Path] = [Path(paths_config)]
        elif isinstance(paths_config, list):
            self.paths: List[Path] = [Path(p) for p in paths_config]
        else:
            raise ValueError("'paths' in config must be a string or a list of strings.")

        self.formatter_name = config.get('formatter_name')
        self.formatter = None
        self.recursive: bool = self.config.get('recursive', False)
        self.val: bool = self.config.get('val', False)

        # Get formatter instance if specified
        if self.formatter_name:
            formatter_class = get_formatter(self.formatter_name)
            self.formatter = formatter_class()
            
        self.num = config.get("num", -1)
        self._samples: List[Dict[str, Any]] = []
        self._load_and_process_data()
    
    def _get_file_paths(self) -> List[Path]:
        all_files = set()
        for path in self.paths:
            if not path.exists():
                logging.warning(f"Path does not exist: {path}")
                continue
            
            if path.is_file():
                # if path.suffix == '.jsonl':
                if path.suffix == f".{self.suffix}":
                    all_files.add(path)
                else:
                    logging.warning(f"Skipping non-{self.suffix} file specified directly: {path}")
            elif path.is_dir():
                glob_pattern = f'**/*.{self.suffix}' if self.recursive else f'*.{self.suffix}'
                all_files.update(path.glob(glob_pattern))
        
        all_files_val = []
        if self.val:
            for file in all_files:
                if 'val' in str(file).split('/')[-1]:
                    all_files_val.append(file)
            all_files = all_files_val

        return sorted(list(all_files))
    
    def _load_and_process_data(self) -> Iterator[Dict[str, Any]]:
        file_paths = self._get_file_paths()
        for data_path in file_paths:
            if not os.path.exists(data_path):
                print(f"Warning: File {str(data_path)} does not exist, skipping...")
                continue
                
            if not str(data_path).lower().endswith('.parquet'):
                print(f"Warning: File {str(data_path)} is not a Parquet file, skipping...")
                continue
            
            try:
                # Read parquet file
                df = pd.read_parquet(data_path)
                
                for row_idx, (_, row) in enumerate(df.iterrows()):
                    try:
                        # Convert pandas Series to dictionary
                        row_dict = row.to_dict()
                        
                        # Use formatter to process data if available
                        if self.formatter:
                            formatted_data = self.formatter.format(row_dict)
                            # Filter out None values
                            if formatted_data is not None:
                                self._samples.append(formatted_data)
                        else:
                            formatted_data = {
                                'raw_data': row_dict,
                                '_source_file': str(data_path),
                                '_row_index': row_idx
                            }
                            self._samples.append(formatted_data)
                            
                    except Exception as e:
                        print(f"Error processing row {row_idx} in {str(data_path)}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error reading file {str(data_path)}: {e}")
                continue
        # Sampling
        if self.num != -1:
            if self.num > len(self._samples):
                logging.warning(f"Requested sample size ({self.num}) is larger than available data ({len(self._samples)}). Using all available data.")
                # No sampling, use all data
            else:
                self._samples = random.sample(self._samples, self.num)
        
        logging.info(f"Finished loading. Total formatted samples: {len(self._samples)}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # Directly iterate over loaded and formatted sample list
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)
    
    def get_total_count(self) -> int:
        """
        Get total data count (optional implementation)
        
        Returns:
            int: Total data count, returns -1 if cannot be determined
        """
        return self.__len__()