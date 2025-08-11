# -*- coding: utf-8 -*-
"""
Provides a flexible DataLoader for reading data from JSON/JSONL files.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Iterator, List, Union
import random
import sys

from .base_dataloader import BaseDataLoader, register_dataloader
from PQAEF.utils.template_registry import get_formatter, BaseFormatter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@register_dataloader("JsonLoader")
class JsonLoader(BaseDataLoader):
    """
    Loads data from JSON/JSONL files from one or more specified paths.

    It can read from a list of paths, and optionally search recursively
    within directories. It uses a registered formatter specified in the
    config to transform raw data into the standard framework format.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Paths can be a single string or a list of strings
        paths_config = self.config.get('paths')
        if isinstance(paths_config, str):
            self.paths: List[Path] = [Path(paths_config)]
        elif isinstance(paths_config, list):
            self.paths: List[Path] = [Path(p) for p in paths_config]
        else:
            raise ValueError("'paths' in config must be a string or a list of strings.")

        self.recursive: bool = self.config.get('recursive', False)
        
        formatter_name = self.config['formatter_name']
        self.num = self.config.get("num", -1)
        formatter_class = get_formatter(formatter_name)
        self.formatter: BaseFormatter = formatter_class()
        
        self._samples: List[Dict[str, Any]] = []
        
        self._load_and_process_data()

    def _get_file_paths(self) -> List[Path]:
        """Identifies all JSON files to be processed based on the configuration."""
        all_files = set()
        for path in self.paths:
            if not path.exists():
                logging.warning(f"Path does not exist: {path}")
                continue
            
            if path.is_file():
                if path.suffix == '.json':
                    all_files.add(path)
                else:
                    logging.warning(f"Skipping non-json file specified directly: {path}")
            elif path.is_dir():
                glob_pattern = '**/*.json' if self.recursive else '*.json'
                all_files.update(path.glob(glob_pattern))
        
        return sorted(list(all_files))
    
    def _load_and_process_data(self):
        file_paths = self._get_file_paths()
        if not file_paths:
            logging.warning(f"No files found to process for the given paths.")
            return

        for file_path in file_paths:
            logging.info(f"Processing file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                logging.error(f"Failed to read or parse file {file_path}: {e}")
                continue

            samples_to_process = raw_data if isinstance(raw_data, list) else [raw_data]

            for i, raw_sample in enumerate(samples_to_process):
                try:
                    formatted_sample = self.formatter.format(raw_sample)
                    self._samples.append(formatted_sample)
                except Exception as e:
                    logging.warning(
                        f"Failed to format sample #{i+1} in file {file_path}. "
                        f"Error: {e}. Skipping sample."
                    )
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