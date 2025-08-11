import csv
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

@register_dataloader("CSVDataLoader")
class CSVDataLoader(BaseDataLoader):
    """
    CSV Data Loader
    
    Supports loading CSV files and converting them to standard format through registered formatters
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CSV data loader
        
        Args:
            config: Configuration dictionary containing:
                - paths: CSV file path (string) or list of paths
                - formatter_name: Formatter name (optional)
                - encoding: File encoding, default utf-8 (optional)
                - skip_header: Whether to skip header row, default True (optional)
                - num: Sample size, -1 means load all data (optional)
        """
        super().__init__(config)
        
        # Paths can be a single string or a list of strings
        paths_config = self.config.get('paths')
        self.suffix = self.config.get("suffix", "csv")
        if isinstance(paths_config, str):
            self.paths: List[Path] = [Path(paths_config)]
        elif isinstance(paths_config, list):
            self.paths: List[Path] = [Path(p) for p in paths_config]
        else:
            raise ValueError("'paths' in config must be a string or a list of strings.")

        self.recursive: bool = self.config.get('recursive', False)
        self.formatter_name = config.get('formatter_name')
        self.encoding = config.get('encoding', 'utf-8')
        self.skip_header = config.get('skip_header', True)  # 默认跳过表头
        self.formatter = None
        
        # Get formatter instance if specified
        if self.formatter_name:
            formatter_class = get_formatter(self.formatter_name)
            self.formatter = formatter_class()
        
        self.num = config.get("num", -1)
        self._samples: List[Dict[str, Any]] = []
        self._load_and_process_data()
        
        # print(json.dumps(self._samples,indent=4))
        # sys.exit(0)
    
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
        
        return sorted(list(all_files))

    def _load_and_process_data(self) -> Iterator[Dict[str, Any]]:
        file_paths = self._get_file_paths()
        for data_path in file_paths:
            if not os.path.exists(data_path):
                print(f"Warning: File {data_path} does not exist, skipping...")
                continue
                
            if not (str(data_path).lower().endswith('.csv') or str(data_path).lower().endswith('.tsv')):
                print(f"Warning: File {data_path} is not a CSV/TSV file, skipping...")
                continue
            
            try:

                with open(data_path, 'r', encoding=self.encoding, newline='') as csvfile:
                    
                    if str(data_path).lower().endswith('.tsv'):
                        delimiter = '\t'  # TSV files use tab separator
                    else:
                        # Try to auto-detect delimiter for CSV files
                        try:
                            sample = csvfile.read(1024)
                            csvfile.seek(0)
                            sniffer = csv.Sniffer()
                            delimiter = sniffer.sniff(sample).delimiter
                        except csv.Error:
                            # Use comma as default if detection fails
                            delimiter = ','
                            csvfile.seek(0)
                    
                    # Read all rows as list
                    reader = csv.reader(csvfile, delimiter=delimiter)
                    
                    # Skip header row
                    if self.skip_header:
                        try:
                            next(reader)  # Skip first row (header)
                        except StopIteration:
                            continue  # File is empty, skip
                    
                    for row_idx, row in enumerate(reader):
                        try:
                            # Skip empty rows
                            if not row or all(not cell.strip() for cell in row):
                                continue
                            
                            # Clean data: remove whitespace
                            cleaned_row = [cell.strip() if isinstance(cell, str) else cell for cell in row]
                            
                            # Use formatter to process data if available
                            if self.formatter:

                                formatted_data = self.formatter.format(cleaned_row)
                            else:
                                formatted_data =  {
                                    'raw_data': cleaned_row,
                                    '_source_file': str(data_path),
                                    '_row_index': row_idx
                                }
                            self._samples.append(formatted_data)
                            if self.num != -1 and len(self._samples) > 100 * self.num:
                                break
                            
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
            int: Total data count, return -1 if cannot be determined
        """
        return self.__len__()