import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Iterator, List, Union
import random

from .base_dataloader import BaseDataLoader, register_dataloader
from PQAEF.utils.template_registry import get_formatter, BaseFormatter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@register_dataloader("JsonlLoader")
class JsonlLoader(BaseDataLoader):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        """
            Acceptable config parameters
                paths: Multiple path locations for storing data
                suffix: File suffix to read
                recursive: Whether to recursively search for files
                formatter_name: Name of the formatter
                num: Limit data quantity, default is -1 which means load all
                seed: Sampling seed
        """
        
        # Paths can be a single string or a list of strings
        paths_config = self.config.get('paths')
        self.suffix = self.config.get("suffix", "jsonl")
        if isinstance(paths_config, str):
            self.paths: List[Path] = [Path(paths_config)]
        elif isinstance(paths_config, list):
            self.paths: List[Path] = [Path(p) for p in paths_config]
        else:
            raise ValueError("'paths' in config must be a string or a list of strings.")

        self.recursive: bool = self.config.get('recursive', False)
        
        formatter_name = self.config['formatter_name']
        formatter_class = get_formatter(formatter_name)
        self.formatter: BaseFormatter = formatter_class()
        
        self.num = config.get("num", -1)
        if not isinstance(self.num, int):
            raise ValueError(f"`num` must be int, but got {type(self.num)}")
        self.seed = config.get("seed", 42)
        random.seed(self.seed)
        
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
        
        return sorted(list(all_files))
    
    def _load_and_process_data(self):
        file_paths = self._get_file_paths()
        if not file_paths:
            logging.warning(f"No files found to process for the given paths.")
            return

        for file_path in file_paths:
            if 'gaokao-mathcloze' in str(file_path) or 'math.jsonl' in str(file_path): # Fill-in-the-blank questions
                continue
            logging.info(f"Loading and processing file: {file_path}")
            try:
                
                # Some dataset labels are in separate files
                labels = []
                if 'PIQA' in str(file_path):
                    label_path = str(file_path).replace('.jsonl', '-labels.lst')
                    with open(label_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()  # Returns a list containing all lines (each line may contain newline characters at the end)
                    labels = [line.strip() for line in lines]

                # Read JSON Lines file line by line
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        try:
                            raw_sample = json.loads(line)
                            # print(json.dumps(raw_sample, indent=4, ensure_ascii=False))
                            # print(raw_sample.keys())
                            # print(raw_sample["context"])
                            # print(raw_sample["discussion"])
                            # print(raw_sample["summary"])
                            # sys.exit(0)
                            if len(labels) > 0:
                                raw_sample['label'] = labels[i]
                            formatted_sample = self.formatter.format(raw_sample)
                            # print(json.dumps(formatted_sample, indent=4, ensure_ascii=False))
                            # sys.exit(0)
                            self._samples.append(formatted_sample)
                        except json.JSONDecodeError as e:
                            logging.warning(f"Failed to parse {self.suffix} in line #{i+1} of file {file_path}. Error: {e}. Skipping line.")
                            continue
                        except Exception as e:
                            logging.warning(
                                f"Failed to format sample from line #{i+1} in file {file_path}. "
                                f"Error: {e}. Skipping sample."
                            )
                            continue
                
                
                
            except IOError as e:
                logging.error(f"Failed to read file {file_path}: {e}")
                continue
        # Sampling
        # Modified around line 127
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