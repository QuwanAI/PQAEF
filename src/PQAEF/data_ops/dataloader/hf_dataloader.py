# -*- coding: utf-8 -*-
"""
Provides a flexible DataLoader for reading data from JSON/JSONL files.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Iterator, List, Union

from datasets import load_dataset
from typing_extensions import override

from .base_dataloader import BaseDataLoader, register_dataloader
from PQAEF.utils.template_registry import get_formatter, BaseFormatter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@register_dataloader("HfDataLoader")
class HfDataLoader(BaseDataLoader):
    """
    Loads a dataset from the Hugging Face Hub and uses a registered
    formatter to process each sample.

    It supports both standard (downloading) and streaming modes. The specific
    formatter to use is determined by the 'formatter_name' in the config.

    Configuration Example:
    ----------------------
    config = {
        "path": "squad",              # The dataset path on Hugging Face Hub
        "name": "plain_text",         # Optional: The specific configuration/subset
        "split": "validation",        # Optional: The split to load
        "streaming": False,           # Optional: Stream the dataset. Defaults to False.
        "formatter_name": "squad_formatter" # REQUIRED: The name of the formatter to use
    }
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # --- Hugging Face Specific Config ---
        self.path = self.config.get('path')
        if not self.path:
            raise ValueError("'path' is a required key in the HfDataLoader config.")
        
        # self.name = self.config.get('name')
        self.split = self.config.get('split', "train")
        self.streaming = self.config.get('streaming', False)
        
        self.num = self.config.get("num", -1)
        self._samples: List[Dict[str, Any]] = []
        
        # --- Formatter Initialization (following JsonLoader pattern) ---
        formatter_name = self.config.get('formatter_name')
        if not formatter_name:
            raise ValueError("'formatter_name' is a required key in the HfDataLoader config.")
            
        try:
            formatter_class = get_formatter(formatter_name)
            self.formatter: BaseFormatter = formatter_class()
        except ValueError as e:
            logging.error(f"Could not initialize formatter: {e}")
            raise
    
        logging.info(f"Preparing to load dataset '{self.path}' from Hugging Face Hub.")
        logging.info(f"Split: {self.split or 'train'}, Streaming: {self.streaming}")
        try:
            # `load_dataset` does the heavy lifting of finding and loading data
            self.dataset = load_dataset(
                path=self.path,
                # name=self.name,
                split=self.split,
                streaming=self.streaming,
                trust_remote_code=True,
            )
            if self.num != -1:
                self.dataset = self.dataset.take(self.num)
        except Exception as e:
            logging.error(f"Fatal error: Failed to load dataset '{self.path}' from Hugging Face Hub. Error: {e}")
            # If we can't load the dataset at all, we stop iteration.
            return

        logging.info(f"Successfully loaded dataset. Now processing and formatting samples...")
        
        for i, raw_sample in enumerate(self.dataset):
            try:
                # Use the formatter to process the raw sample
                formatted_sample = self.formatter.format(raw_sample)
                self._samples.append(formatted_sample)
            except Exception as e:
                # This robust error handling (from JsonLoader) is crucial.
                # It allows the process to continue even if some samples are malformed.
                logging.warning(
                    f"Failed to format sample #{i+1} from dataset '{self.path}'. "
                    f"Error: {e}. Skipping sample."
                )
                continue

    @override
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Loads the dataset, iterates over its samples, formats each one,
        and yields it.
        """
        return iter(self._samples)


    
    @override
    def __len__(self):
        # return len(self.dataset)
        return len(self._samples)


# --- 使用示例 ---
if __name__ == "__main__":
    print("--- HfDataLoader Test ---")
    
    # 我们使用模拟的 "passthrough" 格式化器进行测试
    squad_config = {
        "path": "squad",
        "split": "validation",
        "formatter_name": "passthrough" # Use our test formatter
    }
    
    squad_loader = HfDataLoader(squad_config)
    
    print("\nIterating through the first 3 samples of SQuAD...")
    count = 0
    for data_sample in squad_loader:
        if count < 3:
            print(f"\nFormatted Sample {count + 1} (Keys: {list(data_sample.keys())})")
            print(f"  ID: {data_sample.get('id')}")
            print(f"  Title: {data_sample.get('title')}")
            count += 1
        else:
            break
            
    if count == 0:
        print("\nERROR: No samples were processed.")
    else:
        print(f"\nSuccessfully iterated through {count} samples.")