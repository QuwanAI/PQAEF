# -*- coding: utf-8 -*-
import json
import os
from typing import Dict, Any, List

from .base_dumper import BaseDataDumper

class JsonDataDumper(BaseDataDumper):
    """
    Dumps data into chunked JSON files.
    This class is focused solely on writing data to disk.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.output_dir = self.config['output_dir']
        self.file_prefix = self.config.get('file_prefix', 'processed_data')
        self.chunk_size = self.config.get('chunk_size', 5000)
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def dump(self, data: List[Dict[str, Any]], run_metadata: Dict[str, Any]):
        """
        Dumps the data into chunked JSON files.
        The run_metadata is ignored by this dumper but kept for interface consistency.
        """
        if not data:
            print("WARNING: No data to dump.")
            return
            
        print(f"INFO: Starting data dump. Total samples: {len(data)}, Chunk size: {self.chunk_size}")

        num_chunks = (len(data) + self.chunk_size - 1) // self.chunk_size
        for i in range(num_chunks):
            chunk_data = data[i * self.chunk_size : (i + 1) * self.chunk_size]
            file_name = f"{self.file_prefix}_{i:04d}.json"
            file_path = os.path.join(self.output_dir, file_name)
            
            print(f"INFO: Writing chunk {i+1}/{num_chunks} to {file_path}...")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)

        print("INFO: All data chunks have been successfully saved.")