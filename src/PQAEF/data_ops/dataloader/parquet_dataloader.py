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
    Parquet数据加载器
    
    支持加载Parquet文件并通过注册的格式化器转换为标准格式
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Parquet数据加载器
        
        Args:
            config: 配置字典，包含以下字段：
                - path: Parquet文件路径（字符串）或路径列表
                - formatter_name: 格式化器名称（可选）
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

        # 如果指定了格式化器，则获取格式化器实例
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
                # 读取parquet文件
                df = pd.read_parquet(data_path)
                
                for row_idx, (_, row) in enumerate(df.iterrows()):
                    try:
                        # 将pandas Series转换为字典
                        row_dict = row.to_dict()
                        
                        # 如果有格式化器，则使用格式化器处理数据
                        if self.formatter:
                            formatted_data = self.formatter.format(row_dict)
                            # 过滤掉None值
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
        # 采样
        # 在第127行附近修改
        if self.num != -1:
            if self.num > len(self._samples):
                logging.warning(f"Requested sample size ({self.num}) is larger than available data ({len(self._samples)}). Using all available data.")
                # 不进行采样，使用全部数据
            else:
                self._samples = random.sample(self._samples, self.num)
        
        logging.info(f"Finished loading. Total formatted samples: {len(self._samples)}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # 直接迭代已经加载和格式化好的样本列表
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)
    
    def get_total_count(self) -> int:
        """
        获取数据总数（可选实现）
        
        Returns:
            int: 数据总数，如果无法确定则返回-1
        """
        return self.__len__()