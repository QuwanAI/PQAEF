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
    CSV数据加载器
    
    支持加载CSV文件并通过注册的格式化器转换为标准格式
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化CSV数据加载器
        
        Args:
            config: 配置字典，包含以下字段：
                - paths: CSV文件路径（字符串）或路径列表
                - formatter_name: 格式化器名称（可选）
                - encoding: 文件编码，默认utf-8（可选）
                - skip_header: 是否跳过表头行，默认True（可选）
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
        
        # 如果指定了格式化器，则获取格式化器实例
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
                
            # 修改这里：支持TSV文件
            if not (str(data_path).lower().endswith('.csv') or str(data_path).lower().endswith('.tsv')):
                print(f"Warning: File {data_path} is not a CSV/TSV file, skipping...")
                continue
            
            try:
                # logger.info(f"Loading CSV file: {data_path}")
                with open(data_path, 'r', encoding=self.encoding, newline='') as csvfile:
                    
                    # 根据文件扩展名确定分隔符
                    if str(data_path).lower().endswith('.tsv'):
                        delimiter = '\t'  # TSV文件使用制表符
                    else:
                        # CSV文件尝试自动检测分隔符
                        try:
                            sample = csvfile.read(1024)
                            csvfile.seek(0)
                            sniffer = csv.Sniffer()
                            delimiter = sniffer.sniff(sample).delimiter
                        except csv.Error:
                            # 如果检测失败，默认使用逗号
                            delimiter = ','
                            csvfile.seek(0)
                    
                    # 读取所有行作为列表
                    reader = csv.reader(csvfile, delimiter=delimiter)
                    
                    # 跳过表头行
                    if self.skip_header:
                        try:
                            next(reader)  # 跳过第一行（表头）
                        except StopIteration:
                            continue  # 文件为空，跳过
                    
                    for row_idx, row in enumerate(reader):
                        try:
                            # 跳过空行
                            if not row or all(not cell.strip() for cell in row):
                                continue
                            
                            # 清理数据：移除空白字符
                            cleaned_row = [cell.strip() if isinstance(cell, str) else cell for cell in row]
                            
                            # 如果有格式化器，则使用格式化器处理数据
                            if self.formatter:
                                # logger.info(f"Formatting row {row_idx} in {data_path}")
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
        # 采样
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