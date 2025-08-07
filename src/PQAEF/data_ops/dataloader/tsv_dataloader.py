import csv
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Iterator, List, Union
import random
import os

from .base_dataloader import BaseDataLoader, register_dataloader
from PQAEF.utils.template_registry import get_formatter, BaseFormatter


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@register_dataloader("TSVDataLoader")
class TSVDataLoader(BaseDataLoader):
    """
    TSV数据加载器

    支持加载TSV文件并通过注册的格式化器转换为标准格式
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化TSV数据加载器

        Args:
            config: 配置字典，包含以下字段：
                - paths: TSV文件路径（字符串）或路径列表
                - formatter_name: 格式化器名称（可选）
                - encoding: 文件编码，默认utf-8（可选）
                - skip_header: 是否跳过表头行，默认True（可选）
                - num: 采样数量，-1表示加载全部数据（可选）
        """
        super().__init__(config)

        paths_config = self.config.get('paths')
        # 默认文件后缀为 'tsv'
        self.suffix = self.config.get("suffix", "tsv")
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

    def _get_file_paths(self) -> List[Path]:
        """查找并返回所有符合条件的TSV文件路径"""
        all_files = set()
        for path in self.paths:
            if not path.exists():
                logging.warning(f"Path does not exist: {path}")
                continue

            if path.is_file():
                if path.suffix == f".{self.suffix}":
                    all_files.add(path)
                else:
                    logging.warning(f"Skipping non-{self.suffix} file specified directly: {path}")
            elif path.is_dir():
                glob_pattern = f'**/*.{self.suffix}' if self.recursive else f'*.{self.suffix}'
                all_files.update(path.glob(glob_pattern))

        return sorted(list(all_files))

    def _load_and_process_data(self):
        """加载并处理所有找到的TSV文件"""
        file_paths = self._get_file_paths()
        for data_path in file_paths:
            logging.info(f"Processing TSV file: {data_path}")
            try:
                with open(data_path, 'r', encoding=self.encoding, newline='') as tsvfile:
                    # TSV文件使用制表符作为分隔符
                    # quoting=csv.QUOTE_NONE 表示不处理引号，因为TSV标准中字段内不应包含制表符，从而避免了引号问题
                    reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)

                    # 跳过表头行
                    if self.skip_header:
                        try:
                            next(reader)
                        except StopIteration:
                            logging.warning(f"File is empty or contains only a header: {data_path}")
                            continue

                    for row_idx, row in enumerate(reader):
                        try:
                            # 跳过空行
                            if not row or all(not cell.strip() for cell in row):
                                continue

                            # 清理数据：移除单元格两端的空白字符
                            cleaned_row = [cell.strip() if isinstance(cell, str) else cell for cell in row]

                            # 如果有格式化器，则使用格式化器处理数据
                            if self.formatter:
                                formatted_data = self.formatter.format(cleaned_row)
                            else:
                                formatted_data = {
                                    'raw_data': cleaned_row,
                                    '_source_file': str(data_path),
                                    '_row_index': row_idx
                                }
                            self._samples.append(formatted_data)

                        except Exception as e:
                            logging.error(f"Error processing row {row_idx} in {str(data_path)}: {e}")
                            continue

            except Exception as e:
                logging.error(f"Error reading file {str(data_path)}: {e}")
                continue

        # 进行健壮的采样
        if self.num != -1:
            if self.num > len(self._samples):
                logging.warning(
                    f"Requested sample size ({self.num}) is larger than available data ({len(self._samples)}). Using all available data."
                )
                # 不进行采样，使用全部数据
            else:
                self._samples = random.sample(self._samples, self.num)

        logging.info(f"Finished loading. Total formatted samples: {len(self._samples)}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """返回已加载和格式化好的样本迭代器"""
        return iter(self._samples)

    def __len__(self) -> int:
        """返回加载的样本总数"""
        return len(self._samples)

    def get_total_count(self) -> int:
        """
        获取数据总数

        Returns:
            int: 加载的数据总数
        """
        return self.__len__()