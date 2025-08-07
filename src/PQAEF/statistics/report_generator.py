# -*- coding: utf-8 -*-
import os
import datetime
from pyexpat import model
import re
import json  # 添加json导入
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from ..constant import constant

# 导入此模块会加载所有分析器并填充注册表
from .analysis import __init__ 
from .analysis.registry import ANALYSIS_REGISTRY

class ReportGenerator:
    """
    通过动态加载和运行可配置的分析器模块，分析已处理的数据并生成综合报告。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = self.config['output_dir']
        self.file_prefix = self.config.get('file_prefix', 'statistics')
        # 修改这里：支持两种字段名
        self.analyses_to_run = self.config.get(
            'analyses_to_run', 
            self.config.get('analyzers', list(ANALYSIS_REGISTRY.keys()))
        )

        os.makedirs(self.output_dir, exist_ok=True)
        self.font_props = self._get_font_properties()
        plt.rcParams['axes.unicode_minus'] = False
        print(f"INFO: ReportGenerator configured to run the following analyses: {self.analyses_to_run}")
        print(f"INFO: ReportGenerator config: {self.config}")
    # 移除以下方法，因为路径构建逻辑已经移到run.py中：
    # - _build_complete_output_dir
    # - _extract_model_name  
    # - _extract_dataset_name

    def _get_font_properties(self) -> FontProperties:
        """加载用于在绘图中支持 CJK 字符的捆绑字体文件。"""
        try:
            font_path = os.path.join(
                os.path.dirname(__file__), '..', 'resources', 'fonts', 'NotoSansSC-Regular.ttf'
            )
            if os.path.exists(font_path):
                print(f"INFO: Found CJK font at '{font_path}'.")
                return FontProperties(fname=font_path)
            else:
                print("WARNING: CJK font not found. Chinese characters in plots may appear as squares.")
                return None
        except Exception as e:
            print(f"ERROR: An error occurred while loading the font: {e}")
            return None

    def _flatten_sample_annotations(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """展平嵌套的注释字典并添加轮次计数。"""
        flat_data = {}
        
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        dialogue_ann = sample.get(constant.KEY_DIALOGUE_ANNOTATION, {})
        flat_data.update(flatten_dict(dialogue_ann))
        flat_data['turn_count'] = len(sample.get(constant.KEY_DIALOGUES, []))
        return flat_data

    def _unpack_list_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """查找包含列表的列并将其解包为单独的列。"""
        print("INFO: Checking for list-based columns to unpack...")
        cols_to_unpack = []
        for col in df.columns:
            # 检查非空值是否为列表
            first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(first_val, list):
                cols_to_unpack.append(col)

        for col in cols_to_unpack:
            print(f"INFO: Unpacking list-based column: '{col}'")
            # 确保索引对齐
            unpacked_cols = pd.DataFrame(df[col].tolist(), index=df.index).add_prefix(f'{col}_')
            df = df.join(unpacked_cols)
            df = df.drop(columns=[col])
        return df

    # def _write_json_file(self, data: List[Dict[str, Any]]):
    #     # 保存第一条记录（准确率统计信息）到JSON文件
    #     if data and len(data) > 0:
    #         first_record = data[0]  # 获取第一条记录
    #         # 构建与statistical_analysis同级的JSON文件路径
    #         accuracy_stats_path = os.path.join(os.path.dirname(self.output_dir), "accuracy_stats.json")
            
    #         try:
    #             with open(accuracy_stats_path, 'w', encoding='utf-8') as f:
    #                 json.dump(first_record, f, ensure_ascii=False, indent=2)
    #             print(f"INFO: Accuracy statistics saved to {accuracy_stats_path}")
    #         except Exception as e:
    #             print(f"ERROR: Failed to save accuracy statistics: {e}")

    def analyze(self, data: List[Dict[str, Any]], run_metadata: Dict[str, Any]):
        """执行所有分析并生成报告的主方法。"""
        if not data:
            print("WARNING: No data provided for analysis.")
            return

        # self._write_json_file(data)

        print("\n--- Starting Statistical Analysis ---")
        
        
        # 1. 数据预处理
        print("[Step 1/3] Pre-processing data...")
        flattened_data = [self._flatten_sample_annotations(s) for s in data]
        df = pd.DataFrame(flattened_data)
        df = self._unpack_list_columns(df)
        
        # 打印df
        # print(df)
    
        # 2. 运行已配置的分析器
        print("[Step 2/3] Running configured analyzers...")
        all_results = []
        for name in self.analyses_to_run:
            if name not in ANALYSIS_REGISTRY:
                print(f"WARNING: Analyzer '{name}' is specified in config but not registered. Skipping.")
                continue
            
            AnalyzerClass = ANALYSIS_REGISTRY[name]
            analyzer_instance = AnalyzerClass(self.config, self.output_dir, self.font_props, self.file_prefix)
            # result = analyzer_instance.analyze(df)
            result = analyzer_instance.analyze(df, data) # <--- MODIFIED
            
            all_results.append(result)
    
        # 3. 编写统一的报告
        print("[Step 3/3] Compiling final report...")
        self._write_report_file(data, run_metadata, all_results)
        
        print("--- Statistical Analysis Finished ---")

    def _write_report_file(self, data, run_metadata, all_results: List[Dict[str, Any]]):
        """将所有分析结果编译到一个文本文件中。"""
        report_path = os.path.join(self.output_dir, f"{self.file_prefix}_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # 报告头
            f.write("="*80 + "\n")
            f.write(" PQAEF Modular Statistical Report\n")
            f.write("="*80 + "\n")
            f.write(f"Report Generated On: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples Analyzed: {len(data)}\n\n")

            # 运行元数据
            f.write("-" * 30 + " Tasks Performed " + "-"*31 + "\n")
            for i, task_info in enumerate(run_metadata.get('tasks_run', [])):
                # f.write(f"  {i+1}. {task_info['task_class']} -> '{task_info['config'].get('output_field_name')}'\n")
                f.write(f"  {i+1}. {task_info['task_class']}\n")
            f.write("\n")

            # 汇总所有图表
            all_plots = [plot for res in all_results for plot in res.get("plots", [])]
            if all_plots:
                f.write("-" * 32 + " Generated Plots " + "-"*31 + "\n")
                f.write(f"  Plots are saved in: {os.path.abspath(self.output_dir)}\n")
                for path in all_plots:
                    f.write(f"  - {os.path.basename(path)}\n")
                f.write("\n")

            # 动态写入每个分析器的结果
            for i, result in enumerate(all_results):
                f.write("="*80 + "\n")
                f.write(f" Part {i+1}: {result.get('title', 'Untitled Analysis')}\n")
                f.write("="*80 + "\n")
                
                if result.get('summary'):
                    f.write(result['summary'])
                    f.write("\n\n")

                if result.get('data_tables'):
                    for table_name, table_df in result['data_tables'].items():
                        f.write(f"--- {table_name} ---\n")
                        with pd.option_context('display.width', 120, 'display.max_rows', 100, 'display.float_format', '{:.4f}'.format):
                           f.write(table_df.to_string())
                        f.write("\n\n")

        print(f"INFO: Full statistical report saved to {report_path}")
