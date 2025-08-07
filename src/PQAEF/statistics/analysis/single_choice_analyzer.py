# -*- coding: utf-8 -*-
from typing import Dict, Any, List
import pandas as pd
import json
import os
from sklearn.metrics import f1_score

from .registry import register_analyzer
from .base_analysis import BaseAnalysis
from PQAEF.constant import constant


@register_analyzer(name="single_choice")
class SingleChoiceAnalyzer(BaseAnalysis):
    """
    单选题任务的分析器，计算准确率和F1分数等指标。
    """

    def analyze(self, df: pd.DataFrame, raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        print("INFO: [Analyzer] Running Single Choice Analysis...")
        
        # 检查原始数据是否包含预期的键
        if not raw_data or not all('is_correct' in item for item in raw_data if isinstance(item, dict)):
            return {
                "title": "Single Choice Analysis",
                "summary": "Skipped: Input data does not contain the required 'is_correct' key for this analysis.",
                "plots": [],
                "data_tables": {}
            }
        
        # 过滤出包含is_correct字段的数据项（排除第一个统计项）
        results = [item for item in raw_data if isinstance(item, dict) and 'is_correct' in item]
        
        if not results:
            return {
                "title": "Single Choice Analysis",
                "summary": "No valid results found for analysis.",
                "plots": [],
                "data_tables": {}
            }
        # 打印result
        # print("INFO: [Analyzer] Single Choice Analysis Results:")
        # print(results)
        # print(self.config)
        # 获取评估工具类型，默认为Accuracy
        # 尝试从第一个结果中获取eval_tool信息
        eval_tool = self.config.get('eval_tool') or ['Accuracy']
        
        # 初始化metrics字典
        metrics = {}
        
        # 确保eval_tool是列表格式
        if isinstance(eval_tool, str):
            eval_tool = [eval_tool]
        
        # 遍历所有评估工具，计算相应指标
        for tool in eval_tool:
            if tool == 'Accuracy':
                accuracy_metrics = self._calculate_accuracy(results)
                metrics.update(accuracy_metrics)
        
        # 如果没有指定的评估工具，默认使用准确率
        if not metrics:
            metrics = self._calculate_accuracy(results)
        
        # 将指标写入JSON文件
        self._write_metrics_to_json(metrics)
        
        return {
            "title": "Single Choice Analysis",
            "summary": f"Calculated {eval_tool} metrics: {metrics}",
            "plots": [],
            "data_tables": {}
        }
    
    def _calculate_accuracy(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算准确率指标
        """
        correct_count = sum(1 for result in results if result['is_correct'])
        total_count = len(results)
        
        accuracy = {
            'overall': correct_count / total_count if total_count > 0 else 0,
            'correct_count': correct_count,
            'total_count': total_count
        }
        print(f"总体准确率: {accuracy['overall']:.2%}")
        return accuracy
            
    def _write_metrics_to_json(self, metrics: Dict[str, Any]):
        """
        将指标写入JSON文件
        """
        try:
            # 构建输出文件路径（与statistical_analysis目录同级）
            output_file = os.path.join(self.output_dir, 'result_stats.json')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            print(f"Metrics saved to: {output_file}")
        except Exception as e:
            print(f"Error writing metrics to JSON: {e}")