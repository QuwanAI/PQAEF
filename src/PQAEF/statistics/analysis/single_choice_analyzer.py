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
    Analyzer for single-choice tasks, calculating metrics such as accuracy and F1 score.
    """

    def analyze(self, df: pd.DataFrame, raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        print("INFO: [Analyzer] Running Single Choice Analysis...")
        
        # Check if raw data contains expected keys
        if not raw_data or not all('is_correct' in item for item in raw_data if isinstance(item, dict)):
            return {
                "title": "Single Choice Analysis",
                "summary": "Skipped: Input data does not contain the required 'is_correct' key for this analysis.",
                "plots": [],
                "data_tables": {}
            }
        
        # Filter data items containing is_correct field (excluding the first statistical item)
        results = [item for item in raw_data if isinstance(item, dict) and 'is_correct' in item]
        
        if not results:
            return {
                "title": "Single Choice Analysis",
                "summary": "No valid results found for analysis.",
                "plots": [],
                "data_tables": {}
            }
        # Get evaluation tool type, default to Accuracy
        eval_tool = self.config.get('eval_tool') or ['Accuracy']
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Ensure eval_tool is in list format
        if isinstance(eval_tool, str):
            eval_tool = [eval_tool]
        
        # Iterate through all evaluation tools and calculate corresponding metrics
        for tool in eval_tool:
            if tool == 'Accuracy':
                accuracy_metrics = self._calculate_accuracy(results)
                metrics.update(accuracy_metrics)
        
        # If no specific evaluation tool is specified, use accuracy by default
        if not metrics:
            metrics = self._calculate_accuracy(results)
        
        # Write metrics to JSON file
        self._write_metrics_to_json(metrics)
        
        return {
            "title": "Single Choice Analysis",
            "summary": f"Calculated {eval_tool} metrics: {metrics}",
            "plots": [],
            "data_tables": {}
        }
    
    def _calculate_accuracy(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate accuracy metrics
        """
        correct_count = sum(1 for result in results if result['is_correct'])
        total_count = len(results)
        
        accuracy = {
            'overall': correct_count / total_count if total_count > 0 else 0,
            'correct_count': correct_count,
            'total_count': total_count
        }
        print(f"Overall accuracy: {accuracy['overall']:.2%}")
        return accuracy
            
    def _write_metrics_to_json(self, metrics: Dict[str, Any]):
        """
        Write metrics to JSON file
        """
        try:
            # Build output file path (at the same level as statistical_analysis directory)
            output_file = os.path.join(self.output_dir, 'result_stats.json')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            print(f"Metrics saved to: {output_file}")
        except Exception as e:
            print(f"Error writing metrics to JSON: {e}")