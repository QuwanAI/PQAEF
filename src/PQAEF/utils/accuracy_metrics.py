# -*- coding: utf-8 -*-
"""
准确率评测函数
用于分类任务的准确率计算
"""

from typing import List, Dict, Any


def calculate_accuracy_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    计算准确率
    
    Args:
        results: 评测结果列表
        
    Returns:
        Dict[str, float]: 准确率统计
    """
    if not results:
        return {'accuracy': 0.0, 'correct_count': 0, 'total_count': 0}
    
    correct_count = sum(1 for result in results if result.get('is_correct', False))
    total_count = len(results)
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': total_count
    }