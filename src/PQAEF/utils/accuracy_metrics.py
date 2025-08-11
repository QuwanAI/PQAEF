# -*- coding: utf-8 -*-
"""
Accuracy evaluation functions
For accuracy calculation in classification tasks
"""

from typing import List, Dict, Any


def calculate_accuracy_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate accuracy metrics
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dict[str, float]: Accuracy statistics
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