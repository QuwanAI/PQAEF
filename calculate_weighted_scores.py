# -*- coding: utf-8 -*-

"""
该脚本用于根据层级化的权重配置文件 (weight_config.yaml) 和分散在目录中的数据集
评测结果，递归地计算每一级能力的加权分数，并生成合并的数据集分数JSON文件。
"""

import os
import json
import yaml
import logging
import argparse
from typing import Dict, Any, Optional

# --- 1. 配置区域 (默认值) ---
DEFAULT_OUTPUT_DIR = "result_analyze"
DEFAULT_BASE_RESULT_PATH = "./output/test"
DEFAULT_WEIGHT_CONFIG_FILE = "weight_config.yaml"
RESULT_FILENAME = "statistical_analysis/result_stats.json"
DEFAULT_SCORE = 70.0

SUCCESSFULLY_LOADED_DATASETS = set()

# --- 2. 日志与全局变量 ---
BASE_RESULT_PATH = DEFAULT_BASE_RESULT_PATH 

def setup_logging(log_file_path: str):
    """配置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, 'w', 'utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )

# --- 3. 核心功能函数 ---

def get_dataset_score(dataset_name: str, model_path: str) -> Optional[float]:
    """根据数据集名称和模型路径获取分数。如果文件不存在或解析失败，返回None表示不参与计算。"""
    score_file_path = os.path.join(model_path, dataset_name, RESULT_FILENAME)
    if not os.path.exists(score_file_path):
        logging.warning(f"结果文件未找到: {dataset_name} (模型: {os.path.basename(model_path)})。该数据集不参与计算")
        return None
    try:
        with open(score_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            first_value = next(iter(data.values()))
            if not isinstance(first_value, (int, float)):
                logging.error(f"在 {score_file_path} 中找到的第一个值不是数字，该数据集不参与计算")
                return None
            
            # 转换为百分制
            score_percentage = float(first_value) * 100
            logging.info(f"成功加载分数: {dataset_name} = {score_percentage:.2f}分 (模型: {os.path.basename(model_path)})")
            SUCCESSFULLY_LOADED_DATASETS.add(dataset_name)
            return score_percentage
    except (json.JSONDecodeError, StopIteration, AttributeError) as e:
        logging.error(f"解析JSON文件失败: {score_file_path}，错误: {e}。该数据集不参与计算")
        return None

def calculate_scores(node: Dict[str, Any], node_name: str, model_path: str) -> Optional[float]:
    """
    递归计算节点的分数，采用加权平均，缺失数据不参与计算。
    """
    # 情况1: 中间节点 (有 'sub_tasks' 字典)
    if 'sub_tasks' in node and isinstance(node.get('sub_tasks'), dict):
        sub_tasks_node = node['sub_tasks']
        total_weighted_score_sum = 0.0
        total_weight_sum = 0.0
        has_valid_child = False

        for task_name, task_config in sub_tasks_node.items():
            child_score = None
            weight = 1.0

            if isinstance(task_config, dict):
                child_score = calculate_scores(task_config, task_name, model_path)
                weight = task_config.get('weight', 1.0)
            elif isinstance(task_config, (int, float)):
                child_score = get_dataset_score(task_name, model_path)
                weight = task_config

            # 只有当child_score不为None时才参与计算
            if child_score is not None:
                total_weighted_score_sum += child_score * weight
                total_weight_sum += weight
                has_valid_child = True
        
        if has_valid_child and total_weight_sum > 0:
            final_score = total_weighted_score_sum / total_weight_sum
            node['_score'] = final_score
            return final_score
        return None

    # 情况2: 未细分到数据集的叶子任务 (只有 'weight' 键)
    elif len(node) == 1 and 'weight' in node:
        logging.info(f"任务 '{node_name}' 未细分到数据集，不参与计算。")
        return None
    
    # 情况3: 最底层的节点，直接是数据集:权重的集合
    else:
        total_weighted_score_sum = 0.0
        total_weight_sum = 0.0
        has_valid_child = False
        
        # 存储数据集分数用于报告
        dataset_scores = {}
        
        for dataset_name, weight in node.items():
            if not isinstance(dataset_name, str) or not isinstance(weight, (int, float)):
                continue
            
            score = get_dataset_score(dataset_name, model_path)
            # 只有当score不为None时才参与计算
            if score is not None:
                dataset_scores[dataset_name] = score
                total_weighted_score_sum += score * weight
                total_weight_sum += weight
                has_valid_child = True
            else:
                # 记录未参与计算的数据集
                dataset_scores[dataset_name] = None
        
        # 将数据集分数存储到节点中，用于报告生成
        node['_dataset_scores'] = dataset_scores
        
        if has_valid_child and total_weight_sum > 0:
            final_score = total_weighted_score_sum / total_weight_sum
            node['_score'] = final_score
            return final_score
        return None

def collect_all_datasets(node: Dict[str, Any]) -> set:
    """
    递归收集配置文件中的所有数据集名称
    """
    datasets = set()
    
    # 情况1: 中间节点 (有 'sub_tasks' 字典)
    if 'sub_tasks' in node and isinstance(node.get('sub_tasks'), dict):
        for task_name, task_config in node['sub_tasks'].items():
            if isinstance(task_config, dict):
                datasets.update(collect_all_datasets(task_config))
            elif isinstance(task_config, (int, float)):
                # 如果task_config是数字，说明task_name是数据集名称
                datasets.add(task_name)
    
    # 情况2: 最底层的节点，直接是数据集:权重的集合
    else:
        for key, value in node.items():
            # 跳过特殊键
            if key in ['weight', 'sub_tasks', '_score', '_dataset_scores']:
                continue
            # 如果值是数字，说明这是一个数据集:权重的配置
            if isinstance(value, (int, float)):
                datasets.add(key)
    
    return datasets

def generate_model_datasets_json(model_path: str) -> Dict[str, float]:
    """
    为单个模型生成包含所有数据集分数的字典
    直接遍历模型路径下的所有数据集目录
    """
    model_name = os.path.basename(model_path)
    logging.info(f"开始处理模型: {model_name}")
    
    # 直接遍历模型路径下的所有目录，每个目录代表一个数据集
    all_datasets = set()
    if os.path.exists(model_path):
        for item in os.listdir(model_path):
            item_path = os.path.join(model_path, item)
            if os.path.isdir(item_path):
                # 检查是否存在结果文件
                result_file = os.path.join(item_path, RESULT_FILENAME)
                if os.path.exists(result_file):
                    all_datasets.add(item)
    
    logging.info(f"模型 {model_name} 共发现 {len(all_datasets)} 个数据集")
    
    # 为每个数据集获取分数
    dataset_scores = {}
    for dataset_name in sorted(all_datasets):  # 排序以便输出有序
        score = get_dataset_score(dataset_name, model_path)
        if score is not None:
            dataset_scores[dataset_name] = round(score, 2)  # 保留两位小数
        else:
            dataset_scores[dataset_name] = -1  # 没有分数的设为-1
    
    # 统计信息
    valid_scores = sum(1 for score in dataset_scores.values() if score != -1)
    invalid_scores = len(dataset_scores) - valid_scores
    logging.info(f"模型 {model_name} 统计: 有效分数 {valid_scores} 个，缺失分数 {invalid_scores} 个")
    
    return dataset_scores

def generate_all_models_json() -> None:
    """
    遍历所有模型，生成合并的数据集分数JSON文件
    """
    logging.info("开始遍历所有模型...")
    
    # 检查基础路径是否存在
    if not os.path.exists(BASE_RESULT_PATH):
        logging.error(f"基础结果路径不存在: {BASE_RESULT_PATH}")
        return
    
    # 获取所有模型目录
    model_dirs = []
    for item in os.listdir(BASE_RESULT_PATH):
        item_path = os.path.join(BASE_RESULT_PATH, item)
        if os.path.isdir(item_path):
            model_dirs.append(item_path)
    
    if not model_dirs:
        logging.error(f"在 {BASE_RESULT_PATH} 中未找到任何模型目录")
        return
    
    logging.info(f"找到 {len(model_dirs)} 个模型目录")
    
    # 为每个模型生成数据集分数
    all_models_scores = {}
    for model_path in sorted(model_dirs):
        model_name = os.path.basename(model_path)
        logging.info(f"\n--- 处理模型: {model_name} ---")
        
        # 重置全局变量
        global SUCCESSFULLY_LOADED_DATASETS
        SUCCESSFULLY_LOADED_DATASETS = set()
        
        # 生成该模型的数据集分数（不再需要config_data参数）
        model_scores = generate_model_datasets_json(model_path)
        
        # 只保留有效分数（不为-1的）
        valid_scores = {k: v for k, v in model_scores.items() if v != -1}
        if valid_scores:
            all_models_scores[model_name] = valid_scores
            logging.info(f"模型 {model_name} 添加了 {len(valid_scores)} 个有效数据集分数")
        else:
            logging.warning(f"模型 {model_name} 没有有效的数据集分数")
    
    # 保存合并的JSON文件
    output_file = os.path.join(DEFAULT_OUTPUT_DIR, "scores.json")
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_models_scores, f, ensure_ascii=False, indent=4)
    
    logging.info(f"\n合并的数据集分数JSON文件已保存到: {output_file}")
    logging.info(f"总共处理了 {len(all_models_scores)} 个模型")

# --- 4. 主程序入口 ---

def main():
    """主执行函数"""
    # 设置和解析命令行参数
    parser = argparse.ArgumentParser(description="遍历模型目录并计算数据集分数。")
    parser.add_argument(
        '--results_path',
        type=str,
        default=DEFAULT_BASE_RESULT_PATH,
        help=f"评测结果的根目录路径。默认为: {DEFAULT_BASE_RESULT_PATH}"
    )
    args = parser.parse_args()

    # 使用命令行参数或默认值更新全局变量
    global BASE_RESULT_PATH
    BASE_RESULT_PATH = args.results_path

    # 创建输出目录并配置日志
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    setup_logging(os.path.join(DEFAULT_OUTPUT_DIR, "score_calculation.log"))
    
    logging.info("--- 开始计算所有模型的数据集分数 ---")
    logging.info(f"使用结果路径: {BASE_RESULT_PATH}")
    
    # 生成所有模型的数据集分数JSON文件（不再需要配置文件）
    logging.info("开始生成所有模型的数据集分数JSON文件...")
    generate_all_models_json()
    
    logging.info("--- 全部分析完成 ---")


if __name__ == "__main__":
    main()