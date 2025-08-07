#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量转换配置文件脚本
功能：
1. 支持将配置文件中的模型替换为指定的ApiModel或LocalModel配置
2. 自动更新tasks中的llm_model_name引用
3. 支持从外部传入配置或从JSON文件读取配置
"""

import os
import yaml
import re
import json
import sys
from pathlib import Path
from typing import Dict, Any

def get_default_config() -> Dict[str, Any]:
    """
    返回默认的openai_evaluator配置
    """
    return {
        "model_type": "api",
        "model_name": "openai_evaluator",
        "class": "ApiModel",
        "config": {
            "provider": "url",
            "model_identifier": "YOUR_MODEL",
            "api_key": "YOUR_API_KEY",
            "base_url": "YOUR_BASE_URL",
            "concurrency": 1
        }
    }

def main(external_config: Dict[str, Any] = None):
    """
    主函数：批量处理所有配置文件
    external_config: 外部传入的配置，如果提供则使用该配置
    """
    # 获取test目录下所有YAML文件
    test_dir = Path('./test')
    yaml_files = list(test_dir.glob('*.yaml'))
    
    if not yaml_files:
        print("❌ 在test目录下未找到任何YAML文件")
        return
    
    # 如果提供了外部配置，使用它；否则使用默认配置
    target_config = external_config if external_config else get_default_config()
    
    print(f"找到 {len(yaml_files)} 个YAML配置文件")
    if external_config:
        model_id = external_config.get('config', {}).get('model_identifier', 
                   external_config.get('config', {}).get('model_path', 'unknown'))
        model_type = external_config.get('model_type', 'unknown')
        model_name = external_config.get('model_name', 'unknown')
        print(f"使用外部配置，模型类型: {model_type}, 模型名称: {model_name}, 模型标识: {model_id}")
    print("=" * 60)
    
    modified_yaml_count = 0
    total_yaml_count = len(yaml_files)
    
    # 逐个处理YAML文件
    for yaml_file in sorted(yaml_files):
        print(f"\n处理文件: {yaml_file.name}")
        if update_yaml_config_with_target(yaml_file, target_config):
            modified_yaml_count += 1
    
    # 输出总结
    print("\n" + "=" * 60)
    print(f"处理完成！")
    print(f"YAML文件总数: {total_yaml_count}")
    print(f"YAML文件已修改: {modified_yaml_count}")
    print(f"YAML文件未修改: {total_yaml_count - modified_yaml_count}")
    
    if modified_yaml_count > 0:
        model_name = target_config.get('model_name', 'unknown')
        model_type = target_config.get('model_type', 'unknown')
        print(f"\n🎉 转换完成！已成功更新配置为模型: {model_name} (类型: {model_type})")
    else:
        print(f"\n✅ 所有文件都已是目标配置，无需修改。")

def update_yaml_config_with_target(file_path: Path, target_config: Dict[str, Any]) -> bool:
    """
    使用指定的目标配置更新单个YAML配置文件
    """
    try:
        # 读取YAML文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        modified = False
        target_model_name = target_config.get('model_name')
        target_model_class = target_config.get('class')
        target_model_config = target_config.get('config')
        
        # 检查models配置
        if 'models' in data:
            # 获取当前所有模型名称
            current_model_names = list(data['models'].keys())
            
            # 清空现有模型配置
            data['models'].clear()
            
            # 添加新的模型配置
            data['models'][target_model_name] = {
                'class': target_model_class,
                'name': target_model_name,
                'config': target_model_config
            }
            
            if current_model_names:
                print(f"  ✓ 已将模型配置更新为 {target_model_name} ({target_model_class})")
                modified = True
        
        # 检查并更新tasks中的llm_model_name引用
        if 'tasks' in data:
            # 处理tasks为字典的情况
            if isinstance(data['tasks'], dict):
                for task_name, task_config in data['tasks'].items():
                    if isinstance(task_config, dict) and 'config' in task_config:
                        current_model = task_config['config'].get('llm_model_name')
                        if current_model and current_model != target_model_name:
                            task_config['config']['llm_model_name'] = target_model_name
                            modified = True
                            print(f"  ✓ 已更新任务 {task_name} 的 llm_model_name: {current_model} -> {target_model_name}")
            
            # 处理tasks为列表的情况
            elif isinstance(data['tasks'], list):
                for i, task_config in enumerate(data['tasks']):
                    if isinstance(task_config, dict) and 'config' in task_config:
                        current_model = task_config['config'].get('llm_model_name')
                        if current_model and current_model != target_model_name:
                            task_config['config']['llm_model_name'] = target_model_name
                            modified = True
                            print(f"  ✓ 已更新任务列表第 {i+1} 项的 llm_model_name: {current_model} -> {target_model_name}")
        
        # 如果有修改，写回文件
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, 
                         sort_keys=False, indent=2)
            return True
        else:
            print(f"  - 无需修改")
            return False
            
    except Exception as e:
        print(f"  ❌ 处理文件时出错: {e}")
        return False

if __name__ == '__main__':
    # 检查是否有命令行参数传入配置文件
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                external_config = json.load(f)
            main(external_config)
        except Exception as e:
            print(f"❌ 读取配置文件失败: {e}")
            sys.exit(1)
    else:
        main()