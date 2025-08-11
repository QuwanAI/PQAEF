"""
Batch configuration file conversion script
Features:
1. Support replacing models in configuration files with specified ApiModel or LocalModel configurations
2. Automatically update llm_model_name references in tasks
3. Support external configuration input or reading configuration from JSON files
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
    Return default openai_evaluator configuration
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
    Main function: batch process all configuration files
    external_config: External configuration input, use this configuration if provided
    """
    # Get all YAML files in test directory
    test_dir = Path('./test')
    yaml_files = list(test_dir.glob('*.yaml'))
    
    if not yaml_files:
        print("❌ No YAML files found in test directory")
        return
    
    # If external configuration is provided, use it; otherwise use default configuration
    target_config = external_config if external_config else get_default_config()
    
    print(f"Found {len(yaml_files)} YAML configuration files")
    if external_config:
        model_id = external_config.get('config', {}).get('model_identifier', 
                   external_config.get('config', {}).get('model_path', 'unknown'))
        model_type = external_config.get('model_type', 'unknown')
        model_name = external_config.get('model_name', 'unknown')
        print(f"Using external configuration, model type: {model_type}, model name: {model_name}, model identifier: {model_id}")
    print("=" * 60)
    
    modified_yaml_count = 0
    total_yaml_count = len(yaml_files)
    
    # Process YAML files one by one
    for yaml_file in sorted(yaml_files):
        print(f"\nProcessing file: {yaml_file.name}")
        if update_yaml_config_with_target(yaml_file, target_config):
            modified_yaml_count += 1
    
    # Output summary
    print("\n" + "=" * 60)
    print(f"Processing completed!")
    print(f"Total YAML files: {total_yaml_count}")
    print(f"YAML files modified: {modified_yaml_count}")
    print(f"YAML files unchanged: {total_yaml_count - modified_yaml_count}")
    
    if modified_yaml_count > 0:
        model_name = target_config.get('model_name', 'unknown')
        model_type = target_config.get('model_type', 'unknown')
        print(f"\n🎉 Conversion completed! Successfully updated configuration to model: {model_name} (type: {model_type})")
    else:
        print(f"\n✅ All files are already in target configuration, no modification needed.")

def update_yaml_config_with_target(file_path: Path, target_config: Dict[str, Any]) -> bool:
    """
    Update single YAML configuration file with specified target configuration
    """
    try:
        # Read YAML file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        modified = False
        target_model_name = target_config.get('model_name')
        target_model_class = target_config.get('class')
        target_model_config = target_config.get('config')
        
        # Check models configuration
        if 'models' in data:
            # Get all current model names
            current_model_names = list(data['models'].keys())
            
            # Clear existing model configuration
            data['models'].clear()
            
            # Add new model configuration
            data['models'][target_model_name] = {
                'class': target_model_class,
                'name': target_model_name,
                'config': target_model_config
            }
            
            if current_model_names:
                print(f"  ✓ Updated model configuration to {target_model_name} ({target_model_class})")
                modified = True
        
        # Check and update llm_model_name references in tasks
        if 'tasks' in data:
            # Handle case where tasks is a dictionary
            if isinstance(data['tasks'], dict):
                for task_name, task_config in data['tasks'].items():
                    if isinstance(task_config, dict) and 'config' in task_config:
                        current_model = task_config['config'].get('llm_model_name')
                        if current_model and current_model != target_model_name:
                            task_config['config']['llm_model_name'] = target_model_name
                            modified = True
                            print(f"  ✓ Updated task {task_name} llm_model_name: {current_model} -> {target_model_name}")
            
            # Handle case where tasks is a list
            elif isinstance(data['tasks'], list):
                for i, task_config in enumerate(data['tasks']):
                    if isinstance(task_config, dict) and 'config' in task_config:
                        current_model = task_config['config'].get('llm_model_name')
                        if current_model and current_model != target_model_name:
                            task_config['config']['llm_model_name'] = target_model_name
                            modified = True
                            print(f"  ✓ Updated task list item {i+1} llm_model_name: {current_model} -> {target_model_name}")
        
        # If there are modifications, write back to file
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, 
                         sort_keys=False, indent=2)
            return True
        else:
            print(f"  - No modification needed")
            return False
            
    except Exception as e:
        print(f"  ❌ Error processing file: {e}")
        return False

if __name__ == '__main__':
    # Check if configuration file is passed via command line arguments
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                external_config = json.load(f)
            main(external_config)
        except Exception as e:
            print(f"❌ Failed to read configuration file: {e}")
            sys.exit(1)
    else:
        main()