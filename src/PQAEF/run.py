import yaml
import json
import os
import importlib
from typing import Dict, Any
import argparse
# Set jieba cache directory to user directory
os.environ['TMPDIR'] = os.path.expanduser('~/tmp')
os.makedirs(os.path.expanduser('~/tmp'), exist_ok=True)
from PQAEF.utils.timer import _timer

def setup_environment(config: Dict[str, Any]):
    """
    Sets up CUDA_VISIBLE_DEVICES to the union of all GPUs required by all tasks.
    This ensures all necessary GPUs are visible to the main process.
    """
    all_device_ids = set()
    model_configs = config.get('models', {})
    for model_name, model_conf in model_configs.items():
        device_ids = model_conf.get('config', {}).get('device_ids')
        if device_ids:
            all_device_ids.update(device_ids)
    
    if all_device_ids:
        visible_devices = ','.join(map(str, sorted(list(all_device_ids))))
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
        print(f"INFO: Environment prepared. All required GPUs are visible: CUDA_VISIBLE_DEVICES='{visible_devices}'")
    else:
        print("INFO: No specific 'device_ids' found. Using default visible GPUs or CPU.")

def load_config(path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_dataloader(config: Dict[str, Any]):
    from PQAEF.data_ops.dataloader import BaseDataLoader, get_dataloader
    # print(config)
    data_loaders = {}
    for k, v in config.items():
        data_loaders[k] = get_dataloader(v.get("class", None))(v)
    return data_loaders

def _extract_model_name(config: Dict[str, Any]) -> str:
    """
    Extract model name from configuration
    """
    models = config.get('models', {})
    if models:
        first_model_key = list(models.keys())[0]
        first_model = models[first_model_key]
        
        # Check if it's an API model (e.g., openai_evaluator)
        if first_model_key == 'openai_evaluator':
            model_identifier = first_model.get('config', {}).get('model_identifier', '')
            if model_identifier:
                return model_identifier
        
        # For local models, extract from model_path
        model_path = first_model.get('config', {}).get('model_path', '')
        if model_path:
            return os.path.basename(model_path)
        else:
            return first_model_key
    return 'unknown_model'

def _extract_dataset_name(config: Dict[str, Any]) -> str:
    """
    Extract dataset name from configuration
    """
    data_loaders = config.get('data_loaders', {})
    if data_loaders:
        first_loader = list(data_loaders.values())[0]
        formatter_name = first_loader.get('formatter_name', '')
        if formatter_name:
            if formatter_name.endswith('Formatter'):
                return formatter_name[:-9]
            elif formatter_name.endswith('_format'):
                return formatter_name[:-7]
            else:
                return formatter_name
    return 'unknown_dataset'

def _build_complete_output_dir(base_output_dir: str, config: Dict[str, Any]) -> str:
    """
    Auto-complete output_dir path based on configuration: base_output_dir/{model_name}/{dataset_name}
    """
    model_name = _extract_model_name(config)
    dataset_name = _extract_dataset_name(config)
    complete_path = os.path.join(base_output_dir, model_name, dataset_name)
    print(f"INFO: Built complete output path: {complete_path}")
    return complete_path

def main():
    time_map = {}
    
    with _timer("All Tasks", time_map):
        print("--- PQAEF Full Pipeline Run ---")

        # --- 1. Load Config & Set Environment ---
        parser = argparse.ArgumentParser(description="Run the PQAEF pipeline.")
        parser.add_argument(
            '--config', 
            type=str, 
            # required=True,
            default = './test/test_Cosmos.yaml',
            help='Path to the YAML configuration file (e.g., configs/run_config.yml or configs/run_async_quality.yml)'
        )
        args = parser.parse_args()

        print(f"\n[1/5] Loading configuration from '{args.config}'...")
        config = load_config(args.config)
        
        print("\n[2/5] Setting up run environment...")
        setup_environment(config)

        # Import CUDA-dependent libraries after environment setup
        import asyncio
        from PQAEF.data_ops.dataloader import BaseDataLoader, get_dataloader
        from PQAEF.pipelines.synchronous_pipe import SynchronousPipeline
        from PQAEF.pipelines.asynchronous_pipe import AsynchronousPipeline
        from PQAEF.data_ops.datadumper.json_dumper import JsonDataDumper
        from PQAEF.statistics.report_generator import ReportGenerator

        # Ensure formatter module is loaded to register formatters
        importlib.import_module("PQAEF.data_ops.formatters.formatters")
        
        # --- 3. Initialize DataLoader ---
        print("\n[3/5] Initializing data loaders...")
        data_loader_config = config['data_loaders']
        data_loaders = load_dataloader(data_loader_config)
        
        # --- 4. Initialize and Run Pipeline ---
        print("\n[4/5] Initializing and running the pipeline...")
        
        # Read pipeline_type from config, default to 'synchronous' if not specified
        pipeline_type = config.get('pipeline_type', 'synchronous')
        print(f"INFO: Selected pipeline type: '{pipeline_type}'")

        final_results = None
        if pipeline_type == 'asynchronous':
            # For asynchronous mode, instantiate AsynchronousPipeline
            pipeline = AsynchronousPipeline(config)
            raise NotImplementedError("Not support yet")
        elif pipeline_type == 'synchronous':
            # For synchronous mode, instantiate SynchronousPipeline
            pipeline = SynchronousPipeline(config)
            final_results = pipeline.run(data_loaders)
        else:
            raise ValueError(f"Unknown pipeline_type: '{pipeline_type}'. Must be 'synchronous' or 'asynchronous'.")

        # --- 5. Dump Final Results ---
        print("\n[5/5] Dumping processed data...")
        if 'data_dumper' in config:
            # Build complete output path
            original_output_dir = config['data_dumper']['output_dir']
            complete_output_dir = _build_complete_output_dir(original_output_dir, config)
            
            # Create new data_dumper config with complete path
            data_dumper_config = config['data_dumper'].copy()
            data_dumper_config['output_dir'] = complete_output_dir
            
            dumper = JsonDataDumper(data_dumper_config)
            run_metadata = {'tasks_run': config.get('tasks', [])}
            dumper.dump(final_results, run_metadata)
        else:
            print("WARNING: 'data_dumper' configuration not found. Skipping data dump.")
            print("\n--- Final Processed Data (first 2 samples) ---")
            for i, sample in enumerate(final_results[:2]):
                print(f"\n----- Sample #{i+1} -----")
                print(json.dumps(sample, indent=2, ensure_ascii=False))

        print(f"\n--- Pipeline finished. Total samples processed: {len(final_results)} ---")

        # --- 6. Generate Statistics Report ---
        print("\n[6/6] Generating statistical report...")
        
        # Check if a statistics generator should be run at all. 
        # Can be an empty dict {} to run with defaults.
        if 'statistics_generator' in config:
            stats_config = config['statistics_generator']
            
            # Derive it from data_dumper config
            if 'output_dir' not in stats_config:
                if 'data_dumper' in config and 'output_dir' in config['data_dumper']:
                    # Use the already built complete path and add statistical_analysis subdirectory
                    stats_config['output_dir'] = os.path.join(complete_output_dir, 'statistical_analysis')
                    print(f"INFO: Statistics 'output_dir' not set. Defaulting to: {stats_config['output_dir']}")
                else:
                    print("WARNING: Cannot determine default statistics output path because 'data_dumper' config is missing. Skipping statistics.")
                    stats_config = None
            
            if stats_config:
                # Add eval_tool from tasks config to stats_config
                stats_config['eval_tool'] = config.get('tasks', [])[0].get('eval_tool')
                generator = ReportGenerator(stats_config)
                generator.analyze(final_results, run_metadata)
                
        else:
            print("INFO: 'statistics_generator' configuration not found. Skipping report generation.")
    
    print(json.dumps(time_map, indent=4))

if __name__ == "__main__":
    
    main()
    