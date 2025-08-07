# PQAEF/PQAEF/pipelines/asynchronous_pipe.py
import importlib
from typing import Dict, Any, List
import gc
import torch

from ..data_ops.dataloader.base_dataloader import BaseDataLoader
from ..tasks.base_task import BaseTask
from ..models.base_model import BaseModel

class AsynchronousPipeline:
    """
    A pipeline that executes a series of I/O-bound tasks sequentially,
    but processes the data for each task asynchronously and concurrently.
    Ideal for tasks involving API calls.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_configs = self.config.get('tasks', [])
        
    def _initialize_model(self, model_config: Dict[str, Any]) -> BaseModel:
        # This method is identical to the one in SynchronousPipeline
        class_name = model_config['class']
        # Note: Module path logic assumes a consistent naming convention
        module_path = f"PQAEF.models.{class_name.lower().replace('model', '_model')}"
        try:
            module = importlib.import_module(module_path)
            ModelClass = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import or find model class '{class_name}' from '{module_path}': {e}")
        print(f"\nINFO: Initializing model '{model_config['name']}' of class '{class_name}'...")
        model = ModelClass(model_name=model_config['name'], config=model_config['config'])
        return model

    def _initialize_task_and_models(self, task_config: Dict[str, Any]) -> tuple[BaseTask, List[BaseModel]]:
        # This method is identical to the one in SynchronousPipeline
        class_name = task_config['task_class']
        module_path = task_config['module_path']
        try:
            module = importlib.import_module(module_path)
            TaskClass = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import or find task class '{class_name}' from '{module_path}': {e}")
        
        task_specific_config = task_config.get('config', {})
        initialized_models_for_task = []
        task_init_kwargs = {}
        model_dependency_keys = [key for key in task_specific_config if key.endswith('_model_name')]
        
        for key in model_dependency_keys:
            model_name = task_specific_config[key]
            model_config_to_load = self.config.get('models', {}).get(model_name)
            if not model_config_to_load:
                raise ValueError(f"Model '{model_name}' required by task '{class_name}' not found.")
            
            model_instance = self._initialize_model(model_config_to_load)
            initialized_models_for_task.append(model_instance)
            init_key = key.replace('_name', '')
            task_init_kwargs[init_key] = model_instance

        print(f"INFO: Initializing task '{class_name}'...")
        task = TaskClass(task_config=task_specific_config, **task_init_kwargs)
        return task, initialized_models_for_task

    async def run(self, data_loader: BaseDataLoader) -> List[Dict[str, Any]]:
        """Executes the full asynchronous pipeline."""
        print("INFO: Loading all data into memory...")
        all_data = list(data_loader)
        print(f"INFO: Loaded {len(all_data)} samples.")
        
        if not all_data:
            print("WARNING: No data to process.")
            return []

        for i, task_conf in enumerate(self.task_configs):
            print(f"\n{'='*20} EXECUTING ASYNC TASK {i+1}/{len(self.task_configs)}: {task_conf['task_class']} {'='*20}")
            
            task, models_to_clean = self._initialize_task_and_models(task_conf)
            
            # Here we call the asynchronous processing method
            all_data = await task.aprocess_batch(all_data)

            # Clean up resources
            print("INFO: Cleaning up resources for the completed task...")
            del task
            for model in models_to_clean:
                del model
            del models_to_clean
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("INFO: Cleanup complete.")
            
        print(f"\n{'='*20} ASYNC PIPELINE EXECUTION FINISHED {'='*20}")
        return all_data