# PQAEF/PQAEF/pipelines/synchronous_pipe.py

import importlib
from typing import Dict, Any, List
import gc
import torch

from ..data_ops.dataloader.base_dataloader import BaseDataLoader
from ..tasks.base_task import BaseTask
from ..models.base_model import BaseModel

class SynchronousPipeline:
    """
    A pipeline that executes a series of tasks sequentially on a dataset.
    It is designed to be resource-efficient by loading and unloading resources
    (like models) for each task as needed.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_configs = self.config.get('tasks', [])
        
    def _initialize_model(self, model_config: Dict[str, Any]) -> BaseModel:
        """Dynamically initializes a single model based on its configuration."""
        class_name = model_config['class']
        # module_path = f"PQAEF.models.{class_name.lower().replace('model', '_model')}"
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
        """
        Dynamically initializes a task and its required models.
        
        Returns:
            A tuple containing the initialized task instance and a list of its model instances.
        """
        class_name = task_config['task_class']
        module_path = task_config['module_path']
        
        try:
            module = importlib.import_module(module_path)
            TaskClass = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import or find task class '{class_name}' from '{module_path}': {e}")
        
        task_specific_config = task_config.get('config', {})
        
        # --- Model Initialization for the Task ---
        initialized_models_for_task = []
        task_init_kwargs = {}
        
        model_dependency_keys = [key for key in task_specific_config if key.endswith('_model_name')]
        
        for key in model_dependency_keys:
            model_name = task_specific_config[key]
            model_config_to_load = self.config.get('models', {}).get(model_name)
            if not model_config_to_load:
                raise ValueError(f"Model '{model_name}' required by task '{class_name}' not found in the main 'models' configuration.")
            
            # Initialize the model
            model_instance = self._initialize_model(model_config_to_load)
            initialized_models_for_task.append(model_instance)
            
            # Prepare the key for the task's __init__ (e.g., 'embedding_model')
            init_key = key.replace('_name', '')
            task_init_kwargs[init_key] = model_instance

        print(f"INFO: Initializing task '{class_name}'...")
        task = TaskClass(task_config=task_specific_config, **task_init_kwargs)
        
        return task, initialized_models_for_task

    def _initialize_datas(self, data_loaders: Dict[str, Any], task_conf: Dict[str, Any]):
        loader_names = task_conf.get("loader_names", None)
        if loader_names is None:
            raise ValueError(f"Task should define data_loader name `loader_names`, but nou found.")
        if isinstance(loader_names, str):
            loader_names = [loader_names]
        if not isinstance(loader_names, list):
            raise ValueError(f"Invalid loader_names type: {type(loader_names)}, should be `list`")
        
        all_data = []
        for name in loader_names:
            if name not in data_loaders:
                raise ValueError(f"Could not find `{name}` in data_loaders")
            data_loader = data_loaders[name]
            print(f"INFO: Loaded {len(data_loader)} samples from {name}")
            all_data.extend(list(data_loader))
        # list(data_loader)
        return list(data_loader)

    # def run(self, data_loader: BaseDataLoader) -> List[Dict[str, Any]]:
    def run(self, data_loaders: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Executes the full pipeline, managing resources for each task."""
        print("INFO: Loading all data into memory...")
        # all_data = list(data_loader)
        # print(f"INFO: Loaded {len(all_data)} samples.")
        
        # if not all_data:
        #     print("WARNING: No data to process.")
        #     return []

        for i, task_conf in enumerate(self.task_configs):
            print(f"\n{'='*20} EXECUTING TASK {i+1}/{len(self.task_configs)}: {task_conf['task_class']} {'='*20}")
            
            # 1. Initialize the task AND get a direct handle on its models
            all_data = self._initialize_datas(data_loaders, task_conf)
            task, models_to_clean = self._initialize_task_and_models(task_conf)
            
            # 2. Process all data
            all_data = task.process_batch(all_data)

            # 3. Handle any finalization steps
            if hasattr(task, 'finalize') and callable(task.finalize):
                task.finalize()

            # 4. Clean up resources - THIS IS THE CORRECTED LOGIC
            print("INFO: Cleaning up resources for the completed task...")
            
            # Explicitly delete the task and all models it initialized
            del task
            for model in models_to_clean:
                del model
            del models_to_clean
            
            # This is crucial for releasing GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # gc.collect() helps Python's garbage collector reclaim memory sooner
            gc.collect()
            print("INFO: Cleanup complete.")
            
        print(f"\n{'='*20} PIPELINE EXECUTION FINISHED {'='*20}")
        return all_data