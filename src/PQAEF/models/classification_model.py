# PQAEF/PQAEF/models/classification_model.py

from typing import List, Dict, Any, Union
import torch
from .base_model import BaseModel

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    AutoModelForSequenceClassification = AutoTokenizer = torch = None

class ClassificationModel(BaseModel):
    """
    Handles local inference for text classification, with batching and multi-GPU.
    """
    def __init__(self, model_name: str, config: Dict[str, Any]):
        if AutoModelForSequenceClassification is None:
            raise ImportError("Please install 'transformers' and 'torch' to use ClassificationModel.")
        
        super().__init__(model_name, config)
        self.model_path = self.config['model_path']
        self.batch_size = self.config.get('batch_size', 1)
        
        model_kwargs = self.config.get('model_kwargs', {})
        device_ids = self.config.get('device_ids')

        if device_ids:
            # If specific GPUs are requested, rely on device_map='auto'
            # which will respect the CUDA_VISIBLE_DEVICES set by the runner.
            if "device_map" not in model_kwargs:
                model_kwargs["device_map"] = "auto"
            print(f"INFO: 'device_ids' provided. Using device_map='auto' for model '{self.model_name}'.")
        elif "device_map" not in model_kwargs:
            # If no device_ids, default to CPU to avoid accidentally filling a GPU.
            model_kwargs["device_map"] = "cpu"
            print(f"INFO: No 'device_ids' provided for '{self.model_name}'. Loading on CPU.")
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, **model_kwargs)
        self.model.eval()

    def process(self, inputs: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Classifies input texts in batches."""
        if isinstance(inputs, str):
            inputs = [inputs]

        all_results = []
        with torch.no_grad():
            for i in range(0, len(inputs), self.batch_size):
                batch_texts = inputs[i:i + self.batch_size]
                encoded_inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
                
                outputs = self.model(**encoded_inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class_ids = torch.argmax(probabilities, dim=-1)

                for j in range(len(batch_texts)):
                    label_id = predicted_class_ids[j].item()
                    all_results.append({
                        "label": self.model.config.id2label[label_id],
                        "score": probabilities[j][label_id].item(),
                        "logits": logits[j].cpu().numpy().tolist()
                    })
        return all_results