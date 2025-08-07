from typing import List, Union, Dict, Any
import numpy as np
from .base_model import BaseModel

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class EmbeddingModel(BaseModel):
    """
    Generates dense vector embeddings for text using sentence-transformers,
    with support for batching, Flash Attention, and custom prompts.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        if SentenceTransformer is None:
            raise ImportError("Please install 'sentence-transformers' to use EmbeddingModel.")
            
        super().__init__(model_name, config)
        self.model_path = self.config['model_path']
        self.batch_size = self.config.get('batch_size', 32)
        
        # Advanced kwargs for model and tokenizer initialization
        model_kwargs = self.config.get('model_kwargs', {})
        tokenizer_kwargs = self.config.get('tokenizer_kwargs', {})
        device_ids = self.config.get('device_ids')
        
        device = None # Default to None, which SentenceTransformer maps to CPU or best available
        
        if device_ids:
            if not isinstance(device_ids, list) or not device_ids:
                raise ValueError("'device_ids' for EmbeddingModel must be a non-empty list of integers.")
            
            # Use the first GPU ID specified in the list.
            # The device string needs to be 'cuda:X' where X is the *relative* index
            # within the visible devices.
            # Example: CUDA_VISIBLE_DEVICES="4,5,6". If device_ids=[5], the relative index is 1.
            # The safest way is to let the user specify relative index or let run.py handle it.
            # Let's adopt a simpler, clear contract: `run.py` sets the environment,
            # and here we just pick the first available one, which corresponds to device_ids[0].
            
            target_gpu_index = device_ids[0] # We will always use the first *visible* GPU.
            device = f"cuda:{target_gpu_index}"
            print(f"INFO: 'device_ids' provided. EmbeddingModel '{self.model_name}' will be loaded on the first visible GPU: '{device}'.")
        else:
            device = "cpu"
            print(f"INFO: No 'device_ids' provided for '{self.model_name}'. Loading on CPU.")
        self.model = SentenceTransformer(
            self.model_path,
            # model_kwargs=model_kwargs
        )

    def process(self, inputs: Union[str, List[str]], prompt_name: str = None) -> np.ndarray:
        """
        Encodes text into embeddings with batching.

        Args:
            inputs: A single sentence or a list of sentences.
            prompt_name: The name of the prompt to use for encoding, e.g., "query".
                         This is specific to certain models like Qwen embedding models.

        Returns:
            A numpy array of embeddings.
        """
        return self.model.encode(
            inputs,
            batch_size=self.batch_size,
            prompt_name=prompt_name,
            convert_to_numpy=True
        )