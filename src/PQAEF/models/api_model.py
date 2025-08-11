import os
import asyncio
from typing import List, Dict, Any, Union
from tqdm import tqdm

from .base_model import BaseModel
from ..utils.async_utils import batch_post_requests_async, dispatch_openai_requests
from ..constant import constant

# Lazy load openai
try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    OpenAI, AsyncOpenAI = None, None

class ApiModel(BaseModel):
    """
    Handles interactions with external APIs.
    Supports OpenAI-compatible endpoints (sync/async)
    and generic URL endpoints (sync/async).

    This model can accept simple prompt strings (str or List[str]) and will
    automatically format them into the standard OpenAI 'messages' structure.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.provider = self.config['provider']
        self.concurrency = self.config.get('concurrency', 10)
        self.batch_size = self.config.get('batch_size', 32)

        if self.provider == constant.API_PROVIDER_OPENAI:
            if OpenAI is None:
                raise ImportError("The 'openai' package is required. Please install it with 'pip install openai'.")
            
            api_key = self.config.get('api_key') or os.environ.get(self.config.get('api_key_env_var'))
            if not api_key:
                raise ValueError(f"API key for {model_name} not found.")

            base_url = self.config.get('base_url')
            self.sync_client = OpenAI(api_key=api_key, base_url=base_url)
            self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            self.model_identifier = self.config['model_identifier']
            # Get generation parameters from config, e.g. temperature, max_tokens
            self.generation_kwargs = self.config.get('generation_kwargs', {})


        elif self.provider == constant.API_PROVIDER_URL:
            self.url = self.config['base_url']
            default_headers = {
                'Content-Type': 'application/json'
                # 'Accept': 'application/json'
            }
            self.model_identifier = self.config['model_identifier']
            # Get generation parameters from config, e.g. temperature, max_tokens
            self.generation_kwargs = self.config.get('generation_kwargs', {})

            api_key = self.config.get('api_key') or os.environ.get(self.config.get('api_key_env_var'))

            user_defined_headers = self.config.get('headers', {})
            self.headers = {**default_headers, **user_defined_headers}

            if api_key:
                auth_header_name = self.config.get('auth_header_name', 'Authorization')
                auth_scheme = self.config.get('auth_scheme', 'Bearer')
                
                # This step overrides any 'Authorization' manually written by user in config, ensuring program-generated key takes priority
                self.headers[auth_header_name] = f"{auth_scheme} {api_key}".strip()
        
        else:
            raise ValueError(f"Unsupported API provider: {self.provider}")

    def _prepare_openai_requests(self, inputs: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Helper function: Convert various input formats into unified OpenAI API request body list.
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        requests = []
        if all(isinstance(i, str) for i in inputs):
            # If it's a string list, automatically wrap
            for prompt in inputs:
                request_body = {
                    "messages": [{"role": "user", "content": prompt}]
                }
                # Merge configured generation parameters with request body
                if self.generation_kwargs:
                    request_body.update(self.generation_kwargs)
                requests.append(request_body)
        elif all(isinstance(i, dict) for i in inputs):
            # If it's already a dictionary list, use directly (assuming correct format)
            requests = inputs
        else:
            raise TypeError(f"Unsupported input type for ApiModel. Must be str, List[str], or List[Dict]. Got: {type(inputs[0])}")
        
        return requests

    def process(self, inputs: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Synchronously process inputs.
        Accepts strings or list of dictionaries. If string, automatically wraps into 'messages' format.
        """
        if self.provider == constant.API_PROVIDER_OPENAI:
            # Call helper function for format conversion
            requests = self._prepare_openai_requests(inputs)
            
            all_results = []

            print(f"INFO: Starting synchronous processing for {len(requests)} requests...")
            # for request_body in requests:
            for request_body in tqdm(requests, desc=f"Sync Processing for {self.model_name}", unit="request"):
                try:
                    # model_identifier 和 request_body 分开传递
                    response = self.sync_client.chat.completions.create(model=self.model_identifier, **request_body)
                    all_results.append(response.to_dict())
                except Exception as e:
                    print(f"API call failed for model {self.model_name}: {e}")
                    all_results.append({"error": str(e)})
            return all_results
        
        elif self.provider == constant.API_PROVIDER_URL:
            return asyncio.run(self.aprocess(inputs))
            
        return [] # Should not be reached

    async def aprocess(self, inputs: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Asynchronously process inputs, leveraging high concurrency.
        Accepts strings or list of dictionaries. If string, automatically wraps into 'messages' format.
        """
        if self.provider == constant.API_PROVIDER_OPENAI:
            # Call helper function for format conversion
            requests = self._prepare_openai_requests(inputs)
            
            # Prepare final request parameter list for dispatcher
            final_requests_to_dispatch = [
                {"model": self.model_identifier, **req} for req in requests
            ]

            return await dispatch_openai_requests(
                client=self.async_client,
                requests=final_requests_to_dispatch,
                concurrency=self.concurrency
            )
        
        elif self.provider == constant.API_PROVIDER_URL:
            # Assume URL provider accepts dictionary list as payload
            requests = self._prepare_openai_requests(inputs)
            final_requests_to_dispatch = [
                {"model": self.model_identifier, **req} for req in requests
            ]
            all_results = []
            # for i in range(0, len(inputs), self.batch_size):
            #     batch_payloads = inputs[i:i + self.batch_size]
            #     batch_results = await batch_post_requests_async(
            #         self.url, batch_payloads, self.headers, self.concurrency
            #     )
            #     all_results.extend(batch_results)
            all_results = await batch_post_requests_async(
                url=self.url,
                payloads=final_requests_to_dispatch, # <-- Directly use prepared requests
                headers=self.headers,
                concurrency=self.concurrency
            )
            return all_results

        return [] # Should not be reached