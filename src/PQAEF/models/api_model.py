# import os
# import asyncio
# import math
# from typing import List, Dict, Any, Union

# from .base_model import BaseModel
# from PQAEF.utils.utils import batch_post_requests_async
# from PQAEF.constant import constant

# # Lazily import openai
# try:
#     import openai
# except ImportError:
#     openai = None

# class ApiModel(BaseModel):
#     """
#     Handles interactions with external APIs.
#     Supports OpenAI-compatible endpoints (OpenAI, Claude, Gemini, Grok)
#     and generic URL endpoints with batching.
#     """

#     def __init__(self, model_name: str, config: Dict[str, Any]):
#         super().__init__(model_name, config)
#         self.provider = self.config['provider']

#         if self.provider == constant.API_PROVIDER_OPENAI:
#             if openai is None:
#                 raise ImportError("The 'openai' package is required. Please install it with 'pip install openai'.")
            
#             # API key can be from config or fallback to environment variable
#             api_key = self.config.get('api_key') or os.environ.get(self.config.get('api_key_env_var'))
#             if not api_key:
#                 raise ValueError(f"API key for {model_name} not found. Provide it in config or set the '{self.config.get('api_key_env_var')}' environment variable.")

#             # base_url allows targeting any OpenAI-compatible API
#             base_url = self.config.get('base_url')
#             self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
#             self.model_identifier = self.config['model_identifier'] # e.g., "claude-3-opus-20240229"
#             self.batch_size = self.config.get('batch_size', 1) # For simulating batches

#         elif self.provider == constant.API_PROVIDER_URL:
#             self.url = self.config['url']
#             self.headers = self.config.get('headers', {})
#             self.batch_size = self.config.get('batch_size', 32)
#             self.concurrency = self.config.get('concurrency', 10)
        
#         else:
#             raise ValueError(f"Unsupported API provider: {self.provider}")

#     def _process_openai_compatible(self, inputs: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Processes batches for OpenAI-compatible APIs."""
#         if not isinstance(inputs, list):
#             inputs = [inputs]

#         all_results = []
#         for i in range(0, len(inputs), self.batch_size):
#             batch = inputs[i:i + self.batch_size]
#             # Note: OpenAI API itself doesn't have a "batch" endpoint for chat completions.
#             # We process them sequentially in a loop. For true parallelism, one would
#             # need to use asyncio with multiple requests, but this loop is simpler
#             # and respects API rate limits more gracefully.
#             for single_input in batch:
#                 try:
#                     response = self.client.chat.completions.create(model=self.model_identifier, **single_input)
#                     all_results.append(response.to_dict())
#                 except Exception as e:
#                     print(f"API call failed for model {self.model_name}: {e}")
#                     all_results.append({"error": str(e)})
#         return all_results

#     def _process_url(self, inputs: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Processes batches for generic URL APIs asynchronously."""
#         if not isinstance(inputs, list):
#             inputs = [inputs]
        
#         all_results = []
#         # Process in batches to respect server capacity and manage memory
#         for i in range(0, len(inputs), self.batch_size):
#             batch_payloads = inputs[i:i + self.batch_size]
            
#             try:
#                 loop = asyncio.get_running_loop()
#             except RuntimeError:
#                 loop = asyncio.new_event_loop()
#                 asyncio.set_event_loop(loop)

#             batch_results = loop.run_until_complete(
#                 batch_post_requests_async(self.url, batch_payloads, self.headers, self.concurrency)
#             )
#             all_results.extend(batch_results)
        
#         return all_results

#     def process(self, inputs: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """
#         Processes inputs by calling the configured API. Always returns a list of results.
#         """
#         if self.provider == constant.API_PROVIDER_OPENAI:
#             return self._process_openai_compatible(inputs)
        
#         elif self.provider == constant.API_PROVIDER_URL:
#             return self._process_url(inputs)


# PQAEF/PQAEF/models/api_model.py
# PQAEF/PQAEF/models/api_model.py

import os
import asyncio
from typing import List, Dict, Any, Union
from tqdm import tqdm

from .base_model import BaseModel
from ..utils.async_utils import batch_post_requests_async, dispatch_openai_requests
from ..constant import constant

# 懒加载 openai
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
        self.batch_size = self.config.get('batch_size', 32) # For URL provider batching

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
            # (新增) 从配置中获取生成参数，例如 temperature, max_tokens
            self.generation_kwargs = self.config.get('generation_kwargs', {})


        elif self.provider == constant.API_PROVIDER_URL:
            self.url = self.config['base_url']
            default_headers = {
                'Content-Type': 'application/json'
                # 'Accept': 'application/json'  # 加上 Accept 也是一个好习惯
            }
            self.model_identifier = self.config['model_identifier']
            # (新增) 从配置中获取生成参数，例如 temperature, max_tokens
            self.generation_kwargs = self.config.get('generation_kwargs', {})

            api_key = self.config.get('api_key') or os.environ.get(self.config.get('api_key_env_var'))

            user_defined_headers = self.config.get('headers', {})
            self.headers = {**default_headers, **user_defined_headers}

            if api_key:
                auth_header_name = self.config.get('auth_header_name', 'Authorization')
                auth_scheme = self.config.get('auth_scheme', 'Bearer')
                
                # 这一步会覆盖任何用户在配置中手动写的 'Authorization'，确保了程序生成的key优先
                self.headers[auth_header_name] = f"{auth_scheme} {api_key}".strip()
        
        else:
            raise ValueError(f"Unsupported API provider: {self.provider}")

    def _prepare_openai_requests(self, inputs: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        (新增) 辅助函数：将多种输入格式统一转换为 OpenAI API 请求体列表。
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        requests = []
        if all(isinstance(i, str) for i in inputs):
            # 如果是字符串列表，自动包装
            for prompt in inputs:
                request_body = {
                    "messages": [{"role": "user", "content": prompt}]
                }
                # 将配置的生成参数与请求体合并
                if self.generation_kwargs:
                    request_body.update(self.generation_kwargs)
                requests.append(request_body)
        elif all(isinstance(i, dict) for i in inputs):
            # 如果已经是字典列表，直接使用（假设格式正确）
            requests = inputs
        else:
            raise TypeError(f"Unsupported input type for ApiModel. Must be str, List[str], or List[Dict]. Got: {type(inputs[0])}")
        
        return requests

    def process(self, inputs: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        同步处理输入。
        接受字符串或字典列表。如果是字符串，会自动包装成 'messages' 格式。
        """
        if self.provider == constant.API_PROVIDER_OPENAI:
            # (修改) 调用辅助函数进行格式转换
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
        异步处理输入，利用高并发性。
        接受字符串或字典列表。如果是字符串，会自动包装成 'messages' 格式。
        """
        if self.provider == constant.API_PROVIDER_OPENAI:
            # (修改) 调用辅助函数进行格式转换
            requests = self._prepare_openai_requests(inputs)
            
            # (修改) 准备最终的请求参数列表给分发器
            final_requests_to_dispatch = [
                {"model": self.model_identifier, **req} for req in requests
            ]

            return await dispatch_openai_requests(
                client=self.async_client,
                requests=final_requests_to_dispatch,
                concurrency=self.concurrency
            )
        
        elif self.provider == constant.API_PROVIDER_URL:
            # 假设 URL provider 接受字典列表作为 payload
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
                payloads=final_requests_to_dispatch, # <-- 直接使用准备好的请求
                headers=self.headers,
                concurrency=self.concurrency
            )
            return all_results

        return [] # Should not be reached