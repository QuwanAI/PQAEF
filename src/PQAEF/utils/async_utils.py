# PQAEF/PQAEF/utils/async_utils.py
import asyncio
import logging
from typing import List, Dict, Any, Optional

import aiohttp
from tqdm.asyncio import tqdm # Using tqdm's own asyncio-compatible version
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def batch_post_requests_async(
    url: str,
    payloads: List[Dict[str, Any]],
    headers: Dict[str, str],
    concurrency: int = 10
) -> List[Optional[Dict[str, Any]]]:
    """
    Uses aiohttp to concurrently send POST requests with a tqdm progress bar.
    """
    semaphore = asyncio.Semaphore(concurrency)
    
    async def fetch(session: aiohttp.ClientSession, payload: Dict[str, Any], index: int) -> tuple[int, Optional[Dict[str, Any]]]:
        async with semaphore:
            try:
                async with session.post(url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    return index, await response.json()
            except aiohttp.ClientError as e:
                error_context = payload.get('messages', [{}])[0].get('content', 'N/A')[:50]
                logging.error(f"Request to {url} for content '{error_context}...' failed: {e}")
                return index, {"error": str(e)}
            except Exception as e:
                logging.error(f"An unexpected error occurred for payload {payload}: {e}")
                return index, {"error": str(e)}

    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, payload, i) for i, payload in enumerate(payloads)]
        # tqdm.gather creates a progress bar and runs the tasks concurrently
        results_in_completion_order = await tqdm.gather(
            *tasks, desc="Async POST Requests", unit="request"
        )
        
        # Sort results back into their original order to maintain consistency
        results = [None] * len(payloads)
        for index, result_data in results_in_completion_order:
            results[index] = result_data
            
        return results

async def dispatch_openai_requests(
    client: AsyncOpenAI,
    requests: List[Dict[str, Any]],
    concurrency: int = 10
) -> List[Optional[Dict[str, Any]]]:
    """
    Uses openai.AsyncClient to concurrently send Chat Completion requests with a tqdm progress bar.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def get_completion(request_kwargs: Dict[str, Any], index: int) -> tuple[int, Optional[Dict[str, Any]]]:
        async with semaphore:
            try:
                response: ChatCompletion = await client.chat.completions.create(**request_kwargs)
                return index, response.to_dict()
            except Exception as e:
                error_context = request_kwargs.get('messages', [{}])[0].get('content', 'N/A')[:50]
                logging.error(f"OpenAI API call for content '{error_context}...' failed: {e}")
                return index, {"error": str(e)}

    tasks = [get_completion(req, i) for i, req in enumerate(requests)]
    results_in_completion_order = await tqdm.gather(
        *tasks, desc="Async OpenAI Requests", unit="request"
    )
    
    # Sort results back to their original order
    results = [None] * len(requests)
    for index, result_data in results_in_completion_order:
        results[index] = result_data
        
    return results