# -*- coding: utf-8 -*-
"""
Provides common utility functions for the PQAEF framework.
"""
import hashlib
import asyncio
import aiohttp
from collections import Counter
import jieba

import os
import json
import re
from typing import List, Dict, Any, Tuple

from PQAEF.constant import constant


def calculate_hash(data: Dict[str, Any]) -> str:
    """
    Calculates a SHA256 hash for the dialogue content of a data sample.

    This ensures that the hash is deterministic and based solely on the
    conversational turns, making it ideal for data matching and deduplication.

    Args:
        data (Dict[str, Any]): A data sample in the standard framework format.
                               It must contain the `constant.KEY_DIALOGUES` key.

    Returns:
        str: The calculated hex digest of the hash.
    """
    # Ensure dialogues content is used for hashing for consistency
    dialogues_content = data.get(constant.KEY_DIALOGUES, [])
    
    # Use json.dumps with sort_keys=True to create a canonical string representation
    canonical_string = json.dumps(dialogues_content, sort_keys=True, ensure_ascii=False)
    
    # Encode the string to bytes and calculate the hash
    hash_object = hashlib.sha256(canonical_string.encode('utf-8'))
    return hash_object.hexdigest()



async def post_request_async(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Sends a single asynchronous POST request.

    Args:
        session: An aiohttp.ClientSession object.
        url: The endpoint URL.
        payload: The request payload dictionary.
        headers: The request headers.

    Returns:
        The JSON response from the server as a dictionary.
    
    Raises:
        aiohttp.ClientError: For connection or HTTP status errors.
    """
    async with session.post(url, json=payload, headers=headers) as response:
        response.raise_for_status()  # Raises an exception for 4xx/5xx status codes
        return await response.json()

async def batch_post_requests_async(url: str, payloads: List[Dict[str, Any]], headers: Dict[str, str], concurrency_limit: int = 10) -> List[Dict[str, Any]]:
    """
    Sends a batch of POST requests asynchronously with a concurrency limit.

    Args:
        url: The endpoint URL.
        payloads: A list of request payload dictionaries.
        headers: The request headers.
        concurrency_limit: The maximum number of concurrent requests.

    Returns:
        A list of JSON responses in the same order as the input payloads.
    """
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for payload in payloads:
            # Wait for the semaphore before creating a new task
            await semaphore.acquire()
            task = asyncio.create_task(post_request_async(session, url, payload, headers))
            task.add_done_callback(lambda t: semaphore.release())
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results to handle potential exceptions
        final_results = []
        for res in results:
            if isinstance(res, Exception):
                # Log the error or handle it as needed
                print(f"An async request failed: {res}")
                final_results.append({"error": str(res)}) # Append an error placeholder
            else:
                final_results.append(res)
        
        return final_results
    

def preprocess_text(text: str, remove_stopwords: bool = False, stopwords: set = None) -> List[str]:
    """
    Performs text preprocessing for Chinese: segmentation, cleaning.
    """
    # Remove URLs, @users, punctuation and numbers
    text = re.sub(r'http\S+|@\w+|[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
    # Word segmentation
    words = jieba.lcut(text.strip())
    # Remove empty strings and single characters (optional, can keep for analyzing modal particles)
    words = [word for word in words if word.strip()]
    # Remove stopwords
    if remove_stopwords and stopwords:
        words = [word for word in words if word not in stopwords]
    return words

def calculate_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generates n-grams from a list of tokens."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def calculate_distinct_k(tokens: List[str], k_values: List[int]) -> Dict[str, float]:
    """Calculates Distinct-k metrics for a list of tokens."""
    results = {}
    total_tokens = len(tokens)
    if total_tokens == 0:
        return {f"distinct_{k}": 0.0 for k in k_values}
    
    results["total_tokens"] = total_tokens
    results["unique_tokens"] = len(set(tokens))

    for k in k_values:
        if total_tokens < k:
            results[f"distinct_{k}"] = 0.0
            continue
        ngrams = calculate_ngrams(tokens, k)
        total_ngrams = len(ngrams)
        if total_ngrams == 0:
            results[f"distinct_{k}"] = 0.0
        else:
            unique_ngrams = len(set(ngrams))
            results[f"distinct_{k}"] = unique_ngrams / total_ngrams
            
    return results

def load_stopwords(path: str) -> set:
    """Loads a stopwords file, one word per line."""
    if not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f}
    


def parse_score_from_string(text: str, min_val: int = 0, max_val: int = 9) -> int:
    """
    Safely parses the first integer found in a string and clamps it within a range.
    
    Args:
        text (str): The string to parse (e.g., LLM's raw output).
        min_val (int): The minimum acceptable score.
        max_val (int): The maximum acceptable score.
        
    Returns:
        int: The parsed score, or -1 if no valid score is found.
    """
    # Find all sequences of digits
    numbers = re.findall(r'\d+', text)
    # print(f"\nthis is answer {text} and {numbers}====\n")
    if numbers:
        try:
            # Take the first number found
            score = int(numbers[0])
            # Clamp the score to the valid range
            return max(min_val, min(score, max_val))
        except (ValueError, IndexError):
            return -1 # Should not happen with findall, but for safety
    return -1 # Return -1 to indicate failure to parse


def get_model_response_content(response: Any) -> str:
    """
    Safely extract text content from various possible formats returned by models.
    - LocalModel returns str
    - ApiModel returns dict
    """
    if isinstance(response, str):
        return response
    elif isinstance(response, dict) and not response.get("error"):
        # 标准 OpenAI/兼容 API 的响应格式
        choices = response.get('choices', [])
        if choices and isinstance(choices, list) and len(choices) > 0:
            message = choices[0].get('message', {})
            return message.get('content', '')
    # If error or other format, return empty string
    return ""


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        datas = json.load(f)
    return datas


def read_jsonl(path):
    datas = []
    with open(path, "r", encoding="utf-8") as f:
        for lines in f:
            datas.append(json.loads(lines))
    return datas


def write_json(path, obj, mode="w", encoding="utf-8", indent=4):
    assert path.endswith(".json"), "Please name it as json"
    
    with open(path, mode, encoding=encoding) as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
  
        
def write_jsonl(path: str, obj, mode="w", encoding="utf-8", indent=4):
    assert path.endswith(".jsonl"), "Please name it as jsonl"
    
    with open(path, mode, encoding=encoding) as f:
        for item in obj:
                f.write(json.dumps(item, ensure_ascii=False)+'\n')
                f.flush()