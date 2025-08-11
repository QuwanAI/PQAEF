
import re

def extract_choice_answer(response: str) -> str:
    """
    Extract multiple choice answer (A, B, C, D, etc.) from model response
    
    Args:
        response: Raw response from model
        
    Returns:
        str: Extracted multiple choice answer
    """
    if not response or not response.strip():
        return 'UNKNOWN'
    
    # Original multiple choice extraction logic
    patterns = [
        r'答案[是为]?[:：]?\s*([A-Z])',
        r'选择[:：]?\s*([A-Z])',
        r'([A-Z])选?项',
        r'^\s*([A-Z])\s*$',
        r'选项([A-Z])',
        r'会选择选项([A-Z])',
        r'\b([A-Z])\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    
    return 'UNKNOWN'

def extract_reading_comprehension_answer(response: str) -> str:
    """
    Extract reading comprehension answer (text content) from model response
    
    Args:
        response: Raw response from model
        
    Returns:
        str: Extracted reading comprehension answer
    """
    if not response or not response.strip():
        return 'UNKNOWN'
    
    response = response.strip()
    
    # Remove common answer prefixes
    prefixes_to_remove = [
        '答案是：', '答案：', '答案是', '答案为：', '答案为',
        '回答：', '回答是：', '回答是', 
        '根据文本，', '根据上下文，', '根据材料，',
        '根据文章，', '根据段落，'
    ]
    
    cleaned_response = response
    for prefix in prefixes_to_remove:
        if cleaned_response.startswith(prefix):
            cleaned_response = cleaned_response[len(prefix):].strip()
            break
    
    # Remove quotes
    if (cleaned_response.startswith('"') and cleaned_response.endswith('"')) or \
       (cleaned_response.startswith('"') and cleaned_response.endswith('"')) or \
       (cleaned_response.startswith('"') and cleaned_response.endswith('"')):
        cleaned_response = cleaned_response[1:-1].strip()
    
    # Remove punctuation marks
    cleaned_response = cleaned_response.rstrip('。.,!！？?')
    
    # If answer is too long, extract first sentence
    if len(cleaned_response) > 200:
        sentences = re.split(r'[。！？.!?]', cleaned_response)
        if sentences and sentences[0].strip():
            cleaned_response = sentences[0].strip()
    
    return cleaned_response if cleaned_response else 'UNKNOWN'

def extract_answer(response: str, task_type: str = 'choice') -> str:
    """
    Unified entry point for extracting answers based on task type
    
    Args:
        response: Raw response from model
        task_type: Task type ('choice' or 'reading_comprehension')
        
    Returns:
        str: Extracted answer
    """
    if task_type == 'reading_comprehension':
        return extract_reading_comprehension_answer(response)
    else:
        return extract_choice_answer(response)