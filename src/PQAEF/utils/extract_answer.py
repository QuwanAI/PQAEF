
import re

def extract_choice_answer(response: str) -> str:
    """
    从模型响应中提取选择题答案（A、B、C、D等）
    
    Args:
        response: 模型的原始响应
        
    Returns:
        str: 提取的选择题答案
    """
    if not response or not response.strip():
        return 'UNKNOWN'
    
    # 原有的选择题提取逻辑
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
    从模型响应中提取阅读理解答案（文本内容）
    
    Args:
        response: 模型的原始响应
        
    Returns:
        str: 提取的阅读理解答案
    """
    if not response or not response.strip():
        return 'UNKNOWN'
    
    response = response.strip()
    
    # 移除常见的答案前缀
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
    
    # 移除引号
    if (cleaned_response.startswith('"') and cleaned_response.endswith('"')) or \
       (cleaned_response.startswith('"') and cleaned_response.endswith('"')) or \
       (cleaned_response.startswith('"') and cleaned_response.endswith('"')):
        cleaned_response = cleaned_response[1:-1].strip()
    
    # 移除句号等标点符号
    cleaned_response = cleaned_response.rstrip('。.,!！？?')
    
    # 如果答案过长，提取第一句话
    if len(cleaned_response) > 200:
        sentences = re.split(r'[。！？.!?]', cleaned_response)
        if sentences and sentences[0].strip():
            cleaned_response = sentences[0].strip()
    
    return cleaned_response if cleaned_response else 'UNKNOWN'

def extract_answer(response: str, task_type: str = 'choice') -> str:
    """
    根据任务类型提取答案的统一入口
    
    Args:
        response: 模型的原始响应
        task_type: 任务类型 ('choice' 或 'reading_comprehension')
        
    Returns:
        str: 提取的答案
    """
    if task_type == 'reading_comprehension':
        return extract_reading_comprehension_answer(response)
    else:
        return extract_choice_answer(response)