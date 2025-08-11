import os
from typing import List, Dict, Any, TYPE_CHECKING
from sklearn.metrics import f1_score
from typing_extensions import override
from tqdm import tqdm
import re
from PQAEF.constant import constant

from ..base_task import BaseTask

from ...utils.metrics import calculate_rouge
from ...utils.utils import get_model_response_content


if TYPE_CHECKING:
    from ...models.base_model import BaseModel


def extract_answer(response: str) -> str:
    """
    Extract answer from model response
    
    Args:
        response: Raw response from model
        
    Returns:
        str: Extracted answer (A, B, C, etc.)
    """

    # Match A-Z a-z
    patterns = [
        r'答案[是为]?[:：]?\s*([A-Za-z])',
        r'选择[:：]?\s*([A-Za-z])',
        r'([A-Za-z])选?项',
        r'^\s*([A-Za-z])\s*$',
        r'选项([A-Za-z])',  
        r'会选择选项([A-Za-z])',  
        r'\b([A-Za-z])\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    
    return 'UNKNOWN'

class SingleChoiceTask(BaseTask):
    def __init__(self, task_config: Dict[str, Any], llm_model: "BaseModel"):
        super().__init__(task_config)
        self.llm_model = llm_model
        
        prompt_path = self.task_config["prompt_path"]
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()


    def _response(self, responses: List[Any], batch: List[Dict[str, Any]]):
        
        results = []
        
        for response, sample in zip(responses, batch):
            response_content = get_model_response_content(response)
            predicted_answer = extract_answer(response_content)
            
            result = {
                'question_id': sample.get('question_id', ''),
                'scene': sample.get('scene', ''),
                'question_text': sample.get('question_text', ''),
                'options': sample.get('options', ''),
                'correct_answer': sample.get('correct_answer', ''),
                'predicted_answer': predicted_answer,
                'raw_answer': response_content,
                'is_correct': predicted_answer == sample.get('correct_answer', ''),
                'category': sample.get('category', '')
            }
            
            results.append(result)
        return results
    
    @override
    def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_prompts = []
    
        # Extract all placeholders from prompt template
        placeholders = re.findall(r'\{([^}]+)\}', self.prompt_template)
        format_kwargs = {}
    
        for sample in batch:
            for placeholder in placeholders:
                if placeholder in sample:
                    # Get value directly from sample
                    format_kwargs[placeholder] = sample[placeholder]
                elif placeholder.startswith('option_'):
                    # Handle option placeholders (e.g. option_A, option_B)
                    option_key = placeholder.split('_')[1]  # Extract A, B, C, etc.
                    options = sample.get('options', {})
                    if isinstance(options, dict) and option_key in options:
                        format_kwargs[placeholder] = options[option_key]
                    else:
                        format_kwargs[placeholder] = ''
                else:
                    # Handle other possible mappings
                    value = ''
                    if placeholder == 'question':
                        value = sample.get('question_text', sample.get('question', ''))
                    elif placeholder == 'scene':
                        value = sample.get('scene', '')
                    elif placeholder == 'question_text':
                        value = sample.get('question_text', sample.get('question', ''))
                    elif placeholder == 'options':
                        # If template needs options in string format
                        options = sample.get('options', {})
                        if isinstance(options, dict):
                            options_list = []
                            for key in sorted(options.keys()):
                                options_list.append(f"{key}. {options[key]}")
                            value = '\n'.join(options_list)
                        else:
                            value = str(options)
                    
                    format_kwargs[placeholder] = value
            
            prompt = self.prompt_template.format(**format_kwargs)
            all_prompts.append(prompt)
        print('one sample prompt:\n', prompt)
        responses = self.llm_model.process(all_prompts)
        results = self._response(responses, batch)
        
        # Only return _response results, no longer calculate metrics here
        return results
        