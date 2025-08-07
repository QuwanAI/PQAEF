# -*- coding: utf-8 -*-
import os
from typing import List, Dict, Any, Union, Tuple
from collections import defaultdict
import json

from tqdm import tqdm

from .base_model import BaseModel


# Lazily import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:
    AutoModelForCausalLM = AutoTokenizer = torch = None

class LocalModel(BaseModel):
    """
    Handles local inference for generative models from Hugging Face.
    Assumes the CUDA environment (like CUDA_VISIBLE_DEVICES) has been
    set correctly before instantiation.
    """
    def __init__(self, model_name: str, config: Dict[str, Any]):
        if AutoModelForCausalLM is None:
            raise ImportError("Please install 'transformers' and 'torch' to use LocalModel.")
        
        super().__init__(model_name, config)
        self.model_path = self.config['model_path']
        self.batch_size = self.config.get('batch_size', 1)
        
        print(f"INFO: Initializing tokenizer from '{self.model_path}' with left padding.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side='left')
        
        if self.tokenizer.pad_token is None:
            print("INFO: Tokenizer does not have a pad_token. Setting it to eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = self.config.get('model_kwargs', {})
        device_ids = self.config.get('device_ids') # e.g., [0, 1]
    
        # --- New Device Mapping Logic ---
        if device_ids:
            # If multiple specific GPUs are requested, create a device_map
            if len(device_ids) > 1:
                # Transformers' device_map='auto' will use all VISIBLE devices.
                # To restrict it to a SUBSET of visible devices, we must build the map manually.
                # Note: This requires calculating memory distribution, which is complex.
                # A simpler and very effective approach for loading on specific multiple GPUs
                # is to let transformers handle it via device_map='auto' after setting
                # CUDA_VISIBLE_DEVICES in the main script.
                # Therefore, the logic simplifies: we rely on the main script's setup.
                if "device_map" not in model_kwargs:
                    model_kwargs["device_map"] = "auto"
                print(f"INFO: Model '{self.model_name}' will be loaded across visible GPUs (set to {os.environ.get('CUDA_VISIBLE_DEVICES')}) using device_map='auto'.")
            
            # If a single GPU is requested, we can be more direct.
            elif len(device_ids) == 1:
                # The device will be mapped relative to CUDA_VISIBLE_DEVICES.
                # If CUDA_VISIBLE_DEVICES="4,5,6", asking for device_id 4 corresponds to cuda:0.
                # This logic is tricky. The simplest approach is to trust device_map='auto'
                # when CUDA_VISIBLE_DEVICES is set to a single device.
                # Forcing a specific device can sometimes conflict with `auto`.
                # Let's stick to the reliable pattern.
                if "device_map" not in model_kwargs:
                    model_kwargs["device_map"] = "auto"
                print(f"INFO: Model '{self.model_name}' will be loaded onto the single visible GPU (set to {os.environ.get('CUDA_VISIBLE_DEVICES')}) using device_map='auto'.")

        elif "device_map" not in model_kwargs:
             model_kwargs["device_map"] = 'cpu'
             print(f"INFO: No 'device_ids' provided for '{self.model_name}'. Loading on CPU.")
        # if "device_map" not in model_kwargs:
        #     model_kwargs["device_map"] = "auto" if torch.cuda.is_available() else 'cpu'

        # This logic remains, but it will now operate on the *pre-filtered* list of GPUs
        if "attn_implementation" not in model_kwargs and torch.cuda.is_available():
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                 model_kwargs["attn_implementation"] = "flash_attention_2"


        print(f"INFO: Loading model '{self.model_path}' with kwargs: {model_kwargs}")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        self.model.eval()
    
    
    def _prepare_and_group_inputs(
        self, inputs: Union[str, List[str], List[Dict[str, Any]], Dict[str, Any]]
    ) -> Dict[str, List[Tuple[int, str]]]:
        """
        解析多种格式的输入，将其规范化，并根据生成参数对样本进行分组。

        Returns:
            一个字典，其中：
            - key 是一个代表该组生成参数的 JSON 字符串。
            - value 是一个元组列表，每个元组为 (原始索引, 提示词)。
        """
        # 1. 拷贝模型级别的默认生成参数作为基础
        base_kwargs = self.config.get('generation_kwargs', {}).copy()

        # 2. 创建一个字典来存储分组后的样本
        #    key: 序列化后的 kwargs (str) -> value: list of (original_index, prompt)
        grouped_prompts = defaultdict(list)
        
        # 3. 规范化输入
        samples: List[Dict[str, Any]] = []
        if isinstance(inputs, str):
            # 单个字符串
            samples = [{'prompt': inputs}]
        elif isinstance(inputs, list) and all(isinstance(i, str) for i in inputs):
            # 字符串列表
            samples = [{'prompt': i} for i in inputs]
        elif isinstance(inputs, list) and all(isinstance(i, dict) for i in inputs):
            # 字典列表 (最复杂的情况)
            for item in inputs:
                if 'prompt' not in item:
                    raise ValueError("当输入为字典列表时，每个字典必须包含 'prompt' 键。")
            samples = inputs
        elif isinstance(inputs, dict):
            # 单个字典，包含 prompts 列表和共享的 kwargs
            if 'prompts' not in inputs or not isinstance(inputs['prompts'], list):
                raise ValueError("当输入为字典时，必须包含一个名为 'prompts' 的列表。")
            
            shared_kwargs = {k: v for k, v in inputs.items() if k != 'prompts'}
            samples = [{'prompt': p, **shared_kwargs} for p in inputs['prompts']]
        else:
            raise TypeError(f"不支持的输入类型: {type(inputs)}")

        # 4. 遍历规范化后的样本，进行分组
        for original_index, sample in enumerate(samples):
            # 创建当前样本的最终参数
            current_kwargs = base_kwargs.copy()
            override_kwargs = {k: v for k, v in sample.items() if k != 'prompt'}
            current_kwargs.update(override_kwargs)
            
            # 为了能作为字典的 key，必须将参数字典序列化为不可变的字符串
            # sorted() 确保了 {'a':1, 'b':2} 和 {'b':2, 'a':1} 会得到相同的 key
            kwargs_key = json.dumps(current_kwargs, sort_keys=True)
            
            grouped_prompts[kwargs_key].append((original_index, sample['prompt']))

        return grouped_prompts

    def _run_inference_batch(self, prompts: List[str], generation_kwargs: Dict[str, Any]) -> List[str]:
        """
        对一个具有相同参数的批次执行实际的推理。
        """
        all_generated_texts = []
        with torch.no_grad():
            for i in range(0, len(prompts), self.batch_size):
                batch_prompts = prompts[i:i + self.batch_size]
                messages_batch = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
                
                tokenized_chats = self.tokenizer.apply_chat_template(
                    messages_batch, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", padding=True
                )
                
                input_ids = tokenized_chats.to(self.model.device)
                
                output_ids = self.model.generate(input_ids, **generation_kwargs)
                
                prompt_lengths = [len(x) for x in input_ids]
                
                batch_texts = []
                for j in range(len(output_ids)):
                    response_ids = output_ids[j][prompt_lengths[j]:]
                    decoded_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    batch_texts.append(decoded_text)

                all_generated_texts.extend(batch_texts)
        return all_generated_texts

    def process(self, inputs: Union[str, List[str], List[Dict[str, Any]], Dict[str, Any]]) -> List[str]:
        """
        处理多种格式的输入，支持样本级别的生成参数。
        """
        # 1. 对输入进行解析和分组
        grouped_prompts = self._prepare_and_group_inputs(inputs)
        
        # 2. 准备一个列表，用于按原始顺序存放结果
        total_samples = sum(len(v) for v in grouped_prompts.values())
        results = [None] * total_samples

        # 3. 遍历每个分组，分批次进行推理
        for kwargs_key, group in tqdm(grouped_prompts.items(), desc="Processing groups"):
            # 反序列化 kwargs
            generation_kwargs = json.loads(kwargs_key)
            
            # 提取原始索引和提示词
            indices, prompts_in_group = zip(*group)

            print(f"--- Running group with {len(prompts_in_group)} samples. Kwargs: {generation_kwargs} ---") # 调试信息
            
            # 对当前分组进行推理
            generated_texts = self._run_inference_batch(list(prompts_in_group), generation_kwargs)

            # 4. 将生成的结果放回其在原始输入中的位置
            for original_index, text in zip(indices, generated_texts):
                results[original_index] = text
        
        return results
    
    # 在 LocalModel 类的末尾添加此方法
    def score_options(self, context: str, options: List[str]) -> List[float]:
        """
        为给定的上下文和多个选项计算分数，分数越高越好。
        """
        scores = []
        with torch.no_grad():
            messages_context = [{"role": "user", "content": context}]
            context_template = self.tokenizer.apply_chat_template(
                messages_context, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            for option in options:
                full_text = context_template + option
                inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
                input_ids = inputs.input_ids
                
                outputs = self.model(input_ids=input_ids, labels=input_ids)
                logits = outputs.logits
                
                context_inputs = self.tokenizer(context_template, return_tensors="pt")
                context_length = context_inputs.input_ids.shape[1]
                
                response_logits = logits[:, context_length-1:-1, :]
                response_labels = input_ids[:, context_length:]
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(response_logits.view(-1, self.model.config.vocab_size), response_labels.view(-1))
                
                avg_log_likelihood = -loss.mean().item()
                scores.append(avg_log_likelihood)

        return scores
    
    def get_log_probs(self, question: str, answers: List[str]) -> List[float]:
        """
        Calculates the total log probability for each answer given a question.
        This is the core function for multiple-choice evaluation.

        Args:
            question (str): The question or context part of the prompt.
            answers (List[str]): A list of possible answer strings to evaluate.

        Returns:
            List[float]: A list of total log probability scores, one for each answer.
        """
        all_scores = []
        
        # Determine the device of the model. This handles multi-GPU setups.
        # `self.model.device` might not work with device_map='auto'.
        # A more robust way is to check the device of a parameter.
        device = next(self.model.parameters()).device

        with torch.no_grad():
            # Process answers in batches for efficiency
            for i in tqdm(range(0, len(answers), self.batch_size), desc="Scoring answers..."):
                batch_answers = answers[i:i + self.batch_size]
                
                # --- Step 1: Construct full prompts (Question + Answer) ---
                # We use the chat template to format the input correctly.
                # The "question" is the user's turn, and the "answer" is the start of the assistant's turn.
                messages_batch = [
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": ans}
                    ] for ans in batch_answers
                ]

                # --- Step 2: Tokenize the prompts and the question part separately ---
                # Tokenize the full Q+A prompt. We don't add the generation prompt here.
                full_tokenized = self.tokenizer.apply_chat_template(
                    messages_batch,
                    tokenize=True,
                    add_generation_prompt=False, # We are scoring, not generating
                    return_tensors="pt",
                    padding=True
                ).to(device)

                # Tokenize just the question part to know where the answer begins
                question_messages = [[{"role": "user", "content": question}]]
                question_tokenized = self.tokenizer.apply_chat_template(
                    question_messages,
                    tokenize=True,
                    add_generation_prompt=True, # This gets us the exact prompt fed to the model
                    return_tensors="pt"
                )
                question_len = question_tokenized.shape[1]

                # --- Step 3: Get model's logits ---
                # The model predicts the *next* token for each position in the input.
                # So, the logits for `full_tokenized.input_ids` will have shape
                # [batch_size, sequence_length, vocab_size]
                outputs = self.model(input_ids=full_tokenized)
                logits = outputs.logits

                # --- Step 4: Calculate log probabilities for the answer tokens ---
                # We need the log probability of each *actual* token in the answer.
                # The logits for the token at `input_ids[:, i]` are in `logits[:, i-1]`.
                # So we shift the logits to align with the tokens we are predicting.
                shifted_logits = logits[:, :-1, :]
                shifted_tokens = full_tokenized[:, 1:]

                # Convert logits to log probabilities
                log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
                
                # Gather the log probabilities of the actual tokens that appeared in the input
                # `token_log_probs` will have the same shape as `shifted_tokens`
                token_log_probs = torch.gather(log_probs, 2, shifted_tokens.unsqueeze(-1)).squeeze(-1)
                
                # --- Step 5: Sum log probabilities for only the answer part ---
                for j in range(len(batch_answers)):
                    # We need to find the start and end of the answer for this specific example
                    # in the padded batch.
                    # The question part ends at `question_len - 1` (since we are using shifted tokens).
                    answer_start_index = question_len - 1
                    
                    # The answer ends where the padding begins.
                    # We look for the pad_token_id in the *original* (unshifted) tokens.
                    # The length of the non-padded sequence is needed.
                    sequence_len = torch.sum(full_tokenized[j] != self.tokenizer.pad_token_id).item()
                    answer_end_index = sequence_len - 1 # Use -1 because we shifted tokens

                    # Slice the log probabilities for the answer tokens
                    answer_log_probs = token_log_probs[j, answer_start_index:answer_end_index]
                    
                    # Sum them up to get the total score for this answer
                    total_score = answer_log_probs.sum().item()
                    all_scores.append(total_score)

        return all_scores

    # def process(self, inputs: Union[str, List[str]]) -> List[str]:
    #     # The process method remains exactly the same as the last version.
    #     if isinstance(inputs, str):
    #         inputs = [inputs]

    #     all_generated_texts = []
    #     generation_kwargs = self.config.get('generation_kwargs', {})
        
    #     with torch.no_grad():
    #         # for i in range(0, len(inputs), self.batch_size):
    #         for i in tqdm(range(0, len(inputs), self.batch_size), desc="Inferring..."):
    #             batch_prompts = inputs[i:i + self.batch_size]
    #             messages_batch = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
                
    #             # print(batch_prompts)
    #             tokenized_chats = self.tokenizer.apply_chat_template(
    #                 messages_batch,
    #                 tokenize=True,
    #                 add_generation_prompt=True,
    #                 return_tensors="pt",
    #                 padding=True
    #             )
                
    #             input_ids = tokenized_chats.to(self.model.device)
                
    #             output_ids = self.model.generate(input_ids, **generation_kwargs)
                
    #             prompt_lengths = [len(x) for x in input_ids]
                
    #             batch_texts = []
    #             for j in range(len(output_ids)):
    #                 response_ids = output_ids[j][prompt_lengths[j]:]
    #                 decoded_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
    #                 batch_texts.append(decoded_text)

    #             all_generated_texts.extend(batch_texts)
                
    #     return all_generated_texts