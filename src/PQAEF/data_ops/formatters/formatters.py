"""
Contains specific formatters for different raw dataset structures.
To make a formatter available, simply import this file in your main script.
"""
import sys
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import hashlib

from typing_extensions import override

from PQAEF.constant import constant
from PQAEF.utils.template_registry import register_formatter
from PQAEF.utils.utils import calculate_hash
from PQAEF.utils.template_registry import BaseFormatter

def map_options_to_letters(options):
    # Generate corresponding letter list, cycling from 'a' when exceeding 'Z'
    letters = []
    for i in range(len(options)):
        # Calculate offset, switch to lowercase letters after 26 uppercase letters
        offset = i % 52  # 52 = 26(uppercase) + 26(lowercase)
        if offset < 26:
            # Uppercase letters A-Z
            letters.append(chr(ord('A') + offset))
        else:
            # Lowercase letters a-z
            letters.append(chr(ord('a') + (offset - 26)))
    # Create letter-to-option mapping dictionary
    mapped_options = {letter: options[i] for i, letter in enumerate(letters)}
    return mapped_options

@register_formatter("empty_format")
class EmptyFormatter(BaseFormatter):
    @override
    def format(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        hash_value = calculate_hash(raw_sample)
        raw_sample[constant.KEY_OTHER_INFO] = {}
        raw_sample[constant.KEY_OTHER_INFO][constant.KEY_HASH] = hash_value

        return raw_sample

@register_formatter("AGNewsFormatter")
class AGNewsFormatter(BaseFormatter):
    @override
    def format(self, raw_sample) -> Dict[str, Any]:
        # Process CSV row data
        if not isinstance(raw_sample, list) or len(raw_sample) < 3:
            return None
        
        try:
            label = int(raw_sample[0].strip())
            title = raw_sample[1].strip().strip('"')
            description = raw_sample[2].strip().strip('"')
        except (ValueError, IndexError):
            return None
        
        # Validate label range
        if label not in [1, 2, 3, 4] or not title or not description:
            return None
        
        # Build scenario description
        scene = f"Title: {title}\nDescription: {description}"
        
        # Generate hash value
        hash_value = hashlib.sha256(scene.encode('utf-8')).hexdigest()
        
        # Build options
        options = {
            "A": "World - International news, politics, and global events",
            "B": "Sports - Athletic events, games, and sports-related news", 
            "C": "Business - Economic news, corporate affairs, and financial markets",
            "D": "Sci/Tech - Science, technology, and innovation news"
        }
        
        # Map label to answer
        label_to_answer = {1: "A", 2: "B", 3: "C", 4: "D"}
        correct_answer = label_to_answer.get(label, "")
        
        # Add other information
        other_info = {
            "dataset_name": "agnews",
            "title": title,
            "description": description,
            "original_label": label
        }
        
        formatted_sample = {
            constant.KEY_QUESTION_ID: hash_value,
            constant.KEY_SCENE: scene,
            constant.KEY_QUESTION_TEXT: "Please classify this news article into one of the following categories:",
            constant.KEY_OPTIONS: options,
            constant.KEY_CORRECT_ANSWER: correct_answer,
            constant.KEY_CATEGORY: "AGNews",
            constant.KEY_SUBCATEGORY: "News Topic Classification",
            constant.KEY_OTHER_INFO: other_info
        }
        
        return formatted_sample