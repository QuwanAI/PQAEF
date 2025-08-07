# -*- coding: utf-8 -*-
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
    # 生成对应的字母列表，超过Z时从a开始循环
    letters = []
    for i in range(len(options)):
        # 计算偏移量，26个大写字母后切换到小写字母循环
        offset = i % 52  # 52 = 26(大写) + 26(小写)
        if offset < 26:
            # 大写字母 A-Z
            letters.append(chr(ord('A') + offset))
        else:
            # 小写字母 a-z
            letters.append(chr(ord('a') + (offset - 26)))
    # 创建字母到选项的映射字典
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
        # 处理CSV行数据
        if not isinstance(raw_sample, list) or len(raw_sample) < 3:
            return None
        
        try:
            label = int(raw_sample[0].strip())
            title = raw_sample[1].strip().strip('"')
            description = raw_sample[2].strip().strip('"')
        except (ValueError, IndexError):
            return None
        
        # 验证标签范围
        if label not in [1, 2, 3, 4] or not title or not description:
            return None
        
        # 构建场景描述
        scene = f"Title: {title}\nDescription: {description}"
        
        # 生成哈希值
        hash_value = hashlib.sha256(scene.encode('utf-8')).hexdigest()
        
        # 构建选项
        options = {
            "A": "World - International news, politics, and global events",
            "B": "Sports - Athletic events, games, and sports-related news", 
            "C": "Business - Economic news, corporate affairs, and financial markets",
            "D": "Sci/Tech - Science, technology, and innovation news"
        }
        
        # 映射标签到答案
        label_to_answer = {1: "A", 2: "B", 3: "C", 4: "D"}
        correct_answer = label_to_answer.get(label, "")
        
        # 添加其他信息
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