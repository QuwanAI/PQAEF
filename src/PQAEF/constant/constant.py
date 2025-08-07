# -*- coding: utf-8 -*-
"""
Defines constant key names used throughout the DataCrafter framework
for standardized data sample dictionaries.
"""

# ====== dataset ========

# Top-level keys in a standard data sample
KEY_DIALOGUES = "dialogues"
KEY_DIALOGUE_ANNOTATION = "dialogue_annotation"
KEY_OTHER_INFO = "other_info"

# Keys within the 'dialogues' list items
KEY_ROLE = "role"
KEY_CONTENT = "content"
KEY_SENTENCE_ANNOTATION = "sentence_annotation"

# Keys within the 'other_info' dictionary
KEY_HASH = "hash"

# ======== model =========
# --- Model Related Constants ---
MODEL_TYPE_LOCAL = "local"
MODEL_TYPE_API = "api"
MODEL_TYPE_EMBEDDING = "embedding"
MODEL_TYPE_CLASSIFICATION = "classification"

# API Model subtypes
API_PROVIDER_OPENAI = "openai"
API_PROVIDER_URL = "url"


# --- Task Related Constants ---
TASK_TYPE_LEXICAL_DIVERSITY = "lexical_diversity"
TASK_TYPE_SEMANTIC_DIVERSITY = "semantic_diversity"
TASK_TYPE_LLM_DIVERSITY = "llm_diversity"


# --- Quality Task Constants ---
TASK_TYPE_LLM_QUALITY_SCORE = "llm_quality_score"

# --- Quality Multi task Constants ---
QUALITY_DIM_CONTENT = "content_quality"
QUALITY_DIM_FLUENCY = "fluency"
QUALITY_DIM_DEPTH = "depth"
QUALITY_DIM_INTERACTIVITY = "interactivity"
QUALITY_DIM_SENSITIVITY = "sensitivity"
QUALITY_DIM_SAFETY = "safety"

# 一个包含所有合法维度的列表，用于校验
VALID_QUALITY_DIMENSIONS = [
    QUALITY_DIM_CONTENT,
    QUALITY_DIM_FLUENCY,
    QUALITY_DIM_DEPTH,
    QUALITY_DIM_INTERACTIVITY,
    QUALITY_DIM_SENSITIVITY,
    QUALITY_DIM_SAFETY
]

# --- LLM Evaluation Constants ---
TASK_TYPE_LLM_EVALUATION = "llm_evaluation"

# Evaluation Question Types
QUESTION_TYPE_SINGLE_CHOICE = "single_choice"
QUESTION_TYPE_MULTIPLE_CHOICE = "multiple_choice"
QUESTION_TYPE_SUBJECTIVE = "subjective"
QUESTION_TYPE_READING_COMPREHENSION = "reading_comprehension"

# Evaluation Data Keys
KEY_QUESTION_ID = "question_id"
KEY_SCENE = "scene"
KEY_QUESTION_TEXT = "question_text"
KEY_OPTIONS = "options"
KEY_CORRECT_ANSWER = "correct_answer"
KEY_CATEGORY = "category"
KEY_SUBCATEGORY = "subcategory"
KEY_MODEL_ANSWER = "model_answer"
KEY_IS_CORRECT = "is_correct"
KEY_EVALUATION_RESULT = "evaluation_result"
KEY_CONTEXT = 'context'


# Evaluation tool
KEY_EVALUATION_TOOL = "evaluation_tool"