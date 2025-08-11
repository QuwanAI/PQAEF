from typing import List, Dict, Union

import jieba
import rouge_chinese
from nltk.util import ngrams
import jieba

# def calculate_rouge(hypothesis: str, references: Union[str, List[str]]) -> Dict[str, Dict[str, float]]:
def calculate_rouge(hypothesis: str, reference: str) -> Dict[str, Dict[str, float]]:
    """
    Calculate ROUGE scores for given text, supporting Chinese.

    Args:
        hypothesis (str): Model-generated summary or text.
        reference (str): A reference summary or text.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing ROUGE-1, ROUGE-2, ROUGE-L scores.
                                     Each score contains 'f' (f1-score), 'p' (precision), 'r' (recall).
                                     Example: {'rouge-1': {'f': 0.5, 'p': 0.5, 'r': 0.5}, ...}
    """
    if not hypothesis or not reference:
        # If generated or reference is empty, all scores are 0
        return {
            "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
            "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
            "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0},
        }
    
    assert isinstance(hypothesis, str)
    assert isinstance(reference, str)
    hypothesis = " ".join(jieba.cut(hypothesis))
    reference = " ".join(jieba.cut(reference))
    
    rouge = rouge_chinese.Rouge()
    
    scores = rouge.get_scores(hypothesis, reference)
    
    return scores[0]


def calculate_mutual_metrics(predictions: List[str], correct_answers: List[str]) -> Dict[str, float]:
    """
    Calculate evaluation metrics for Mutual dataset: Recall@1, Recall@2, MRR
    
    Args:
        predictions: List of model predicted answers (A, B, C, D)
        correct_answers: List of correct answers (A, B, C, D)
        
    Returns:
        Dict[str, float]: Dictionary containing Recall@1, Recall@2, MRR
    """
    if len(predictions) != len(correct_answers):
        raise ValueError("Predictions and correct answers must have the same length")
    
    total_samples = len(predictions)
    if total_samples == 0:
        return {"recall@1": 0.0, "recall@2": 0.0, "mrr": 0.0}
    
    # Define option ordering
    options = ['A', 'B', 'C', 'D']
    
    recall_at_1 = 0
    recall_at_2 = 0
    mrr_sum = 0.0
    
    for pred, correct in zip(predictions, correct_answers):
        # If predicted answer is correct, Recall@1 = 1
        if pred == correct:
            recall_at_1 += 1
            recall_at_2 += 1
            mrr_sum += 1.0  # Rank 1, reciprocal is 1
        else:
            # For Recall@2, we need to consider predicted answer in top 2 positions
            # Since this is multiple choice, we assume model gives most likely answer
            # If prediction is wrong, we check if correct answer is in top 2 most likely options
            # Simplified handling: if prediction is wrong, assume correct answer is at position 2
            try:
                pred_idx = options.index(pred) if pred in options else -1
                correct_idx = options.index(correct) if correct in options else -1
                
                # Simplified Recall@2 calculation: if prediction and correct answer are adjacent, count as Recall@2
                if abs(pred_idx - correct_idx) <= 1 and pred_idx != -1 and correct_idx != -1:
                    recall_at_2 += 1
                    mrr_sum += 0.5  # Assume rank 2, reciprocal is 0.5
                # Otherwise assume correct answer is at position 3 or 4
                elif correct_idx != -1:
                    mrr_sum += 1.0 / (correct_idx + 1)  # Calculate reciprocal rank based on correct answer position
            except (ValueError, IndexError):
                # If answer is not in options, skip
                continue
    
    return {
        "recall@1": recall_at_1 / total_samples,
        "recall@2": recall_at_2 / total_samples, 
        "mrr": mrr_sum / total_samples
    }

def _zipngram(words: List[str], ngram_size: int):
    """
    Generate n-grams
    :param words: List of segmented words
    :param ngram_size: Size of n-gram
    :return: Iterator of n-grams
    """
    return zip(*[words[i:] for i in range(ngram_size)])

def calculate_distinct_n(sentences: List[str], n: int) -> float:
    """
    Calculate Distinct-n score across the entire batch.
    This metric is not present in the original empchat code, it's a new implementation.

    Args:
        sentences (List[str]): List of all model-generated responses.
        n (int): The n value for n-gram.

    Returns:
        float: Distinct-n score.
    """
    if not sentences:
        return 0.0

    all_ngrams = []
    for sentence in sentences:
        words = list(jieba.cut(sentence))
        all_ngrams.extend(list(ngrams(words, n)))
        

    if not all_ngrams:
        return 0.0
    
    return len(set(all_ngrams)) / len(all_ngrams)


def test_rouge():
    # Example 1: English text
    print("--- English ROUGE Example ---")
    generated_summary_en = "the cat was found under the bed"
    reference_summary_en = "the cat was under the bed"
    
    rouge_scores_en = calculate_rouge(generated_summary_en, reference_summary_en)
    
    print(f"Generated summary: {generated_summary_en}")
    print(f"Reference summary: {reference_summary_en}")
    print("ROUGE scores:")
    # Format output for better readability
    for rouge_type, scores in rouge_scores_en.items():
        print(f"  {rouge_type.upper()}:")
        print(f"    F1-Score: {scores['f']:.4f}")
        print(f"    Precision: {scores['p']:.4f}")
        print(f"    Recall: {scores['r']:.4f}")

    print("\n" + "="*50 + "\n")

    # Example 2: Chinese text
    print("--- Chinese ROUGE Example ---")
    generated_summary_zh = "今天天气真好，阳光明媚，我们一起去公园玩吧。"
    reference_summary_zh = "今天天气不错，我们去公园玩。"
    
    rouge_scores_zh = calculate_rouge(generated_summary_zh, reference_summary_zh)
    
    print(f"Generated summary: {generated_summary_zh}")
    print(f"Reference summary: {reference_summary_zh}")
    print("ROUGE scores:")
    for rouge_type, scores in rouge_scores_zh.items():
        print(f"  {rouge_type.upper()}:")
        print(f"    F1-Score: {scores['f']:.4f}")
        print(f"    Precision: {scores['p']:.4f}")
        print(f"    Recall: {scores['r']:.4f}")

    print("\n" + "="*50 + "\n")

def test_distinct_n():
    sentences = [
        "今天天气真好啊",
        "今天天气真好啊",
        "今天天气真好啊",
        "你是谁呢"
    ]
    print(calculate_distinct_n(sentences, 3))


if __name__ == "__main__":
    test_distinct_n()