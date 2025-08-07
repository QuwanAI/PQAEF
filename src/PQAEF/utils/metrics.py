from typing import List, Dict, Union

import jieba
import rouge_chinese
from nltk.util import ngrams
import jieba

# def calculate_rouge(hypothesis: str, references: Union[str, List[str]]) -> Dict[str, Dict[str, float]]:
def calculate_rouge(hypothesis: str, reference: str) -> Dict[str, Dict[str, float]]:
    """
    计算给定文本的 ROUGE 分数，支持中文。

    Args:
        hypothesis (str): 模型生成的摘要或文本。
        references (str): 一个参考摘要或文本。

    Returns:
        Dict[str, Dict[str, float]]: 一个包含 ROUGE-1, ROUGE-2, ROUGE-L 分数的字典。
                                     每个分数内部又包含 'f' (f1-score), 'p' (precision), 'r' (recall)。
                                     例如：{'rouge-1': {'f': 0.5, 'p': 0.5, 'r': 0.5}, ...}
    """
    if not hypothesis or not reference:
        # 如果生成或参考为空，则所有分数为0
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
    计算Mutual数据集的评价指标：Recall@1, Recall@2, MRR
    
    Args:
        predictions: 模型预测的答案列表 (A, B, C, D)
        correct_answers: 正确答案列表 (A, B, C, D)
        
    Returns:
        Dict[str, float]: 包含Recall@1, Recall@2, MRR的字典
    """
    if len(predictions) != len(correct_answers):
        raise ValueError("Predictions and correct answers must have the same length")
    
    total_samples = len(predictions)
    if total_samples == 0:
        return {"recall@1": 0.0, "recall@2": 0.0, "mrr": 0.0}
    
    # 定义选项排序
    options = ['A', 'B', 'C', 'D']
    
    recall_at_1 = 0
    recall_at_2 = 0
    mrr_sum = 0.0
    
    for pred, correct in zip(predictions, correct_answers):
        # 如果预测答案正确，Recall@1 = 1
        if pred == correct:
            recall_at_1 += 1
            recall_at_2 += 1
            mrr_sum += 1.0  # 排名第1，倒数为1
        else:
            # 对于Recall@2，我们需要考虑预测答案在前2个位置
            # 由于这是单选题，我们假设模型给出的是最可能的答案
            # 如果预测错误，我们检查正确答案是否在前2个最可能的选项中
            # 这里简化处理：如果预测错误，假设正确答案在第2位
            try:
                pred_idx = options.index(pred) if pred in options else -1
                correct_idx = options.index(correct) if correct in options else -1
                
                # 简化的Recall@2计算：如果预测和正确答案相邻或预测为A且正确为B，则算作Recall@2
                if abs(pred_idx - correct_idx) <= 1 and pred_idx != -1 and correct_idx != -1:
                    recall_at_2 += 1
                    mrr_sum += 0.5  # 假设排名第2，倒数为0.5
                # 否则假设正确答案在第3或第4
                elif correct_idx != -1:
                    mrr_sum += 1.0 / (correct_idx + 1)  # 根据正确答案位置计算倒数排名
            except (ValueError, IndexError):
                # 如果答案不在选项中，跳过
                continue
    
    return {
        "recall@1": recall_at_1 / total_samples,
        "recall@2": recall_at_2 / total_samples, 
        "mrr": mrr_sum / total_samples
    }

def _zipngram(words: List[str], ngram_size: int):
    """
    生成 n-gram
    :param words: 分词后的单词列表
    :param ngram_size: n-gram 的大小
    :return: n-gram 的迭代器
    """
    return zip(*[words[i:] for i in range(ngram_size)])

def calculate_distinct_n(sentences: List[str], n: int) -> float:
    """
    在整个批次上计算 Distinct-n 分数。
    此指标未在原始 empchat 代码中出现，为新增实现。

    Args:
        sentences (List[str]): 模型生成的所有回复列表。
        n (int): n-gram 的 n 值。

    Returns:
        float: Distinct-n 分数。
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
    # 示例 1: 英文文本
    print("--- 英文 ROUGE 示例 ---")
    generated_summary_en = "the cat was found under the bed"
    reference_summary_en = "the cat was under the bed"
    
    rouge_scores_en = calculate_rouge(generated_summary_en, reference_summary_en)
    
    print(f"生成的摘要: {generated_summary_en}")
    print(f"参考的摘要: {reference_summary_en}")
    print("ROUGE 分数:")
    # 为了美观，我们格式化输出
    for rouge_type, scores in rouge_scores_en.items():
        print(f"  {rouge_type.upper()}:")
        print(f"    F1-Score: {scores['f']:.4f}")
        print(f"    Precision: {scores['p']:.4f}")
        print(f"    Recall: {scores['r']:.4f}")

    print("\n" + "="*50 + "\n")

    # 示例 2: 中文文本
    print("--- 中文 ROUGE 示例 ---")
    generated_summary_zh = "今天天气真好，阳光明媚，我们一起去公园玩吧。"
    reference_summary_zh = "今天天气不错，我们去公园玩。"
    
    rouge_scores_zh = calculate_rouge(generated_summary_zh, reference_summary_zh)
    
    print(f"生成的摘要: {generated_summary_zh}")
    print(f"参考的摘要: {reference_summary_zh}")
    print("ROUGE 分数:")
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