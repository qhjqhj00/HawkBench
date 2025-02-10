import re
import string
import jieba
import difflib
from fuzzywuzzy import fuzz
from typing import List
from collections import Counter, defaultdict
from rouge import Rouge
import random
import numpy as np
import torch
import json
from .prompts import *
from nltk.tokenize import sent_tokenize

def set_seed(seed: int):
    """
    Set a fixed seed for reproducibility across multiple libraries.

    Parameters:
    seed (int): The seed value to use for random number generation.
    """
    # Set the seed for Python's random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    
    # Ensures deterministic behavior when using CUDA (may affect performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz.ratio(prediction, ground_truth) / 100)

def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if em_match_list != 0:
        if ground_truth in em_match_list:
            score = (1.0 / len(em_match_list))
        else:
            score = 0.0
    else:
        best_match = None
        highest_similarity = 0
        for string in all_classes:
            similarity = difflib.SequenceMatcher(None, string, prediction).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = string
        score = float(best_match == ground_truth)
    return score
    
def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def rouge_score_zh(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
    score = rouge_score(prediction, ground_truth)
    return score

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_score_zh(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)

def scorer(predictions, answers):
    
    score_func = [qa_f1_score, rouge_score]
    total_score = [0. for _ in range(len(score_func))]
    for (prediction, ground_truths) in zip(predictions, answers):
        for i, func in enumerate(score_func):
            score = 0.
            for ground_truth in ground_truths:
                score = max(score, func(prediction, ground_truth))
            total_score[i] += score
    for i in range(len(score_func)):
        total_score[i] = round(100 * total_score[i] / len(predictions), 2)
    rtn = {}
    for i, func in enumerate(score_func):
        rtn[func.__name__] = total_score[i]
    return rtn

def compute_multichoice_score(predicted_choices: list, reference_choices: list) -> float:
    predicted_set = set(predicted_choices)
    reference_set = set(reference_choices)
    
    if predicted_set != reference_set:
        return 0.0
    else:
        return 1

def get_gpt_score(agent, questions: list, gold_answers: list, predicted_answers: list):
    prompts = []
    for question, gold_answer, predicted_answer in zip(questions, gold_answers, predicted_answers):
        eval_prompt = ANSWER_EVALUATION_PROMPT.format(reference=gold_answer, predicted=predicted_answer)
        prompts.append(eval_prompt)

    gpt_scores = agent.batch_completion(prompts, max_completion_tokens=16)
    normalized_gpt_scores = []
    for gpt_score in gpt_scores:
        try:
            normalized_gpt_scores.append(float(gpt_score))
        except:
            normalized_gpt_scores.append(0.0)
    return round(np.mean(normalized_gpt_scores), 2)

def normalize_multichoice(multichoice):
    # Handle string input
    if isinstance(multichoice, str):
        # Try parsing as JSON first
        try:
            multichoice = json.loads(multichoice)
        except:
            # Clean and split on comma or period
            multichoice = multichoice.replace(' ', '').replace('.', ',').split(',')
            # Filter out empty strings
            multichoice = [c for c in multichoice if c]
    return multichoice

def get_multichoice_score(agent, predicted_answers, selections, gold_choices):
    if not gold_choices or any(not choices for choices in gold_choices):
        return 0
    all_prompts = []
    for predicted_answer, selection in zip(predicted_answers, selections):
        selection = json.loads(selection)
        str_selections = ""
        for k, v in selection.items():
            str_selections += f"{k}. {v}\n"

        eval_prompt = ANSWER_MULTICHOICE_PROMPT.format(answer=predicted_answer, selections=str_selections)
        all_prompts.append(eval_prompt)
    
    multichoices = agent.batch_completion(all_prompts, max_completion_tokens=32)
    scores = []
    for multichoice, gold_choice in zip(multichoices, gold_choices):
        gold_choice = json.loads(gold_choice)
        multichoice = normalize_multichoice(multichoice)
        score = compute_multichoice_score(multichoice, gold_choice)
        scores.append(score)
    return round(np.mean(scores), 2)

def get_sentence_level_score(agent, gold_answers, pred_answers):
    prompts = []
    sample_index = []
    for i, (gold_answer, pred_answer) in enumerate(zip(gold_answers, pred_answers)):
        gold_answer = gold_answer[0]
        sents = sent_tokenize(pred_answer)
        for sent in sents:
            eval_prompt = ANSWER_CONTAINS_PROMPT.format(reference=gold_answer, sentence=sent)
            prompts.append(eval_prompt)
            sample_index.append(i+1)

        sents = sent_tokenize(gold_answer)
        for sent in sents:
            eval_prompt = ANSWER_CONTAINS_PROMPT.format(reference=pred_answer, sentence=sent)
            prompts.append(eval_prompt)
            sample_index.append(-i-1)
            
    scores = agent.batch_completion(prompts, max_completion_tokens=16)
    score_counter = defaultdict(list)
    for score, index in zip(scores, sample_index):
        try:
            score_counter[index].append(float(score))
        except:
            continue
    precision = []
    recall = []
    for index in score_counter:
        if not score_counter[index]:
            continue
        if index < 0:
            recall.append(np.mean(score_counter[index]))
        else:
            precision.append(np.mean(score_counter[index]))

    return round(np.mean(precision)*100, 2), round(np.mean(recall)*100, 2)
