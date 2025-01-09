import re
import evaluate
from rouge_score import rouge_scorer
import numpy as np
import copy
from collections import OrderedDict, Counter
from .utils import clean_output_text

def get_answer(text):
    # text = text.lower()
    label = text.split('Response:')[-1].strip()
    return label

def check_data_state(preds, targets):
    assert len(preds) == len(targets)

def get_rouge_score(predictions, targets):
    check_data_state(predictions, targets)
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    scores = {
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0,
        'rougeLsum': 0.0
    }
    for prediction, target in zip(predictions, targets):
        prediction = get_answer(clean_output_text(prediction))
        target = get_answer(clean_output_text(target))
        print(f"Prediction: {prediction} \nTarget: {target}")
        rouge_output = rouge.score(prediction=prediction, target=target)
        scores['rouge1'] += round(rouge_output["rouge1"].fmeasure, 4)
        scores['rouge2'] += round(rouge_output["rouge2"].fmeasure, 4)
        scores['rougeL'] += round(rouge_output["rougeL"].fmeasure, 4)
        scores['rougeLsum'] += round(rouge_output["rougeLsum"].fmeasure, 4)
    
    
    scores['rouge1'] /= len(predictions)
    scores['rouge2'] /= len(predictions)
    scores['rougeL'] /= len(predictions)
    scores['rougeLsum'] /= len(predictions)
    return scores

def exact_match(predictions, targets):
    check_data_state(predictions, targets)
    predictions = [get_answer(clean_output_text(prediction)) for prediction in predictions]
    targets = [get_answer(clean_output_text(target)) for target in targets]

    preds, targets = np.asarray(predictions, dtype="<U16"), np.asarray(targets, dtype="<U16")

    # print(preds, targets)
    return {"exact_match": np.sum(preds == targets) / preds.size}

def _f1_score(prediction, target):
    prediction_tokens = prediction.split()
    target_tokens = target.split()
    common = Counter(prediction_tokens) & Counter(target_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(target_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1(predictions, targets):
    check_data_state(predictions, targets)
    f1_score = 0.0
    for prediction, target in zip(predictions, targets):
        prediction = get_answer(clean_output_text(prediction))
        target = get_answer(clean_output_text(target))
        f1_score += _f1_score(prediction=prediction, target=target)
       
    f1_score /= len(predictions)
    return {'f1': f1_score}