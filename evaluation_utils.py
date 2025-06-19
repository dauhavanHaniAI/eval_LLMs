from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np
from typing import Dict, List

def calculate_bleu(reference: str, candidate: str) -> float:
    reference = [reference.split()]
    candidate = candidate.split()
    senten_bleu =  sentence_bleu(reference, candidate)
    return senten_bleu

def calculate_rouge(reference: str, candidate: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return  {k: v.fmeasure for k, v in scores.items()}

def evaluate_task_rag(data: List[dict], api_call_func: callable) -> List[float]:
    results = []
    for item in data:
        context = item["context"]
        question = item["question"]
        prompt = f"Dựa trên: {context}. {question}"
        answer = api_call_func(prompt)
        results.append(1 if answer.strip().lower() in item["answers"][0].lower() else 0)
    return results

def evaluate_task_economy(data: List[dict], api_call_func: callable) -> List[float]:
    results = []
    for item in data:
        prompt = item["question"]
        answer = api_call_func(prompt)
        results.append(1 if any(kw in answer.lower() for kw in ["lạm phát", "gdp", "tăng trưởng"]) else 0)
    return results

def evaluate_task_summary(data: List[dict], api_call_func: callable) -> List[Dict[str, float]]:
    results = []
    for item in data:
        text = item["text"]
        summary_ref = item["summary"]
        prompt = f"Tóm tắt đoạn sau trong 50 từ: {text}"
        summary = api_call_func(prompt)
        rouge = calculate_rouge(summary_ref, summary)
        results.append(rouge)
    return results

def evaluate_task_translation(data: List[dict], api_call_func: callable) -> List[float]:
    results = []
    for item in data:
        en_text = item["en"]
        vi_ref = item["vi"]
        prompt_en_vi = f"Dịch sang tiếng Việt: {en_text}"
        vi_pred = api_call_func(prompt_en_vi)
        prompt_vi_en = f"Dịch sang tiếng Anh: {vi_ref}"
        en_pred = api_call_func(prompt_vi_en)
        bleu_vi = calculate_bleu(vi_ref, vi_pred)
        bleu_en = calculate_bleu(en_text, en_pred)
        results.append((bleu_vi + bleu_en) / 2)
    return results

def evaluate_task_reasoning(data: List[dict], api_call_func: callable) -> List[float]:
    results = []
    for item in data:
        question = item["question"]
        answer_ref = item["answer"]
        prompt = question
        answer = api_call_func(prompt)
        results.append(1 if answer.strip().lower() == answer_ref.lower() else 0)
    return results