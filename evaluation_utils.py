from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np
import re
from typing import Dict, List

def is_clain_suppoted(context: str, claim: str) -> bool:

    return claim.lower() in context.lower()

def calculate_faithfulness_score(context: str, claims: List[str])->float:
    if not claims:
        return 0.0 
    
    supported = sum(1 for claim in claims if is_clain_suppoted(context, claim))

    return supported / len(claims)

def extract_claims_by_llm(response_text: str, extraction_api_func: callable[[str], str])->List[str]:
    prompt = (
        "Liệt kê các khẳng định (claims) có trong đoạn sau. "
        "Mỗi khẳng định là một câu hoàn chỉnh, rõ ràng, khách quan.\n"
        f"Đoạn văn:\n{response_text}"
    )
    result = extraction_api_func(prompt)
    return [claim.strip() for claim in result.split("\n") if claim.strip()]

def calculate_bleu(reference: str, candidate: str) -> float:
    reference = [reference.split()]
    candidate = candidate.split()
    senten_bleu =  sentence_bleu(reference, candidate)
    return senten_bleu

def calculate_rouge(reference: str, candidate: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return  {k: v.fmeasure for k, v in scores.items()}

def evaluate_task_rag(data: List[dict], api_call_func: callable[[str], str], extraction_api_func: callable[[str], str]) -> List[float]:
    result = []
    for item in data:
        context = item["context"]
        question = item["question"]
        prompt = f"Dựa trên: {context}.{question}"

        response = api_call_func(prompt)
        claims = extract_claims_by_llm(response, extraction_api_func)
        score = calculate_faithfulness_score(context, claims)
        result.append(score)

        return result 
    
def evaluate_task_economy(data: List[dict], api_call_func: callable)-> List[float]:
    results = []

    for item in data:
        if "question" not in item or "answers" not in item:
            results.append(0)
            continue

        question = item["question"]
        answers = item["answers"]

        keywords = ["lạm phát", "gdp", "tăng trưởng", "kinh tế", "lãi suất", "thương mại"]
        if not any(kw in question.lower() for kw in keywords):
            results.append(0)
            continue

        if isinstance(answers, list):
            answers_ref = answers[0].lower().strip() if answers else ""
        else:
            answers_ref = answers.lower().strip()

        prompt = question
        answers = api_call_func(prompt).lower().strip()

        is_correct = 1 if answers == answers_ref else 0 

        results.append(is_correct)

    return results

def evaluate_task_summary(data: List[dict], api_call_func: callable)-> List[Dict[str, float]]:
    resutls = []

    for item in data:
        ducument = item["Document"]
        summary_ref = item["Summary"]

        prompt= f"Tóm tắt đoạn văn sau một cách ngắn gọn và súc tích: {ducument}"
        summary = api_call_func(prompt)

        summary = re.sub(r'\s+', ' ', summary.strip()).lower()
        summary_ref = re.sub(r's+', ' ', summary_ref.strip()).lower()

        rouge = calculate_rouge(summary_ref, summary)
        resutls.append(rouge)

    return resutls

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
        text = item["text"]
        target = item["target_word"].lower().strip()

        prompt = f"Dựa trên đoạn văn sau, từ cuối cùng hợp lý là gì? {text} [Điền từ]"
        answer = api_call_func(prompt).lower().strip()

        predicted_words = re.findall(r'\w', answer)
        predicted_word = predicted_words[-1] if predicted_words else ""

        is_correct = 1 if predicted_words == target else 0 
        results.append(is_correct)

    return results
