from typing import List, Dict, Callable
from ragas.metrics import faithfulness
from ragas import evaluate
from datasets import Dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
import logging
import re
import numpy as np
from rouge_score import rouge_scorer
from langchain.llms import OpenAI


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_rouge(reference: str, candidate: str) -> Dict[str, float]:
    try:
        reference = re.sub(r'\s+', ' ', reference.lower().strip())
        candidate = re.sub(r'\s+', ' ', candidate.lower().strip())


        if not reference or not candidate:
            logger.warning("Reference or candidate is empty")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)

        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        }
    except Exception as e:
        logger.error(f"Error calculating ROUGE: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    

def calculate_bleu(reference: str, candidate: str) -> float:
    try:
        reference = [reference.split()]
        candidate = candidate.split()
        smoothie = SmoothingFunction().method4
        return sentence_bleu(reference, candidate, smoothing_function=smoothie)
    except Exception as e:
        logger.error(f"Error calculating BLEU: {e}")
        return 0.0

def evaluate_task_rag(data: List[dict], api_call_func: Callable[[str], str], llm, embeddings) -> List[float]:
    results = []
    questions = []
    contexts = []
    answers = []

    for item in data:
        try:
            context = item.get("context", "")
            question = item.get("question", "")
            if not context or not question:
                logger.warning("Missing context or question, skipping item")
                results.append(0.0)
                continue

            prompt = f"""Bạn là một trợ lý AI. Chỉ trả lời dựa trên ngữ cảnh dưới đây. Nếu không tìm thấy thông tin trong ngữ cảnh, hãy trả lời: "Tôi không tìm thấy thông tin trong ngữ cảnh."

            Ngữ cảnh: {context}
            Câu hỏi: {question}
            Trả lời:"""

            response = api_call_func(prompt).strip()

            if not response:
                response = "Tôi không tìm thấy thông tin trong ngữ cảnh."

            questions.append(question)
            contexts.append([context])
            answers.append(response)
        except Exception as e:
            logger.error(f"Error processing RAG item: {e}")
            results.append(0.0)
            continue

    try:
        dataset = Dataset.from_dict({
            "question": questions,
            "contexts": contexts,
            "answer": answers
        })
        result = evaluate(dataset, metrics=[faithfulness], llm=llm, embeddings=embeddings)
        faithfulness_scores = result.scores['faithfulness'].tolist()
        results.extend(faithfulness_scores)
    except Exception as e:
        logger.error(f"Error evaluating RAG with RAGAS: {e}")
        results.extend([0.0] * len(questions))

    return results

def evaluate_task_economy(data: List[dict], api_call_func: Callable[[str], str]) -> List[float]:
    results = []

    for item in data:
        try:
            question = item.get("question", "")
            answers = item.get("answer") or item.get("answers", [])
            if not question:
                logger.warning("Empty question, skipping item")
                results.append(0.0)
                continue

            prompt = f"""Bạn là một chuyên gia kinh tế Việt Nam năm 2024. Trả lời ngắn gọn, chỉ khi chắc chắn. Nếu không có đủ thông tin, hãy trả lời: "Tôi không chắc chắn về dữ kiện này."
            Câu hỏi: {question}
            Trả lời:"""

            pred = api_call_func(prompt).lower().strip()

            if isinstance(answers, list) and answers and isinstance(answers[0], str):
                ref = answers[0].lower().strip()
            elif isinstance(answers, str):
                ref = answers.lower().strip()
            else:
                logger.warning("Invalid reference answer format")
                results.append(0.0)
                continue

            if not pred:
                logger.warning("Empty prediction")
                results.append(0.0)
                continue

            pred_numbers = re.findall(r"\d+(?:[\.,]\d+)?", pred)
            ref_numbers = re.findall(r"\d+(?:[\.,]\d+)?", ref)
            numeric_score = 1.0 if pred_numbers and ref_numbers and pred_numbers[0] == ref_numbers[0] else 0.0

            bleu_score = calculate_bleu(ref, pred)
            ref_embedding = semantic_model.encode(ref)
            pred_embedding = semantic_model.encode(pred)
            semantic_score = util.cos_sim(ref_embedding, pred_embedding)[0][0].item()
            combined_score = (bleu_score + semantic_score + numeric_score) / 3
            results.append(combined_score)
        except Exception as e:
            logger.error(f"Error evaluating economy item: {e}")
            results.append(0.0)

    return results

def evaluate_task_summary(data: List[dict], api_call_func: Callable[[str], str]) -> List[Dict[str, float]]:
    results = []
    for item in data:
        document = item.get("Document", "")
        summary_ref = item.get("Summary", "")
        prompt = f"""Bạn là một chuyên gia tóm tắt văn bản tiếng Việt. Hãy tóm tắt đoạn văn sau thành 1-2 câu, giữ lại các ý chính, sử dụng ngôn ngữ tự nhiên và ngắn gọn.

        **Ví dụ**:
        Đoạn văn: Việt Nam đang đẩy mạnh xuất khẩu nông sản như cà phê, gạo và trái cây. Năm 2023, kim ngạch xuất khẩu đạt 50 tỷ USD. Chính phủ đang hỗ trợ nông dân bằng các chính sách ưu đãi.
        Tóm tắt: Việt Nam tăng cường xuất khẩu nông sản, đạt 50 tỷ USD năm 2023, với sự hỗ trợ từ chính sách chính phủ.

        **Đoạn văn**: {document}
        **Tóm tắt**: """
        summary_pred = api_call_func(prompt)
        rouge = calculate_rouge(summary_ref.lower().strip(), summary_pred.lower().strip())
        results.append(rouge)
    return results


def evaluate_task_translation(data: List[dict], api_call_func: Callable[[str], str]) -> List[float]:
    results = []
    for item in data:
        en_text = item["en"]
        vi_ref = item["vi"]
        prompt_vi = f"""Bạn là một chuyên gia dịch thuật Anh-Việt. Hãy dịch câu sau sang tiếng Việt, giữ nguyên nghĩa và sử dụng ngôn ngữ tự nhiên, phù hợp với văn hóa Việt Nam.

        **Ví dụ**:
        Câu tiếng Anh: The sun is shining brightly today.
        Bản dịch: Mặt trời đang chiếu sáng rực rỡ hôm nay.

        **Câu cần dịch**: {en_text}
        **Bản dịch**: """
        vi_pred = api_call_func(prompt_vi)

        prompt_en = f"""Bạn là một chuyên gia dịch thuật Việt-Anh. Hãy dịch câu sau sang tiếng Anh, giữ nguyên nghĩa và sử dụng ngôn ngữ tự nhiên.

        **Ví dụ**:
        Câu tiếng Việt: Việt Nam là một đất nước xinh đẹp.
        Bản dịch: Vietnam is a beautiful country.

        **Câu cần dịch**: {vi_ref}
        **Bản dịch**: """
        en_pred = api_call_func(prompt_en)

        bleu_vi = calculate_bleu(vi_ref, vi_pred)
        bleu_en = calculate_bleu(en_text, en_pred)
        results.append((bleu_vi + bleu_en) / 2)
    return results

def evaluate_task_reasoning(data: List[dict], api_call_func: Callable[[str], str]) -> List[float]:
    results = []

    for item in data:
        try:
            text = item.get("text", "")
            target = item.get("target_word", "").lower().strip()
            if not text or not target:
                logger.warning("Missing text or target word, skipping item")
                results.append(0.0)
                continue

            prompt = f"Dựa trên đoạn văn sau, từ cuối cùng hợp lý là gì? {text} [Điền từ]"
            answer = api_call_func(prompt).lower().strip()
            if not answer:
                logger.warning("Empty answer")
                results.append(0.0)
                continue

            predicted_words = re.findall(r'\w+', answer)
            last_word = predicted_words[-1] if predicted_words else ""

            target_embedding = semantic_model.encode(target)
            answer_embedding = semantic_model.encode(answer)
            semantic_score = util.cos_sim(target_embedding, answer_embedding)[0][0].item()
            exact_match = 1.0 if last_word == target else 0.0
            combined_score = (semantic_score + exact_match) / 2
            results.append(combined_score)
        except Exception as e:
            logger.error(f"Error evaluating reasoning item: {e}")
            results.append(0.0)

    return results
