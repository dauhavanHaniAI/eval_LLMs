from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data() -> tuple:
    dataset_rag = load_dataset("taidng/UIT-ViQuAD2.0", split="validation[:100]")
    dataset_economy = load_dataset("taidng/UIT-ViQuAD2.0", split="validation[:100]")
    dataset_summary = load_dataset("OpenHust/vietnamese-summarization", split="train[:50]")
    dataset_translation = load_dataset("ncduy/mt-en-vi", split="train[:50]")
    dataset_reasoning = load_dataset("vlsp-2023-vllm/lambada_vi", split="validation[:50]")

    rag = [{"context": i["context"], "question": i["question"]} for i in dataset_rag]
    econ = [{"context": i["context"], "question": i["question"], "answer": i.get("answers", [""]) if i.get("answers") else ""} for i in dataset_economy]
    summ = [{"Document": i["Document"], "Summary": i["Summary"]} for i in dataset_summary]
    trans = [{"en": i["en"], "vi": i["vi"]} for i in dataset_translation]
    reason = [{"text": i["text"], "target_word": i.get("target_word", "")} for i in dataset_reasoning]

    return rag, econ, summ, trans, reason