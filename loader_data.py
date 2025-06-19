from datasets import load_dataset

def load_data() -> tuple:
    dataset_rag = load_dataset("taidng/UIT-ViQuAD2.0", split="train[:100]")
    dataset_economy = [
        {"question": "Tình hình lạm phát Việt Nam năm 2025 là gì?", "context": ""},
        {"question": "GDP Việt Nam tăng trưởng bao nhiêu trong quý 1/2025?", "context": ""}
    ]
    dataset_summary = load_dataset("", split="train[:50]")
    dataset_translation = load_dataset("ncduy/mt-en-vi", split="train[:50]")
    dataset_reasoning = load_dataset("", split="train[:50]")
    return dataset_rag, dataset_economy, dataset_summary, dataset_translation, dataset_reasoning