from datasets import load_dataset

def load_data() -> tuple:
    dataset_rag = load_dataset("taidng/UIT-ViQuAD2.0", split="train[:100]")
    dataset_economy = load_dataset("taidng/UIT-ViQuAD2.0", split="train[:100]")
    dataset_summary = load_dataset("OpenHust/vietnamese-summarization", split="train[:50]")
    dataset_translation = load_dataset("ncduy/mt-en-vi", split="train[:50]")
    dataset_reasoning = load_dataset("vlsp-2023-vllm/lambada_vi", split="train[:50]")
    return dataset_rag, dataset_economy, dataset_summary, dataset_translation, dataset_reasoning