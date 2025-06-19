import numpy as np
from api_utils import call_asko_api, call_qwen
from evaluation_utils import evaluate_task_rag, evaluate_task_economy, evaluate_task_summary, evaluate_task_translation, evaluate_task_reasoning
from loader_data import load_data

def run_evaluation():
    dataset_rag, dataset_economy, dataset_summary, dataset_translation, dataset_reasoning = load_data()

    print("Evaluating  Model asko...")
    asko_results = {
        "RAG": evaluate_task_rag(dataset_rag,call_asko_api),
        "Economy": evaluate_task_economy(dataset_economy, call_asko_api),
        "Summary": evaluate_task_summary(dataset_summary, call_asko_api),
        "Translation": evaluate_task_translation(dataset_translation, call_asko_api),
        "Reasoning": evaluate_task_reasoning(dataset_reasoning, call_asko_api)
    }

    print("Evaluating Qwen3-8B...")
    qwen_results = {
        "RAG": evaluate_task_rag(dataset_rag, call_qwen),
        "Economy": evaluate_task_economy(dataset_economy, call_qwen),
        "Summary": evaluate_task_summary(dataset_summary, call_qwen),
        "Translation": evaluate_task_translation(dataset_translation, call_qwen),
        "Reasoning": evaluate_task_reasoning(dataset_reasoning, call_qwen)
    }

    metrics = ["RAG", "Economy", "Summary", "Translation", "Reasoning"]
    for metric in metrics:
        if metric in ["RAG", "Economy", "Reasoning"]:
            your_score = np.mean(asko_results[metric])
            qwen_score = np.mean(qwen_results[metric])
            print(f"{metric} -model asko: {your_score:.2f}, model Qwen3-8B: {qwen_score:.2f}")
        elif metric == "Summary":
            your_score = np.mean([d["rouge1"] for d in asko_results[metric]])
            qwen_score = np.mean([d["rouge1"] for d in qwen_results[metric]])
            print(f"{metric} (ROUGE-1) -model asko: {your_score:.2f},model Qwen3-8B: {qwen_score:.2f}")
        else: 
            your_score = np.mean(asko_results[metric])
            qwen_score = np.mean(qwen_results[metric])
            print(f"{metric} (BLEU) -model asko: {your_score:.2f},model Qwen3-8B: {qwen_score:.2f}")


if __name__ == "__main__":
    run_evaluation()