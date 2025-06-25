from datasets import load_dataset
import numpy as np
from api_utils import  call_qwen
from evaluation_utils import (
    evaluate_task_rag,  
    evaluate_task_economy,
    evaluate_task_summary,
    evaluate_task_translation,
    evaluate_task_reasoning,
    )
from loader_data import load_data 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

def run_evaluation():
    dataset_rag, dataset_economy, dataset_summary,dataset_translation,  dataset_reasoning = load_data()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key="")

    print("\nEvaluating Model Qwen...")
    results = {
        "RAG": evaluate_task_rag(dataset_rag[:5], call_qwen, llm = llm, embeddings = embeddings),  
        "Economy": evaluate_task_economy(dataset_economy[:5], call_qwen),
        "Summary": evaluate_task_summary(dataset_summary[:5], call_qwen),
        "Translation": evaluate_task_translation(dataset_translation[:5], call_qwen),
        "Reasoning": evaluate_task_reasoning(dataset_reasoning[:5], call_qwen),
    }

    for metric, scores in results.items():
        print(f"\n{metric} scores: {scores}")
        if not scores:
            print(f"{metric} - No valid scores")
            continue
        if metric == "Summary":
            avg = np.mean([s["rouge1"] for s in scores])
            print(f"{metric} (ROUGE-1) - Model Qwen: {avg:.2f}")
        elif metric == "Translation":
            print(f"{metric} (BLEU) - Model Qwen: {np.mean(scores):.2f}")
        else:
            print(f"{metric} - Model Qwen: {np.mean(scores):.2f}")

if __name__ == "__main__":
    run_evaluation()