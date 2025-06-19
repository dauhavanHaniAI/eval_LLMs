import requests
import json
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_URL = "https://asko-llm.blaze.vn"
HEADERS = {"Content-Type": "application/json"}
MODEL = "" 


MODEL_NAME = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def call_asko_api(prompt: str, max_tokens: int = 100) -> str:
    """
    gửi prompt tới model LLM được triển khai qua API

    Args: 
        prompt(srt): câu hỏi hoặc nội dung đầu vào
        max_tokens (int): số lượng token tối đa được sinh ra từ mô hình

    Return:
        str: nội dung phản hồi từ mô hình hoặc thông báo lỗi nếu có
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }
    response = requests.post(f"{BASE_URL}/v1/chat/completions", headers=HEADERS, json=payload)
    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Lỗi: {e} | {response.status_code} | {response.text}"

def call_qwen(prompt: str, max_tokens: int = 100) -> str:
    """
    Sinh văn bản từ model Qwen3

    Args:
        prompt(str): câu hỏi hoặc nội dung đầu vào
        max_tokens: số lượng token tối đa được sinh ra từ mô hình

    Return:
        str: chuỗi đàu ra của mô hình
    """
    inputs = tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        max_length=max_tokens, 
                        truncation=True)

    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=max_tokens, 
                        do_sample=False)
    outputs_token =  tokenizer.decode(outputs[0], skip_special_tokens=True)

    return outputs_token