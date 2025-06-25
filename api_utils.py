import logging
import requests
import re
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Qwen/Qwen3-1.7B"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

def call_qwen(prompt: str, max_tokens: int = 512, 
              temperature: float = 0.7, top_p: float = 0.95, top_k: int = 20,
              do_sample: bool = True) -> str:
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}  

    with torch.no_grad():  
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    

    if generated.startswith(prompt):
        return generated[len(prompt):].strip()
    return generated.strip()
