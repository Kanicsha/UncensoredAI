import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2b-it"
token = os.getenv("HF_TOKEN")

if not token:
    raise RuntimeError("HF_TOKEN is not set in environment")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(model_id, token=token)

print("MODEL LOADED")