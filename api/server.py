from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml

# 加载配置
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

app = FastAPI(title="LLM API Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL_PATH"], trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    cfg["MODEL_PATH"],
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"prompt": prompt, "response": response}
