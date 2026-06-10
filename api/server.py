from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


def build_model(cfg: dict):
    quant_bits = cfg.get("QUANT_BITS", 0)
    if quant_bits in (4, 8):
        model_path = cfg["QUANT_MODEL_PATH"]
        quantization_config = GPTQConfig(bits=quant_bits, disable_exllama=False)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL_PATH"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["MODEL_PATH"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
    return tokenizer, model


app = FastAPI(title="LLM API Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

tokenizer, model = build_model(cfg)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"prompt": prompt, "response": response}
