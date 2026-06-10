from fastapi import FastAPI
from vllm import LLM, SamplingParams
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

app = FastAPI(title="vLLM 加速服务")

quant_bits = cfg.get("QUANT_BITS", 0)
if quant_bits in (4, 8):
    llm = LLM(
        model=cfg["QUANT_MODEL_PATH"],
        quantization="gptq",
        trust_remote_code=True,
    )
else:
    llm = LLM(model=cfg["MODEL_PATH"], trust_remote_code=True)

params = SamplingParams(max_tokens=1024, temperature=0.7)


@app.post("/chat")
def chat(prompt: str):
    res = llm.generate(prompt, params)
    return {"prompt": prompt, "response": res[0].outputs[0].text}
