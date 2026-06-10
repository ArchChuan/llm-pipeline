import yaml
import json


def load_cfg():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def should_quantize(cfg: dict) -> bool:
    bits = cfg["QUANT_BITS"]
    if bits not in (0, 4, 8):
        raise ValueError("QUANT_BITS must be 0, 4, or 8")
    return bits != 0


def load_calibration_data(cfg: dict, tokenizer) -> list:
    with open(cfg["QUANT_DATASET"]) as f:
        raw = json.load(f)
    samples = raw[: cfg["QUANT_SAMPLES"]]
    texts = [item.get("instruction", "") + item.get("input", "") for item in samples]
    return [
        tokenizer(t, return_tensors="pt", truncation=True, max_length=512)
        for t in texts
        if t.strip()
    ]


def quantize(cfg: dict):
    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    if not should_quantize(cfg):
        print("QUANT_BITS=0，跳过量化")
        return

    bits = cfg["QUANT_BITS"]
    print(f"开始 INT{bits} 量化: {cfg['MODEL_PATH']} -> {cfg['QUANT_MODEL_PATH']}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL_PATH"], trust_remote_code=True)
    calib_data = load_calibration_data(cfg, tokenizer)

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=128,
        desc_act=False,
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        cfg["MODEL_PATH"],
        quantize_config=quantize_config,
        trust_remote_code=True,
    )

    model.quantize(calib_data)
    model.save_quantized(cfg["QUANT_MODEL_PATH"], use_safetensors=True)
    tokenizer.save_pretrained(cfg["QUANT_MODEL_PATH"])
    print(f"量化完成，已保存至: {cfg['QUANT_MODEL_PATH']}")


if __name__ == "__main__":
    quantize(load_cfg())
