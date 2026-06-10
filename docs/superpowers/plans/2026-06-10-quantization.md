# 模型量化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 llm-pipeline 集成 GPTQ 量化，支持 INT4/INT8 可选，降低显存占用并提升推理速度，同时兼容 HuggingFace server 和 vLLM server。

**Architecture:** 在 `configs/config.yaml` 新增量化相关配置项（`QUANT_BITS`/`QUANT_MODEL_PATH`），新增离线量化脚本 `scripts/quantize.py`，修改两个 API 服务在启动时按配置选择加载原始模型或量化模型。`QUANT_BITS=0` 时行为与现在完全一致。

**Tech Stack:** auto-gptq>=0.7, optimum>=1.16, transformers GPTQConfig, vLLM quantization="gptq"

---

## 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `configs/config.yaml` | 新增量化配置项 |
| 修改 | `requirements.txt` | 新增 auto-gptq, optimum |
| 新增 | `scripts/quantize.py` | GPTQ 离线量化脚本 |
| 修改 | `api/server.py` | 按 QUANT_BITS 选择加载方式 |
| 修改 | `api/vllm_server.py` | 按 QUANT_BITS 选择加载方式 |
| 修改 | `Makefile` | 新增 quantize 目标 |
| 新增 | `tests/test_quantize.py` | 量化脚本单元测试 |
| 新增 | `tests/test_server_quant.py` | server 量化加载逻辑测试 |

---

### Task 1: 更新配置和依赖

**Files:**
- Modify: `configs/config.yaml`
- Modify: `requirements.txt`

- [ ] **Step 1: 更新 config.yaml，新增量化配置**

将 `configs/config.yaml` 改为：

```yaml
# 服务配置
HOST: 0.0.0.0
API_PORT: 8000
VLLM_PORT: 8001

# 模型配置
MODEL_NAME: qwen/Qwen-7B-Chat
MODEL_PATH: ./models/qwen/Qwen-7B-Chat

# 量化配置
QUANT_BITS: 0                                        # 0=不量化, 4=INT4, 8=INT8
QUANT_MODEL_PATH: ./models/qwen/Qwen-7B-Chat-GPTQ-int4
QUANT_DATASET: ./data/train_data.json
QUANT_SAMPLES: 128

# 微调配置
DATA_PATH: ./data/train_data.json
LORA_OUTPUT: ./lora_weights
EPOCHS: 3
```

- [ ] **Step 2: 更新 requirements.txt，新增量化依赖**

在 `requirements.txt` 末尾追加：

```
# 量化
auto-gptq>=0.7
optimum>=1.16
```

- [ ] **Step 3: 提交**

```bash
git add configs/config.yaml requirements.txt
git commit -m "feat: 新增量化配置项和依赖"
```

---

### Task 2: 编写量化脚本测试

**Files:**
- Create: `tests/test_quantize.py`

- [ ] **Step 1: 创建测试文件**

```python
import pytest
import os
import yaml
from unittest.mock import patch, MagicMock


def load_cfg():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def test_config_has_quant_fields():
    cfg = load_cfg()
    assert "QUANT_BITS" in cfg
    assert "QUANT_MODEL_PATH" in cfg
    assert "QUANT_SAMPLES" in cfg
    assert cfg["QUANT_BITS"] == 0  # 默认不量化


def test_quant_bits_zero_skips_quantization():
    """QUANT_BITS=0 时应直接返回，不执行量化"""
    cfg = load_cfg()
    cfg["QUANT_BITS"] = 0

    with patch("scripts.quantize.AutoGPTQForCausalLM") as mock_gptq:
        from scripts.quantize import should_quantize
        assert should_quantize(cfg) is False
        mock_gptq.assert_not_called()


def test_quant_bits_4_triggers_quantization():
    cfg = load_cfg()
    cfg["QUANT_BITS"] = 4
    from scripts.quantize import should_quantize
    assert should_quantize(cfg) is True


def test_quant_bits_8_triggers_quantization():
    cfg = load_cfg()
    cfg["QUANT_BITS"] = 8
    from scripts.quantize import should_quantize
    assert should_quantize(cfg) is True


def test_invalid_quant_bits_raises():
    cfg = load_cfg()
    cfg["QUANT_BITS"] = 3
    from scripts.quantize import should_quantize
    with pytest.raises(ValueError, match="QUANT_BITS must be 0, 4, or 8"):
        should_quantize(cfg)
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_quantize.py -v
```

期望：`ImportError: cannot import name 'should_quantize' from 'scripts.quantize'`（文件尚未创建）

---

### Task 3: 实现量化脚本

**Files:**
- Create: `scripts/quantize.py`
- Create: `scripts/__init__.py`（如不存在）

- [ ] **Step 1: 创建 scripts/__init__.py（如不存在）**

```bash
touch scripts/__init__.py
```

- [ ] **Step 2: 创建 scripts/quantize.py**

```python
import yaml
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
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
```

- [ ] **Step 3: 运行测试，确认通过**

```bash
python -m pytest tests/test_quantize.py -v
```

期望：5 个测试全部 PASS

- [ ] **Step 4: 提交**

```bash
git add scripts/__init__.py scripts/quantize.py tests/test_quantize.py
git commit -m "feat: 新增 GPTQ 量化脚本及测试"
```

---

### Task 4: 修改 api/server.py 支持量化模型加载

**Files:**
- Modify: `api/server.py`
- Create: `tests/test_server_quant.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_server_quant.py`：

```python
import pytest
from unittest.mock import patch, MagicMock
import yaml


def load_cfg():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def test_build_model_no_quant():
    """QUANT_BITS=0 时使用原始模型路径，无 GPTQConfig"""
    cfg = load_cfg()
    cfg["QUANT_BITS"] = 0

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("api.server.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_load, \
         patch("api.server.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):

        from api.server import build_model
        tokenizer, model = build_model(cfg)

        args, kwargs = mock_load.call_args
        assert args[0] == cfg["MODEL_PATH"]
        assert "quantization_config" not in kwargs


def test_build_model_int4():
    """QUANT_BITS=4 时使用量化模型路径和 GPTQConfig(bits=4)"""
    cfg = load_cfg()
    cfg["QUANT_BITS"] = 4
    cfg["QUANT_MODEL_PATH"] = "./models/qwen/Qwen-7B-Chat-GPTQ-int4"

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("api.server.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_load, \
         patch("api.server.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("api.server.GPTQConfig") as mock_gptq_cfg:

        from api.server import build_model
        tokenizer, model = build_model(cfg)

        mock_gptq_cfg.assert_called_once_with(bits=4, disable_exllama=False)
        args, kwargs = mock_load.call_args
        assert args[0] == cfg["QUANT_MODEL_PATH"]
        assert "quantization_config" in kwargs


def test_build_model_int8():
    """QUANT_BITS=8 时使用量化模型路径和 GPTQConfig(bits=8)"""
    cfg = load_cfg()
    cfg["QUANT_BITS"] = 8
    cfg["QUANT_MODEL_PATH"] = "./models/qwen/Qwen-7B-Chat-GPTQ-int8"

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("api.server.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_load, \
         patch("api.server.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("api.server.GPTQConfig") as mock_gptq_cfg:

        from api.server import build_model
        tokenizer, model = build_model(cfg)

        mock_gptq_cfg.assert_called_once_with(bits=8, disable_exllama=False)
        args, kwargs = mock_load.call_args
        assert args[0] == cfg["QUANT_MODEL_PATH"]
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_server_quant.py -v
```

期望：`ImportError: cannot import name 'build_model' from 'api.server'`

- [ ] **Step 3: 修改 api/server.py**

```python
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
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_server_quant.py -v
```

期望：3 个测试全部 PASS

- [ ] **Step 5: 提交**

```bash
git add api/server.py tests/test_server_quant.py
git commit -m "feat: server.py 支持 GPTQ 量化模型加载"
```

---

### Task 5: 修改 api/vllm_server.py 支持量化模型

**Files:**
- Modify: `api/vllm_server.py`

- [ ] **Step 1: 修改 api/vllm_server.py**

```python
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
```

- [ ] **Step 2: 提交**

```bash
git add api/vllm_server.py
git commit -m "feat: vllm_server.py 支持 GPTQ 量化模型加载"
```

---

### Task 6: 更新 Makefile

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: 在 Makefile 中新增 quantize 目标**

在 `finetune` 目标后添加：

```makefile
# 模型量化
quantize:
	@echo "=== 开始 GPTQ 量化 ==="
	python scripts/quantize.py
```

- [ ] **Step 2: 验证 make 命令可解析**

```bash
make --dry-run quantize
```

期望输出：`python scripts/quantize.py`（不报语法错误）

- [ ] **Step 3: 提交**

```bash
git add Makefile
git commit -m "feat: Makefile 新增 quantize 目标"
```

---

### Task 7: 更新 README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 在 README 的"Makefile 命令速查"表格中新增量化行**

找到：
```markdown
make finetune   # 启动 LoRA 微调
```

改为：
```markdown
make finetune   # 启动 LoRA 微调
make quantize   # GPTQ 量化（INT4/INT8，约 10-30 分钟）
```

- [ ] **Step 2: 在"配置"章节新增量化字段说明**

找到：
```yaml
EPOCHS: 3
```

改为：
```yaml
EPOCHS: 3
QUANT_BITS: 0              # 量化精度：0=不量化, 4=INT4, 8=INT8
QUANT_MODEL_PATH: ./models/qwen/Qwen-7B-Chat-GPTQ-int4
QUANT_SAMPLES: 128         # 量化校准样本数
```

- [ ] **Step 3: 提交**

```bash
git add README.md
git commit -m "docs: README 新增量化配置和命令说明"
```

---

### Task 8: 推送到远端

- [ ] **Step 1: 运行全部测试，确认全绿**

```bash
python -m pytest tests/ -v
```

期望：所有测试 PASS，无 FAIL

- [ ] **Step 2: 推送**

```bash
git push origin master
```
