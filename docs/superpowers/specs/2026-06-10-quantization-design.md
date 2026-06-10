# 模型量化设计方案

**日期**: 2026-06-10
**项目**: llm-pipeline
**标签**: #quantization #gptq #inference

## 目标

在现有流水线中集成 GPTQ 量化，实现：
- 显存占用降低（INT4 约 4GB，INT8 约 8GB）
- 推理速度提升（离线量化权重，vLLM 原生支持）
- INT4 / INT8 运行时可选，不破坏原有非量化流程

## 方案选型

采用 **GPTQ（auto-gptq）**，放弃 bitsandbytes 动态量化和 AWQ：
- GPTQ 量化权重可同时被 HuggingFace server 和 vLLM 加载，两个服务都受益
- 比 bitsandbytes 推理更快（离线量化 vs 运行时动态量化）
- vLLM 原生支持 `quantization="gptq"` 参数，改动极小

## 架构

```
configs/config.yaml          新增 QUANT_BITS / QUANT_MODEL_PATH
scripts/quantize.py          新增：GPTQ 离线量化脚本
api/server.py                修改：按 QUANT_BITS 决定加载方式
api/vllm_server.py           修改：按 QUANT_MODEL_PATH 加载量化模型
Makefile                     新增 quantize 目标
requirements.txt             新增 auto-gptq
```

## 数据流

```
config.yaml
  QUANT_BITS: 0/4/8
  QUANT_MODEL_PATH: ./models/qwen/Qwen-7B-Chat-GPTQ
       │
       ├─→ scripts/quantize.py
       │     加载原始模型 → GPTQ 校准（128条样本）→ 保存量化权重
       │
       ├─→ api/server.py
       │     QUANT_BITS=0  → AutoModelForCausalLM.from_pretrained(MODEL_PATH)
       │     QUANT_BITS=4/8 → GPTQConfig(bits=N) + from_pretrained(QUANT_MODEL_PATH)
       │
       └─→ api/vllm_server.py
             QUANT_BITS=0  → LLM(model=MODEL_PATH)
             QUANT_BITS=4/8 → LLM(model=QUANT_MODEL_PATH, quantization="gptq")
```

## 文件变更

### configs/config.yaml
新增字段：
```yaml
QUANT_BITS: 4                                          # 0=不量化, 4=INT4, 8=INT8
QUANT_MODEL_PATH: ./models/qwen/Qwen-7B-Chat-GPTQ-int4
QUANT_DATASET: ./data/train_data.json                  # 校准数据集
QUANT_SAMPLES: 128                                     # 校准样本数
```

### scripts/quantize.py（新增）
1. 读取 config.yaml，加载 `MODEL_PATH` 原始模型
2. 用 `QUANT_DATASET` 的前 `QUANT_SAMPLES` 条数据做校准
3. 用 `auto_gptq.AutoGPTQForCausalLM` 量化，bits 取 `QUANT_BITS`
4. 保存到 `QUANT_MODEL_PATH`

### api/server.py（修改）
- 启动时读取 `QUANT_BITS`
- `QUANT_BITS > 0`：用 `GPTQConfig(bits=QUANT_BITS, disable_exllama=False)` 加载 `QUANT_MODEL_PATH`
- `QUANT_BITS == 0`：保持原有逻辑不变

### api/vllm_server.py（修改）
- `QUANT_BITS > 0`：`LLM(model=QUANT_MODEL_PATH, quantization="gptq")`
- `QUANT_BITS == 0`：保持原有逻辑不变

### Makefile（修改）
新增：
```makefile
quantize:
    python scripts/quantize.py
```

### requirements.txt（修改）
新增：
```
auto-gptq>=0.7
optimum>=1.16
```

## 关键约束

- `QUANT_BITS: 0` 为默认值，完全向后兼容
- `MODEL_PATH` 和 `QUANT_MODEL_PATH` 独立，原始权重不被覆盖
- 量化过程需要原始模型完整加载到显存，约耗时 10-30 分钟
- exllama 内核（`disable_exllama=False`）在 CUDA 可用时自动启用，速度更快
