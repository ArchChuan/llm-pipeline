# LLM Pipeline

基于 Qwen-7B-Chat 的完整 LLM 自动化流水线，涵盖模型下载、推理服务、LoRA 微调、GPTQ 量化和测试评估。

## 功能模块

| 模块 | 路径 | 说明 |
|------|------|------|
| 模型下载 | `scripts/download_model.py` | 从 ModelScope 下载模型权重 |
| 基础推理 API | `api/server.py` | HuggingFace Transformers + FastAPI |
| vLLM 加速 API | `api/vllm_server.py` | vLLM 引擎，高吞吐推理，支持 GPTQ 量化模型 |
| LoRA 微调 | `scripts/finetune.py` | PEFT LoRA 微调，输出适配器权重 |
| GPTQ 量化 | `scripts/quantize.py` | INT4/INT8 量化，压缩模型体积 |

## 环境要求

- Python 3.10+
- CUDA 11.8+（推荐 A100/3090/4090）
- 显存：≥24GB（Qwen-7B-Chat bfloat16）；量化后 ≥8GB（INT4）

## 快速开始

### 1. 安装依赖

```bash
python -m venv llm_env
source llm_env/bin/activate
pip install -r requirements.txt
```

> `flash-attn` 已从默认依赖中移除（需要 nvcc 编译）。如需安装：
> ```bash
> pip install flash-attn --no-build-isolation
> ```

### 2. 下载模型

```bash
make download
# 或者
python scripts/download_model.py
```

模型默认下载至 `./models/qwen/Qwen-7B-Chat`。

### 3. 启动推理服务

**基础服务（HuggingFace Transformers）：**

```bash
make api
# 访问: http://127.0.0.1:8000
```

**vLLM 加速服务（推荐生产使用）：**

```bash
make vllm
# 访问: http://127.0.0.1:8001
```

### 4. LoRA 微调

准备训练数据（JSON 格式，字段：`instruction` / `input` / `output`）至 `./data/train_data.json`，然后：

```bash
make finetune
```

微调后的 LoRA 权重保存至 `./lora_weights`。

### 5. GPTQ 量化

```bash
make quantize
```

量化参数在 `configs/config.yaml` 中配置，量化模型输出至 `QUANT_MODEL_PATH`。约需 10–30 分钟（取决于 GPU）。

## API 接口

### 健康检查（仅 server.py）

```
GET /health
```

### 对话推理

```
POST /chat
Content-Type: application/json

{
  "prompt": "你好，请介绍一下自己"
}
```

响应：

```json
{
  "prompt": "你好，请介绍一下自己",
  "response": "..."
}
```

## 配置

所有参数通过 `configs/config.yaml` 统一管理（同时作为 Makefile 变量导入）：

```yaml
HOST: 127.0.0.1
API_PORT: 8000
VLLM_PORT: 8001

MODEL_NAME: qwen/Qwen-7B-Chat      # ModelScope 模型 ID
MODEL_PATH: ./models/qwen/Qwen-7B-Chat

QUANT_BITS: 0                      # 0=不量化, 4=INT4, 8=INT8
QUANT_MODEL_PATH: ./models/qwen/Qwen-7B-Chat-GPTQ-int4
QUANT_DATASET: ./data/train_data.json
QUANT_SAMPLES: 128

DATA_PATH: ./data/train_data.json
LORA_OUTPUT: ./lora_weights
EPOCHS: 3
```

## 项目结构

```
llm-pipeline/
├── api/
│   ├── server.py               # Transformers 推理服务（含 /health）
│   └── vllm_server.py          # vLLM 加速服务（支持 GPTQ）
├── configs/
│   └── config.yaml             # 唯一配置源
├── scripts/
│   ├── download_model.py       # 模型下载
│   ├── finetune.py             # LoRA 微调
│   └── quantize.py             # GPTQ 量化
├── tests/
│   ├── perf/
│   │   ├── latency_benchmark.py  # 延迟基准测试
│   │   ├── stress_test.py        # 并发压力测试
│   │   └── throughput_test.py    # 吞吐量测试
│   ├── accuracy/
│   │   ├── eval_chat.py          # 业务准确率评估
│   │   └── consistency_test.py   # 输出一致性测试
│   ├── compare_quant.sh          # 量化前后对比
│   ├── test_quantize.py          # 量化单元测试
│   └── test_server_quant.py      # 服务量化单元测试
├── data/                       # 训练数据（自行准备）
├── models/                     # 模型权重（下载后生成）
├── lora_weights/               # LoRA 适配器输出
├── logs/                       # 服务日志
├── Makefile
└── requirements.txt
```

## Makefile 命令速查

```bash
make install          # 安装依赖
make download         # 下载模型
make api              # 后台启动基础 API（端口 8000，日志→logs/api.log）
make vllm             # 后台启动 vLLM API（端口 8001，日志→logs/vllm.log）
make finetune         # LoRA 微调
make quantize         # GPTQ 量化（INT4/INT8）
make logs             # 查看 API 日志
make stop             # 停止所有 uvicorn 进程
make clean            # 清空 logs/

# 性能测试
make bench-latency    # 延迟基准测试
make bench-stress     # 并发压力测试（10 并发，100 请求）
make bench-throughput # 吞吐量测试

# 准确率测试
make eval             # 业务准确率评估
make eval-consistency # 输出一致性测试
make compare-quant    # 量化前后效果对比

# 单元测试
make test             # 运行所有单元测试（pytest）
```
