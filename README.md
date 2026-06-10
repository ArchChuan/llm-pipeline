# LLM Pipeline

基于 Qwen-7B-Chat 的完整 LLM 自动化流水线，涵盖模型下载、推理服务和 LoRA 微调。

## 功能模块

| 模块 | 路径 | 说明 |
|------|------|------|
| 模型下载 | `scripts/download_model.py` | 从 ModelScope 下载模型权重 |
| 基础推理 API | `api/server.py` | HuggingFace Transformers + FastAPI |
| vLLM 加速 API | `api/vllm_server.py` | vLLM 引擎，高吞吐推理 |
| LoRA 微调 | `scripts/finetune.py` | PEFT LoRA 微调，输出适配器权重 |

## 环境要求

- Python 3.10+
- CUDA 11.8+（推荐 A100/3090/4090）
- 显存：≥24GB（Qwen-7B-Chat bfloat16）

## 快速开始

### 1. 安装依赖

```bash
python -m venv llm_env
source llm_env/bin/activate
pip install -r requirements.txt
```

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
make serve
# 访问: http://0.0.0.0:8000
```

**vLLM 加速服务（推荐生产使用）：**

```bash
make vllm
# 访问: http://0.0.0.0:8001
```

### 4. LoRA 微调

准备训练数据（JSON 格式）至 `./data/train_data.json`，然后：

```bash
make finetune
```

微调后的 LoRA 权重保存至 `./lora_weights`。

## API 接口

### 健康检查

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

所有参数通过 `configs/config.yaml` 统一管理：

```yaml
MODEL_NAME: qwen/Qwen-7B-Chat      # ModelScope 模型 ID
MODEL_PATH: ./models/qwen/Qwen-7B-Chat
DATA_PATH: ./data/train_data.json
LORA_OUTPUT: ./lora_weights
EPOCHS: 3
HOST: 0.0.0.0
API_PORT: 8000
VLLM_PORT: 8001
```

## 项目结构

```
llm-pipeline/
├── api/
│   ├── server.py          # Transformers 推理服务
│   └── vllm_server.py     # vLLM 加速服务
├── configs/
│   └── config.yaml        # 统一配置
├── scripts/
│   ├── download_model.py  # 模型下载
│   └── finetune.py        # LoRA 微调
├── data/                  # 训练数据（自行准备）
├── models/                # 模型权重（下载后生成）
├── lora_weights/          # LoRA 适配器输出
├── logs/                  # 服务日志
├── Makefile
└── requirements.txt
```

## Makefile 命令速查

```bash
make download   # 下载模型
make install    # 安装依赖
make serve      # 启动基础 API（端口 8000）
make vllm       # 启动 vLLM API（端口 8001）
make finetune   # 启动 LoRA 微调
make clean      # 清理日志
```
