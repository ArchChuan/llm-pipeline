# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install       # 安装依赖（pip install -r requirements.txt）
make download      # 从 ModelScope 下载模型到 ./models/
make api           # 后台启动 HuggingFace 推理 API（端口 8000，日志→logs/api.log）
make vllm          # 后台启动 vLLM 加速 API（端口 8001，日志→logs/vllm.log）
make finetune      # 运行 LoRA 微调
make logs          # tail -f logs/api.log
make stop          # pkill uvicorn
make clean         # 清空 logs/
```

直接运行（前台，便于调试）：

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000


uvicorn api.vllm_server:app --host 0.0.0.0 --port 8001
python scripts/download_model.py
python scripts/finetune.py
```

## 架构

单配置源：`configs/config.yaml` —— 所有脚本和服务在启动时读取此文件，无运行时覆盖机制。

```
configs/config.yaml     ← 唯一配置（HOST/PORT/MODEL_PATH/LoRA 参数）
api/
  server.py             ← HuggingFace Transformers 推理，POST /chat + GET /health
  vllm_server.py        ← vLLM 引擎推理，POST /chat（无 /health 端点）
scripts/
  download_model.py     ← ModelScope snapshot_download → ./models/
  finetune.py           ← PEFT LoRA 微调，输出到 LORA_OUTPUT
```

两个 API 服务互相独立，共享同一 `MODEL_PATH`。`server.py` 在模块导入时加载模型（启动慢）；`vllm_server.py` 用 vLLM `LLM` 对象，吞吐更高。

`finetune.py` 读取 `DATA_PATH`（JSON 格式），LoRA target modules 硬编码为 `q_proj`/`v_proj`（适配 Qwen 架构）。

## 关键约定

- 虚拟环境：`llm_env`（uv 管理），激活用 `source llm_env/bin/activate`
- 模型默认：`qwen/Qwen-7B-Chat`（ModelScope ID）→ 本地路径 `./models/qwen/Qwen-7B-Chat`
- 训练数据格式：JSON，字段由 `datasets.load_dataset("json", ...)` 自动解析
- `bitsandbytes` 已在依赖中，可用于动态 INT8 量化（`load_in_8bit=True`）
