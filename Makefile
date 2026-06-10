include configs/config.yaml

# 安装依赖
install:
	@echo "=== 安装依赖 ==="
	pip install -r requirements.txt

# 下载模型
download:
	@echo "=== 下载模型 ==="
	python scripts/download_model.py

# 启动基础API
api:
	@echo "=== 启动API服务 ==="
	nohup uvicorn api.server:app --host $(HOST) --port $(API_PORT) > logs/api.log 2>&1 &
	@echo "服务已启动: http://$(HOST):$(API_PORT)"

# 启动vLLM加速API
vllm:
	@echo "=== 启动vLLM加速服务 ==="
	nohup uvicorn api.vllm_server:app --host $(HOST) --port $(VLLM_PORT) > logs/vllm.log 2>&1 &
	@echo "vLLM服务已启动: http://$(HOST):$(VLLM_PORT)"

# LoRA微调
finetune:
	@echo "=== 开始LoRA微调 ==="
	python scripts/finetune.py

# 查看日志
logs:
	tail -f logs/api.log

# 停止服务
stop:
	@pkill -f uvicorn
	@echo "服务已停止"

# 清理日志
clean:
	rm -rf logs/*
