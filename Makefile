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

# 模型量化
quantize:
	@echo "=== 开始 GPTQ 量化 ==="
	python scripts/quantize.py

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

# ── 测试 ──────────────────────────────────────────────────────
# 性能测试
bench-latency:
	python tests/perf/latency_benchmark.py --url http://$(HOST):$(API_PORT)

bench-stress:
	python tests/perf/stress_test.py --url http://$(HOST):$(API_PORT) --concurrency 10 --total 100

bench-throughput:
	python tests/perf/throughput_test.py --url http://$(HOST):$(API_PORT)

# 准确率测试
eval:
	python tests/accuracy/eval_chat.py --url http://$(HOST):$(API_PORT)

eval-consistency:
	python tests/accuracy/consistency_test.py --url http://$(HOST):$(API_PORT)

# 量化前后对比（需要服务已停止，脚本会自动启停）
compare-quant:
	bash tests/compare_quant.sh

# 跑全部单元测试
test:
	python -m pytest tests/ -v --ignore=tests/perf --ignore=tests/accuracy
