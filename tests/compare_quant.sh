#!/usr/bin/env bash
#
# 量化前后对比测试脚本
# 用法：./tests/compare_quant.sh
# 说明：自动修改 config.yaml 的 QUANT_BITS，重启服务，运行性能+准确率测试
#

set -e

echo "===== 量化前后对比测试 ====="
echo ""

CONFIG_PATH="configs/config.yaml"
BACKUP_PATH="configs/config.yaml.bak"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "错误: 未找到 $CONFIG_PATH"
    exit 1
fi

cp "$CONFIG_PATH" "$BACKUP_PATH"
echo "[info] 已备份配置到 $BACKUP_PATH"

# ── 测试函数 ──────────────────────────────────────────────────────────────────
run_tests() {
    local mode=$1
    local url=$2
    echo ""
    echo "=========================================="
    echo "  测试配置: $mode"
    echo "=========================================="

    # 等待服务启动
    for i in {1..30}; do
        if curl -s "$url/health" >/dev/null 2>&1 || curl -s "$url/chat?prompt=test" >/dev/null 2>&1; then
            echo "[√] 服务已就绪"
            break
        fi
        echo "[wait] 等待服务启动... ($i/30)"
        sleep 2
    done

    echo ""
    echo "--- 1. 延迟基准 ---"
    python tests/perf/latency_benchmark.py --url "$url" --rounds 10 2>&1 | tee "logs/latency_$mode.log"

    echo ""
    echo "--- 2. 压力测试 ---"
    python tests/perf/stress_test.py --url "$url" --concurrency 5 --total 50 2>&1 | tee "logs/stress_$mode.log"

    echo ""
    echo "--- 3. 准确率评估 ---"
    python tests/accuracy/eval_chat.py --url "$url" 2>&1 | tee "logs/accuracy_$mode.log"
}

# ── 基线测试（QUANT_BITS=0）───────────────────────────────────────────────────
echo ""
echo "===== 阶段 1/2: 基线测试（无量化）====="
sed -i.tmp 's/^QUANT_BITS:.*/QUANT_BITS: 0/' "$CONFIG_PATH"
make stop 2>/dev/null || true
make api
sleep 3

run_tests "baseline" "http://localhost:8000"

# ── 量化测试（QUANT_BITS=4）───────────────────────────────────────────────────
echo ""
echo "===== 阶段 2/2: 量化测试（INT4）====="
sed -i.tmp 's/^QUANT_BITS:.*/QUANT_BITS: 4/' "$CONFIG_PATH"
make stop 2>/dev/null || true

# 检查量化模型是否存在
QUANT_MODEL=$(grep 'QUANT_MODEL_PATH:' "$CONFIG_PATH" | awk '{print $2}')
if [[ ! -d "$QUANT_MODEL" ]]; then
    echo "[warn] 量化模型不存在: $QUANT_MODEL"
    echo "[info] 运行量化脚本..."
    make quantize
fi

make api
sleep 3

run_tests "quant_int4" "http://localhost:8000"

# ── 恢复配置 ──────────────────────────────────────────────────────────────────
mv "$BACKUP_PATH" "$CONFIG_PATH"
make stop 2>/dev/null || true
rm -f "$CONFIG_PATH.tmp"

# ── 生成对比报告 ──────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  对比总结"
echo "=========================================="
echo ""
echo "延迟对比（Mean）："
grep "Mean :" logs/latency_baseline.log | tail -1 || echo "  baseline: N/A"
grep "Mean :" logs/latency_quant_int4.log | tail -1 || echo "  quant_int4: N/A"
echo ""
echo "QPS 对比："
grep "实际 QPS" logs/stress_baseline.log | tail -1 || echo "  baseline: N/A"
grep "实际 QPS" logs/stress_quant_int4.log | tail -1 || echo "  quant_int4: N/A"
echo ""
echo "准确率对比（ROUGE-L）："
grep "ROUGE-L" logs/accuracy_baseline.log | tail -1 || echo "  baseline: N/A"
grep "ROUGE-L" logs/accuracy_quant_int4.log | tail -1 || echo "  quant_int4: N/A"
echo ""
echo "详细日志见 logs/ 目录"
echo "配置已恢复到初始状态"
