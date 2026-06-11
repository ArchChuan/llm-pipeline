"""
单请求延迟基准测试
指标：E2E 延迟、tokens/s
用法：python tests/perf/latency_benchmark.py --url http://localhost:8000 --rounds 20
"""
import argparse
import time
import statistics
import httpx
import yaml

PROMPTS = [
    "用一句话介绍量子计算",
    "什么是机器学习？",
    "请解释 Python 的 GIL",
    "写一个冒泡排序的 Python 实现",
    "Linux 和 Windows 的主要区别是什么？",
]


def count_tokens_approx(text: str) -> int:
    # 粗估：中文约 1.5 字/token，英文约 4 char/token
    return max(1, len(text) // 2)


def single_request(client: httpx.Client, url: str, prompt: str) -> dict:
    t0 = time.perf_counter()
    resp = client.post(f"{url}/chat", params={"prompt": prompt}, timeout=120)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    response_text = data.get("response", "")
    output_tokens = count_tokens_approx(response_text)
    return {
        "latency_s": elapsed,
        "output_tokens": output_tokens,
        "tokens_per_s": output_tokens / elapsed if elapsed > 0 else 0,
    }


def run(url: str, rounds: int):
    results = []
    with httpx.Client() as client:
        # 先检查服务是否可用
        try:
            health = client.get(f"{url}/health", timeout=5)
            print(f"[health] {health.json()}")
        except Exception:
            print("[health] 无 /health 端点，跳过检查")

        for i in range(rounds):
            prompt = PROMPTS[i % len(PROMPTS)]
            try:
                r = single_request(client, url, prompt)
                results.append(r)
                print(
                    f"[{i+1:>3}/{rounds}] latency={r['latency_s']:.3f}s "
                    f"tokens={r['output_tokens']} tps={r['tokens_per_s']:.1f}"
                )
            except Exception as e:
                print(f"[{i+1:>3}/{rounds}] ERROR: {e}")

    if not results:
        print("无成功请求")
        return

    latencies = [r["latency_s"] for r in results]
    tps_list = [r["tokens_per_s"] for r in results]

    print("\n===== 延迟统计 =====")
    print(f"  P50  : {statistics.median(latencies)*1000:.0f} ms")
    print(f"  P95  : {sorted(latencies)[int(len(latencies)*0.95)]*1000:.0f} ms")
    print(f"  P99  : {sorted(latencies)[int(len(latencies)*0.99)]*1000:.0f} ms")
    print(f"  Mean : {statistics.mean(latencies)*1000:.0f} ms")
    print(f"  Min  : {min(latencies)*1000:.0f} ms")
    print(f"  Max  : {max(latencies)*1000:.0f} ms")
    print(f"\n===== 吞吐统计 =====")
    print(f"  Mean tokens/s : {statistics.mean(tps_list):.1f}")
    print(f"  Max  tokens/s : {max(tps_list):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()
    run(args.url, args.rounds)
