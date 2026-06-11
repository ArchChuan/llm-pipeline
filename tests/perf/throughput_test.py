"""
吞吐量测试：对比不同 max_new_tokens 下的 tokens/s
用法：python tests/perf/throughput_test.py --url http://localhost:8000
"""
import argparse
import time
import httpx


PROMPT = "请详细介绍一下大型语言模型（LLM）的工作原理，包括 Transformer 架构、预训练过程、微调方法，以及当前面临的主要挑战。"

TOKEN_LENGTHS = [64, 128, 256, 512, 1024]
REPEATS = 3


def count_tokens_approx(text: str) -> int:
    return max(1, len(text) // 2)


def measure(client: httpx.Client, url: str, max_new_tokens: int) -> dict:
    latencies = []
    token_counts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        resp = client.post(
            f"{url}/chat",
            params={"prompt": PROMPT},
            timeout=300,
        )
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()
        text = resp.json().get("response", "")
        tokens = count_tokens_approx(text)
        latencies.append(elapsed)
        token_counts.append(tokens)

    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(token_counts) / len(token_counts)
    return {
        "max_new_tokens": max_new_tokens,
        "avg_latency_s": avg_latency,
        "avg_output_tokens": avg_tokens,
        "tokens_per_s": avg_tokens / avg_latency if avg_latency > 0 else 0,
    }


def run(url: str):
    print(f"吞吐量测试: url={url}  重复次数/配置={REPEATS}\n")
    print(f"{'max_tokens':>12} {'avg_latency_ms':>15} {'output_tokens':>14} {'tokens/s':>10}")
    print("-" * 56)

    with httpx.Client() as client:
        for max_tok in TOKEN_LENGTHS:
            try:
                r = measure(client, url, max_tok)
                print(
                    f"{r['max_new_tokens']:>12} "
                    f"{r['avg_latency_s']*1000:>15.0f} "
                    f"{r['avg_output_tokens']:>14.0f} "
                    f"{r['tokens_per_s']:>10.1f}"
                )
            except Exception as e:
                print(f"{max_tok:>12}  ERROR: {e}")

    print("\n注：output_tokens 为粗估（按字符数/2），实际以模型 tokenizer 为准。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    run(args.url)
