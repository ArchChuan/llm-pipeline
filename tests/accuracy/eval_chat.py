"""
业务准确率评估
指标：ROUGE-L、BLEU、Exact Match
测试集：data/eval_data.json（格式见下方 EVAL_DATA_SCHEMA）
用法：python tests/accuracy/eval_chat.py --url http://localhost:8000 --data data/eval_data.json

eval_data.json 格式：
[
    {"input": "你好", "expected": "你好！我是你的AI助手"},
    {"input": "介绍Linux", "expected": "Linux是开源服务器操作系统..."}
]
"""
import argparse
import json
import math
import re
import httpx


# ── 内置 fallback 测试集（data/eval_data.json 不存在时使用）──────────────────
BUILTIN_EVAL = [
    {
        "input": "用一句话介绍机器学习",
        "expected": "机器学习是让计算机通过数据自动学习规律的技术",
        "type": "factual",
    },
    {
        "input": "Python 列表推导式怎么写？",
        "expected": "[x for x in range(10)]",
        "type": "code",
    },
    {
        "input": "1+1等于多少",
        "expected": "2",
        "type": "exact",
    },
    {
        "input": "HTTP 和 HTTPS 的区别",
        "expected": "HTTPS 在 HTTP 基础上加了 TLS/SSL 加密，保证传输安全",
        "type": "factual",
    },
    {
        "input": "什么是 Docker？",
        "expected": "Docker 是容器化平台，用于打包和运行应用",
        "type": "factual",
    },
]


# ── ROUGE-L ──────────────────────────────────────────────────────────────────
def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(pred: str, ref: str) -> float:
    p_tokens = list(pred.strip())
    r_tokens = list(ref.strip())
    if not p_tokens or not r_tokens:
        return 0.0
    lcs = _lcs_length(p_tokens, r_tokens)
    precision = lcs / len(p_tokens)
    recall = lcs / len(r_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── BLEU-1 ───────────────────────────────────────────────────────────────────
def bleu1(pred: str, ref: str) -> float:
    p_tokens = list(pred.strip())
    r_tokens = set(ref.strip())
    if not p_tokens:
        return 0.0
    matches = sum(1 for t in p_tokens if t in r_tokens)
    bp = 1.0 if len(p_tokens) >= len(list(ref.strip())) else math.exp(1 - len(list(ref.strip())) / len(p_tokens))
    return bp * (matches / len(p_tokens))


# ── Exact Match ───────────────────────────────────────────────────────────────
def exact_match(pred: str, ref: str) -> float:
    def normalize(s):
        return re.sub(r"\s+", "", s.strip().lower())
    return 1.0 if normalize(pred) == normalize(ref) else 0.0


# ── 主流程 ───────────────────────────────────────────────────────────────────
def get_response(client: httpx.Client, url: str, prompt: str) -> str:
    resp = client.post(f"{url}/chat", params={"prompt": prompt}, timeout=120)
    resp.raise_for_status()
    text = resp.json().get("response", "")
    # 去掉 prompt 复读部分（某些模型会把 prompt 回显）
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text


def run(url: str, data_path: str | None):
    if data_path:
        with open(data_path, encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        dataset = BUILTIN_EVAL
        print("[info] 使用内置测试集，如需自定义请提供 --data 参数\n")

    rouge_scores, bleu_scores, em_scores = [], [], []

    print(f"{'#':>3}  {'ROUGE-L':>8} {'BLEU-1':>8} {'EM':>4}  input / pred / expected")
    print("-" * 80)

    with httpx.Client() as client:
        for i, item in enumerate(dataset):
            inp = item["input"]
            exp = item["expected"]
            try:
                pred = get_response(client, url, inp)
            except Exception as e:
                print(f"{i+1:>3}  ERROR: {e}")
                continue

            rl = rouge_l(pred, exp)
            b1 = bleu1(pred, exp)
            em = exact_match(pred, exp)
            rouge_scores.append(rl)
            bleu_scores.append(b1)
            em_scores.append(em)

            pred_display = pred[:40].replace("\n", " ")
            exp_display = exp[:40].replace("\n", " ")
            print(
                f"{i+1:>3}  {rl:>8.3f} {b1:>8.3f} {em:>4.0f}"
                f"  [{inp[:20]}] → [{pred_display}…] (期望: {exp_display}…)"
            )

    if not rouge_scores:
        print("无成功评估结果")
        return

    n = len(rouge_scores)
    print(f"\n===== 准确率汇总（n={n}）=====")
    print(f"  ROUGE-L     : {sum(rouge_scores)/n:.4f}")
    print(f"  BLEU-1      : {sum(bleu_scores)/n:.4f}")
    print(f"  Exact Match : {sum(em_scores)/n:.4f}  ({int(sum(em_scores))}/{n})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--data", default=None, help="测试集 JSON 路径")
    args = parser.parse_args()
    run(args.url, args.data)
