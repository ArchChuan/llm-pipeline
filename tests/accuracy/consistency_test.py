"""
输出一致性测试：同一 prompt 重复 N 次，测量输出相似度
指标：字符级 Jaccard 相似度、完全一致率
用法：python tests/accuracy/consistency_test.py --url http://localhost:8000 --repeats 5
"""
import argparse
import httpx


PROMPTS = [
    "用一句话介绍机器学习",
    "Python 的 GIL 是什么？",
    "Docker 和虚拟机的区别",
]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.strip()), set(b.strip())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def get_response(client: httpx.Client, url: str, prompt: str) -> str:
    resp = client.post(f"{url}/chat", params={"prompt": prompt}, timeout=120)
    resp.raise_for_status()
    text = resp.json().get("response", "")
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text


def run(url: str, repeats: int):
    print(f"一致性测试: url={url}  repeats={repeats}\n")

    with httpx.Client() as client:
        for prompt in PROMPTS:
            print(f"Prompt: {prompt}")
            responses = []
            for i in range(repeats):
                try:
                    r = get_response(client, url, prompt)
                    responses.append(r)
                    print(f"  [{i+1}] {r[:60].replace(chr(10), ' ')}…")
                except Exception as e:
                    print(f"  [{i+1}] ERROR: {e}")

            if len(responses) < 2:
                print("  结果不足，跳过统计\n")
                continue

            # 所有 pair 的 Jaccard
            scores = []
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    scores.append(jaccard(responses[i], responses[j]))

            exact = sum(1 for i in range(1, len(responses)) if responses[i] == responses[0])
            print(f"  → Jaccard 均值: {sum(scores)/len(scores):.3f}  完全一致: {exact}/{repeats-1}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    run(args.url, args.repeats)
