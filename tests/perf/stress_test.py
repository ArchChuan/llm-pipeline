"""
并发压力测试
指标：QPS、P50/P95/P99 延迟、错误率
用法：python tests/perf/stress_test.py --url http://localhost:8000 --concurrency 10 --total 100
"""
import argparse
import asyncio
import time
import statistics
import httpx


PROMPTS = [
    "用一句话介绍量子计算",
    "什么是机器学习？",
    "请解释 Python 的 GIL",
    "写一个冒泡排序的 Python 实现",
    "Linux 和 Windows 的主要区别是什么？",
    "解释一下 TCP 三次握手",
    "什么是 Docker？",
    "简述 REST API 的设计原则",
]

results: list[dict] = []
errors: list[str] = []


async def single_request(
    client: httpx.AsyncClient, url: str, prompt: str, idx: int
):
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{url}/chat", params={"prompt": prompt}, timeout=120
        )
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()
        results.append({"latency_s": elapsed, "status": resp.status_code})
    except httpx.TimeoutException:
        elapsed = time.perf_counter() - t0
        errors.append(f"[{idx}] timeout after {elapsed:.1f}s")
        results.append({"latency_s": elapsed, "status": 408})
    except Exception as e:
        elapsed = time.perf_counter() - t0
        errors.append(f"[{idx}] {type(e).__name__}: {e}")
        results.append({"latency_s": elapsed, "status": 0})


async def worker(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    url: str,
    tasks: list[tuple[int, str]],
):
    for idx, prompt in tasks:
        async with sem:
            await single_request(client, url, prompt, idx)


async def run(url: str, concurrency: int, total: int):
    sem = asyncio.Semaphore(concurrency)
    task_list = [(i, PROMPTS[i % len(PROMPTS)]) for i in range(total)]

    print(f"压测配置: url={url} concurrency={concurrency} total={total}")
    wall_start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(
                worker(sem, client, url, [task_list[i]])
            )
            for i in range(total)
        ]
        await asyncio.gather(*tasks)

    wall_time = time.perf_counter() - wall_start

    success = [r for r in results if 200 <= r["status"] < 300]
    failed = [r for r in results if r["status"] < 200 or r["status"] >= 300]
    latencies = sorted([r["latency_s"] for r in success])

    print(f"\n===== 压测结果 =====")
    print(f"  总请求数  : {total}")
    print(f"  成功      : {len(success)}")
    print(f"  失败      : {len(failed)}")
    print(f"  错误率    : {len(failed)/total*100:.1f}%")
    print(f"  总耗时    : {wall_time:.2f}s")
    print(f"  实际 QPS  : {total/wall_time:.2f}")

    if latencies:
        def pct(p):
            idx = min(int(len(latencies) * p), len(latencies) - 1)
            return latencies[idx] * 1000

        print(f"\n===== 延迟分布（成功请求）=====")
        print(f"  P50  : {pct(0.50):.0f} ms")
        print(f"  P90  : {pct(0.90):.0f} ms")
        print(f"  P95  : {pct(0.95):.0f} ms")
        print(f"  P99  : {pct(0.99):.0f} ms")
        print(f"  Mean : {statistics.mean(latencies)*1000:.0f} ms")
        print(f"  Max  : {max(latencies)*1000:.0f} ms")

    if errors:
        print(f"\n===== 错误样本（前 10 条）=====")
        for e in errors[:10]:
            print(f"  {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数")
    parser.add_argument("--total", type=int, default=100, help="总请求数")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.concurrency, args.total))
