#!/usr/bin/env python3
"""
Task 6: Multi-User Throughput Under Load
Stress-test the vLLM server with concurrent requests.
"""

import os
import sys
import time
import json
import asyncio

# Project root is the directory containing this script; markers and the
# other task scripts live beside it (no hard-coded /root paths).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MARKERS_DIR = os.path.join(PROJECT_ROOT, "markers")


async def send_request(session, url, model, prompt, max_tokens=50):
    """Send a single completion request and return timing info."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    start = time.time()
    try:
        async with session.post(
            f"{url}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            data = await resp.json()
            end = time.time()
            if resp.status != 200:
                return {
                    "latency": end - start,
                    "tokens": 0,
                    "success": False,
                    "error": f"HTTP {resp.status}: {str(data)[:150]}",
                }
            completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
            return {
                "latency": end - start,
                "tokens": completion_tokens,
                "success": True,
            }
    except Exception as e:
        return {"latency": time.time() - start, "tokens": 0, "success": False, "error": str(e)}


async def run_load_test(url, model, prompts, num_concurrent):
    """Run a load test with the given number of concurrent users."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_concurrent):
            prompt = prompts[i % len(prompts)]
            tasks.append(send_request(session, url, model, prompt))

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

    return results, total_time


def main():
    print("=" * 65)
    print("Task 6: Multi-User Throughput Under Load")
    print("=" * 65)

    model_name = "HuggingFaceTB/SmolLM-135M"
    server_url = "http://localhost:8000"

    # Verify server is running
    import requests
    try:
        resp = requests.get(f"{server_url}/health")
        if resp.status_code != 200:
            raise Exception("Server not healthy")
    except Exception:
        print("\nERROR: vLLM server is not running on port 8000.")
        print(f"Run Task 5 first: python {os.path.join(PROJECT_ROOT, 'task_5_api_server.py')}")
        return

    print(f"\nServer: {server_url}")
    print(f"Model: {model_name}")
    print("-" * 65)

    # Diverse prompts to simulate real users
    prompts = [
        "What is machine learning?",
        "Explain neural networks briefly.",
        "How does a transformer model work?",
        "What is natural language processing?",
        "Describe deep learning in one paragraph.",
        "What are tokens in the context of LLMs?",
        "How is AI used in healthcare?",
        "What is the difference between AI and ML?",
        "Explain what fine-tuning means.",
        "What is transfer learning?",
    ]

    # Create the list of concurrent user counts to test
    # Hint: Start small and increase to see how throughput scales
    concurrent_users = [1, 5, 10, 20]  # Set to [1, 5, 10, 20]

    print(f"\nLoad test plan: {concurrent_users} concurrent users")
    print(f"Each user sends 1 request with max_tokens=50\n")

    results_table = []

    for num_users in concurrent_users:
        print(f"  Testing with {num_users} concurrent user(s)...", end=" ")

        test_results, total_time = asyncio.run(
            run_load_test(server_url, model_name, prompts, num_users)
        )

        successful = [r for r in test_results if r["success"]]
        total_tokens = sum(r["tokens"] for r in successful)
        avg_latency = sum(r["latency"] for r in successful) / len(successful) if successful else 0

        #  Calculate aggregate throughput
        # Hint: Divide total tokens by total time
        throughput = total_tokens / total_time if total_time > 0 else 0

        results_table.append({
            "users": num_users,
            "total_tokens": total_tokens,
            "total_time": total_time,
            "throughput": throughput,
            "avg_latency": avg_latency,
            "success_rate": len(successful) / len(test_results) * 100,
        })

        print(f"done ({throughput:.1f} tok/s, {avg_latency:.2f}s avg latency)")

        failed = [r for r in test_results if not r["success"]]
        if failed:
            print(f"    WARNING: {len(failed)} request(s) failed.")
            print(f"    First error: {failed[0].get('error', 'unknown')}")

    # --- RESULTS TABLE ---
    print(f"\n--- LOAD TEST RESULTS ---")
    print(f"{'Users':>6} {'Total Tokens':>13} {'Time (s)':>9} {'Throughput':>12} {'Avg Latency':>12} {'Success':>8}")
    print("-" * 66)
    for r in results_table:
        print(
            f"{r['users']:>6} "
            f"{r['total_tokens']:>13} "
            f"{r['total_time']:>8.2f}s "
            f"{r['throughput']:>9.1f} tok/s "
            f"{r['avg_latency']:>10.2f}s "
            f"{r['success_rate']:>7.0f}%"
        )

    # --- SCALING ANALYSIS ---
    if len(results_table) >= 2:
        baseline = results_table[0]
        peak = max(results_table, key=lambda r: r["throughput"])
        scaling = peak["throughput"] / baseline["throughput"] if baseline["throughput"] > 0 else 0

        print(f"\n--- SCALING ANALYSIS ---")
        print(f"  Baseline (1 user): {baseline['throughput']:.1f} tok/s")
        print(f"  Peak ({peak['users']} users): {peak['throughput']:.1f} tok/s")
        print(f"  Scaling factor: {scaling:.1f}x throughput improvement")

    # Save results for dashboard
    os.makedirs(MARKERS_DIR, exist_ok=True)
    with open(os.path.join(MARKERS_DIR, "load_test_results.json"), "w") as f:
        json.dump(results_table, f, indent=2)

    # --- KEY INSIGHT ---
    print("\n" + "=" * 65)
    print("KEY INSIGHT:")
    print("- Throughput SCALES with concurrent users")
    print("- vLLM uses continuous batching - does not wait for batch to fill")
    print("- PagedAttention allows efficient KV cache sharing across requests")
    print("- Per-request latency increases but total throughput improves")
    print("- This is the core value of vLLM: high-throughput multi-user serving")
    print("=" * 65)

    # Create marker
    with open(os.path.join(MARKERS_DIR, "task6_complete.txt"), "w") as f:
        f.write("TASK_6_COMPLETE\n")

    print("\nTask 6 Complete!")
    print(f"Next: python {os.path.join(PROJECT_ROOT, 'task_7_tuning.py')}")


if __name__ == "__main__":
    main()
