#!/usr/bin/env python3
"""
Task 2: vLLM Offline Inference - See the Difference
Compare vLLM inference speed against the HuggingFace baseline.
"""

import os
import time

# Project root is the directory containing this script; markers and the
# other task scripts live beside it (no hard-coded /root paths).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MARKERS_DIR = os.path.join(PROJECT_ROOT, "markers")

# Configure vLLM for CPU-only execution.
# The lab VM has a 4GB memory limit, so run the engine in-process
# (VLLM_ENABLE_V1_MULTIPROCESSING=0): the default multi-process mode
# spawns 2 extra Python processes that each cost over 1GB.
# VLLM_CPU_KVCACHE_SPACE is deliberately unset - it would override the
# kv_cache_memory_bytes argument and only accepts whole GiB values.
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.pop("VLLM_CPU_KVCACHE_SPACE", None)
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# 128MB KV cache - holds ~43 full-length sequences for SmolLM-135M at
# max_model_len=128 (~22.5KB KV per token). Kept small because vLLM's
# startup check counts the pod's page cache (model downloads) as used
# memory, leaving only a few hundred MB "available" on the 4GB VM.
KV_CACHE_BYTES = 128 * 1024 * 1024


def main():
    print("=" * 65)
    print("Task 2: vLLM Offline Inference - See the Difference")
    print("=" * 65)

    from vllm import LLM, SamplingParams

    model_name = "HuggingFaceTB/SmolLM-135M"
    prompt = "Explain what a large language model is in simple terms."

    print(f"\nModel: {model_name}")
    print(f"Prompt: \"{prompt}\"")
    print("-" * 65)

    # --- INITIALIZE vLLM ---
    print("\nInitializing vLLM engine...")

    # Initialize the vLLM engine
    # Hint: Pass the model_name variable to the LLM constructor
    # Note: enforce_eager=True skips torch.compile to save memory on CPU
    llm = LLM(model=model_name, max_model_len=128, enforce_eager=True,
              kv_cache_memory_bytes=KV_CACHE_BYTES)  # Set to model_name

    # Create SamplingParams for generation
    # Hint: Set temperature and max_tokens for text generation
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)  # Set to 0.7 and 50

    print("vLLM engine ready.")

    # --- GENERATE ---
    print("\nGenerating with vLLM...")
    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    end_time = time.time()

    # Extract results
    generated_text = outputs[0].outputs[0].text
    generated_tokens = len(outputs[0].outputs[0].token_ids)
    total_time = end_time - start_time
    tokens_per_second = generated_tokens / total_time

    # --- vLLM RESULTS ---
    print("\n--- vLLM RESULTS ---")
    print(f"Generated text: {generated_text[:200]}...")
    print(f"\nGenerated tokens: {generated_tokens}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.1f} tok/s")

    # --- COMPARISON ---
    print("\n--- COMPARISON: HuggingFace vs vLLM ---")
    hf_tps = None
    hf_time = None
    baseline_file = os.path.join(MARKERS_DIR, "hf_baseline.txt")
    if os.path.exists(baseline_file):
        with open(baseline_file, "r") as f:
            for line in f:
                key, value = line.strip().split("=")
                if key == "tokens_per_second":
                    hf_tps = float(value)
                elif key == "total_time":
                    hf_time = float(value)

    if hf_tps:
        print(f"{'Metric':<20} {'HuggingFace':>12} {'vLLM':>12}")
        print("-" * 46)
        print(f"{'Tokens/sec':<20} {hf_tps:>12.1f} {tokens_per_second:>12.1f}")
        print(f"{'Total time':<20} {hf_time:>11.2f}s {total_time:>11.2f}s")
        if tokens_per_second > hf_tps:
            speedup = tokens_per_second / hf_tps
            print(f"\nvLLM is {speedup:.1f}x faster in tokens/sec")
        else:
            print("\nNote: For single requests, results may be similar.")
            print("The real advantage shows under concurrent load (Task 6).")
    else:
        print("(HuggingFace baseline not found - run Task 1 first)")

    # Save vLLM metrics for later comparison
    os.makedirs(MARKERS_DIR, exist_ok=True)
    with open(os.path.join(MARKERS_DIR, "vllm_baseline.txt"), "w") as f:
        f.write(f"tokens_per_second={tokens_per_second:.2f}\n")
        f.write(f"total_time={total_time:.4f}\n")
        f.write(f"generated_tokens={generated_tokens}\n")

    # --- KEY INSIGHT ---
    print("\n" + "=" * 65)
    print("KEY INSIGHT:")
    print("- vLLM optimizes inference even for single requests")
    print("- The REAL advantage is under concurrent load (Task 6)")
    print("- vLLM handles batching natively - no manual queue management")
    print("- Before that, let's understand WHY vLLM is faster (Tasks 3-4)")
    print("=" * 65)

    # Create marker
    with open(os.path.join(MARKERS_DIR, "task2_complete.txt"), "w") as f:
        f.write("TASK_2_COMPLETE\n")

    print("\nTask 2 Complete!")
    print(f"Next: python {os.path.join(PROJECT_ROOT, 'task_3_kv_cache_problem.py')}")

    # Clean up
    del llm


if __name__ == "__main__":
    main()
