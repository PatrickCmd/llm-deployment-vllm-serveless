#!/usr/bin/env python3
"""
Task 1: Naive HuggingFace Inference - The Baseline
Measure baseline inference speed using raw HuggingFace transformers.
"""

import os
import time

# Project root is the directory containing this script; markers and the
# other task scripts live beside it (no hard-coded /root paths).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MARKERS_DIR = os.path.join(PROJECT_ROOT, "markers")


def main():
    print("=" * 65)
    print("Task 1: Naive HuggingFace Inference - The Baseline")
    print("=" * 65)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "HuggingFaceTB/SmolLM-135M"
    prompt = "Explain what a large language model is in simple terms."

    print(f"\nModel: {model_name}")
    print(f"Prompt: \"{prompt}\"")
    print("-" * 65)

    # --- LOAD MODEL ---
    print("\nLoading model with HuggingFace transformers...")

    # Load the model
    # Hint: Use the model_name variable ("HuggingFaceTB/SmolLM-135M")
    model = AutoModelForCausalLM.from_pretrained(model_name)  # Set to model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully.")

    # --- GENERATE ---
    print("\nGenerating with HuggingFace transformers...")
    inputs = tokenizer(prompt, return_tensors="pt")

    # Set the max_new_tokens for generation
    # Hint: Controls how many tokens the model generates
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,  # Set to 50
        do_sample=True,
        temperature=0.7,
    )
    end_time = time.time()

    # Calculate metrics
    input_tokens = inputs["input_ids"].shape[1]
    total_tokens = outputs.shape[1]
    generated_tokens = total_tokens - input_tokens
    total_time = end_time - start_time
    tokens_per_second = generated_tokens / total_time

    # Decode output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- RESULTS ---
    print("\n--- RESULTS ---")
    print(f"Generated text: {output_text[:200]}...")
    print(f"\nGenerated tokens: {generated_tokens}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.1f} tok/s")

    # Save baseline for later comparison
    baseline_file = os.path.join(MARKERS_DIR, "hf_baseline.txt")
    os.makedirs(MARKERS_DIR, exist_ok=True)
    with open(baseline_file, "w") as f:
        f.write(f"tokens_per_second={tokens_per_second:.2f}\n")
        f.write(f"total_time={total_time:.4f}\n")
        f.write(f"generated_tokens={generated_tokens}\n")

    # --- KEY INSIGHT ---
    print("\n" + "=" * 65)
    print("KEY INSIGHT:")
    print("- This is SINGLE-REQUEST performance")
    print("- There is no batching - one request at a time")
    print("- Under load with multiple users, requests would queue up")
    print("- Next: See how vLLM improves this (Task 2)")
    print("=" * 65)

    # Create marker file
    with open(os.path.join(MARKERS_DIR, "task1_complete.txt"), "w") as f:
        f.write("TASK_1_COMPLETE\n")

    print("\nTask 1 Complete!")
    print(f"Next: python {os.path.join(PROJECT_ROOT, 'task_2_vllm_inference.py')}")

    # Clean up
    del model
    del tokenizer


if __name__ == "__main__":
    main()
