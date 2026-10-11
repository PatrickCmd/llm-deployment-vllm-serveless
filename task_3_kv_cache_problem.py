#!/usr/bin/env python3
"""
Task 3: The KV Cache Problem - Why Memory Matters
Simulate KV cache fragmentation with contiguous memory allocation.
"""

import os

# Project root is the directory containing this script; markers and the
# other task scripts live beside it (no hard-coded /root paths).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MARKERS_DIR = os.path.join(PROJECT_ROOT, "markers")


def main():
    print("=" * 65)
    print("Task 3: The KV Cache Problem - Why Memory Matters")
    print("=" * 65)

    # Simulated requests with different prompt lengths
    requests = [
        {"id": 1, "prompt_tokens": 45,  "description": "Short question"},
        {"id": 2, "prompt_tokens": 128, "description": "Medium paragraph"},
        {"id": 3, "prompt_tokens": 23,  "description": "Quick greeting"},
        {"id": 4, "prompt_tokens": 256, "description": "Long document"},
        {"id": 5, "prompt_tokens": 67,  "description": "Code snippet"},
    ]

    # Set the maximum sequence length for worst-case allocation
    # Hint: Traditional systems use values like 512, 2048, or 4096
    max_seq_len = 512  # Set to 512

    print(f"\nMax sequence length (pre-allocated per request): {max_seq_len}")
    print(f"Number of concurrent requests: {len(requests)}")

    # --- SIMULATE CONTIGUOUS ALLOCATION ---
    print("\n--- SIMULATING CONTIGUOUS ALLOCATION ---")
    print(f"(Each request gets {max_seq_len} token slots, regardless of actual usage)\n")

    total_allocated = 0
    total_used = 0

    for req in requests:
        actual = req["prompt_tokens"]
        allocated = max_seq_len
        total_allocated += allocated
        total_used += actual

        # Create visual bar
        bar_width = 50
        used_chars = int((actual / allocated) * bar_width)
        wasted_chars = bar_width - used_chars
        bar = "#" * used_chars + "." * wasted_chars

        #  Calculate the wasted memory percentage
        # Hint: Subtract actual from allocated, then divide by allocated
        wasted_pct = (allocated - actual) / allocated * 100  # Set to (allocated - actual) / allocated * 100

        print(f"  Request {req['id']} ({req['description']}):")
        print(f"    [{bar}] {actual}/{allocated} used ({wasted_pct:.1f}% wasted)")

    # --- SUMMARY ---
    overall_utilization = total_used / total_allocated * 100
    overall_waste = 100 - overall_utilization

    print(f"\n--- SUMMARY ---")
    print(f"Total allocated: {total_allocated} token slots")
    print(f"Total actually used: {total_used} token slots")
    print(f"Memory utilization: {overall_utilization:.1f}%")
    print(f"Overall waste: {overall_waste:.1f}%")

    # Visual comparison
    print(f"\n--- WHY THIS IS A PROBLEM ---")
    print(f"  With {max_seq_len}-token pre-allocation:")
    print(f"  - You allocated {total_allocated} slots but only used {total_used}")
    print(f"  - {overall_waste:.1f}% of your memory is WASTED")
    print(f"  - That wasted memory could serve MORE users")
    if overall_waste > 60:
        print(f"  - This matches vLLM's finding: 60-80% memory waste in traditional systems")

    # Max users comparison
    hypothetical_memory = 10000  # Assume 10000 token slots of total memory
    max_users_contiguous = hypothetical_memory // max_seq_len
    avg_actual = total_used // len(requests)
    max_users_ideal = hypothetical_memory // avg_actual

    print(f"\n--- CONCURRENT USER IMPACT ---")
    print(f"  With {hypothetical_memory} total memory slots:")
    print(f"  - Contiguous allocation: {max_users_contiguous} concurrent users max")
    print(f"  - Ideal (no waste): {max_users_ideal} concurrent users max")
    print(f"  - You are serving {max_users_contiguous}x fewer users than possible!")

    # --- KEY INSIGHT ---
    print("\n" + "=" * 65)
    print("KEY INSIGHT:")
    print("- Traditional systems pre-allocate WORST-CASE memory per request")
    print("- Short prompts waste massive amounts of memory")
    print("- This limits how many concurrent requests you can serve")
    print("- This is the EXACT problem vLLM's PagedAttention solves (Task 4)")
    print("=" * 65)

    # Create marker
    os.makedirs(MARKERS_DIR, exist_ok=True)
    with open(os.path.join(MARKERS_DIR, "task3_complete.txt"), "w") as f:
        f.write("TASK_3_COMPLETE\n")

    print("\nTask 3 Complete!")
    print(f"Next: python {os.path.join(PROJECT_ROOT, 'task_4_paged_attention.py')}")


if __name__ == "__main__":
    main()
