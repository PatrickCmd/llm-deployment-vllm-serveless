#!/usr/bin/env python3
"""
Task 4: PagedAttention - vLLM's Solution
Compare paged allocation vs contiguous allocation for KV cache.
"""

import os
import math

# Project root is the directory containing this script; markers and the
# other task scripts live beside it (no hard-coded /root paths).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MARKERS_DIR = os.path.join(PROJECT_ROOT, "markers")


def main():
    print("=" * 65)
    print("Task 4: PagedAttention - vLLM's Solution")
    print("=" * 65)

    # Same requests from Task 3
    requests = [
        {"id": 1, "prompt_tokens": 45,  "description": "Short question"},
        {"id": 2, "prompt_tokens": 128, "description": "Medium paragraph"},
        {"id": 3, "prompt_tokens": 23,  "description": "Quick greeting"},
        {"id": 4, "prompt_tokens": 256, "description": "Long document"},
        {"id": 5, "prompt_tokens": 67,  "description": "Code snippet"},
    ]

    max_seq_len = 512  # From Task 3

    #  Set the page size for paged allocation
    # Hint: A typical page holds 16 tokens (like OS 4KB pages)
    page_size = 16  # Set to 16

    print(f"\nPage size: {page_size} tokens per page")
    print(f"Contiguous allocation: {max_seq_len} tokens per request (worst-case)")

    # --- PAGED ALLOCATION ---
    print("\n--- PAGED ALLOCATION (like vLLM's PagedAttention) ---\n")

    total_paged_allocated = 0
    total_contiguous_allocated = 0
    total_used = 0

    for req in requests:
        actual = req["prompt_tokens"]
        total_used += actual

        # Contiguous: worst-case allocation
        contiguous_alloc = max_seq_len
        total_contiguous_allocated += contiguous_alloc

        #  Calculate how many pages are needed
        # Hint: Round up to nearest page using math.ceil
        pages_needed = math.ceil(actual / page_size)  # Set to page_size

        paged_alloc = pages_needed * page_size
        total_paged_allocated += paged_alloc

        paged_waste = (paged_alloc - actual) / paged_alloc * 100 if paged_alloc > 0 else 0

        # Visual: show pages
        page_blocks = "|".join(["##" if i < pages_needed else ".." for i in range(pages_needed)])

        print(f"  Request {req['id']}: {actual} tokens -> {pages_needed} pages ({paged_alloc} slots)")
        print(f"    Pages: [{page_blocks}]  waste: {paged_waste:.1f}%")

    #  Calculate paged memory utilization
    # Hint: Divide total used by total paged allocated
    paged_utilization = total_used / total_paged_allocated * 100  # Set to total_paged_allocated
    contiguous_utilization = total_used / total_contiguous_allocated * 100

    # --- SIDE-BY-SIDE COMPARISON ---
    print(f"\n--- SIDE-BY-SIDE COMPARISON ---")
    print(f"{'Method':<14} {'Total Allocated':>16} {'Total Used':>12} {'Utilization':>13}")
    print("-" * 57)
    print(f"{'Contiguous':<14} {total_contiguous_allocated:>12} slots {total_used:>8} slots {contiguous_utilization:>12.1f}%")
    print(f"{'Paged':<14} {total_paged_allocated:>12} slots {total_used:>8} slots {paged_utilization:>12.1f}%")

    memory_saved = total_contiguous_allocated - total_paged_allocated
    savings_ratio = total_contiguous_allocated / total_paged_allocated if total_paged_allocated > 0 else 0

    print(f"\nMemory saved: {memory_saved} slots ({savings_ratio:.1f}x less memory)")

    # --- CONCURRENT USER IMPACT ---
    hypothetical_memory = 10000
    max_users_contiguous = hypothetical_memory // max_seq_len
    avg_paged = total_paged_allocated // len(requests)
    max_users_paged = hypothetical_memory // avg_paged if avg_paged > 0 else 0

    print(f"\n--- CONCURRENT USER IMPACT ---")
    print(f"  With {hypothetical_memory} total memory slots:")
    print(f"  - Contiguous: {max_users_contiguous} concurrent users")
    print(f"  - Paged:      {max_users_paged} concurrent users")
    print(f"  - Improvement: {max_users_paged / max_users_contiguous:.1f}x more users!")

    # --- OS PAGING ANALOGY ---
    print(f"\n--- OS PAGING ANALOGY ---")
    print(f"  Contiguous = reserving an entire row of seats for each person")
    print(f"  Paged      = giving seats one at a time as people sit down")
    print(f"")
    print(f"  Just like OS virtual memory:")
    print(f"  - OS divides RAM into fixed-size pages (typically 4KB)")
    print(f"  - Processes get pages on demand, not large contiguous blocks")
    print(f"  - vLLM does the same for KV cache during LLM inference")

    # --- KEY INSIGHT ---
    print("\n" + "=" * 65)
    print("KEY INSIGHT:")
    print("- PagedAttention uses small pages (like OS virtual memory)")
    print("- No worst-case pre-allocation needed")
    print(f"- Memory utilization: {contiguous_utilization:.0f}% -> {paged_utilization:.0f}%")
    print("- This frees memory to serve MORE concurrent users")
    print("- Next: Let's use this in practice with vLLM's API server (Task 5)")
    print("=" * 65)

    # Create marker
    os.makedirs(MARKERS_DIR, exist_ok=True)
    with open(os.path.join(MARKERS_DIR, "task4_complete.txt"), "w") as f:
        f.write("TASK_4_COMPLETE\n")

    print("\nTask 4 Complete!")
    print(f"Next: python {os.path.join(PROJECT_ROOT, 'task_5_api_server.py')}")


if __name__ == "__main__":
    main()
