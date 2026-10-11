#!/usr/bin/env python3
"""
Task 8: Production Monitoring Dashboard (Capstone)
Build a live Gradio dashboard to monitor vLLM inference metrics.
"""

import os
import sys
import json
import time
import asyncio
import threading

# Project root is the directory containing this script; markers and the
# other task scripts live beside it (no hard-coded /root paths).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MARKERS_DIR = os.path.join(PROJECT_ROOT, "markers")

# Configure vLLM for CPU-only execution (in case server needs restart)
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.pop("VLLM_CPU_KVCACHE_SPACE", None)
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Disable Gradio analytics to avoid CORS errors behind reverse proxy
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


def main():
    print("=" * 65)
    print("Task 8: Production Monitoring Dashboard (Capstone)")
    print("=" * 65)

    import gradio as gr
    import requests

    server_url = "http://localhost:8000"
    model_name = "HuggingFaceTB/SmolLM-135M"

    # Verify server is running
    try:
        resp = requests.get(f"{server_url}/health")
        if resp.status_code != 200:
            raise Exception("Server not healthy")
    except Exception:
        print("\nERROR: vLLM server is not running on port 8000.")
        print(f"Run Task 5 first: python {os.path.join(PROJECT_ROOT, 'task_5_api_server.py')}")
        return

    print(f"\nServer: {server_url} (running)")
    print(f"Dashboard will be available at: http://localhost:7860")
    print("-" * 65)

    # Load previous results
    hf_baseline = {}
    if os.path.exists(os.path.join(MARKERS_DIR, "hf_baseline.txt")):
        with open(os.path.join(MARKERS_DIR, "hf_baseline.txt"), "r") as f:
            for line in f:
                key, value = line.strip().split("=")
                hf_baseline[key] = float(value)

    vllm_baseline = {}
    if os.path.exists(os.path.join(MARKERS_DIR, "vllm_baseline.txt")):
        with open(os.path.join(MARKERS_DIR, "vllm_baseline.txt"), "r") as f:
            for line in f:
                key, value = line.strip().split("=")
                vllm_baseline[key] = float(value)

    load_test_results = []
    if os.path.exists(os.path.join(MARKERS_DIR, "load_test_results.json")):
        with open(os.path.join(MARKERS_DIR, "load_test_results.json"), "r") as f:
            load_test_results = json.load(f)

    tuning_results = []
    if os.path.exists(os.path.join(MARKERS_DIR, "tuning_results.json")):
        with open(os.path.join(MARKERS_DIR, "tuning_results.json"), "r") as f:
            tuning_results = json.load(f)

    # Send a test request to the vLLM server
    # Hint: Use the requests library to send a POST request
    def get_live_metrics():
        """Send a test request and return latency and token count."""
        try:
            payload = {
                "model": model_name,
                "prompt": "Hello, how are you?",
                "max_tokens": 20,
                "temperature": 0.7,
            }
            start = time.time()
            resp = requests.post(  # Set to requests.post
                f"{server_url}/v1/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            latency = time.time() - start
            data = resp.json()
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            tps = tokens / latency if latency > 0 else 0
            return {
                "latency": round(latency, 3),
                "tokens": tokens,
                "tokens_per_second": round(tps, 1),
                "status": "healthy",
            }
        except Exception as e:
            return {
                "latency": 0,
                "tokens": 0,
                "tokens_per_second": 0,
                "status": f"error: {str(e)}",
            }

    # Build the comparison chart data
    # Hint: Use the baseline values loaded above
    hf_tps = hf_baseline.get("tokens_per_second", 0)
    vllm_tps = vllm_baseline.get("tokens_per_second", 0)
    comparison_data = {
        "Engine": ["HuggingFace", "vLLM"],
        "Tokens per Second": [hf_tps, vllm_tps],
    }

    # Calculate the improvement ratio
    # Hint: Divide vLLM speed by HuggingFace speed
    improvement = vllm_tps / hf_tps if hf_tps > 0 else 0

    # --- THEME AND CSS (following llm-settings-lab pattern) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def make_comparison_chart():
        """Create HF vs vLLM comparison bar chart."""
        fig, ax = plt.subplots(figsize=(5, 3.5))
        engines = ["HuggingFace", "vLLM"]
        values = [hf_tps, vllm_tps]
        colors = ["#ef4444", "#22c55e"]
        bars = ax.bar(engines, values, color=colors, width=0.5, edgecolor="white")
        ax.set_ylabel("Tokens per Second")
        ax.set_title("Single Request Throughput")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", fontweight="bold")
        ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 10)
        fig.tight_layout()
        return fig

    def make_load_chart():
        """Create load test throughput chart."""
        if not load_test_results:
            return None
        fig, ax = plt.subplots(figsize=(6, 3.5))
        users = [str(r["users"]) for r in load_test_results]
        throughputs = [r["throughput"] for r in load_test_results]
        bars = ax.bar(users, throughputs, color="#3b82f6", width=0.5, edgecolor="white")
        ax.set_xlabel("Concurrent Users")
        ax.set_ylabel("Tokens per Second")
        ax.set_title("Throughput by Concurrent Users")
        for bar, val in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", fontsize=8)
        ax.set_ylim(0, max(throughputs) * 1.3 if max(throughputs) > 0 else 10)
        fig.tight_layout()
        return fig

    def make_tuning_chart():
        """Create tuning results chart."""
        if not tuning_results:
            return None
        fig, ax = plt.subplots(figsize=(6, 3.5))
        configs = [r["config"] for r in tuning_results]
        throughputs = [r["throughput"] for r in tuning_results]
        bars = ax.bar(configs, throughputs, color="#a855f7", width=0.5, edgecolor="white")
        ax.set_ylabel("Tokens per Second")
        ax.set_title("Throughput by Configuration")
        for bar, val in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", fontsize=8)
        ax.set_ylim(0, max(throughputs) * 1.3 if max(throughputs) > 0 else 10)
        fig.tight_layout()
        return fig

    # Custom CSS - matching llm-settings-lab pattern with !important
    custom_css = """
    .gradio-container {
        max-width: 100% !important;
        padding: 40px 80px !important;
    }

    h1, h2, h3 {
        color: #29ddff !important;
    }

    table {
        border-collapse: collapse !important;
        width: 100% !important;
        margin-bottom: 16px !important;
    }

    th {
        background: #161b22 !important;
        color: #29ddff !important;
        padding: 10px 12px !important;
        text-align: left !important;
    }

    td {
        padding: 10px 12px !important;
        border-bottom: 1px solid #29ddff22 !important;
    }

    tr:last-child td {
        border-bottom: none !important;
    }

    blockquote {
        border-left: 4px solid #29ddff !important;
        padding-left: 12px !important;
    }

    button.primary {
        background: linear-gradient(135deg, #a5fecb, #12d8fa, #1fa2ff) !important;
        color: #0a0e14 !important;
        font-weight: 600 !important;
    }

    button.primary:hover {
        background: linear-gradient(135deg, #12d8fa, #1fa2ff, #7c3aed) !important;
    }
    """

    # KodeKloud Brand Theme (same as llm-settings-lab)
    kk_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#0a0e14",
        body_background_fill_dark="#0a0e14",
        background_fill_primary="#161b22",
        background_fill_primary_dark="#161b22",
        background_fill_secondary="#1e293b",
        background_fill_secondary_dark="#1e293b",
        body_text_color="#ffffff",
        body_text_color_dark="#ffffff",
        body_text_color_subdued="#94a3b8",
        body_text_color_subdued_dark="#94a3b8",
        border_color_primary="#29ddff33",
        border_color_primary_dark="#29ddff33",
        input_background_fill="#161b22",
        input_background_fill_dark="#161b22",
        input_border_color="#29ddff33",
        input_border_color_dark="#29ddff33",
        button_primary_background_fill="#29ddff",
        button_primary_background_fill_dark="#29ddff",
        button_primary_background_fill_hover="#12d8fa",
        button_primary_background_fill_hover_dark="#12d8fa",
        button_primary_text_color="#0a0e14",
        button_primary_text_color_dark="#0a0e14",
        block_background_fill="#161b22",
        block_background_fill_dark="#161b22",
        block_border_color="#29ddff22",
        block_border_color_dark="#29ddff22",
        block_label_background_fill="#1e293b",
        block_label_background_fill_dark="#1e293b",
        block_label_text_color="#29ddff",
        block_label_text_color_dark="#29ddff",
    )

    # Build the dashboard
    with gr.Blocks(
        title="vLLM Monitoring Dashboard",
        theme=kk_theme,
        css=custom_css,
    ) as dashboard:

        gr.Markdown("# vLLM Production Monitoring Dashboard")
        gr.Markdown("*InferenceIO - SmolLM-135M Inference Server*")

        # --- ROW 1: Live Status ---
        with gr.Row():
            with gr.Column(scale=1):
                status_text = gr.Textbox(
                    label="Server Status",
                    value="Checking...",
                    interactive=False,
                )
            with gr.Column(scale=1):
                live_tps = gr.Number(
                    label="Live Tokens/sec",
                    value=0,
                )
            with gr.Column(scale=1):
                live_latency = gr.Number(
                    label="Live Latency (s)",
                    value=0,
                )

        refresh_btn = gr.Button("Refresh Live Metrics", variant="primary", size="lg")

        def refresh_metrics():
            metrics = get_live_metrics()
            return (
                metrics["status"],
                metrics["tokens_per_second"],
                metrics["latency"],
            )

        refresh_btn.click(
            fn=refresh_metrics,
            outputs=[status_text, live_tps, live_latency],
        )

        gr.Markdown("")
        gr.Markdown("---")

        # --- ROW 2: Before/After Comparison ---
        gr.Markdown("## HuggingFace vs vLLM Comparison")
        gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Plot(value=make_comparison_chart())
            with gr.Column(scale=1):
                gr.Markdown(f"""
### Performance Summary

| Metric | HuggingFace | vLLM |
|--------|-------------|------|
| Tokens/sec | {hf_tps:.1f} | {vllm_tps:.1f} |
| Improvement | - | {improvement:.1f}x |

**vLLM is {improvement:.1f}x faster** for single-request inference.
The advantage grows significantly under concurrent load.
""")

        gr.Markdown("")
        gr.Markdown("---")

        # --- ROW 3: Load Test Results ---
        if load_test_results:
            gr.Markdown("## Multi-User Load Test Results")
            gr.Markdown("")
            load_fig = make_load_chart()
            if load_fig:
                gr.Plot(value=load_fig)
            gr.Markdown("")
            gr.Markdown("---")

        # --- ROW 4: Tuning Results ---
        if tuning_results:
            gr.Markdown("## Parameter Tuning Results")
            gr.Markdown("")
            tuning_fig = make_tuning_chart()
            if tuning_fig:
                gr.Plot(value=tuning_fig)
            gr.Markdown("")
            gr.Markdown("---")

        # --- ROW 5: Lab Journey Summary ---
        gr.Markdown("## Lab Journey Summary")
        gr.Markdown(f"""
| Task | What You Did | Key Result |
|------|-------------|------------|
| 1 | HuggingFace baseline | {hf_tps:.1f} tok/s (single user) |
| 2 | vLLM offline inference | {vllm_tps:.1f} tok/s ({improvement:.1f}x faster) |
| 3 | KV cache simulation | ~80% memory waste with contiguous allocation |
| 4 | PagedAttention simulation | ~95% memory utilization with paging |
| 5 | OpenAI-compatible API | Server on port 8000 |
| 6 | Multi-user load test | Throughput scales with concurrent users |
| 7 | Parameter tuning | Optimized for workload |
| 8 | Monitoring dashboard | This dashboard! |
""")

        gr.Markdown("")

        gr.Markdown("""
### Key Takeaways

1. **Inference engines matter** — same model, different speeds
2. **KV cache is the bottleneck** — traditional systems waste 60-80% of memory
3. **PagedAttention solves it** — inspired by OS virtual memory paging
4. **vLLM scales** — throughput improves with concurrent users
5. **Production needs monitoring** — always track tokens/sec and latency
""")

        gr.Markdown("")
        gr.Markdown("---")
        gr.Markdown("**vLLM Monitoring Dashboard** — InferenceIO Production Monitoring")

    # Verify the live metrics path works before marking the task
    # complete - this catches an unfilled TODO 1, which would otherwise
    # be swallowed by get_live_metrics' error handling.
    print("\nTesting live metrics against the vLLM server...")
    metrics = get_live_metrics()
    if metrics["status"] != "healthy":
        print(f"\nERROR: Live metrics check failed: {metrics['status']}")
        print(f"Check TODO 1 in {os.path.abspath(__file__)} and try again.")
        return

    # Create marker
    os.makedirs(MARKERS_DIR, exist_ok=True)
    with open(os.path.join(MARKERS_DIR, "task8_complete.txt"), "w") as f:
        f.write("TASK_8_COMPLETE\n")

    print("\nBuilding Gradio dashboard...")
    print("\nTask 8 Complete!")
    print("\nLaunching dashboard on port 7860...")
    print("Click the 'Gradio UI' button (top-right) to view the dashboard.")
    print("Press Ctrl+C to stop.\n")

    # Kill any existing process on port 7860
    import subprocess as _sp
    try:
        result = _sp.run(["ss", "-tlnp", "sport", "=", ":7860"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "pid=" in line:
                import re
                pids = re.findall(r'pid=(\d+)', line)
                for pid in pids:
                    try:
                        os.kill(int(pid), 9)
                        print(f"  Killed old process on port 7860 (PID {pid})")
                        time.sleep(1)
                    except ProcessLookupError:
                        pass
    except Exception:
        pass

    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
