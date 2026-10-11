# vLLM Explained Lab - Assets

## Lab Overview

**Scenario:** You are an ML engineer at InferenceIO, building an LLM-as-a-Service platform.
**Mission:** Use vLLM to go from single-user HuggingFace inference to production-ready serving.
**Model:** SmolLM-135M (HuggingFaceTB/SmolLM-135M)

## Task Summary

| Task  | Script                     | TODOs | Key Skill                            |
| ----- | -------------------------- | ----- | ------------------------------------ |
| Setup | verify_environment.py      | 0     | Environment check and model download |
| 1     | task_1_hf_baseline.py      | 2     | Measure baseline inference speed     |
| 2     | task_2_vllm_inference.py   | 2     | Compare vLLM vs HuggingFace          |
| 3     | task_3_kv_cache_problem.py | 2     | Understand KV cache fragmentation    |
| 4     | task_4_paged_attention.py  | 3     | See PagedAttention efficiency        |
| 5     | task_5_api_server.py       | 2     | Launch OpenAI-compatible API         |
| 6     | task_6_multi_user_load.py  | 2     | Load test concurrent users           |
| 7     | task_7_tuning.py           | 2     | Tune production parameters           |
| 8     | task_8_dashboard.py        | 3     | Build Gradio monitoring UI           |

## Directory Structure

```
code/
├── verify_environment.py       # Setup: check env + download model
├── task_1_hf_baseline.py       # HuggingFace baseline inference
├── task_2_vllm_inference.py    # vLLM offline inference
├── task_3_kv_cache_problem.py  # KV cache fragmentation simulation
├── task_4_paged_attention.py   # PagedAttention comparison
├── task_5_api_server.py        # vLLM API server
├── task_6_multi_user_load.py   # Concurrent load testing
├── task_7_tuning.py            # Parameter tuning
└── task_8_dashboard.py         # Gradio monitoring dashboard
```

## Task Files Reference

### verify_environment.py

- Checks Python venv, packages (vllm, transformers, gradio, aiohttp)
- Downloads SmolLM-135M model
- Runs quick test generation
- Creates marker: `markers/environment_verified.txt`

### task_1_hf_baseline.py

- Loads SmolLM-135M with HuggingFace AutoModelForCausalLM
- Generates 50 tokens and measures tok/s
- Saves baseline to `markers/hf_baseline.txt`
- TODO 1: Load model with from_pretrained()
- TODO 2: Set max_new_tokens=50

### task_2_vllm_inference.py

- Loads SmolLM-135M with vLLM LLM class
- Generates 50 tokens and compares with baseline
- Saves metrics to `markers/vllm_baseline.txt`
- TODO 1: Initialize LLM engine
- TODO 2: Set SamplingParams

### task_3_kv_cache_problem.py

- Simulates 5 concurrent requests with different lengths
- Shows contiguous allocation waste (~80%)
- TODO 1: Set max_seq_len
- TODO 2: Calculate waste percentage

### task_4_paged_attention.py

- Simulates same requests with paged allocation
- Shows utilization improvement (~95%)
- TODO 1: Set page_size
- TODO 2: Calculate pages needed
- TODO 3: Compute utilization

### task_5_api_server.py

- Starts vLLM OpenAI-compatible server on port 8000
- Sends chat completion via OpenAI client
- Server stays running for subsequent tasks
- TODO 1: Configure OpenAI client
- TODO 2: Send completion request

### task_6_multi_user_load.py

- Sends concurrent requests (1/5/10/20 users)
- Measures throughput scaling
- Saves results to `markers/load_test_results.json`
- TODO 1: Set concurrent user counts
- TODO 2: Calculate throughput

### task_7_tuning.py

- Tests 3 vLLM configurations (default, shorter context, limited concurrency)
- Benchmarks each and compares
- Saves to `markers/tuning_results.json`
- TODO 1: Set max_model_len
- TODO 2: Set max_num_seqs

### task_8_dashboard.py

- Builds Gradio dashboard on port 7860
- Shows HF vs vLLM comparison, load test results, tuning results
- Live metrics refresh button
- TODO 1: Create metrics function
- TODO 2: Build comparison chart data
- TODO 3: Calculate improvement ratio

## Troubleshooting

- **Model not found:** Run verify_environment.py first
- **Server already running:** Use `fuser -k 8000/tcp` to stop
- **Server not responding:** Check logs at `markers/vllm_server.log`
- **Port conflict:** Check `lsof -i :8000` or `lsof -i :7860`
- **Memory issues:** Reduce max_model_len or max_num_seqs
- **"Device string must not be empty":** The installed vLLM is not the CPU build. Run verify_environment.py for the fix command.

## Running Locally (uv, macOS or Linux)

`verify_environment.py` installs [uv](https://docs.astral.sh/uv/) if needed, creates `.venv` in the
project root, re-executes itself inside it, and checks every dependency:

```bash
python3 verify_environment.py
```

All task scripts write their results to `markers/` next to the scripts, and the dashboard reads from there.

### vLLM on Apple Silicon

There is no prebuilt vLLM wheel for macOS on Apple Silicon, so build the CPU backend from source:

```bash
source .venv/bin/activate
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
git apply ../patches/vllm-cpu-slot-mapping-fallback.patch   # see note below
uv pip install -r requirements/cpu.txt
uv pip install -e .
```

`patches/vllm-cpu-slot-mapping-fallback.patch` fixes a regression on vLLM `main` (October 2026) where the
CPU fallback for the slot-mapping kernel was not updated after the Triton kernel's signature changed,
which crashes every non-Triton CPU build with "Boolean value of Tensor with more than one value is
ambiguous". Skip the patch if upstream has already fixed it (the apply step will tell you).

On Linux x86_64, the prebuilt CPU wheel works instead; the fix hint printed by `verify_environment.py`
has the exact command.
