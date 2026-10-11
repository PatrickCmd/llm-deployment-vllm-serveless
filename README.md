# Deploying Custom LLMs with vLLM

Companion repository for the PyCon Africa 2026 talk **"Deployment of Custom LLMs with vLLM using
serverless architectures"** by Patrick Walukagga, Software Engineer at [Sunbird AI](https://sunbird.ai).

The talk has two parts:

| Part | Where | What |
| ---- | ----- | ---- |
| 1 | Live, 30 minutes | What LLM serving is, why the KV cache is the bottleneck, how vLLM's PagedAttention and continuous batching fix it. Eight runnable demos. |
| 2 | This repository, self-paced | Deploying a custom model with vLLM on serverless GPU infrastructure. In progress; watch this repo. |

Every demo runs on a laptop CPU with a 135M-parameter model, so you can follow along without a GPU.

## Quick start

Requirements: Python 3.11 to 3.13, macOS or Linux, about 2 GB of free disk for the model and packages.

```bash
git clone https://github.com/PatrickCmd/llm-deployment-vllm-serveless.git
cd llm-deployment-vllm-serveless
python3 verify_environment.py
```

`verify_environment.py` installs [uv](https://docs.astral.sh/uv/) if it is missing, creates `.venv` in the
project root, re-executes itself inside that environment, and checks each dependency. On the first run it
will stop at the missing packages and print the exact `uv pip install` command to fix each one. Install
torch and transformers first, then vLLM (see below), then the rest:

```bash
source .venv/bin/activate
uv pip install torch transformers gradio aiohttp requests openai matplotlib
```

Run the verifier again once vLLM is installed. It downloads SmolLM-135M and runs a test generation.

### Installing vLLM on CPU

**Linux x86_64**: a prebuilt CPU wheel exists. Replace the version with the latest vLLM release:

```bash
uv pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_34_x86_64.whl" --torch-backend cpu
```

**macOS on Apple Silicon**: there is no prebuilt wheel, so build the CPU backend from source. You need
Xcode with the Command Line Tools.

```bash
source .venv/bin/activate
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
git apply ../patches/vllm-cpu-slot-mapping-fallback.patch
uv pip install -r requirements/cpu.txt
uv pip install -e .
```

The patch fixes a regression on vLLM `main` (October 2026): the CPU fallback for the slot-mapping kernel
was not updated when the Triton kernel's signature changed, and every non-Triton CPU build crashes with
`Boolean value of Tensor with more than one value is ambiguous`. If `git apply` reports the patch no longer
applies, upstream has most likely fixed it and you can skip that step.

The generic `vllm` package on PyPI is a CUDA build and fails on a machine without a GPU. The verifier checks
that the installed version string ends in `+cpu`.

## The demos

Run them in order from the project root with the virtual environment active. Each prints a short
"key insight" at the end and writes its results to `markers/`, which later tasks read.

| Task | Script | What it shows | Result on an M-series MacBook |
| ---- | ------ | ------------- | ----------------------------- |
| 1 | `task_1_hf_baseline.py` | Plain HuggingFace generation, one request at a time | 67 tok/s |
| 2 | `task_2_vllm_inference.py` | The same model and prompt through vLLM's engine | 88 tok/s, 1.3x |
| 3 | `task_3_kv_cache_problem.py` | Contiguous KV cache allocation and how much it wastes | 80% of memory wasted |
| 4 | `task_4_paged_attention.py` | The same requests with 16-token pages | 95% utilization, 4.7x less memory |
| 5 | `task_5_api_server.py` | vLLM as an OpenAI-compatible server on port 8000 | Works with the OpenAI SDK unchanged |
| 6 | `task_6_multi_user_load.py` | 1, 5, 10 and 20 concurrent users | 4x throughput at 20 users |
| 7 | `task_7_tuning.py` | `max_model_len` and `max_num_seqs` benchmarked across three configs | No single best setting |
| 8 | `task_8_dashboard.py` | A Gradio monitoring dashboard on port 7860 | Charts from every task plus live metrics |

Tasks 1 to 4 are standalone. Task 5 starts a server that tasks 6, 7 and 8 talk to; task 7 restarts it
with each configuration and leaves the last one running. Stop it with the PID in
`markers/vllm_server_pid.txt`, or `lsof -ti :8000 | xargs kill`.

The tasks are adapted from a vLLM lab exercise. The lines marked `TODO` in the comments are the
blanks a learner fills in; this copy has them completed so the demos run as-is.

## What is in the repository

```
verify_environment.py      environment check, uv and .venv bootstrap, model download
task_1 ... task_8           the eight demos, in talk order
patches/                    vLLM CPU fallback fix for Apple Silicon builds
logos/                      PyCon Africa 2026 and Sunbird AI logos used in the slides
slides/                     reference decks, notes and a recorded explainer used while preparing the talk
markers/                    created at runtime; results, baselines, server PID and log (git-ignored)
vllm_source/                created by you if you build vLLM from source (git-ignored)
```

## How vLLM works, in one paragraph

A serving engine holds the model weights on the GPU and a KV cache for every active request. The cache
grows by one entry per token, and a naive engine reserves one contiguous block per request sized for the
longest possible answer, so most of that memory sits idle and the rest fragments. PagedAttention splits the
cache into small fixed-size blocks allocated on demand and tracked through a block table, which lifts
utilization from roughly 20% to 95% and lets requests share identical prefixes. Continuous batching swaps
finished requests out for waiting ones at every decode step instead of waiting for a whole batch. The two
together are why one GPU serves several times more users, and vLLM exposes all of it behind an
OpenAI-compatible API. The source for this framing is
[How does vLLM work?](https://outcomeschool.com/blog/how-does-vllm-work).

## Troubleshooting

- **`Device string must not be empty`**: the installed vLLM is a CUDA build. Reinstall the CPU wheel or
  build from source as above.
- **`Boolean value of Tensor ... is ambiguous` on Apple Silicon**: apply the patch in `patches/` and
  reinstall with `uv pip install -e .` from `vllm_source`.
- **Server not responding**: check `markers/vllm_server.log`. Port 8000 may be held by an earlier run.
- **Dashboard shows old numbers**: it reads `markers/` at launch. Restart it after rerunning tasks.
- **Out of memory**: lower `max_model_len` or `KV_CACHE_BYTES` at the top of tasks 2, 5 and 7.

## Links

- Sunbird AI products: [Sunflower](https://sunflower.sunbird.ai), [Speech](https://speech.sunbird.ai),
  [API](https://api.sunbird.ai)
- vLLM: [documentation](https://docs.vllm.ai), [CPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html)
- PyCon Africa 2026: [africa.pycon.org](https://africa.pycon.org/)
