# Part 2: Deploying a custom LLM with vLLM on serverless GPUs

A step-by-step guide to taking a fine-tuned model, **Sunbird AI's `Sunbird/Sunflower-Gemma4-E2B`**,
and serving it with vLLM from a **RunPod Serverless** endpoint that scales to zero, with an
OpenAI-compatible API in front of it. It continues the PyCon Africa 2026 talk
"Deploying Custom LLMs with vLLM": Part 1 showed *why* vLLM serves many users from one GPU;
this part shows *how* to run it in production without paying for an idle GPU.

Everything in this guide was executed on 2026-10-12 against a real RunPod account. Terminal
output shown in the text is real; where a screenshot of the RunPod console belongs, the guide
marks it and [images/README.md](images/README.md) lists what to capture.

**What you will build**

```
 your app / notebook                 RunPod Serverless (queue-based endpoint)
 ─────────────────────               ──────────────────────────────────────────────────
 OpenAI SDK                          ┌────────────────────────────────────────────────┐
   base_url = https://api.runpod.ai  │ endpoint  sunflower-gemma-2b                   │
     /v2/<ENDPOINT_ID>/openai/v1 ───▶│   template: runpod/worker-v1-vllm (vLLM 0.30)  │
   api_key  = RUNPOD_API_KEY         │   env: MODEL_NAME, HF_TOKEN, MAX_MODEL_LEN …   │
                                     │   model cache: Sunflower-Gemma4-E2B @ 44e5d28  │
 queue API (/run, /status, /health) ▶│   GPU: RTX 4090 24 GB, workers 0 → 2           │
                                     └────────────────────────────────────────────────┘
                                       idle: 0 workers, $0/h   busy: ~$1.10/h per worker
```

**Time and money.** About 60 minutes the first time. Creating the template and the endpoint
costs nothing. Each cold start of this 10 GB model bills about 2 to 3 minutes of RTX 4090
time (roughly $0.04 to $0.06), and each request a few seconds. The whole walkthrough,
including the multimodal tests, cost under $1. Idle endpoints cost $0.

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Set up a RunPod account](#2-set-up-a-runpod-account)
3. [Set up your computer](#3-set-up-your-computer)
4. [Package the model for vLLM](#4-package-the-model-for-vllm)
5. [Create the template](#5-create-the-template)
6. [Create the scale-to-zero endpoint](#6-create-the-scale-to-zero-endpoint)
7. [First request and verification](#7-first-request-and-verification)
8. [Cold starts](#8-cold-starts)
9. [Cost](#9-cost)
10. [Monitoring](#10-monitoring)
11. [Updating the model and tearing down](#11-updating-the-model-and-tearing-down)
12. [Troubleshooting](#12-troubleshooting)
13. [Other serverless GPU platforms](#13-other-serverless-gpu-platforms)
14. [References](#14-references)

## 1. Prerequisites

| You need | Why | Notes |
| --- | --- | --- |
| A RunPod account with about $10 of credit | The GPU runs there | Section 2 |
| A Hugging Face account and access token | The Sunflower checkpoints are gated | Request access on the [model page](https://huggingface.co/Sunbird/Sunflower-Gemma4-E2B); the token must be able to read it. Any other vLLM-supported model works the same way |
| `runpodctl` 2.4.0 or newer | Creates templates and endpoints from the terminal | Section 3. The console works too; the guide shows both |
| Python 3.11 to 3.13 with the `openai` package | Calling the endpoint | The repo's `.venv` from Part 1 already has it |
| Optional: `ffmpeg`, Docker, a GitHub account | Audio tests; building a custom worker image | Only for section 4.4 and the audio check |

Clone this repository if you have not, then work from the `part2/` folder:

```bash
git clone https://github.com/PatrickCmd/llm-deployment-vllm-serveless.git
cd llm-deployment-vllm-serveless
python3 verify_environment.py        # creates .venv (see the top-level README)
source .venv/bin/activate
cd part2
cp .env.example .env                 # fill in the keys as you go
```

## 2. Set up a RunPod account

1. **Sign up** at <https://console.runpod.io/signup> (Google, GitHub or email).
   *Screenshot `01-signup.png`.*
2. **Add credits.** Console → **Billing** (<https://console.runpod.io/user/billing>) → pick an
   amount or **Other**. $10 is enough for this whole guide. Cards via Stripe and crypto are
   accepted; credits are prepaid and non-refundable. Optionally enable **Auto-pay** so a
   demo never dies on an empty balance. *Screenshot `02-billing.png`.*
3. **Create an API key.** Console → **Settings** → **Credentials** → **API Keys** tab → **Create API Key**. Give it
   a name such as `pycon-serverless` and a permission level. **Restricted** with Serverless
   set to *Read/Write* is enough for this guide; the key is shown once, so paste it into
   `part2/.env` as `RUNPOD_API_KEY` immediately. *Screenshot `03-api-key.png`.*
4. **Know your spend limit.** New accounts are capped at $80/hour across all resources.
   This guide stays far below that. The Settings page also has low-balance email alerts;
   turn them on.

> The RunPod API key is what clients use as the *OpenAI* API key later. There is no separate
> endpoint secret.

## 3. Set up your computer

### 3.1 Install runpodctl

```bash
# macOS / Linux, official installer
curl -sSL https://cli.runpod.net | bash
# or macOS Homebrew (the tap can lag behind releases)
brew install runpod/runpodctl/runpodctl
# Windows PowerShell
Invoke-WebRequest -Uri https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.zip -OutFile runpodctl.zip
Expand-Archive runpodctl.zip -DestinationPath $env:LOCALAPPDATA\runpodctl
```

Then get on a recent build and confirm it. `--model-reference` (the model cache, section 6)
needs 2.4.0 or newer; the latest release on 2026-10-12 was v2.15.0.

```bash
runpodctl update
runpodctl version
```

```
runpodctl 2.7.2-309512b        # the version this guide was tested with
```

### 3.2 Authenticate

Either export the key (good for scripts and CI) or let the interactive doctor save it:

```bash
export RUNPOD_API_KEY=rpa_...    # from section 2
runpodctl doctor                 # checks the key, API connectivity and your SSH key
runpodctl user                   # account email and balance
```

```json
{
  "checks": [
    {"name": "api_key", "status": "pass"},
    {"details": "user: info@sunbird.ai", "name": "api_connectivity", "status": "pass"},
    {"details": "local_key: true, synced_to_cloud: true", "name": "ssh_key", "status": "pass"}
  ],
  "healthy": true
}
```

*Screenshot `04-runpodctl-doctor.png` of this terminal output.*

### 3.3 Hugging Face token

Create a **read** token at <https://huggingface.co/settings/tokens>, request access to the
model, then confirm the token can actually read it. Workers will use this exact token.

```bash
export HF_TOKEN=hf_...
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/models/Sunbird/Sunflower-Gemma4-E2B" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['id'], '| gated:', d['gated'], '| sha:', d['sha'], '| modified:', d['lastModified'])"
```

```
Sunbird/Sunflower-Gemma4-E2B | gated: auto | sha: 44e5d28f2012b4a76fe1855a62b1f184d2b6cfe0 | modified: 2026-07-27T12:17:21.000Z
```

A `401` or `403` here means the workers will fail too, with a misleading
"Unable to access model" error. Fix access first.

Put both keys in `part2/.env` (it is git-ignored):

```bash
RUNPOD_API_KEY=rpa_...
HF_TOKEN=hf_...
```

## 4. Package the model for vLLM

With vLLM, "packaging" a model is not building a binary. vLLM loads a standard Hugging Face
checkpoint directory at start-up, so packaging means four decisions:

### 4.1 Make the checkpoint vLLM-readable

A vLLM-ready repo is a transformers-format folder. Check the files on the Hub:

```bash
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/models/Sunbird/Sunflower-Gemma4-E2B?blobs=true" \
  | python3 -c "import json,sys; [print(f\"{s['rfilename']:30s} {s.get('size',0)/1e9:6.2f} GB\") for s in json.load(sys.stdin)['siblings']]"
```

```
.gitattributes                   0.00 GB
README.md                        0.00 GB
chat_template.jinja              0.00 GB      ← chat format; vLLM applies it for /chat/completions
config.json                      0.00 GB      ← architecture: Gemma4ForConditionalGeneration
generation_config.json           0.00 GB
model.safetensors               10.21 GB      ← bf16 weights (5.1 B raw params; ~2 B effective)
processor_config.json            0.00 GB      ← image/audio preprocessing (multimodal only)
tokenizer.json                   0.03 GB
tokenizer_config.json            0.00 GB
```

If you fine-tuned with LoRA, **merge the adapter into the base weights** first
(`PeftModel.merge_and_unload()` then `save_pretrained`) and push the merged folder with
`hf upload <org>/<name> ./merged`. vLLM can also serve LoRA adapters dynamically, but a merged
checkpoint is the simplest thing to cache and version. Save in **safetensors**, keep the
tokenizer and the chat template in the same repo, and set `torch_dtype` in `config.json`.

### 4.2 Check that vLLM supports the architecture

`config.json` says `"architectures": ["Gemma4ForConditionalGeneration"]`, `"model_type":
"gemma4"`. Look it up in vLLM's [supported models list](https://docs.vllm.ai/en/latest/models/supported_models.html)
and note the **first vLLM version that supports it**. This decides which worker image you can
use (4.3). Two things we learned the hard way with this checkpoint:

- **Gemma 4 E-variants share KV cache across layers** (`num_kv_shared_layers: 20`), so the
  checkpoint legitimately omits `k_proj`, `v_proj`, `k_norm` for layers 15 to 34. vLLM's strict
  loader check rejects this with `weights were not initialized from checkpoint`. The fix is the
  engine option `model_loader_extra_config={"enable_weights_track": false}` (vLLM ≥ 0.22;
  still present in 0.30). It only relaxes the load-time bookkeeping; the forward pass never
  touches those weights.
- **Audio** needs vLLM's `[audio]` extras (`av soundfile soxr scipy`), which the July 2026
  worker image lacked and the current `vllm/vllm-openai`-based image has, plus a workaround for
  a transformers Gemma 4 processor bug: `mm_processor_kwargs={"max_length": 480000,
  "truncation": true}` (30 s at 16 kHz). Clips longer than 30 s must be chunked by the client.

### 4.3 Choose a worker image

RunPod's [`worker-vllm`](https://github.com/runpod-workers/worker-vllm) is the official vLLM
OpenAI image (`vllm/vllm-openai:<version>`) plus a thin serverless wrapper. The wrapper starts
`vllm serve`, waits for `/health`, and then pulls jobs from the endpoint queue. Every
**UPPERCASED environment variable becomes a vLLM engine argument** (`MAX_MODEL_LEN` →
`--max-model-len`, `MODEL_LOADER_EXTRA_CONFIG` → `--model-loader-extra-config`), which is how a
template can configure the engine without a start command.

| Image | vLLM | Use it when |
| --- | --- | --- |
| `runpod/worker-v1-vllm:v2.28.0` (used here) | 0.30.0, CUDA 13.0 | The architecture is supported by this vLLM and you need no extra system packages |
| `runpod/worker-v1-vllm:<newer tag>` | see [releases](https://github.com/runpod-workers/worker-vllm/releases) | A newer architecture; check the release notes, RunPod occasionally rolls a vLLM bump back |
| Your own image `FROM runpod/worker-v1-vllm:...` | same | You need extras baked in (audio libs, a plugin), see 4.4 |
| `vllm/vllm-openai:<version>` directly | any | A **load-balancing** endpoint that exposes vLLM's own `/v1` (console-only endpoint type; no queue, no Requests tab) |

Pick the GPU with the weights size: 10.2 GB bf16 + KV cache for 8 k context fits a **24 GB
card** (RTX 4090, L4). A 9B model (18.8 GB bf16) only fits 24 GB with FP8 online quantization
(`QUANTIZATION=fp8`, Ada/Hopper GPUs only), and a 14B (29.5 GB) needs FP8 or a 48 GB card.

### 4.4 When you need a custom image: bake the extras

In July 2026 the stable worker did not bundle a vLLM new enough for Gemma 4, and installing the
audio extras at container start took about 6 minutes on serverless hosts, long enough for the
platform to recycle the worker mid-install. The fix was a two-line Dockerfile built by GitHub
Actions and pushed to GHCR, then used as the template image
([PatrickCmd/sunflower-vllm-worker](https://github.com/PatrickCmd/sunflower-vllm-worker)):

```dockerfile
FROM runpod/worker-v1-vllm:v2.28.0
RUN export PATH=$PATH:/root/.local/bin && \
    uv pip install --system av soundfile soxr scipy
```

```yaml
# .github/workflows/build.yml (trimmed): build on push, publish to ghcr.io/<you>/<image>:<tag>
- uses: docker/login-action@v3
  with: {registry: ghcr.io, username: ${{ github.actor }}, password: ${{ secrets.GITHUB_TOKEN }}}
- uses: docker/build-push-action@v6
  with: {context: ., push: true, tags: ghcr.io/<you>/sunflower-vllm-worker:0.30.0-audio}
```

Make the GHCR package public (or add a registry credential in RunPod: **Settings → Container
Registry Auth**, then `--registry-auth-id` on the template). The base image is about 9 GB
compressed, so the workflow needs the "free disk space" step the repo shows. Baking the
*model weights* into the image is also possible (`--build-arg MODEL_NAME=...`) but bloats
every pull; the model cache in section 6 is the better default.

## 5. Create the template

A template is the container spec: image, disk, environment. Creating one is free. The env block
below is the whole engine configuration; every key is explained in the table.

```bash
set -a; source .env; set +a

ENV_JSON=$(python3 - <<'PY'
import json, os
print(json.dumps({
    "MODEL_NAME": "Sunbird/Sunflower-Gemma4-E2B",
    "HF_TOKEN": os.environ["HF_TOKEN"],
    "OPENAI_SERVED_MODEL_NAME_OVERRIDE": "sunflower-gemma-2b",
    "MAX_MODEL_LEN": "8192",
    "GPU_MEMORY_UTILIZATION": "0.90",
    "RAW_OPENAI_OUTPUT": "1",
    "ENFORCE_EAGER": "true",
    "MODEL_LOADER_EXTRA_CONFIG": json.dumps({"enable_weights_track": False}),
    "MM_PROCESSOR_KWARGS": json.dumps({"max_length": 480000, "truncation": True}),
}))
PY
)

runpodctl template create --name sunflower-gemma-2b-vllm-pycon --serverless \
  --image "runpod/worker-v1-vllm:v2.28.0" --container-disk-in-gb 40 --env "$ENV_JSON"
```

```
TEMPLATE_ID 1py0k4aw3b | image runpod/worker-v1-vllm:v2.28.0 | disk 40
```

| Variable | Value | Why |
| --- | --- | --- |
| `MODEL_NAME` | `Sunbird/Sunflower-Gemma4-E2B` | The Hugging Face repo. Must match the cached model attached in section 6 |
| `HF_TOKEN` | your token | Gated repo. Also needed by RunPod's validator when the model is attached |
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | `sunflower-gemma-2b` | The `model=` clients send. Decouples the API from the repo name |
| `MAX_MODEL_LEN` | `8192` | Context window. Lower means less KV cache per request, more concurrent users (Part 1, task 7) |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Share of VRAM vLLM may claim for weights + KV blocks |
| `ENFORCE_EAGER` | `true` | Skip CUDA-graph capture: faster cold start, small steady-state cost |
| `RAW_OPENAI_OUTPUT` | `1` | Pass vLLM's OpenAI responses and SSE stream through unchanged |
| `MODEL_LOADER_EXTRA_CONFIG` | `{"enable_weights_track": false}` | Gemma 4 KV-sharing loader check (section 4.2) |
| `MM_PROCESSOR_KWARGS` | `{"max_length": 480000, "truncation": true}` | Gemma 4 audio processor bug (section 4.2). Harmless for text |

Container disk 40 GB: the image unpacks to roughly 20 GB and vLLM writes compile caches; the
model itself lives in the host cache, not on this disk.

> **Secret hygiene.** `worker-v1-vllm:v2.28.0` prints the full `vllm serve` command line,
> including `--hf-token hf_…`, in the container log at start-up. Anyone with console access to
> the endpoint can read it. Release v2.29.0 (2026-10-07) keeps the token out of argv and masks
> it in the launch log, but its image tag was not on Docker Hub when this guide was tested, so
> check `docker.io/runpod/worker-v1-vllm` tags and move to it when it lands. Until then, use a
> dedicated fine-grained Hugging Face token that can only read this one repo.

Not set here: `QUANTIZATION` (the model fits in bf16), `LIMIT_MM_PER_PROMPT` (the default
allows images and audio), `DTYPE` (taken from `config.json`).

**Console alternative.** Serverless → **Templates** → **New Template**: Template Type
*Serverless*, Container Image `runpod/worker-v1-vllm:v2.28.0`, Container Disk 40 GB, then add
the same environment variables. *Screenshot `05-template-console.png` of the saved template
page showing the image and env list.*

## 6. Create the scale-to-zero endpoint

```bash
runpodctl serverless create --template-id 1py0k4aw3b --name sunflower-gemma-2b-pycon \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --workers-min 0 --workers-max 2 --idle-timeout 120 \
  --scale-by delay --scale-threshold 300 \
  --min-cuda-version 13.0 \
  --model-reference "https://huggingface.co/Sunbird/Sunflower-Gemma4-E2B:44e5d28f2012b4a76fe1855a62b1f184d2b6cfe0"
```

```
ENDPOINT_ID oy9jxtbeq3c4ks | name sunflower-gemma-2b-pycon | gpus ADA_24 | workers 0 - 2 | idle 120 | minCuda 13.0
{"computeType": "GPU", "scalerType": "QUEUE_DELAY", "scalerValue": 300, "templateId": "1py0k4aw3b"}
```

| Flag | Why |
| --- | --- |
| `--gpu-id "NVIDIA GeForce RTX 4090"` | 24 GB Ada card, $1.10/h flex. Up to three GPU types can be set in the console as fallbacks |
| `--workers-min 0` | **Scale to zero.** No worker, no bill. `--workers-min 1` keeps one warm and bills about $800/month |
| `--workers-max 2` | Cost ceiling and concurrency ceiling. Raise for real traffic |
| `--idle-timeout 120` | Keep a warm worker for 2 minutes after the last request, so a burst of demo requests does not pay a cold start each time. Default is 5 s |
| `--scale-by delay --scale-threshold 300` | Add a worker only when requests have waited 300 s. Conservative on purpose; use `requests` with a small count for latency-sensitive traffic |
| `--min-cuda-version 13.0` | The vLLM 0.30 image is built on CUDA 13.0. Without this floor the scheduler can land on old-driver hosts and the container crash-loops |
| `--model-reference …:<commit>` | **RunPod model cache**: the host already holds the weights, so a worker loads them from local disk instead of downloading 10 GB on every cold start. Download time is not billed. The ref is resolved to a commit once and pinned; use an explicit hash so git records what was deployed |
| FlashBoot (default on) | Snapshots worker state so repeat cold starts are faster |
| No `--data-center-ids`, no network volume | Pinning to one data centre starved GPU supply in our July deployment; the model cache gives the caching benefit without the pin |

The whole thing is in [deploy_gemma_e2b.sh](deploy_gemma_e2b.sh), which runs exactly these two
commands (`NAME_SUFFIX=-pycon ./deploy_gemma_e2b.sh` reproduces the names above). Put the
printed id in `.env`:

```bash
ENDPOINT_ID=oy9jxtbeq3c4ks
SERVED_MODEL=sunflower-gemma-2b
```

**Console alternative.** Serverless → **New Endpoint** → *Import from template* (or deploy the
**vLLM** Hub repo and type the model name). Set GPU, Active workers 0, Max workers 2, Idle
timeout 120, FlashBoot on, **Model** = the Hugging Face URL (this is the cache), and under
Advanced the CUDA version filter and scaling type. *Screenshots `06-new-endpoint.png` (the
form) and `07-endpoint-overview.png` (the endpoint page with its id, GPU and worker counts).*

## 7. First request and verification

A "ready" worker is not proof of a working endpoint. Only a real request is. The endpoint
exposes two kinds of routes:

| Route | Use |
| --- | --- |
| `POST /v2/<id>/run` → `GET /v2/<id>/status/<job>` | Async queue. Survives cold starts of any length. Submitting a job is what makes a scaled-to-zero endpoint start a worker |
| `POST /v2/<id>/runsync` | Sync queue, up to about 90 s |
| `/v2/<id>/openai/v1/{models,chat/completions,completions}` | OpenAI-compatible. Behind an edge timeout of about 100 s, so **cold** calls here fail with 524 |
| `GET /v2/<id>/health` | Worker and queue counts, no auth needed beyond the API key |

### 7.1 Run the verifier

[verify_endpoint.py](verify_endpoint.py) does the cold start through the queue route, then the
OpenAI checks (`models.list`, chat, streaming), then optional vision and audio:

```bash
set -a; source .env; set +a
python verify_endpoint.py --vision
```

```
=== sunflower-gemma-2b @ oy9jxtbeq3c4ks ===
  health before: {'jobs': {...'inQueue': 0}, 'workers': {'idle': 0, 'initializing': 1, 'ready': 0, ...}}
  submitted job f01b2484-4ec4-40ec-8604-efa18e19729e-e2
  [    1s] job IN_QUEUE     workers: init=1 running=0 idle=0 throttled=0 unhealthy=0
  ...
  [  538s] job COMPLETED    workers: init=0 running=1 idle=1 throttled=0 unhealthy=0
  cold output: {'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'content': 'Eddwaliro erisinga okumpi liri ludda wa?', 'role': 'assistant'}, ...
  cold start + first job: 538s
  models.list: ['sunflower-gemma-2b']
  chat (1.8s): Eddwaliro erisinga okumpi liri ludda wa?
  usage: 46 prompt + 16 completion tokens
  streaming: OK (11 chunks, first token after 0.66s)
  vision: A yellow sunflower with a dark brown center.

All checks passed.
```

The 538 s first run is the **first-ever** start of this model on RunPod, most of it unbilled
cache placement (section 8 breaks it down). The translation matches the Sunflower model card
exactly, and the vision check described the generated test image correctly.

*Screenshot `08-requests-tab.png`: the endpoint's **Requests** tab showing this job as
COMPLETED with its delay and execution time.*

### 7.2 Call it from the OpenAI SDK

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["RUNPOD_API_KEY"],
    base_url=f"https://api.runpod.ai/v2/{os.environ['ENDPOINT_ID']}/openai/v1",
)
r = client.chat.completions.create(
    model="sunflower-gemma-2b",
    messages=[
        {"role": "system", "content": "You are Sunflower, a helpful assistant made by Sunbird AI "
                                      "who knows many African languages."},
        {"role": "user", "content": "Translate from English to Luganda: Where is the nearest hospital?"},
    ],
    max_tokens=128,
    temperature=0.0,      # the Sunflower models use a fixed translation template; greedy decoding
)
print(r.choices[0].message.content)
```

Streaming is `stream=True` on the same call, exactly as in Part 1, task 5. The only things
that changed from the laptop demo are the base URL and the key.

### 7.3 Multimodal: images and audio

Gemma 4 E2B takes images and audio in the same chat request. Two constraints found live:
**media must be embedded as base64 data URIs** (the worker's outbound fetch of remote URLs
fails with 500), and **audio clips are capped at 30 s** by the encoder, so split longer
recordings client-side:

```bash
ffmpeg -i talk.mp3 -f segment -segment_time 28 -ar 16000 -ac 1 chunk_%03d.mp3
python verify_endpoint.py --skip-cold --audio chunk_000.mp3
```

```python
{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
    {"type": "text", "text": "Describe this image in one sentence."}]}

{"role": "user", "content": [
    {"type": "input_audio", "input_audio": {"data": mp3_b64, "format": "mp3"}},
    {"type": "text", "text": "Please transcribe this Luganda audio."}]}
```

Verified live against a warm worker with a 6-second clip synthesised by macOS `say`
("Good morning. Welcome to PyCon Africa. Today we are deploying a language model with vLLM on
serverless GPUs."):

```
python verify_endpoint.py --skip-cold --audio hello_16k.mp3
  chat (1.4s): Eddwaliro erisinga okumpi liri ludda wa?
  streaming: OK (11 chunks, first token after 0.60s)
  audio transcript: Olukusa olulungi, tukwaniriza ku Pikan Africa, leero tukuleetawo enkola
  y'olulimi nga tukozesa ebyuma bya BLLM ku bintu eby'enjawulo.
```

The clip is English but the prompt asks for Luganda, so the model rendered the speech in
Luganda ("PyCon" became "Pikan", "vLLM" became "BLLM"). Prompt for the language the audio is
actually in when you want a verbatim transcript. The audio path worked on the stock
`worker-v1-vllm:v2.28.0` image, so the baked audio image of section 4.4 is no longer needed
for this model; keep that section as the recipe for the next time a dependency is missing.

## 8. Cold starts

A cold start is everything between "a job arrives and no worker is running" and "the first
token". RunPod bills from the moment the worker starts to the moment it stops, **except** image
pull and cached-model download, which are free. What you pay for is the model load into VRAM
and the engine start.

```
 request ─▶ scheduler finds a host ─▶ pull image (free) ─▶ container start ─▶ vLLM: load weights
            (seconds; can be minutes      (cached on          │                 from host cache (~10 s),
             if the GPU pool is dry)       most hosts)        │                 build engine (~60–90 s)
                                                              ▼
                                                   worker "ready" ─▶ job runs ─▶ idle timeout ─▶ scale to 0
```

Measured on this deployment (RTX 4090 in US-TX-3, 10 GB bf16 model, eager mode), from the
worker's system and container logs:

| Phase of the **first-ever** start (2026-10-12) | Clock | Duration | Billed? |
| --- | --- | --- | --- |
| Job submitted, scheduler picks hosts | 06:18:24 | | no |
| `image ready, initializing model files` (the host downloads the 10 GB snapshot into the cache) | 06:21:52 → 06:25:37 | 3 min 45 s | **no** |
| `model ready` → `start container` | 06:26:09 | 30 s | yes |
| `vllm serve` starts, resolves `Gemma4ForConditionalGeneration` | 06:26:11 → 06:27:24 | 1 min 13 s | yes |
| `Loading weights took 2.28 seconds` from the local cache | 06:27:37 | 2 s | yes |
| `init engine (profile, create kv cache, warmup model) took 38.23 s`, multimodal warm-up 32 s | 06:28:16 | 40 s | yes |
| `vLLM is healthy`, serverless worker starts, job picked up and finished | 06:28:29 | 7 s | yes |
| **Billed start time (container start → first job done)** | | **2 min 20 s ≈ $0.04** | |
| Wall time seen by the client (`/run` → `COMPLETED`) | | 8 min 58 s | |

| **Second** cold start, 7 minutes later, weights already on the hosts | Clock | Duration |
| --- | --- | --- |
| `/run` submitted with zero running containers (idle timeout had stopped the worker) | 06:35:10 | |
| Job `COMPLETED`, translation returned | 06:37:33 | **2 min 23 s** |
| Warm requests right after: chat 1.5 s, first streamed token 0.72 s | | |

```
python verify_endpoint.py          # second run, same endpoint
  submitted job 124d7e4c-4627-47b3-808c-c1d9b36eefcb-e1
  [  143s] job COMPLETED    workers: init=0 running=1 idle=1 throttled=0 unhealthy=0
  cold start + first job: 143s
  chat (1.5s): Eddwaliro erisinga okumpi liri ludda wa?
  streaming: OK (11 chunks, first token after 0.72s)
```

Reading the tables: the expensive part of a first-ever start is RunPod placing the model on
hosts, and it is free. Once the hosts hold the snapshot, a cold start is the billed part only:
about 2 min 20 s from an idle endpoint to the first answer, both times. After that, loading 10 GB of weights takes 2 seconds because they come
from local disk. The billed 2 minutes 20 seconds are almost entirely vLLM starting up (Python
imports, engine profile, KV-cache allocation, multimodal warm-up), which is why the levers
below are about the engine, not the download. The KV cache the engine reported, 10.7 GiB for
876,679 tokens, is "107 concurrent requests at 8,192 tokens each": Part 1's PagedAttention
numbers on a real GPU.

Levers, roughly in order of impact:

1. **Model cache** (`--model-reference`). Without it every cold start downloads the weights
   from Hugging Face first; a 10 GB download alone is 1 to 3 minutes and larger models
   scale linearly. With it, the weights are on local disk.
2. **FlashBoot** (on by default). Repeat cold starts on a host that recently ran the worker
   restore faster.
3. **Smaller or quantized model.** Load time is roughly linear in bytes. FP8 halves a bf16
   checkpoint; E2B is already small.
4. **`ENFORCE_EAGER=true`.** Skips CUDA-graph capture (30 to 60 s on larger models) at a small
   throughput cost. Flip it off once a workload is steady and throughput matters more.
5. **Idle timeout.** 120 s here. Longer keeps the worker warm between bursts at
   $0.00031/s; the break-even against a cold start is a few minutes of idle.
6. **An active worker** (`--workers-min 1`) removes cold starts entirely and costs the full
   hourly rate around the clock. Right for production with steady traffic, wrong for a demo.
7. **Client side.** Make the *first* call through the async queue route or with a
   retry-on-timeout loop (the verifier does the former; `chat_with_retry` in the Sunbird repo
   does the latter), and send a tiny warm-up request a few minutes before a demo.

## 9. Cost

Serverless is billed per second of worker time, rounded up. Prices on the RunPod pricing page
on 2026-10-12 (flex workers; active workers are discounted through sales):

| GPU | VRAM | $/hour | $/second | Fits (bf16 / fp8) |
| --- | --- | --- | --- | --- |
| RTX 4090 | 24 GB | 1.10 | 0.00031 | E2B bf16; 9B and 14B with FP8 |
| L4, A5000, RTX 3090 | 24 GB | 0.69 | 0.00019 | E2B bf16 (no native FP8 on Ampere) |
| L40S, 6000 Ada | 48 GB | 1.75 | 0.00049 | 14B bf16 |
| A6000, A40 | 48 GB | 1.22 | 0.00034 | 14B bf16 |
| A100 | 80 GB | 2.72 | 0.00076 | 30B+ |
| H100 | 80 GB | 4.79 | 0.00133 | 30B+, highest throughput |

Storage: container disk about $0.10/GB/month while a worker exists, network volumes
$0.07/GB/month. There is no charge for an endpoint with zero workers.

Worked examples for this endpoint on an RTX 4090:

| Scenario | Billed time | Cost |
| --- | --- | --- |
| One cold start (about 2.5 min) | 150 s | $0.05 |
| One warm translation request (about 2 s) plus its share of idle timeout | 2 s + up to 120 s | $0.0006 + up to $0.04 |
| A 20-minute live demo, one worker warm the whole time | 1,200 s | $0.37 |
| 1,000 requests/day arriving in bursts, about 30 cold starts | ≈ 2.5 h | ≈ $2.75/day |
| One active worker, 24/7 | 730 h | ≈ $800/month |

The idle timeout is the knob that trades cold starts for idle spend. The equivalent of one
cold start is about 150 s of idle, so a 120 s idle timeout pays for itself whenever a second
request arrives within two minutes. Watch spending in Console → **Billing**, or:

```bash
runpodctl billing serverless --endpoint-id $ENDPOINT_ID --bucket-size day   # per-endpoint spend
```

Billing records appear with a lag of a few hours; right after the walkthrough the command
returned no rows for this endpoint yet.

## 10. Monitoring

What to watch, and where:

| Signal | Where | What good looks like |
| --- | --- | --- |
| Worker states | `GET /v2/<id>/health` → `workers`; **Workers** tab for the detail | `running` while a job executes. `idle` in `/health` also counts provisioned hosts whose container is stopped (not billed); the Workers tab shows which worker actually has a running container and for how long. `throttled` means the GPU pool is dry; `unhealthy` means the container is crash-looping |
| Queue depth | `/health` → `jobs.inQueue`, console **Requests** tab | Near zero. Growing with workers idle means the worker image is broken or the model placement is pending (section 11) |
| Per-request delay and execution time | **Requests** tab (per job), **Metrics** tab (percentiles) | Delay is the cold start; execution time is prefill + decode |
| Worker logs | **Workers** tab → a worker → **Logs** | Look for the model revision line, `Loading weights took …`, and `vllm serve` reporting the KV cache size. Start-up errors appear here first |
| Spend | **Billing**, `runpodctl billing` | Matches the worked examples above |
| Balance alerts | **Settings** → notifications | Low-balance email on |

*Screenshots `09-workers-tab.png` (worker list with states), `10-worker-logs.png` (the log
showing weights loaded from the cache), `11-metrics-tab.png` (requests and execution time over
time).*

```bash
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.ai/v2/$ENDPOINT_ID/health
```

```json
{"jobs":{"completed":0,"failed":0,"inProgress":0,"inQueue":1,"retried":0},
 "workers":{"idle":0,"initializing":1,"ready":0,"running":0,"throttled":0,"unhealthy":0}}
```

For the two numbers Part 1 said every LLM service should show, **tokens per second** and
**latency per request**, vLLM exposes Prometheus metrics at `/metrics` inside the worker. On a
queue-based endpoint that port is not public, so measure from the client instead: the
`usage` block of every chat completion gives token counts, and the Requests tab gives time.
The Part 1 dashboard (`task_8_dashboard.py`) works unchanged against this endpoint if you point
its base URL and key at RunPod. Programmatic monitoring: the same `/health` endpoint on a
schedule, plus `runpodctl serverless get <id>` for configuration drift.

## 11. Updating the model and tearing down

**New weights on the Hub do not reach the endpoint by themselves.** The model cache pins a
commit. To roll forward, change the model reference to the new commit; RunPod then creates a
new release and must place the new snapshot on each worker's host. Until a host shows the
snapshot as *deployed*, its worker sits idle and jobs queue silently. The Sunbird production
repo has a script that does this safely (resolve ref → hash, update, watch placement, roll
back with the previous hash): see `update_model.py` in
[sunflower-multi-models-deployment-runpod](https://github.com/SunbirdAI/sunflower-multi-models-deployment-runpod).
Policy that came out of a 45-minute outage: detect drift automatically, promote manually,
off-peak, with a verifier run afterwards.

**Change engine settings** by editing the template env (`runpodctl template update <id> --env
'<json>'`); every template change rolls the workers.

**Teardown:**

```bash
runpodctl serverless delete oy9jxtbeq3c4ks     # endpoint (idle endpoints cost $0, so this is optional)
runpodctl template delete 1py0k4aw3b           # template
```

## 12. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| First request returns 524 / times out | Cold sync call behind the ~100 s edge timeout | Use `/run` + `/status` for the first touch, or a retry loop; keep a worker warm before demos |
| `Unable to access model … at branch/tag` when creating or updating | RunPod's validator could not read the gated repo with the token in the **endpoint** env | Check the token with the curl in 3.3; make sure `HF_TOKEN` is in the template env when the model is attached |
| Workers `unhealthy`, restarting every minute | Container crash-loop: driver too old for the image (`nvidia-container-cli: unsatisfied condition`), or a vLLM start-up error | Set `--min-cuda-version 13.0`; read Workers → Logs for the real error |
| `weights were not initialized from checkpoint` | Gemma 4 KV-shared layers (section 4.2) | `MODEL_LOADER_EXTRA_CONFIG={"enable_weights_track": false}` |
| `When setting truncation=True, make sure that max_length is defined` on audio | transformers Gemma 4 processor bug | `MM_PROCESSOR_KWARGS={"max_length": 480000, "truncation": true}`; chunk audio to ≤ 30 s |
| 500 on a request with an `image_url`/audio URL | Worker cannot fetch remote media | Embed media as base64 data URIs |
| Workers `throttled`, jobs stuck `IN_QUEUE` | GPU pool exhausted, often because of a data-centre or volume pin | Unpin; add fallback GPU types; use the model cache instead of a volume |
| Workers `idle` but jobs never start after a model update | New snapshot still *assigned*, not *deployed*, on those hosts | Roll back to the previous commit, retry later (section 11) |
| Out of memory at start-up | `MAX_MODEL_LEN` too large for the KV cache that is left | Lower `MAX_MODEL_LEN` or `GPU_MEMORY_UTILIZATION`, quantize, or pick a bigger GPU. The worker's log shows a specific message |
| Gibberish after enabling `QUANTIZATION=fp8` | FP8 on an Ampere GPU falls back to a weight-only path that corrupted Qwen3 output | FP8 only on Ada/Hopper (RTX 4090, L40S, H100) |
| Streaming arrives as one blob | `RAW_OPENAI_OUTPUT` not `1` | Set it in the template |
| `model not found` from the OpenAI route | `model=` does not match `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | `client.models.list()` shows the served name |

## 13. Other serverless GPU platforms

Everything above is RunPod-specific only in the control plane. vLLM, the checkpoint, and the
OpenAI-compatible client are the same everywhere. What differs is how the platform runs the
container, how it scales to zero, what a cold start looks like from the client, and the price.

| Platform | How vLLM runs | Scale to zero | Cold start from the client | Billing | Good fit |
| --- | --- | --- | --- | --- | --- |
| **RunPod Serverless** (this guide) | Prebuilt `worker-vllm` image, env-var config, queue or load-balancer endpoint | Yes, `workers-min 0`; idle timeout; FlashBoot; host model cache | Queue buffers requests; sync route times out at ~100 s | Per second of worker time, model download free; RTX 4090 $1.10/h, L4 $0.69/h, H100 $4.79/h | Cheapest consumer-GPU tier, prebuilt vLLM worker, no code |
| **Modal** | Python-defined container (`modal.Image ... uv_pip_install("vllm==...")`), `@app.server` class that launches `vllm serve`; weights in a `modal.Volume` | Yes, `scaledown_window` (e.g. 15 min); `min_containers` to keep warm | Requests get **503** while no container exists; client retries | Per second of GPU + CPU + memory; L4 $0.80/h, A10 $1.10/h, L40S $1.95/h, A100-80 $2.50/h, H100 $3.95/h; $30/month free credit | Infra-as-Python, fast iteration, strong cold-start tooling (memory snapshots) |
| **Vast.ai Serverless** | Endpoint + workergroup over a serverless template (vLLM images provided); workers run a PyWorker that proxies to vLLM | Autoscaler with `min_workers`, `inactivity_timeout`, cold-worker pool; marketplace GPUs | SDK asks the engine for a worker address, then calls it directly | Per second, marketplace pricing (often lowest), on-demand/interruptible/reserved | Lowest raw GPU price; more moving parts; autoscaler was marked beta |
| **Google Cloud Run (GPU)** | Any container (Google's `pytorch-vllm-serve` or `vllm/vllm-openai`) with `--gpu 1 --gpu-type nvidia-l4` or `nvidia-rtx-pro-6000`; `--command vllm --args serve,...` | Yes, min instances 0, instance-based billing | HTTP request waits while the instance starts (startup probe up to 240 s); GPU driver ready in ~5 s, model load dominates | Per second of instance (CPU + memory + GPU); L4 needs ≥ 4 vCPU/16 GiB, RTX PRO 6000 20 vCPU/80 GiB; zonal redundancy off is cheaper | Already on GCP, want IAM, VPC, Cloud Logging, one Dockerfile |
| **AWS** | No GPU on Lambda or SageMaker Serverless Inference. Options: **SageMaker real-time endpoint with inference components** running the LMI (vLLM) container and *scale to zero* via Application Auto Scaling (`MinInstanceCount 0`); or **Bedrock Custom Model Import** (upload safetensors to S3, fully managed, on-demand) | SageMaker: yes, but scale-in takes ~25 min and scale-out ~5 min; Bedrock: managed | SageMaker: invocations **fail** with `NoCapacityInvocationFailures` until capacity exists, so queue (SQS) or retry; Bedrock: managed | SageMaker per instance-second (g5/g6 for 24 GB); Bedrock per model-copy per 5-minute window | Enterprise AWS estates. Note Bedrock import supports Llama, Mistral, Qwen2/2.5/3 and others but **not Gemma 4** today |
| **Hugging Face Inference Endpoints** | Pick the repo, pick **vLLM** as engine, pick a GPU; no Dockerfile | **Automatic Scale-to-Zero** after 15 min idle (fixed) | **502** while the replica starts; no queue, retry client-side | Per minute of uptime, from $0.50/GPU/h (T4) up; H100 about $4.50/h | Model already on the Hub, zero infra, want the Hub UI |

Honourable mentions with the same pattern (container or Python handler, scale to zero, per-second
GPU): Baseten (Truss), Beam, Koyeb, Fly.io GPUs, Replicate (Cog), and self-managed Kubernetes
with KEDA + vLLM's production stack for teams that already run a cluster.

Two snippets to show the same model on two other platforms:

```python
# Modal (modal deploy serve.py): the whole deployment is one Python file
import modal
app = modal.App("sunflower-gemma-2b")
image = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
         .uv_pip_install("vllm==0.30.0").env({"HF_TOKEN": "..."}))   # use modal.Secret in practice
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.server(image=image, gpu="L4:1", scaledown_window=15 * 60, startup_timeout=10 * 60,
            volumes={"/root/.cache/huggingface": hf_cache}, port=8000, target_concurrency=32)
class Server:
    @modal.enter()
    def start(self):
        import subprocess
        self.p = subprocess.Popen(["vllm", "serve", "Sunbird/Sunflower-Gemma4-E2B",
            "--served-model-name", "sunflower-gemma-2b", "--max-model-len", "8192",
            "--model-loader-extra-config", '{"enable_weights_track": false}',
            "--host", "0.0.0.0", "--port", "8000"])
    @modal.exit()
    def stop(self):
        self.p.terminate()
```

```bash
# Google Cloud Run with an L4 (scale to zero, one GPU per instance)
gcloud beta run deploy sunflower-gemma-2b --region europe-west4 \
  --image vllm/vllm-openai:v0.30.0 --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy \
  --cpu 8 --memory 32Gi --no-cpu-throttling --max-instances 2 --concurrency 32 --timeout 600 \
  --set-secrets HF_TOKEN=hf-token:latest --no-allow-unauthenticated \
  --startup-probe tcpSocket.port=8000,initialDelaySeconds=240,failureThreshold=1,timeoutSeconds=240,periodSeconds=240 \
  --command vllm \
  --args "serve,Sunbird/Sunflower-Gemma4-E2B,--served-model-name,sunflower-gemma-2b,--max-model-len,8192,--model-loader-extra-config,{\"enable_weights_track\": false},--host,0.0.0.0,--port,8000"
```

The decision usually comes down to: where is your data and identity already (cloud), how
much you want to write (Python vs YAML vs clicking), what GPU price you can live with, and
whether you need a request queue in front of cold starts (RunPod and Vast have one; Modal,
Cloud Run and Hugging Face make the client retry).

## 14. References

RunPod

- [Deploy vLLM on Runpod Serverless](https://docs.runpod.io/serverless/vllm/get-started) and
  [vLLM environment variables](https://docs.runpod.io/serverless/vllm/environment-variables)
- [Endpoint configurations](https://docs.runpod.io/serverless/endpoints/endpoint-configurations),
  [Cached models](https://docs.runpod.io/serverless/endpoints/model-caching),
  [Serverless pricing](https://docs.runpod.io/serverless/pricing), [GPU pricing](https://www.runpod.io/pricing)
- [Create API keys](https://docs.runpod.io/get-started/credentials),
  [Billing](https://docs.runpod.io/get-started/billing-information),
  [runpodctl](https://docs.runpod.io/runpodctl/overview) and its [releases](https://github.com/runpod/runpodctl/releases)
- [runpod-workers/worker-vllm](https://github.com/runpod-workers/worker-vllm): the env table,
  `src/engine_args.py` (env → engine args), release notes with the bundled vLLM version
- [Building Runpod's Model Store](https://www.runpod.io/blog/building-runpods-model-store): why the cache pins a commit

vLLM and the model

- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html),
  [OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html),
  [engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- [Sunbird/Sunflower-Gemma4-E2B](https://huggingface.co/Sunbird/Sunflower-Gemma4-E2B) model card
- Sunbird's production deployment of the 9B and 14B Sunflower models, with the full
  operational history this guide distils:
  [SunbirdAI/sunflower-multi-models-deployment-runpod](https://github.com/SunbirdAI/sunflower-multi-models-deployment-runpod)
- [PatrickCmd/sunflower-vllm-worker](https://github.com/PatrickCmd/sunflower-vllm-worker): the baked audio image

Other platforms

- Modal: [vLLM inference example](https://modal.com/docs/examples/vllm_inference), [pricing](https://modal.com/pricing)
- Vast.ai: [Serverless architecture](https://docs.vast.ai/serverless/architecture)
- Google Cloud Run: [GPU configuration](https://docs.cloud.google.com/run/docs/configuring/services/gpu),
  [vLLM on Cloud Run GPUs codelab](https://codelabs.developers.google.com/codelabs/how-to-run-inference-cloud-run-gpu-vllm)
- AWS: [SageMaker scale-down-to-zero](https://aws.amazon.com/blogs/machine-learning/unlock-cost-savings-with-the-new-scale-down-to-zero-feature-in-amazon-sagemaker-inference/),
  [Bedrock Custom Model Import](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html)
- Hugging Face: [Inference Endpoints autoscaling](https://huggingface.co/docs/inference-endpoints/en/autoscaling)
