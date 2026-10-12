#!/usr/bin/env bash
# Create the Sunflower-Gemma4-E2B vLLM serverless endpoint on RunPod: one template + one
# scale-to-zero endpoint with the model host-cached. Companion to part2/README.md, which
# explains every flag. Requires runpodctl >= 2.4.0 (tested with 2.7.2 and 2.15.0 flags) and a
# .env with RUNPOD_API_KEY + HF_TOKEN next to this script.
#
#   cd part2 && ./deploy_gemma_e2b.sh
#
# The script prints the template id and the endpoint id; put the endpoint id in .env as
# ENDPOINT_ID, then run `python verify_endpoint.py`.
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a

MODEL_REPO="Sunbird/Sunflower-Gemma4-E2B"
# Pin an explicit commit, not `main`: RunPod resolves the ref ONCE at create time anyway, and a
# hash in version control tells you later which weights a deployment served.
MODEL_COMMIT="${MODEL_COMMIT:-44e5d28f2012b4a76fe1855a62b1f184d2b6cfe0}"
SERVED_NAME="sunflower-gemma-2b"              # what clients pass as model=
WORKER_IMAGE="${WORKER_IMAGE:-runpod/worker-v1-vllm:v2.28.0}"   # vLLM 0.30.0, CUDA 13.0
NAME_SUFFIX="${NAME_SUFFIX:-}"                # e.g. "-staging" to deploy a second copy

# Template env. Every UPPERCASED key that is not a worker setting becomes a vLLM engine
# argument (MAX_MODEL_LEN -> --max-model-len, MODEL_LOADER_EXTRA_CONFIG ->
# --model-loader-extra-config, ...). See the README for why each one is here.
ENV_JSON=$(python3 - "$MODEL_REPO" "$SERVED_NAME" <<'PY'
import json, os, sys
print(json.dumps({
    "MODEL_NAME": sys.argv[1],
    "HF_TOKEN": os.environ["HF_TOKEN"],
    "OPENAI_SERVED_MODEL_NAME_OVERRIDE": sys.argv[2],
    "MAX_MODEL_LEN": "8192",
    "GPU_MEMORY_UTILIZATION": "0.90",
    "RAW_OPENAI_OUTPUT": "1",
    "ENFORCE_EAGER": "true",
    "MODEL_LOADER_EXTRA_CONFIG": json.dumps({"enable_weights_track": False}),
    "MM_PROCESSOR_KWARGS": json.dumps({"max_length": 480000, "truncation": True}),
}))
PY
)

TEMPLATE_ID=$(runpodctl template create --name "sunflower-gemma-2b-vllm${NAME_SUFFIX}" --serverless \
  --image "$WORKER_IMAGE" --container-disk-in-gb 40 --env "$ENV_JSON" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "template: $TEMPLATE_ID"

ENDPOINT_JSON=$(runpodctl serverless create --template-id "$TEMPLATE_ID" \
  --name "sunflower-gemma-2b${NAME_SUFFIX}" \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --workers-min 0 --workers-max 2 --idle-timeout 120 \
  --scale-by delay --scale-threshold 300 \
  --min-cuda-version 13.0 \
  --model-reference "https://huggingface.co/${MODEL_REPO}:${MODEL_COMMIT}")
ENDPOINT_ID=$(echo "$ENDPOINT_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "endpoint: $ENDPOINT_ID"
echo
echo "Add to .env:  ENDPOINT_ID=$ENDPOINT_ID  SERVED_MODEL=$SERVED_NAME"
echo "OpenAI base_url: https://api.runpod.ai/v2/$ENDPOINT_ID/openai/v1"
echo "Then: python verify_endpoint.py --vision"
