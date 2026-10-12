# Screenshots for Part 2

The guide in [../README.md](../README.md) references these files. Capture them from your own
RunPod console while following the steps (the CLI output in the guide is already real). Crop
to the relevant panel, blur or crop API keys and billing details, and keep PNG at 1x or 2x.

| File | Where in the console | What must be visible | Guide section |
| --- | --- | --- | --- |
| `01-signup.png` | console.runpod.io/signup | The sign-up page | 2 |
| `02-billing.png` | Billing | The "Add credits" amounts and the Auto-pay toggle (hide the card) | 2 |
| `03-api-key.png` | Settings → API Keys → Create API Key | The name field and the permission picker (All / Restricted / Read Only) **before** the key is shown | 2 |
| `04-runpodctl-doctor.png` | your terminal | `runpodctl version`, `runpodctl doctor` healthy, `runpodctl user` (hide the balance if you like) | 3.2 |
| `05-template-console.png` | Serverless → Templates → sunflower-gemma-2b-vllm | Image `runpod/worker-v1-vllm:v2.28.0`, 40 GB disk, the environment variable list with `HF_TOKEN` value hidden | 5 |
| `06-new-endpoint.png` | Serverless → New Endpoint | GPU choice, Active workers 0, Max workers 2, Idle timeout 120, FlashBoot, Model (cached) field, CUDA filter | 6 |
| `07-endpoint-overview.png` | Serverless → the endpoint | Endpoint id, GPU, worker counts, the Model field showing the pinned commit | 6 |
| `08-requests-tab.png` | endpoint → Requests | The first job COMPLETED with its Delay (cold start) and Execution time columns | 7.1 |
| `09-workers-tab.png` | endpoint → Workers | Worker list with states (initializing / idle / running) and data centres | 10 |
| `10-worker-logs.png` | endpoint → Workers → a worker → Logs | The system lines `image ready, initializing model files` and vLLM's `Loading weights took` line | 10 |
| `11-metrics-tab.png` | endpoint → Metrics | Requests, execution time and worker count charts over the test period | 10 |
| `12-billing-after.png` | Billing → Serverless | The spend line for this endpoint after the walkthrough (cents) | 9 |

Optional, for the "other platforms" section: a screenshot of the Modal dashboard or the Cloud
Run service page if you deploy the same model there.
