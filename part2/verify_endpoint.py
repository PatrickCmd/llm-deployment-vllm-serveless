"""Verify a RunPod Serverless vLLM endpoint end to end: cold start, models.list, chat, streaming,
and (optionally) vision and audio for a multimodal model such as Sunflower-Gemma4-E2B.

    export RUNPOD_API_KEY=rpa_...   ENDPOINT_ID=...   SERVED_MODEL=sunflower-gemma-2b
    python part2/verify_endpoint.py                 # cold start + text checks
    python part2/verify_endpoint.py --vision        # + describe a generated test image
    python part2/verify_endpoint.py --audio clip.mp3   # + transcribe a <=30 s clip

Why the cold start goes through the async queue route (/run + /status) and not through the
OpenAI route: a scaled-to-zero endpoint needs 1-4 minutes to boot a worker and load the
model, and the synchronous /openai/v1 route sits behind a ~100 s edge timeout. Submitting a
queue job is also what makes the endpoint start a worker in the first place.
"""

import argparse
import base64
import io
import os
import sys
import time

import httpx
from openai import OpenAI

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["ENDPOINT_ID"]
SERVED_MODEL = os.environ.get("SERVED_MODEL", "sunflower-gemma-2b")
BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
SYSTEM_PROMPT = ("You are Sunflower, a helpful assistant made by Sunbird AI who knows many "
                 "African languages.")
PROMPT = "Translate from English to Luganda: Where is the nearest hospital?"

http = httpx.Client(headers={"Authorization": f"Bearer {API_KEY}"}, timeout=60.0)


def health() -> dict:
    return http.get(f"{BASE}/health").json()


def cold_start(timeout: int = 1800) -> float:
    """Submit one queue job and poll until it completes. Returns the seconds it took."""
    t0 = time.time()
    job = http.post(f"{BASE}/run", json={"input": {
        "messages": [{"role": "user", "content": PROMPT}],
        "sampling_params": {"max_tokens": 64, "temperature": 0},
    }}).json()
    print(f"  submitted job {job['id']}")
    while time.time() - t0 < timeout:
        s = http.get(f"{BASE}/status/{job['id']}").json()
        w = health().get("workers", {})
        print(f"  [{time.time() - t0:5.0f}s] job {s['status']:12s} workers: "
              f"init={w.get('initializing', 0)} running={w.get('running', 0)} "
              f"idle={w.get('idle', 0)} throttled={w.get('throttled', 0)} "
              f"unhealthy={w.get('unhealthy', 0)}")
        if s["status"] == "COMPLETED":
            out = s.get("output")
            text = out
            if isinstance(out, list) and out and isinstance(out[0], dict):
                text = out[0].get("choices", [{}])[0].get("tokens") or out[0]
            print(f"  cold output: {str(text)[:300]}")
            return time.time() - t0
        if s["status"] == "FAILED":
            sys.exit(f"cold-start job FAILED: {s}")
        time.sleep(15)
    sys.exit("cold start timed out; read the worker logs in the console before assuming it is broken")


def openai_checks(client: OpenAI) -> None:
    served = [m.id for m in client.models.list()]
    assert SERVED_MODEL in served, f"{SERVED_MODEL} not in {served}"
    print(f"  models.list: {served}")

    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=SERVED_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": PROMPT}],
        max_tokens=128, temperature=0.0)
    print(f"  chat ({time.perf_counter() - t0:.1f}s): {r.choices[0].message.content.strip()[:200]}")
    print(f"  usage: {r.usage.prompt_tokens} prompt + {r.usage.completion_tokens} completion tokens")

    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=SERVED_MODEL,
        messages=[{"role": "user", "content": "Translate from English to Luganda: Good morning, how are you?"}],
        max_tokens=64, temperature=0.0, stream=True)
    chunks, first = 0, None
    for c in stream:
        if c.choices and c.choices[0].delta.content:
            chunks += 1
            first = first or time.perf_counter() - t0
    assert chunks > 1, "streaming came back as one blob; check RAW_OPENAI_OUTPUT=1"
    print(f"  streaming: OK ({chunks} chunks, first token after {first:.2f}s)")


def test_png_b64() -> str:
    """A small image that needs no extra packages: a yellow sunflower-ish disc on a blue sky."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        sys.exit("pip install pillow for --vision")
    img = Image.new("RGB", (256, 256), (120, 180, 240))
    d = ImageDraw.Draw(img)
    for i in range(12):
        import math
        a = i * math.pi / 6
        d.ellipse([128 + 70 * math.cos(a) - 28, 128 + 70 * math.sin(a) - 28,
                   128 + 70 * math.cos(a) + 28, 128 + 70 * math.sin(a) + 28], fill=(250, 200, 30))
    d.ellipse([88, 88, 168, 168], fill=(90, 50, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def vision_check(client: OpenAI) -> None:
    r = client.chat.completions.create(
        model=SERVED_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{test_png_b64()}"}},
            {"type": "text", "text": "Describe this image in one sentence."},
        ]}],
        max_tokens=96, temperature=0.0)
    print(f"  vision: {r.choices[0].message.content.strip()[:300]}")


def audio_check(client: OpenAI, path: str) -> None:
    fmt = os.path.splitext(path)[1].lstrip(".").lower() or "mp3"
    audio_b64 = base64.b64encode(open(path, "rb").read()).decode()
    r = client.chat.completions.create(
        model=SERVED_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": [
                      {"type": "input_audio", "input_audio": {"data": audio_b64, "format": fmt}},
                      {"type": "text", "text": "Please transcribe this Luganda audio."},
                  ]}],
        max_tokens=256, temperature=0.0)
    print(f"  audio transcript: {r.choices[0].message.content.strip()[:500]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", action="store_true", help="also send a base64 PNG")
    ap.add_argument("--audio", metavar="FILE", help="also transcribe a <=30 s audio clip")
    ap.add_argument("--skip-cold", action="store_true", help="skip the queue cold-start step")
    args = ap.parse_args()

    print(f"=== {SERVED_MODEL} @ {ENDPOINT_ID} ===")
    print(f"  health before: {health()}")
    if not args.skip_cold:
        secs = cold_start()
        print(f"  cold start + first job: {secs:.0f}s")

    client = OpenAI(api_key=API_KEY, base_url=f"{BASE}/openai/v1")
    openai_checks(client)
    if args.vision:
        vision_check(client)
    if args.audio:
        audio_check(client, args.audio)
    print("\nAll checks passed.")
