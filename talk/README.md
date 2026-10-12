# Talk: Deploying Custom LLMs with vLLM

Slides for the PyCon Africa 2026 talk. Part 1 is the live session; Part 2 is the self-paced
serverless deployment deck that accompanies [`part2/README.md`](../part2/README.md).

## Part 1

- **Live deck (present, page through, download as PDF or PPTX):**
  https://claude.ai/artifact/MF9zZcWEMK33hp7TouBW91
- **Source:** `deck/project/deck.json` is the slide index (order, sections, typefaces) and
  `deck/project/slides/*.html` holds one slide per file on a 1920x1080 canvas with inline styles.
  Images live in `deck/project/assets/` and are referenced by relative path.

The demo numbers on the slides come from running `task_1` to `task_8` on an Apple Silicon MacBook with
the CPU build of vLLM; the dashboard panels in `assets/dash-*.png` are screenshots of `task_8_dashboard.py`.

## Part 2

- **Live deck (present, page through, download as PDF or PPTX):**
  https://claude.ai/artifact/7MD1i6aBFWz9eUd25uYA6G
- **Source:** `part2-deck/project/deck.json` plus `part2-deck/project/slides/*.html`, same format as Part 1,
  18 slides: recap, architecture, why serverless, account and CLI, the model, packaging, template,
  endpoint, verification, client, cold starts, cost, monitoring, other platforms, lessons, resources.

Every number on the Part 2 slides (cold-start timings, prices, request latencies, the verifier output)
comes from the live run recorded in `part2/README.md` on 2026-10-12.
