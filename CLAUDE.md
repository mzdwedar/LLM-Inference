# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A benchmarking/experiment workspace (not a library or deployable app) for measuring LLM inference performance on a
RAG-style workload. There is no package, no test suite, and no build step for the Python code — the "product" is the
measurement pipeline plus the result JSONs and `notebooks/analysis.ipynb`.

Target model throughout: `meta-llama/Meta-Llama-3.1-8B-Instruct` (and its FP8 derivatives), served on a
single **NVIDIA L4 24GB (Ada)** Vast.ai instance. The 24GB ceiling is the defining constraint of every
result — bf16 weights alone eat ~16GB of the ~21.6GB budget, leaving room for only ~23 concurrent
requests' worth of KV cache. `README.md` is the results summary and explains this.

## Layout

```
docs/       PLAN.md (intent), RUNBOOK.md (operational log — authoritative for how runs were executed)
scripts/    generate_prompts.py, scrape_metrics.py, quantize_fp8.py, quantize_fp8_kv.py
notebooks/  analysis.ipynb
data/       rag_workload.jsonl (generated)
results/    exp1/ (prior run), exp2/ (current run)
deploy/     Dockerfile, build_and_push.sh, benchmark_config.yaml, nim/ (NGC profile probing)
baseline/   server.py (naive FastAPI strawman)
```

Local scripts resolve paths against the repo root via `__file__`, so they run from any cwd. The notebook uses
`../results/exp2/...` relative paths and therefore must stay in `notebooks/`.

## The experiment loop

Everything follows the same four-stage loop; understanding it explains why the files exist:

1. **Generate workload** — `scripts/generate_prompts.py` writes `data/rag_workload.jsonl` (2000 rows of
   `{"prompt", "output_len"}`). Prompts are deliberately built as *bursts of 3–6 requests sharing an identical long
   prefix* (system instruction + shuffled "retrieved documents"), with only the trailing `User Query:` differing. This
   structure is the point: it makes vLLM's automatic prefix caching (APC) measurable. Changing the prompt assembly
   changes the cache-hit rate and invalidates comparison against existing result JSONs.
2. **Serve** — a GPU host (rented Vast.ai instance, SSH port-forwarded to `localhost:8000`) runs `vllm serve` with a
   per-experiment flag set. `docs/RUNBOOK.md` has the exact server + client commands for each configuration, in order.
3. **Measure** — two independent recorders run concurrently, **both locally over the SSH tunnel** (the
   `metadata.platform` field in every report reads `macOS-arm64`, confirming the client ran here, not on
   the GPU host — so result JSONs land straight in the repo and nothing is copied back):
   - *Client side*: `guidellm` writes `vllm-rag-<config>.json` (per-request latency/throughput).
   - *Server side*: `scripts/scrape_metrics.py` polls
     `http://localhost:8000/metrics` once per second, and writes `results/<exp>/vllm_metrics_<config>.json` on Ctrl-C
     (KV-cache usage, queue depth, preemptions, prefix-cache hit rate).
4. **Analyze** — `notebooks/analysis.ipynb` loads both files per config and compares them.

### Experiment configurations (the naming convention)

Result filenames encode the config, and the pairing between the two recorders is by convention only:

| Config | guidellm output | scrape_metrics output | Server flags |
|---|---|---|---|
| baseline | `vllm-rag-baseline-fp16-cache.json` | `vllm_metrics_baseline.json` | bf16, no chunked prefill, no prefix caching |
| optimized | `vllm-rag-optimized-fp16-cache.json` | `vllm_metrics_optimized.json` | bf16, chunked prefill + prefix caching |
| quantized | `vllm-rag-quantized-8fp-cache.json` | `vllm_metrics_quantized.json` | FP8-Dynamic weights + chunked prefill + prefix caching |
| quantized/concurrent | `vllm-rag-optimized-8fp-concurrent.json` | `vllm_metrics_quantized_concurrent.json` | as above, concurrency sweep profile |

Gotcha worth knowing: Poisson-rate runs use `--profile kind=poisson,rate=8.0` with `max_requests=2000`; concurrency
sweeps use `--profile '{"kind":"concurrent","streams":[...]}'` with a `max_duration` constraint. The two produce
differently shaped guidellm JSON (single vs. many `benchmarks[]` entries) — the notebook has a separate parser for
each (`load_guidellm_multidim` vs `parse_guidellm_results`).

`results/exp1/` holds an earlier full set of results under the same names; keep it untouched as the prior run. New
runs should go to a new `results/expN/` rather than overwriting — `scrape_metrics.py` refuses to clobber an existing
file and takes `--exp` for this.

## Commands

Local (`.venv` is a Python 3.12 venv with guidellm/transformers/jupyter):

```bash
source .venv/bin/activate
python scripts/generate_prompts.py             # regenerate data/rag_workload.jsonl
python scripts/scrape_metrics.py baseline      # poll /metrics; Ctrl-C writes results/exp2/vllm_metrics_baseline.json
python scripts/scrape_metrics.py foo --exp exp3
jupyter lab notebooks/analysis.ipynb
ssh -p <port> root@<host> -L 8000:localhost:8000   # tunnel so both localhost:8000 users work
```

Remote GPU host — copy the exact server/client command blocks from `docs/RUNBOOK.md` rather than reconstructing them.
Recurring pieces:

```bash
ulimit -n 65536        # required before vllm serve; guidellm at high concurrency exhausts the default fd limit
pkill -9 -f "vllm"     # tear down between configurations
scp -P <port> scripts/quantize_fp8.py root@<host>:/workspace/       # quantization runs on the GPU host
```

Quantization scripts run on the GPU box and output a directory that `vllm serve` is pointed at directly:
- `scripts/quantize_fp8.py`: FP8_DYNAMIC weights+activations, no calibration data, saves to
  `/workspace/Meta-Llama-3.1-8B-Instruct-FP8-Dynamic`. This is the one used for the results above.
- `scripts/quantize_fp8_kv.py`: adds per-attention-head FP8 KV-cache quantization, requires calibration on
  ultrachat_200k. Marked as the next step in the runbook; no result files from it yet.

## Side tracks (not part of the vLLM result set)

- `baseline/server.py` — naive FastAPI + HF `pipeline` OpenAI-compatible endpoint, the "no batching" strawman from
  `docs/PLAN.md`. Uses a small Qwen model, not Llama. Its trailing comment block holds aiperf commands.
- `deploy/Dockerfile` + `deploy/build_and_push.sh` — Triton + TensorRT-LLM image
  (`mzdwedar/triton-trtllm-light:25.01`), built `--platform linux/amd64` from macOS.
- `deploy/nim/` — probing NGC/NIM for available TensorRT-LLM engine profiles. `get_profiles.sh` pulls ~700MB of OCI
  blobs into `deploy/nim/llama-nim/` and `deploy/nim/manifest_out/`; both are gitignored and re-derivable.
- `deploy/benchmark_config.yaml` — an unused guidellm config-file form of the poisson run; the runbook passes flags
  instead.

## Secrets

`docs/RUNBOOK.md` and `deploy/nim/get_profile.py` contain a live HF token and an NVIDIA NGC API key inline, and
`vastssh_key` is an unencrypted private key in the working tree. The key files are now gitignored; the inline tokens
are not (they live inside files that should stay tracked). Do not propagate them into new files; prefer
`export HF_TOKEN=...` / env lookups when editing these.
