import argparse
import json
import re
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(
    description="Poll a vLLM server's /metrics endpoint once per second; "
    "write the collected history to JSON on Ctrl-C."
)
parser.add_argument(
    "config",
    help="Configuration label for this run, e.g. baseline, optimized, "
    "quantized, quantized_concurrent. Determines the output filename.",
)
parser.add_argument(
    "--exp",
    default="exp2",
    help="Experiment directory under results/ to write into (default: exp2).",
)
parser.add_argument(
    "--url",
    default="http://localhost:8000/metrics",
    help="vLLM metrics endpoint (default: %(default)s).",
)
args = parser.parse_args()

url = args.url
out_path = REPO_ROOT / "results" / args.exp / f"vllm_metrics_{args.config}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
if out_path.exists():
    parser.error(
        f"{out_path} already exists - refusing to overwrite a previous run. "
        "Pass a different config label or --exp."
    )

print(f"Polling {url}; will write to {out_path} on Ctrl-C.")

print(
    f"{'Time':<8} | {'GPU%':<5} | {'CPU%':<5} | {'Run':<4} | {'Wait':<4} | {'Swap':<4} |"
    f" {'Preempt':<7} | {'Queue(s)':<8} | {'TTFT(ms)':<8} | {'E2E(s)':<6} |"
    f" {'Gen Tok/s':<9} | {'Hit Rate':<8}"
)
print("-" * 115)

metrics_history = []
prev_time = None
prev_gen_tokens = None

try:
    while True:
        vllm_metrics = {}

        try:
            res = requests.get(url, timeout=0.8).text
            for line in res.split("\n"):
                if line.startswith("#") or not line.strip():
                    continue

                # Strip labels inside {...} to get clean metric key
                parts = line.split()
                raw_key = parts[0]
                clean_metric_name = re.sub(r"\{.*\}", "", raw_key)
                val = parts[-1]

                try:
                    parsed_val = (
                        float(val) if ("." in val or "e" in val.lower()) else int(val)
                    )
                except ValueError:
                    parsed_val = val

                # Store clean metric values
                vllm_metrics[clean_metric_name] = parsed_val

        except Exception:
            pass

        timestamp = time.time()

        # 1. Cache Usages
        raw_gpu_kv = vllm_metrics.get(
            "vllm:kv_cache_usage_perc",
            vllm_metrics.get("vllm:gpu_cache_usage_factor", 0.0),
        )
        gpu_kv = round(float(raw_gpu_kv) * 100, 1)

        raw_cpu_kv = vllm_metrics.get("vllm:cpu_cache_usage_perc", 0.0)
        cpu_kv = round(float(raw_cpu_kv) * 100, 1)

        # 2. Request States
        running = int(vllm_metrics.get("vllm:num_requests_running", 0))
        waiting = int(vllm_metrics.get("vllm:num_requests_waiting", 0))
        swapped = int(vllm_metrics.get("vllm:num_requests_swapped", 0))
        preemptions = int(vllm_metrics.get("vllm:num_preemptions_total", 0))

        # 3. Token Counters
        prompt_tokens = int(vllm_metrics.get("vllm:prompt_tokens_total", 0))
        gen_tokens = int(vllm_metrics.get("vllm:generation_tokens_total", 0))

        # 4. Latencies (Extracting _sum and _count metrics accurately)
        q_sum = vllm_metrics.get("vllm:request_queue_time_seconds_sum")
        q_count = vllm_metrics.get("vllm:request_queue_time_seconds_count")
        queue_time = (
            round(q_sum / q_count, 3)
            if (q_sum is not None and q_count and q_count > 0)
            else 0.0
        )

        ttft_sum = vllm_metrics.get("vllm:time_to_first_token_seconds_sum")
        ttft_count = vllm_metrics.get("vllm:time_to_first_token_seconds_count")
        avg_ttft_ms = (
            round((ttft_sum / ttft_count) * 1000, 2)
            if (ttft_sum is not None and ttft_count and ttft_count > 0)
            else 0.0
        )

        e2e_sum = vllm_metrics.get("vllm:e2e_request_latency_seconds_sum")
        e2e_count = vllm_metrics.get("vllm:e2e_request_latency_seconds_count")
        avg_e2e_s = (
            round(e2e_sum / e2e_count, 3)
            if (e2e_sum is not None and e2e_count and e2e_count > 0)
            else 0.0
        )

        # 5. Throughput Rate Calculation
        if prev_time is not None and prev_gen_tokens is not None:
            dt = timestamp - prev_time
            d_tokens = gen_tokens - prev_gen_tokens
            gen_throughput = round(d_tokens / dt, 2) if dt > 0 else 0.0
        else:
            gen_throughput = 0.0

        prev_time = timestamp
        prev_gen_tokens = gen_tokens

        # 6. Prefix Cache Hit Rate
        hits = vllm_metrics.get("vllm:prefix_cache_hits_total", 0)
        queries = vllm_metrics.get("vllm:prefix_cache_queries_total", 0)
        hit_rate = round(hits / queries, 3) if queries > 0 else 0.0

        # Output formatting
        ts_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
        print(
            f"{ts_str:<8} | {gpu_kv:<4}% | {cpu_kv:<4}% | {running:<4} | {waiting:<4} |"
            f" {swapped:<4} | {preemptions:<7} | {queue_time:<8} |"
            f" {avg_ttft_ms:<8} | {avg_e2e_s:<6} | {gen_throughput:<9} | {hit_rate:<8}"
        )

        metrics_history.append({
            "timestamp": timestamp,
            "gpu_cache_usage_perc": gpu_kv,
            "cpu_cache_usage_perc": cpu_kv,
            "num_requests_running": running,
            "num_requests_waiting": waiting,
            "num_requests_swapped": swapped,
            "num_preemptions_total": preemptions,
            "request_queue_time_seconds": queue_time,
            "avg_ttft_ms": avg_ttft_ms,
            "avg_e2e_latency_s": avg_e2e_s,
            "avg_generation_throughput_toks_per_s": gen_throughput,
            "prefix_cache_hit_rate": hit_rate,
            "prompt_tokens_total": prompt_tokens,
            "generation_tokens_total": gen_tokens,
        })

        time.sleep(1)

except KeyboardInterrupt:
    print(f"\nSaving {len(metrics_history)} samples to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(metrics_history, f, indent=4)
    print("Done!")