Hardware: NVIDIA L4 24GB (Ada), single GPU, Vast.ai instance.
Results from these runs are summarized in ../README.md

## Credentials
No secrets in this file. Everything below reads $HF_TOKEN, $NGC_API_KEY, $VAST_HOST and
$VAST_PORT from the repo-root .env, which is gitignored. First time:

    cp .env.example .env        # then fill in the real values

Before running any command here:

    set -a; source .env; set +a

VAST_HOST/VAST_PORT change every time you rent a new instance - update .env, not this file.
The remote host has no access to your .env: paste HF_TOKEN there manually when step 4 asks.

ssh -p $VAST_PORT root@$VAST_HOST -L 8000:localhost:8000


# 1. start up the server
1.0 start instance
1.1 increase the limit
    ulimit -n 65536

    - Allow reuse of TIME_WAIT sockets for new connections
        sysctl -w net.ipv4.tcp_tw_reuse=1 2>/dev/null || echo "net.ipv4.tcp_tw_reuse=1" >> /etc/sysctl.conf

    - Increase ephemeral port range
        sysctl -w net.ipv4.tcp_fin_timeout=15 2>/dev/null || true

2.2 check
View current soft limit (what applications will hit first)
    ulimit -n

    # View current hard limit (the maximum a non-root user can raise ulimit to)
    ulimit -Hn

2. activate main env
    source /venv/main/bin/activate # pre-configured virtual environment located at /venv/main/
3. upgrade pip & install vllm
    - pip install --upgrade pip wheel
    - pip install vllm

4. export token: 
    export HF_TOKEN=...   # paste the value from your local .env (see Credentials, top of file)
5. vllm serve

5.2 check limit
cat /proc/<PID>/limits | grep "Max open files"

6. forward port:
    ssh -p $VAST_PORT root@$VAST_HOST -L 8000:localhost:8000

7. run metrics scraping script (locally, through the tunnel):
    python scripts/scrape_metrics.py <config>     # e.g. baseline | optimized | quantized
    -> writes results/exp2/vllm_metrics_<config>.json on Ctrl-C
8. rnu guidellm inference workload
9. kill it & repeat from (5)

## Paths
Commands below run in two places - do not mix them up:
- REMOTE (the GPU host, cwd /workspace): vllm serve and the quantize scripts only.
      scp -P <port> scripts/quantize_fp8.py root@<host>:/workspace/
- LOCAL (this repo, run from the repo root): guidellm, scripts/scrape_metrics.py, the notebook.
  Both talk to the server over the SSH tunnel on localhost:8000, so `--data ./rag_workload.jsonl`
  and the `--output` paths below are LOCAL paths - the result JSONs are written straight into
  this repo, nothing needs to be copied back. (Confirmed by `metadata.platform` in every report
  in results/exp2/, which records macOS-arm64, i.e. the client ran here.)
  Adjust the paths in the guidellm commands below to data/rag_workload.jsonl and
  results/expN/... , or cd into the relevant directory first.



# 2. baseline 

## Server
ulimit -n 65536 && vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port 8000 \
    --host 0.0.0.0

## client
guidellm run \
    --backend '{"kind":"openai_http","target":"http://localhost:8000/v1","model":"meta-llama/Meta-Llama-3.1-8B-Instruct","timeout":120.0}' \
    --data '{"kind": "json_file", "path": "./rag_workload.jsonl", "load_kwargs": {"split": "train"}}' \
    --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":"prompt","target_output_len_column":"output_len"}}' \
    --data-loader kind=pytorch,samples=-1 \
    --profile kind=poisson,rate=8.0,max_concurrency=128 \
    --constraint kind=max_requests,count=2000 \
    --seed kind=static,value=42 \
    --output kind=json,path=./vllm-rag-baseline-fp16-cache.json

* concurrency
guidellm benchmark run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000/v1","model":"meta-llama/Meta-Llama-3.1-8B-Instruct","timeout":120.0}' \
  --profile '{"kind": "concurrent", "streams": [1, 2, 4, 8, 16, 32, 64]}' \
  --data '{"kind": "json_file", "path": "./rag_workload.jsonl", "load_kwargs": {"split": "train"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":"prompt","target_output_len_column":"output_len"}}' \
  --data-loader kind=pytorch,samples=-1 \
  --constraint '{"kind": "max_duration", "seconds": 60}' \
  --seed kind=static,value=42 \
    --output kind=json,path=./vllm-rag-baseline-fp16-concurrent.json

### kills server: 
    pkill -9 -f "vllm"


## 3. chunked prefil and prefix caching
### server
ulimit -n 65536 && vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --port 8000 \
    --host 0.0.0.0

### client
##### 8 RPS
guidellm run \
    --backend '{"kind":"openai_http","target":"http://localhost:8000/v1","model":"meta-llama/Meta-Llama-3.1-8B-Instruct","timeout":120.0}' \
    --data '{"kind": "json_file", "path": "./rag_workload.jsonl", "load_kwargs": {"split": "train"}}' \
    --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":"prompt","target_output_len_column":"output_len"}}' \
    --data-loader kind=pytorch,samples=-1 \
    --profile kind=poisson,rate=8.0,max_concurrency=128 \
    --constraint kind=max_requests,count=2000 \
    --seed kind=static,value=42 \
    --output kind=json,path=./vllm-rag-optimized-fp16-cache.json

* concurrency
guidellm benchmark run \
  --backend '{"kind":"openai_http","target":"http://localhost:8000/v1","model":"meta-llama/Meta-Llama-3.1-8B-Instruct","timeout":120.0}' \
  --profile '{"kind": "concurrent", "streams": [1, 2, 4, 8, 16, 32, 64]}' \
  --data '{"kind": "json_file", "path": "./rag_workload.jsonl", "load_kwargs": {"split": "train"}}' \
  --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":"prompt","target_output_len_column":"output_len"}}' \
  --data-loader kind=pytorch,samples=-1 \
  --constraint '{"kind": "max_duration", "seconds": 60}' \
  --seed kind=static,value=42 \
  --output kind=json,path=./vllm-rag-optimized-fp16-concurrent.json

### kills server: 
    pkill -9 -f "vllm"

## 4. qunatization (weight, kvcahce, PagedAttention)
- pip install llmcompressor

- copy script to remote instance & run it quantize the model
    * given that: ssh -p $VAST_PORT root@$VAST_HOST -L 8000:localhost:8000
    scp -P $VAST_PORT -o StrictHostKeyChecking=no scripts/quantize_fp8.py root@$VAST_HOST:/workspace/

- notice: the number of kv cache blocks

Ahead-of-Time (AOT) quantization (fp 8; weights and activation)

clean up cache
    rm -rf ~/.cache/huggingface/hub/*

command:
    server:
        vllm serve ./Meta-Llama-3.1-8B-Instruct-FP8-Dynamic \
            --dtype auto \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.90 \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --port 8000 \
            --host 0.0.0.0

    client:
        guidellm run \
        --backend '{"kind":"openai_http","target":"http://localhost:8000/v1","model":"./Meta-Llama-3.1-8B-Instruct-FP8-Dynamic","timeout":120.0}' \
        --data '{"kind": "json_file", "path": "./rag_workload.jsonl", "load_kwargs": {"split": "train"}}' \
        --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":"prompt","target_output_len_column":"output_len"}}' \
        --data-loader kind=pytorch,samples=-1 \
        --profile kind=poisson,rate=8.0,max_concurrency=128 \
        --constraint kind=max_requests,count=2000 \
        --seed kind=static,value=42 \
        --output kind=json,path=./vllm-rag-quantized-8fp-cache.json

    * concurrency
        guidellm run \
        --backend '{"kind":"openai_http","target":"http://localhost:8000/v1","model":"./Meta-Llama-3.1-8B-Instruct-FP8-Dynamic","timeout":120.0}' \
        --profile '{"kind": "concurrent", "streams": [1, 2, 4, 8, 16, 32, 64, 128, 256, 500]}' \
        --data '{"kind": "json_file", "path": "./rag_workload.jsonl", "load_kwargs": {"split": "train"}}' \
        --data-column-mapper '{"kind":"generative_column_mapper","column_mappings":{"text_column":"prompt","target_output_len_column":"output_len"}}' \
        --data-loader kind=pytorch,samples=-1 \
        --constraint '{"kind": "max_duration", "seconds": 30}' \
        --seed kind=static,value=42 \
        --output kind=json,path=./vllm-rag-optimized-fp16-concurrent.json

