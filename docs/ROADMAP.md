# Roadmap (pre-benchmark planning notes)

These notes were written before any runs and are kept verbatim as the original intent. The vLLM
stages are done and reported in [../README.md](../README.md); the TensorRT-LLM and KV-cache
quantization stages have not been executed.

---

# Highly Optimized inference pipeline
We'll tackle the following bottleneck:
- Inefficient memory access.
- Memory requirement (model and KV cache)
- Scalability

# Phase 1: baseline
- run the model with pytorch / HF

identify issues / metrics:

nsights systems
- memory latency [kernel launch]
- CPU bound latency
- sync stalls
- KV fragmentation

nsights compute
- memory bandwidth [roofline]
- unfused element-wise kernels





Milestones
* first establish latency SLA (batch=1), then increase batch while preserving SLO

### Metrics: TTFT; TOFT

- PyTorch [eager mode]
- PyTorch [compiled]
- TensorRT-LLM
- vLLM [plain] - KV Cache optimization [PagedAttention, cont. Attention, ]
- VLLM [Quantized (llmCompressor)]
- 