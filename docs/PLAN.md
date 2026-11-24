# Optimization loop
Where Nsight Fits: The Optimization Loop
You should never start with Nsight. You use it only after you have established your Eager Mode baseline.
1.	Phase 1: Measure (AIPerf) Run your baseline. You discover your Time Per Output Token (TPOT) is significantly higher than the theoretical peak of your H100.
2.	Phase 2: Analyze (Nsight Systems - nsys) You profile the execution. The timeline shows massive gaps between GPU kernels. You realize your Python Eager loop is "starving" the GPU—the GPU finishes a kernel in 2ms but waits 5ms for Python to launch the next one.
3.	Phase 3: Deep Dive (Nsight Compute - ncu) You identify that the Attention kernel is taking 80% of the GPU time. You run Nsight Compute on that specific kernel. It shows you are "Memory Bound" and hitting only 30% of the DRAM bandwidth because of an unoptimized memory access pattern.
4.	Phase 4: Optimize & Verify You swap the vanilla attention for Flash Attention 2. You re-run AIPerf to verify the stopwatch shows a faster time.






A. TTFT vs. Concurrency
Plot Concurrency on the X-axis and TTFT on the Y-axis.
•	The Baseline: You will see a sharp, linear spike. Because the FastAPI/HF setup uses static batching (or no batching), requests queue up. If Request B arrives just after Request A, Request B waits for A to completely finish generating before its prefill phase even begins.
•	vLLM / TensorRT-LLM: The line will remain remarkably flat as concurrency scales, only curving upward when you finally saturate the GPU's compute capacity. Continuous batching allows new requests to piggyback onto the prefill phase of the current batch.

B. ITL vs. Concurrency
Plot Concurrency on the X-axis and ITL on the Y-axis.
•	The Baseline: ITL degrades quickly as the system struggles with fragmented memory and unoptimized memory bandwidth.
•	vLLM / TensorRT-LLM: PagedAttention keeps memory access highly efficient. ITL will stay low and constant. If you use TensorRT-LLM with quantization (e.g., INT4 AWQ), the baseline ITL will be physically lower on the Y-axis compared to FP16 vLLM, demonstrating the power of reduced memory bandwidth requirements.

C. The "Golden Metric": Throughput vs. Latency
Plot System Throughput (‭‬‭‬) on the X-axis and average Latency on the Y-axis.
•	The Baseline: The curve will shoot upward almost immediately. It cannot achieve high throughput without latency becoming unacceptable.
•	vLLM / TensorRT-LLM: The curve will push far to the right. This visually demonstrates that the optimized engine can sustain massive throughput while keeping latency under a strict Service Level Agreement (SLA) threshold (e.g., ‭‬ per token).