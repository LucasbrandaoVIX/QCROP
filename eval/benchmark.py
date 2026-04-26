"""Wall-clock + memory + FLOPs benchmark skeleton — wired up at M5.

A FLOP win that does not translate to wall-clock kills the paper. Every speedup
claim must be accompanied by:
  - prefill latency at fixed context length, varying K
  - peak GPU memory at the same conditions
  - tokens/sec on a representative batch

Reference baseline is dense attention via Flash Attention 2.

NOT YET IMPLEMENTED. Held off until GPU access — wall-clock on a Mac is meaningless.
"""

from __future__ import annotations


def benchmark(*args, **kwargs):
    """TODO(M5): implement wall-clock + peak memory + FLOPs profiler.

    Steps:
      1. Build dense baseline (Flash Attention 2) and QCROP variant at K=N/2, N/4, N/8.
      2. Warm up CUDA (10 forward passes), then time 100 forward passes per condition.
      3. `torch.cuda.max_memory_allocated()` for peak memory.
      4. `torch.profiler` (CUDA) for FLOP counts.
      5. Report: {condition: {latency_ms, peak_mem_mb, flops, tokens_per_sec}}.
    """
    raise NotImplementedError("Benchmark is stubbed — wire up at M5 on the GPU machine.")
