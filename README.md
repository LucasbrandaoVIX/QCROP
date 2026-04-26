# QCROP — Query-Conditioned Reduction Of Prefix-tokens

A two-pass attention scheme that pre-filters tokens before transformer self-attention to cut O(n²) cost on long contexts:

1. A FiLM-conditioned 1D conv gate scores every token for relevance to the current query span — cost O(N·d).
2. Hard top-K keeps the M ≪ N most relevant tokens; the rest are pooled into a small set of residual summary vectors.
3. Standard self-attention runs over `K survivors + summaries + query` — quadratic cost drops by ~(N/M)².

The headline contribution is the **residual summary path** (not the gate alone). Without working summaries, the contribution shrinks to "Quest with a conv."

## Status

Scaffold only — no training has been run. Core modules (`model/gate.py`, `model/summary.py`, `model/qcrop_block.py`) are implemented; baseline LM, training loop, and benchmarks are skeletons awaiting compute access.

## Layout

```
model/
  gate.py             FiLM-conditioned conv gate + score head + top-K
  summary.py          Chunked attention-pooling for dropped tokens
  qcrop_block.py      Gate + summary + attention integration, RoPE preservation
  nano_gpt.py         Baseline nanoGPT (skeleton — vendor karpathy/nanoGPT)
train/
  loop.py             Joint training with τ annealing + sparsity loss (skeleton)
  schedules.py        Temperature, sparsity, LR schedules
eval/
  needle.py           Synthetic passkey + multi-needle harness (M2 kill-gate)
  ppl.py              PG19 / arXiv perplexity (skeleton)
  benchmark.py        Wall-clock, peak memory, FLOPs profiler (skeleton)
configs/              YAML configs for each milestone
scripts/              Data download, baseline runners
tests/                CPU smoke tests for shapes and wiring
```

## Milestones (kill points marked)

| # | Week | Goal | Go criterion |
|---|------|------|--------------|
| M1 | 1 | 125M nanoGPT baseline + FLOP profiler | PPL within 5% of published nanoGPT |
| **M2** | **2** | **Frozen-LM gate-only on synthetic passkey** | **Gate top-K ≥ 80%, beats recency by ≥ 15 pts — KILL POINT** |
| M3 | 3–4 | Joint training, soft mask, PG19 PPL at 2k | Joint PPL within 0.5 of dense at K=N/2 |
| **M4** | **5** | **Hard top-K + summaries + ablation** | **Hard within 1.0 PPL of soft AND ablating summaries hurts ≥ 0.3 PPL** |
| M5 | 6–8 | Scale to 8k, full eval, wall-clock | ≥ 1.5× speedup at K=N/4, PPL drop ≤ 0.5 |

Approved plan: `~/.claude/plans/help-me-brainstorm-this-velvet-plum.md`.

## Smoke test

```bash
python -m tests.smoke
```

Runs gate + summary + qcrop_block on tiny dims (CPU) to verify shapes and wiring. No GPU needed.
