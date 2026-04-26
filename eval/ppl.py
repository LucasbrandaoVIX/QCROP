"""Perplexity evaluation skeleton — wired up at M3.

Plan: PG19 / arXiv long-context corpora at 2k–8k window, sliding window stride
= window/2 (standard for long-context PPL). Numbers reported per K (the gate's
keep ratio) and against the dense baseline at matched parameter count.

NOT YET IMPLEMENTED. Held off until GPU access — running PPL on a Mac would
take days even on a tiny model.
"""

from __future__ import annotations


def evaluate_ppl(*args, **kwargs):
    """TODO(M3): implement sliding-window PPL on PG19 / arXiv.

    Steps:
      1. Load PG19 validation split (HuggingFace `datasets`).
      2. Tokenize with model's tokenizer (tiktoken GPT-2 BPE).
      3. Slide a window of size W with stride W/2; for each window, forward and
         accumulate cross-entropy on the second half (avoids double-counting).
      4. PPL = exp(total_nll / total_tokens).
      5. Return: {"ppl": float, "tokens": int, "windows": int}.
    """
    raise NotImplementedError("PPL eval is stubbed — wire up at M3 on the GPU machine.")
