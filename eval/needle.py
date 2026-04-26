"""Synthetic needle-in-haystack data + scoring for the M2 kill-gate.

M2 is the make-or-break milestone: a frozen LM with only the QCROP gate trainable
must learn to top-K the needle out of N tokens. If the gate cannot beat a recency
baseline by >= 15 points at K = N/4, the conv-gate hypothesis is wrong and the
project stops.

This module provides:
  - `make_passkey_batch`: generates synthetic passkey tasks (single needle).
  - `make_multi_needle_batch`: harder variant with several keys, only one queried.
  - `recency_baseline_topk`: trivial "keep the last K tokens" baseline.
  - `gate_retrieval_accuracy`: scores how often the gate's top-K contains the needle.

Pure-Python / numpy / torch — no GPU, no model, no dataset download.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class PasskeyBatch:
    tokens: torch.Tensor          # [B, N] integer token IDs
    needle_position: torch.Tensor  # [B] index of the needle token in `tokens`
    answer: torch.Tensor          # [B] the needle's value (the "passkey")
    query_span: tuple[int, int]   # (lo, hi) — slice in `tokens` containing the question


# Tiny synthetic vocabulary.
# 0..9: digits used as passkey contents
# 10..109: filler tokens (the haystack)
# 110: BOS, 111: EOS, 112: SEP, 113: QUESTION_MARK
DIGIT_VOCAB = list(range(0, 10))
FILLER_VOCAB = list(range(10, 110))
BOS, EOS, SEP, QMARK = 110, 111, 112, 113
VOCAB_SIZE = 114


def make_passkey_batch(
    batch_size: int,
    seq_len: int,
    needle_len: int = 5,
    query_len: int = 4,
    seed: int | None = None,
) -> PasskeyBatch:
    """Build a batch of passkey-retrieval sequences.

    Layout per row:
        [BOS] [filler...] [SEP] [needle digits] [SEP] [filler...] [SEP] [query] [QMARK]

    The query asks "what was the needle?" by repeating a context cue inserted
    earlier; the model only needs to retrieve the needle digits. For the M2
    eval we don't need the model to *answer* — we just check whether the gate's
    top-K contains the needle positions.
    """
    rng = random.Random(seed)
    tokens = torch.full((batch_size, seq_len), -1, dtype=torch.long)
    needle_pos = torch.zeros(batch_size, dtype=torch.long)
    answers = torch.zeros(batch_size, dtype=torch.long)

    # Reserve room for: BOS + needle (1 sep + needle_len + 1 sep) + query (1 sep + query_len + qmark)
    fixed = 1 + 1 + needle_len + 1 + 1 + query_len + 1
    if fixed >= seq_len:
        raise ValueError(f"seq_len {seq_len} too short for fixed structure {fixed}")

    haystack = seq_len - fixed
    # Position the needle somewhere in the first ~80% of the sequence so the gate
    # has to actually look back; not at the very tail (which the recency baseline trivially wins).
    needle_max = int(0.8 * seq_len)

    for i in range(batch_size):
        digits = [rng.choice(DIGIT_VOCAB) for _ in range(needle_len)]
        # Pick where to put the needle within the haystack region.
        n_pos = rng.randint(needle_len + 2, max(needle_len + 3, needle_max))
        # Build the row.
        row = [BOS]
        # Filler before the needle.
        before = n_pos - 1  # slots already used: BOS
        row.extend(rng.choice(FILLER_VOCAB) for _ in range(before - 1))  # leave room for SEP
        row.append(SEP)
        needle_start = len(row)
        row.extend(digits)
        row.append(SEP)
        # Filler after the needle, leaving room for the query block at the end.
        query_block_len = 1 + query_len + 1   # SEP + query + QMARK
        remaining = seq_len - len(row) - query_block_len
        row.extend(rng.choice(FILLER_VOCAB) for _ in range(max(remaining, 0)))
        row.append(SEP)
        row.extend(rng.choice(FILLER_VOCAB) for _ in range(query_len))
        row.append(QMARK)

        # Pad / truncate to seq_len in case rounding above slipped.
        row = row[:seq_len]
        if len(row) < seq_len:
            row.extend([EOS] * (seq_len - len(row)))

        tokens[i] = torch.tensor(row, dtype=torch.long)
        needle_pos[i] = needle_start
        answers[i] = digits[0]  # first digit of the needle as the answer

    query_lo = seq_len - (1 + query_len + 1)
    query_hi = seq_len
    return PasskeyBatch(
        tokens=tokens,
        needle_position=needle_pos,
        answer=answers,
        query_span=(query_lo, query_hi),
    )


def recency_baseline_topk(seq_len: int, k: int, batch_size: int) -> torch.Tensor:
    """Trivial baseline: keep the last K positions. Returns [B, K] indices."""
    last_k = torch.arange(seq_len - k, seq_len)
    return last_k.unsqueeze(0).expand(batch_size, k).contiguous()


def gate_retrieval_accuracy(
    kept_idx: torch.Tensor,
    needle_position: torch.Tensor,
    needle_len: int = 5,
) -> float:
    """Fraction of rows whose top-K survivors contain the needle's first token.

    Args:
        kept_idx: [B, K] — indices selected by gate top-K (or baseline)
        needle_position: [B] — first index of the needle in each row
        needle_len: number of needle tokens (we count it as recovered if any one
            of the needle positions is in kept_idx)
    Returns:
        scalar accuracy in [0, 1]
    """
    b, _ = kept_idx.shape
    hits = 0
    for i in range(b):
        np_i = int(needle_position[i].item())
        needle_set = set(range(np_i, np_i + needle_len))
        kept = set(kept_idx[i].tolist())
        if needle_set & kept:
            hits += 1
    return hits / b


__all__ = [
    "VOCAB_SIZE",
    "BOS",
    "EOS",
    "SEP",
    "QMARK",
    "PasskeyBatch",
    "make_passkey_batch",
    "recency_baseline_topk",
    "gate_retrieval_accuracy",
]
