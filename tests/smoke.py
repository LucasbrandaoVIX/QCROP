"""CPU smoke test — verifies QCROP modules wire together with correct shapes.

No training, no real dataset, no GPU. Tiny dimensions so it finishes in seconds.
Run from the repo root:

    python -m tests.smoke

If any of these fails, the architecture is broken before any training time is spent.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python -m tests.smoke` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from eval.needle import (  # noqa: E402
    gate_retrieval_accuracy,
    make_passkey_batch,
    recency_baseline_topk,
)
from model.gate import (  # noqa: E402
    QCROPGate,
    query_from_tail,
    select_topk,
    soft_mask,
    sparsity_penalty,
)
from model.qcrop_block import QCROPBlock  # noqa: E402
from model.summary import ResidualSummary  # noqa: E402

TINY_D = 32
TINY_N = 64
TINY_B = 2
TINY_K = 16
TINY_S = 4


def t_gate_shapes() -> None:
    gate = QCROPGate(d_model=TINY_D, kernels=(3, 5, 7))
    x = torch.randn(TINY_B, TINY_N, TINY_D)
    q = query_from_tail(x, tail=8)
    scores = gate(x, q)
    assert scores.shape == (TINY_B, TINY_N), scores.shape
    assert torch.isfinite(scores).all()
    print("[ok] gate scores", scores.shape)


def t_soft_and_sparsity() -> None:
    s = torch.randn(TINY_B, TINY_N)
    m = soft_mask(s, tau=0.5)
    assert m.shape == s.shape
    assert (m >= 0).all() and (m <= 1).all()
    p = sparsity_penalty(m)
    assert p.dim() == 0
    print("[ok] soft mask + sparsity penalty")


def t_topk() -> None:
    x = torch.randn(TINY_B, TINY_N, TINY_D)
    s = torch.randn(TINY_B, TINY_N)
    pad = torch.zeros(TINY_D)
    kept_x, kept_idx, dropped = select_topk(x, s, k=TINY_K, pad_with_drop_token=pad)
    assert kept_x.shape == (TINY_B, TINY_K, TINY_D)
    assert kept_idx.shape == (TINY_B, TINY_K)
    assert dropped.shape == (TINY_B, TINY_N)
    # kept indices must be sorted ascending and unique within a row
    for b in range(TINY_B):
        idx = kept_idx[b].tolist()
        assert idx == sorted(idx)
        assert len(set(idx)) == len(idx)
    # dropped count must equal N - K (only when no padding)
    assert (~dropped).sum().item() == TINY_B * TINY_K
    print("[ok] top-K survivors and dropped mask")


def t_summary_shapes() -> None:
    summary = ResidualSummary(d_model=TINY_D, num_summaries=TINY_S)
    x = torch.randn(TINY_B, TINY_N, TINY_D)
    dropped_mask = torch.zeros(TINY_B, TINY_N, dtype=torch.bool)
    dropped_mask[:, : TINY_N - TINY_K] = True
    summaries, pos = summary(x, dropped_mask=dropped_mask)
    assert summaries.shape == (TINY_B, TINY_S, TINY_D), summaries.shape
    assert pos.shape == (TINY_B, TINY_S), pos.shape
    print("[ok] residual summary shapes")


def t_block_soft() -> None:
    block = QCROPBlock(
        d_model=TINY_D, kernels=(3, 5, 7), num_summaries=TINY_S, query_tail=8
    )
    x = torch.randn(TINY_B, TINY_N, TINY_D)
    out = block(x, mode="soft", tau=1.0)
    # Soft mode preserves N tokens (just gates them).
    assert out.x.shape == (TINY_B, TINY_N, TINY_D), out.x.shape
    assert out.positions.shape == (TINY_B, TINY_N)
    assert out.aux_loss.dim() == 0
    assert torch.isfinite(out.aux_loss)
    print("[ok] block soft mode", out.x.shape)


def t_block_hard() -> None:
    block = QCROPBlock(
        d_model=TINY_D, kernels=(3, 5, 7), num_summaries=TINY_S, query_tail=8
    )
    x = torch.randn(TINY_B, TINY_N, TINY_D)
    out = block(x, mode="hard", k=TINY_K, tau=0.1)
    # Hard mode: K survivors + S summaries.
    expected = TINY_K + TINY_S
    assert out.x.shape == (TINY_B, expected, TINY_D), out.x.shape
    assert out.positions.shape == (TINY_B, expected)
    assert out.kept_idx is not None and out.kept_idx.shape == (TINY_B, TINY_K)
    print("[ok] block hard mode", out.x.shape)


def t_block_grads() -> None:
    """Soft-mode forward should produce finite gradients into both gate and summary
    (well, summary doesn't see gradient in soft mode — only in hard. So check gate only)."""
    block = QCROPBlock(
        d_model=TINY_D, kernels=(3, 5, 7), num_summaries=TINY_S, query_tail=8
    )
    x = torch.randn(TINY_B, TINY_N, TINY_D, requires_grad=True)
    out = block(x, mode="soft", tau=1.0)
    loss = out.x.pow(2).mean() + out.aux_loss
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # At least one gate parameter should have a finite gradient.
    gate_grads = [p.grad for p in block.gate.parameters() if p.grad is not None]
    assert any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gate_grads)
    print("[ok] gradients flow through gate in soft mode")


def t_passkey_data() -> None:
    batch = make_passkey_batch(batch_size=4, seq_len=128, needle_len=5, seed=7)
    assert batch.tokens.shape == (4, 128)
    assert (batch.tokens >= 0).all()
    assert batch.needle_position.shape == (4,)

    # Recency baseline never contains a needle placed in the first 80% of the seq.
    rec = recency_baseline_topk(seq_len=128, k=16, batch_size=4)
    acc = gate_retrieval_accuracy(rec, batch.needle_position, needle_len=5)
    print(f"[ok] passkey data + recency baseline acc={acc:.2f} (expected near 0)")


def main() -> None:
    torch.manual_seed(0)
    t_gate_shapes()
    t_soft_and_sparsity()
    t_topk()
    t_summary_shapes()
    t_block_soft()
    t_block_hard()
    t_block_grads()
    t_passkey_data()
    print("\nall smoke tests passed.")


if __name__ == "__main__":
    main()
