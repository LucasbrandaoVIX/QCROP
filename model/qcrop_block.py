"""QCROP block: gate -> top-K -> residual summaries -> attention.

Two modes:
- Train (soft): scores -> sigmoid mask -> elementwise multiply embeddings ->
  full self-attention over all N tokens. Gradients flow normally.
- Eval (hard): scores -> top-K -> concat with summaries -> attention over (K + S + L_q).
  No gradient through the discrete selection at inference.

The block is designed to be inserted ONCE near the input of a transformer stack,
not per-layer (M3 ablations may revisit). Position IDs are preserved across
pruning so RoPE/learned-pos-embed semantics survive.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gate import (
    QCROPGate,
    query_from_tail,
    select_topk,
    soft_mask,
    sparsity_penalty,
)
from model.summary import ResidualSummary


@dataclass
class QCROPOutput:
    x: torch.Tensor                  # token features after pruning/gating
    positions: torch.Tensor          # position IDs aligned with x
    attention_mask: torch.Tensor     # 1 where token is real, 0 padding
    aux_loss: torch.Tensor           # sparsity penalty (scalar) — add to LM loss
    scores: torch.Tensor | None      # raw gate scores [B, N], for logging/eval
    kept_idx: torch.Tensor | None    # [B, K] indices of survivors (hard mode only)


class QCROPBlock(nn.Module):
    """Integrates gate + summary + (downstream) attention.

    Important: this block does NOT run attention itself — it returns the (possibly
    pruned) sequence and the model is responsible for feeding it through its
    transformer stack. Keeping attention out of this module lets us swap in
    Flash Attention, MLA, or any other backend without touching QCROP.
    """

    def __init__(
        self,
        d_model: int = 768,
        kernels: tuple[int, ...] = (7, 15, 31),
        num_summaries: int = 8,
        query_tail: int = 32,
        sparsity_lambda: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.query_tail = query_tail
        self.sparsity_lambda = sparsity_lambda

        self.gate = QCROPGate(d_model=d_model, kernels=kernels)
        self.summary = ResidualSummary(d_model=d_model, num_summaries=num_summaries)
        # Pad token used for sequences shorter than K real tokens (rare).
        self.pad_drop_token = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        mode: str = "soft",
        k: int | None = None,
        tau: float = 1.0,
        query: torch.Tensor | None = None,
    ) -> QCROPOutput:
        """
        Args:
            x: [B, N, d_model] token embeddings (post embedding layer, pre block-0)
            attention_mask: [B, N] 1 for real tokens, 0 for padding
            mode: "soft" (training) or "hard" (inference / late training)
            k: number of survivors to keep in hard mode (defaults to N//4)
            tau: sigmoid temperature for soft mode
            query: optional explicit query embedding [B, d_model]; otherwise mean-pool
                the last `query_tail` tokens of x
        Returns:
            QCROPOutput
        """
        b, n, d = x.shape
        device = x.device
        if attention_mask is None:
            attention_mask = torch.ones(b, n, dtype=torch.long, device=device)
        if query is None:
            query = query_from_tail(x, tail=self.query_tail)

        scores = self.gate(x, query, attention_mask=attention_mask)  # [B, N]

        if mode == "soft":
            mask = soft_mask(scores, tau)                            # [B, N]
            x_out = x * mask.unsqueeze(-1)
            positions = torch.arange(n, device=device).unsqueeze(0).expand(b, n)
            aux = self.sparsity_lambda * sparsity_penalty(mask)
            return QCROPOutput(
                x=x_out,
                positions=positions,
                attention_mask=attention_mask,
                aux_loss=aux,
                scores=scores,
                kept_idx=None,
            )

        if mode == "hard":
            if k is None:
                k = max(1, n // 4)
            kept_x, kept_idx, dropped_mask = select_topk(
                x, scores, k=k, pad_with_drop_token=self.pad_drop_token
            )
            # Dropped mask must respect padding: padded positions are not "really dropped".
            dropped_mask = dropped_mask & (attention_mask == 1)

            summaries, summary_pos = self.summary(x, dropped_mask=dropped_mask)
            # Concatenate: [survivors | summaries]. Summaries get the chunk-centroid pos IDs.
            x_out = torch.cat([kept_x, summaries], dim=1)              # [B, K+S, d]
            positions = torch.cat([kept_idx, summary_pos], dim=1)      # [B, K+S]

            # All survivor + summary slots are valid attention targets.
            new_mask = torch.ones(x_out.size(0), x_out.size(1), dtype=torch.long, device=device)

            # In hard mode the sparsity term is degenerate (mask is {0, 1}); skip it.
            zero = torch.zeros((), device=device)
            return QCROPOutput(
                x=x_out,
                positions=positions,
                attention_mask=new_mask,
                aux_loss=zero,
                scores=scores,
                kept_idx=kept_idx,
            )

        raise ValueError(f"unknown mode: {mode!r}")
