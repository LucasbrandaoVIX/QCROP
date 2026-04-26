"""Residual summary path: pool dropped tokens into S compact summary vectors.

Dropped tokens are partitioned into S contiguous chunks by ORIGINAL position
(not by score) so the spatial locality signal survives. Each chunk is reduced
via learned attention pooling — a single learned query vector cross-attends
over the chunk's dropped tokens.

Headline contribution of QCROP: without this path the gate is "Quest with a
conv." The summary path is what reviewers should see first.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttnPool1(nn.Module):
    """Single-query learned attention pool: produces one vector from a sequence."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model] — the chunk's tokens
            valid_mask: [B, L] True where the token is valid (i.e. truly dropped, not padding)
        Returns:
            summary: [B, d_model]
        """
        k = self.k_proj(x)                                 # [B, L, d]
        v = self.v_proj(x)                                 # [B, L, d]
        scores = (k @ self.q) * self.scale                 # [B, L]
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, float("-inf"))
        # If a chunk has no valid tokens, softmax is undefined — return zeros.
        if valid_mask is not None:
            empty = ~valid_mask.any(dim=-1, keepdim=True)  # [B, 1]
        else:
            empty = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
        weights = torch.softmax(scores, dim=-1)            # [B, L]
        weights = torch.where(
            empty, torch.zeros_like(weights), weights
        )
        return (weights.unsqueeze(-1) * v).sum(dim=1)      # [B, d]


class ResidualSummary(nn.Module):
    """Chunked attention-pooling residual summary.

    Partitions dropped tokens into S equal chunks by ORIGINAL position and
    pools each chunk to a single summary vector. Output is S vectors plus
    their position IDs (the chunk midpoint of source tokens).
    """

    def __init__(self, d_model: int = 768, num_summaries: int = 8) -> None:
        super().__init__()
        self.s = num_summaries
        self.pools = nn.ModuleList([AttnPool1(d_model) for _ in range(num_summaries)])

    def forward(
        self,
        x: torch.Tensor,
        dropped_mask: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, d_model] — full original sequence
            dropped_mask: [B, N] True where token was dropped (and is therefore eligible)
            positions: [B, N] integer position IDs (defaults to 0..N-1 per row).
                Surviving tokens still occupy these positions; we ignore them via mask.
        Returns:
            summaries:    [B, S, d_model]
            summary_pos:  [B, S] position IDs for each summary (chunk midpoint over sources)
        """
        b, n, d = x.shape
        if positions is None:
            positions = torch.arange(n, device=x.device).unsqueeze(0).expand(b, n)

        # Chunk by original position: split [0, N) into S equal-width chunks.
        # A token belongs to chunk c iff floor(c * N / S) <= pos < floor((c+1) * N / S),
        # AND it was actually dropped.
        edges = torch.linspace(0, n, self.s + 1, device=x.device).long()  # [S+1]

        summaries = []
        summary_pos = []
        for c in range(self.s):
            lo, hi = edges[c].item(), edges[c + 1].item()
            in_chunk = (positions >= lo) & (positions < hi)             # [B, N]
            valid = in_chunk & dropped_mask                             # [B, N]

            # Slice the chunk window. All chunks have width hi-lo (constant per c),
            # which keeps shapes simple.
            chunk_x = x[:, lo:hi, :]                                    # [B, hi-lo, d]
            chunk_valid = valid[:, lo:hi]                               # [B, hi-lo]

            summary = self.pools[c](chunk_x, valid_mask=chunk_valid)    # [B, d]
            summaries.append(summary)

            # Use the centroid of valid positions inside this chunk as the summary's pos.
            # Fallback: chunk midpoint when no valid tokens exist.
            chunk_pos = positions[:, lo:hi].float()                     # [B, hi-lo]
            valid_f = chunk_valid.float()
            denom = valid_f.sum(dim=-1).clamp(min=1.0)
            centroid = (chunk_pos * valid_f).sum(dim=-1) / denom        # [B]
            mid = torch.full_like(centroid, (lo + hi - 1) / 2.0)
            has_any = chunk_valid.any(dim=-1)
            summary_pos.append(torch.where(has_any, centroid, mid))

        summaries_t = torch.stack(summaries, dim=1)                     # [B, S, d]
        summary_pos_t = torch.stack(summary_pos, dim=1).long()          # [B, S]
        return summaries_t, summary_pos_t
