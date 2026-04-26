"""FiLM-conditioned 1D conv gate for query-aware token pruning.

Cost: O(N * d) — depthwise-separable convs over the sequence with FiLM modulation
from the query embedding. Returns per-token relevance scores that downstream
top-K selection consumes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Feature-wise linear modulation: h = gamma(q) * x + beta(q)."""

    def __init__(self, d_model: int, d_query: int | None = None) -> None:
        super().__init__()
        d_query = d_query or d_model
        self.proj = nn.Sequential(
            nn.Linear(d_query, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # x: [B, N, d_model]   q: [B, d_query]
        gamma_beta = self.proj(q)  # [B, 2 * d_model]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise (per-channel) conv followed by 1x1 pointwise mixing."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N]
        return self.pointwise(self.depthwise(x))


class QCROPGate(nn.Module):
    """Query-conditioned conv gate producing per-token relevance scores.

    Pipeline:
        FiLM(q) -> [conv7 -> conv15 -> conv31] -> linear -> scores [B, N]

    Top-K selection lives outside this module — call `select_topk` to apply it.
    """

    def __init__(
        self,
        d_model: int = 768,
        kernels: tuple[int, ...] = (7, 15, 31),
        d_query: int | None = None,
    ) -> None:
        super().__init__()
        if len(kernels) < 1:
            raise ValueError("kernels must have at least one entry")
        self.film = FiLM(d_model, d_query=d_query)

        # Channel schedule: d -> d/2 -> d/2 -> d/4 (when 3 kernels), generalised:
        chans = [d_model] + [max(d_model // 2, 1)] * (len(kernels) - 1) + [max(d_model // 4, 1)]
        layers: list[nn.Module] = []
        for k, c_in, c_out in zip(kernels, chans[:-1], chans[1:], strict=True):
            layers.append(DepthwiseSeparableConv1d(c_in, c_out, kernel_size=k))
            layers.append(nn.GELU())
            layers.append(nn.GroupNorm(num_groups=1, num_channels=c_out))  # LN over channels
        self.conv_stack = nn.Sequential(*layers)
        self.score_head = nn.Linear(chans[-1], 1)

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: token embeddings [B, N, d_model]
            q: query embedding  [B, d_query]
            attention_mask: 1 for real tokens, 0 for padding [B, N]; padding gets -inf score
        Returns:
            scores: [B, N]
        """
        h = self.film(x, q)                  # [B, N, d_model]
        h = h.transpose(1, 2)                # [B, d_model, N]
        h = self.conv_stack(h)               # [B, d_out, N]
        h = h.transpose(1, 2)                # [B, N, d_out]
        scores = self.score_head(h).squeeze(-1)  # [B, N]
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        return scores


def soft_mask(scores: torch.Tensor, tau: float) -> torch.Tensor:
    """Train-time soft mask: sigmoid(s / tau). Lower tau -> sharper, closer to step."""
    return torch.sigmoid(scores / max(tau, 1e-6))


def select_topk(
    x: torch.Tensor,
    scores: torch.Tensor,
    k: int,
    pad_with_drop_token: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard top-K selection.

    Args:
        x: [B, N, d_model]
        scores: [B, N]
        k: number of tokens to keep
        pad_with_drop_token: optional learned [d_model] vector used to pad sequences
            that have fewer than k finite-score tokens. Required if any sequence may
            have fewer than k real tokens; otherwise top-K returns garbage at -inf
            slots.

    Returns:
        kept_x:        [B, k, d_model] surviving token features (in score order)
        kept_idx:      [B, k] original positions of survivors (sorted ascending so
                       downstream RoPE / position-IDs stay monotonic)
        dropped_mask:  [B, N] True where the token was dropped, False where kept
    """
    b, n, d = x.shape
    if k > n:
        raise ValueError(f"k={k} cannot exceed sequence length N={n}")

    topk = scores.topk(k, dim=-1)
    idx = topk.indices  # [B, k]

    # sort kept indices ascending so positional structure is preserved
    sorted_idx, _ = idx.sort(dim=-1)
    kept_x = torch.gather(x, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, d))

    # Build dropped mask
    kept_onehot = torch.zeros(b, n, dtype=torch.bool, device=x.device)
    kept_onehot.scatter_(1, sorted_idx, True)
    dropped_mask = ~kept_onehot

    # Replace any -inf padding survivors with a learned drop token (rare path)
    if pad_with_drop_token is not None:
        kept_scores = torch.gather(scores, 1, sorted_idx)
        invalid = ~torch.isfinite(kept_scores)
        if invalid.any():
            kept_x = torch.where(
                invalid.unsqueeze(-1),
                pad_with_drop_token.view(1, 1, d).expand_as(kept_x),
                kept_x,
            )

    return kept_x, sorted_idx, dropped_mask


def query_from_tail(x: torch.Tensor, tail: int = 32) -> torch.Tensor:
    """Mean-pool the last `tail` tokens to form the query embedding.

    Args:
        x: [B, N, d_model]
        tail: number of trailing tokens to pool (clipped to N)
    Returns:
        q: [B, d_model]
    """
    n = x.size(1)
    take = min(tail, n)
    return x[:, n - take :, :].mean(dim=1)


def sparsity_penalty(mask: torch.Tensor) -> torch.Tensor:
    """L1-style penalty on soft mask values; encourages decisive scores."""
    return mask.mean()
