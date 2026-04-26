"""Schedules for QCROP training: temperature, sparsity, learning rate."""

from __future__ import annotations

import math


def linear_anneal(step: int, total: int, start: float, end: float) -> float:
    """Linear interpolation from `start` to `end` over `total` steps."""
    if total <= 0:
        return end
    t = min(max(step / total, 0.0), 1.0)
    return start + (end - start) * t


def cosine_anneal(step: int, total: int, start: float, end: float) -> float:
    """Cosine schedule from `start` to `end`."""
    if total <= 0:
        return end
    t = min(max(step / total, 0.0), 1.0)
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))


def temperature_schedule(step: int, total: int, start: float = 1.0, end: float = 0.1) -> float:
    """Anneal sigmoid temperature from `start` to `end` over `total` steps.

    Sharper temperature at the end of training narrows the train/inference gap
    between soft mask and hard top-K.
    """
    return cosine_anneal(step, total, start, end)


def sparsity_lambda_schedule(
    step: int,
    warmup: int,
    target: float = 0.01,
) -> float:
    """Ramp sparsity penalty from 0 to `target` over `warmup` steps.

    Holding lambda at zero during early training lets the LM converge before the
    gate is asked to be decisive.
    """
    return linear_anneal(step, warmup, 0.0, target)


def lr_schedule(
    step: int,
    warmup: int,
    total: int,
    peak: float,
    floor: float = 0.0,
) -> float:
    """Linear warmup, cosine decay to floor."""
    if step < warmup:
        return peak * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    t = min(max(t, 0.0), 1.0)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * t))
