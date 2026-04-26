"""Training loop skeleton for QCROP — joint LM + gate training.

NOT YET RUNNABLE. Heavy compute is held off until the user explicitly signals
they are on a GPU machine. This file documents the planned shape of the loop
so M3 can be wired up quickly when that time comes.

Flow:
    1. Sample batch from data iterator.
    2. Forward through embedding -> QCROPBlock(mode=soft, tau=tau(step)).
    3. Forward through transformer stack on the (gated) sequence.
    4. LM cross-entropy + lambda(step) * gate.aux_loss -> backprop.
    5. Step optimizer; advance schedules.

Open questions to resolve at M3 kickoff:
    - Where does the QCROP block live in the stack? Pre-block-0 only, or every K layers?
      Plan defaults to pre-block-0 single insertion. Revisit on M3 ablation.
    - Do we use straight-through estimator from step 0, or only after M4(a) failure?
      Plan defaults to soft sigmoid throughout M3, switch to STE only if M4(a) fails.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # data
    dataset: str = "openwebtext"
    seq_len: int = 1024
    batch_size: int = 32
    # model
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    # qcrop
    use_qcrop: bool = True
    qcrop_mode: str = "soft"          # soft -> hard at M4
    qcrop_k_ratio: float = 0.25        # K = N * ratio, only used in hard mode
    qcrop_tau_start: float = 1.0
    qcrop_tau_end: float = 0.1
    qcrop_lambda: float = 0.01
    qcrop_lambda_warmup: int = 2000
    qcrop_kernels: tuple[int, ...] = (7, 15, 31)
    qcrop_summaries: int = 8
    # optim
    lr_peak: float = 3e-4
    lr_warmup: int = 500
    total_steps: int = 50_000
    grad_clip: float = 1.0
    # logging
    log_every: int = 50
    eval_every: int = 1000
    ckpt_every: int = 5000


def train(cfg: TrainConfig) -> None:
    """TODO(M1-M3): wire up nanoGPT baseline + QCROP block + schedules.

    Implementation order:
      1. Build dataloader (HuggingFace `datasets` for OpenWebText/PG19 + tiktoken).
      2. Instantiate model: nanoGPT with optional QCROPBlock at the input.
      3. Build optimizer (AdamW, weight decay 0.1, betas (0.9, 0.95)).
      4. Step loop: forward, loss = CE + qcrop_aux, backward, clip, step.
      5. At eval_every: run held-out PPL and (optionally) needle eval.
      6. At ckpt_every: save model + optimizer + step.
    """
    raise NotImplementedError(
        "Training loop is intentionally stubbed — wire up at M1 kickoff on the GPU machine."
    )
