"""Baseline nanoGPT skeleton with optional QCROP block at the input.

NOT YET IMPLEMENTED. The plan is to vendor karpathy/nanoGPT
(https://github.com/karpathy/nanoGPT) and add a single QCROPBlock between the
embedding layer and block 0. We are deferring the actual implementation until
the user is on the GPU machine — there is no value in stubbing 200 lines of
attention/MLP code that's already public domain.

When implementing, the only QCROP-specific wiring is:

    h = wte(idx) + wpe(positions)             # standard nanoGPT embedding
    if self.qcrop is not None:
        out = self.qcrop(
            h,
            attention_mask=attn_mask,
            mode=self.qcrop_mode,
            k=self.qcrop_k,
            tau=self.qcrop_tau,
        )
        h = out.x
        positions = out.positions
        attn_mask = out.attention_mask
        aux_loss = out.aux_loss
    else:
        aux_loss = torch.zeros((), device=h.device)

    for block in self.blocks:
        h = block(h, attn_mask)
    logits = self.lm_head(self.ln_f(h))

    loss = F.cross_entropy(logits.view(-1, V), targets.view(-1)) + aux_loss
"""

from __future__ import annotations
