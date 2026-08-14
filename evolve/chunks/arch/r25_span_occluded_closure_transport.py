import math

import torch

from evolve.chunks.arch.r24_reference_closure_transport import R24ReferenceClosureTransport

NAME = "r25_span_occluded_closure_transport"
DESCRIPTION = (
    "The reference-closure transport trunk trained under ramped CONTIGUOUS-SPAN observation "
    "occlusion. In training mode a per-row set of spans is drawn -- span lengths from a clamped "
    "geometric law with mean span_mean, count set so the expected occluded share reaches occ_p -- "
    "and every observation token inside a span is zeroed and key-padded, so the model faces gaps "
    "of several consecutive missing observations rather than isolated holes and cannot bridge a "
    "gap by copying a surviving neighbouring observation. Eval forward is bit-for-bit the "
    "reference-closure forward; no new trainable parameters; loss, head and batcher untouched."
)


class R25SpanOccludedClosureTransport(R24ReferenceClosureTransport):
    def __init__(self, occ_p=0.15, span_mean=3.0, span_max=6, occ_ramp_start=300,
                 occ_ramp_end=1000, **params):
        super().__init__(**params)
        self.occ_p = max(0.0, min(0.9, float(occ_p)))
        self.span_mean = max(1.0, float(span_mean))
        self.span_max = max(1, int(span_max))
        self.occ_ramp_start = max(0, int(occ_ramp_start))
        self.occ_ramp_end = max(self.occ_ramp_start + 1, int(occ_ramp_end))
        self.register_buffer("occ_seen", torch.zeros((), dtype=torch.long))

    def _occ_prob(self):
        s = int(self.occ_seen)
        if s <= self.occ_ramp_start:
            return 0.0
        if s >= self.occ_ramp_end:
            return self.occ_p
        x = (s - self.occ_ramp_start) / float(self.occ_ramp_end - self.occ_ramp_start)
        return self.occ_p * (x * x * (3.0 - 2.0 * x))

    def _span_drop(self, B, n_pair, p, device):
        n_spans = max(1, int(round(p * n_pair / self.span_mean)))
        max_len = float(min(self.span_max, n_pair))
        q = 1.0 - 1.0 / self.span_mean
        u = torch.rand(B, n_spans, device=device).clamp_min(1e-6)
        if q <= 0.0:
            length = torch.ones(B, n_spans, device=device)
        else:
            length = 1.0 + torch.log(u) / math.log(q)
        length = length.floor().clamp(1.0, max_len)
        start = (torch.rand(B, n_spans, device=device) * n_pair).floor().clamp(0.0, n_pair - 1.0)
        pos = torch.arange(n_pair, device=device, dtype=length.dtype).view(1, 1, n_pair)
        inside = (pos >= start.unsqueeze(-1)) & (pos < (start + length).unsqueeze(-1))
        return inside.any(dim=1)

    def forward(self, tok_emb, types, key_pad):
        if self.training:
            self.occ_seen += 1
            p = self._occ_prob()
            B, L = tok_emb.shape[0], tok_emb.shape[1]
            n_pair = L // 2
            if p > 0.0 and n_pair >= 1 and B > 0:
                if key_pad is None:
                    key_pad = torch.zeros(B, L, dtype=torch.bool, device=tok_emb.device)
                key_pad = key_pad.bool()
                drop = self._span_drop(B, n_pair, p, tok_emb.device)
                drop_full = torch.zeros(B, L, dtype=torch.bool, device=tok_emb.device)
                drop_full[:, 1:2 * n_pair:2] = drop
                tok_emb = tok_emb.masked_fill(drop_full.unsqueeze(-1), 0.0)
                key_pad = key_pad | drop_full
        return super().forward(tok_emb, types, key_pad)


def build(**params):
    return R25SpanOccludedClosureTransport(**params)
