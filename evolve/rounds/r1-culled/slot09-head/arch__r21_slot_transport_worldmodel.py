import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    D,
    R18PathStateLatentTransition,
)

NAME = "r21_slot_transport_worldmodel"
DESCRIPTION = (
    "The R18 path-state latent-transition model plus an always-on SLOT-TRANSPORT register: "
    "each command emits a source address and a destination address from the RAW (position-"
    "invariant) command embedding, reads the content currently bound to the destination as its "
    "prediction contribution, and on a gated 'move' rebinds the source content onto the "
    "destination while erasing the source. Observation writes bind the step's observation to the "
    "destination and are suppressed by the move gate, so a silent move carries content instead of "
    "overwriting it with an empty observation. Injected through a zero-init (D,D) readout with a "
    "radial-shell correction, so the forward is exactly the R18 function at initialization."
)


class R21SlotTransport(R18PathStateLatentTransition):
    def __init__(
        self,
        tp_addr_hidden=128,
        tp_shell_max=1.0,
        tp_move_bias=-2.0,
        tp_write_bias=1.0,
        tp_erase_bias=1.0,
        tp_edit_bias=-4.0,
        tp_shell_bias=-0.5,
        tp_decay=0.999,
        **params,
    ):
        super().__init__(**params)
        self.tp_shell_max = float(tp_shell_max)
        self.tp_decay = min(1.0, max(0.5, float(tp_decay)))
        self._tp_cache = None

        rng_state = torch.get_rng_state()
        try:
            h = max(16, int(tp_addr_hidden))
            self.tp_addr_in = nn.Linear(D, h)
            self.tp_src = nn.Linear(h, self.key_d, bias=False)
            self.tp_dst = nn.Linear(h, self.key_d, bias=False)
            self.tp_ctl = nn.Linear(h, 5)
            nn.init.zeros_(self.tp_ctl.weight)
            with torch.no_grad():
                self.tp_ctl.bias.copy_(
                    torch.tensor(
                        [
                            float(tp_move_bias),
                            float(tp_write_bias),
                            float(tp_erase_bias),
                            float(tp_edit_bias),
                            float(tp_shell_bias),
                        ]
                    )
                )
            self.tp_read = nn.Linear(D, D)
            nn.init.zeros_(self.tp_read.weight)
            nn.init.zeros_(self.tp_read.bias)
            self.tp_out_gate = nn.Linear(self.d, 1)
            nn.init.zeros_(self.tp_out_gate.weight)
            nn.init.zeros_(self.tp_out_gate.bias)
        finally:
            torch.set_rng_state(rng_state)

        self.supports_slot_transport = True

    def _transport(self, tok_emb, types, key_pad, move_scale=1.0):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        if n_cmd == 0:
            return tok_emb.new_zeros(B, 0, D)

        if key_pad is not None:
            valid = ~key_pad.bool()
        else:
            valid = torch.ones(B, L, dtype=torch.bool, device=device)
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        z_cmd = torch.nan_to_num(tok_emb[:, 0::2, :], nan=0.0, posinf=1e4, neginf=-1e4)
        obs_tok = torch.nan_to_num(tok_emb[:, 1::2, :], nan=0.0, posinf=1e4, neginf=-1e4)

        a_h = F.gelu(self.tp_addr_in(z_cmd))
        src = self._unit(self.tp_src(a_h))
        dst = self._unit(self.tp_dst(a_h))
        ctl = self.tp_ctl(a_h)
        move = torch.sigmoid(ctl[..., 0:1]) * float(move_scale)
        write = torch.sigmoid(ctl[..., 1:2])
        erase = torch.sigmoid(ctl[..., 2:3])
        edit = torch.sigmoid(ctl[..., 3:4])
        shell = self.tp_shell_max * torch.sigmoid(ctl[..., 4])

        idx0 = torch.zeros(B, n_cmd, dtype=torch.long, device=device)
        cmd_feat = self.in_norm(self.cmd_proj(z_cmd) + self.type_emb(idx0))
        gb = self.tr_out(F.gelu(self.tr_in(cmd_feat)))
        tr_gamma = torch.tanh(gb[..., :D]) * self.tr_gscale
        tr_beta = gb[..., D:]

        live = valid_cmd.to(dtype).unsqueeze(-1)
        dot_ab = (src * dst).sum(dim=-1, keepdim=True)
        obs_active = (valid_cmd[:, :n_pair] & valid_obs).to(dtype).unsqueeze(-1)

        mem = z_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            a = src[:, i, :]
            b = dst[:, i, :]

            # The emitted read precedes every write of step i, which is what keeps position i
            # blind to observation i; reordering these lines breaks the causality guard.
            r = torch.bmm(b.unsqueeze(1), mem).squeeze(1).clamp(-1e3, 1e3)
            reads.append(r)

            c = torch.bmm(a.unsqueeze(1), mem).squeeze(1).clamp(-1e3, 1e3)
            mv = (move[:, i, :] * live[:, i, :]).to(dtype)
            ed = edit[:, i, :].to(dtype)

            moved = (c + ed * ((c * (1.0 + tr_gamma[:, i, :]) + tr_beta[:, i, :]) - c))
            moved = torch.nan_to_num(moved, nan=0.0).clamp(-1e3, 1e3)
            put = mv * (moved - r)
            take = mv * erase[:, i, :] * c

            if i < n_pair:
                wr = write[:, i, :] * (1.0 - move[:, i, :]) * obs_active[:, i, :]
                cur = r + put - dot_ab[:, i, :] * take
                obs_i = obs_tok[:, i, :].to(dtype)
                keys = torch.stack([b, a, b], dim=2)
                vals = torch.stack([put, -take, wr * (obs_i - cur)], dim=1)
            else:
                keys = torch.stack([b, a], dim=2)
                vals = torch.stack([put, -take], dim=1)

            vals = torch.nan_to_num(vals, nan=0.0, posinf=1e3, neginf=-1e3)
            mem = torch.baddbmm(mem, keys, vals, beta=self.tp_decay, alpha=1.0)

        out = torch.stack(reads, dim=1)
        radial = F.normalize(out, dim=-1, eps=1e-6) * (float(D) ** 0.5)
        out = out + shell.unsqueeze(-1) * (radial - out)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e3, 1e3)
        return out * valid_cmd.unsqueeze(-1).to(out.dtype)

    def transport_reads(self, tok_emb, types, key_pad, move_scale=1.0):
        if float(move_scale) == 1.0:
            cache = self._tp_cache
            if cache is not None and cache[0] is tok_emb:
                return cache[1]
        return self._transport(tok_emb, types, key_pad, move_scale)

    def forward(self, tok_emb, types, key_pad):
        pred, hidden = super().forward(tok_emb, types, key_pad)
        L = tok_emb.size(1)
        if L == 0:
            self._tp_cache = None
            return pred, hidden

        reads = self._transport(tok_emb, types, key_pad, 1.0)
        self._tp_cache = (tok_emb, reads) if self.training else None

        gate = torch.sigmoid(self.tp_out_gate(hidden[:, 0::2, :])).to(reads.dtype)
        contrib = torch.nan_to_num(
            gate * self.tp_read(reads), nan=0.0, posinf=1e4, neginf=-1e4
        )

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        if key_pad is not None:
            pred = pred * (~key_pad.bool()).unsqueeze(-1).to(pred.dtype)
        return torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4), hidden


def build(**params):
    return R21SlotTransport(**params)
