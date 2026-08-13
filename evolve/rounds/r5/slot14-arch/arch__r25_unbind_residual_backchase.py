import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.arch.r22_retrieval_composition_renderer import (
    R22RetrievalCompositionRenderer,
)

D = 768

NAME = "r25_unbind_residual_backchase"
DESCRIPTION = (
    "The r18 path-state trunk with r22's retrieval composition renderer, plus a BACKWARD "
    "CHASE over the command sequence whose step transition is an ALGEBRAIC UNBIND rather than "
    "two independent role heads. One shared GELU feature layer over each command's raw 768-d "
    "embedding feeds two same-shape address heads: a BAG address (meant to span every path the "
    "command names) and a KEY address (one distinguished path). The chase state leaving a "
    "command is the learned residual alpha*bag - beta*key + bias, so a two-path command's "
    "outgoing address is structurally what is LEFT OVER after its key address is removed from "
    "its bag, while a one-path command's residual collapses to a constant. Matching is a masked "
    "softmax over strictly-earlier commands combining cosine of residual-vs-key, a learned "
    "multiple of the raw command-embedding cosine, a learned recency penalty, and a learned NULL "
    "column that lets unmatched mass leave the chase instead of smearing. Each step also emits an "
    "absorb probability from its command feature and its own observation feature. The chase starts "
    "from the queried command's BAG address, iterates the residual follow matrix K times, and "
    "accumulates absorbed mass; the accumulated mass over strictly-earlier steps is renormalized "
    "into a convex combination of their raw observation embeddings and added to that command's "
    "prediction through an identity-scaled (D,D) readout under a sigmoid gate fed the absorbed "
    "mass, the chase sharpness and the copy scale. All mass moves strictly backward by explicit "
    "multiplicative masks, so a command's prediction is an exact constant in its own and later "
    "observations."
)

_NEG = -1e9


class R25UnbindResidualBackchase(R22RetrievalCompositionRenderer):
    def __init__(
        self,
        chase_hops=6,
        addr_hidden=256,
        obs_feat_d=64,
        absorb_hidden=96,
        chase_temp=0.4,
        chase_gate_bias=-2.0,
        lex_w_init=3.0,
        recency_init=-1.8,
        absorb_bias=-0.5,
        unbind_alpha=1.0,
        unbind_beta=1.0,
        readout_init=0.2,
        **params,
    ):
        super().__init__(**params)
        self.chase_hops = max(1, int(chase_hops))
        self.addr_hidden = max(32, int(addr_hidden))
        self.obs_feat_d = max(8, int(obs_feat_d))

        self.chase_ln = nn.LayerNorm(D)
        self.chase_in = nn.Linear(D, self.addr_hidden)
        self.addr_bag = nn.Linear(self.addr_hidden, self.key_d, bias=False)
        self.addr_key = nn.Linear(self.addr_hidden, self.key_d, bias=False)

        self.unbind_alpha = nn.Parameter(torch.tensor(float(unbind_alpha)))
        self.unbind_beta = nn.Parameter(torch.tensor(float(unbind_beta)))
        self.unbind_bias = nn.Parameter(torch.zeros(self.key_d))

        self.chase_obs_feat = nn.Linear(D, self.obs_feat_d)
        ah = max(16, int(absorb_hidden))
        self.absorb_in = nn.Linear(self.addr_hidden + self.obs_feat_d, ah)
        self.absorb_out = nn.Linear(ah, 1)
        nn.init.normal_(self.absorb_out.weight, std=0.01)
        nn.init.constant_(self.absorb_out.bias, float(absorb_bias))

        self.chase_log_temp = nn.Parameter(
            torch.tensor(math.log(max(1e-2, float(chase_temp))))
        )
        self.chase_lex_w = nn.Parameter(torch.tensor(float(lex_w_init)))
        self.chase_recency = nn.Parameter(torch.tensor(float(recency_init)))
        self.chase_null = nn.Parameter(torch.tensor(0.0))

        self.chase_out = nn.Linear(D, D)
        with torch.no_grad():
            self.chase_out.weight.copy_(torch.eye(D) * float(readout_init))
            self.chase_out.bias.zero_()
        self.chase_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.chase_gate.bias, float(chase_gate_bias))

    def _chase_weights(self, query_unit, key_unit, lex, gap, allow, allow_f):
        temp = self.chase_log_temp.float().exp().clamp(0.05, 5.0)
        s = torch.bmm(query_unit, key_unit.transpose(1, 2)) / temp
        s = s + self.chase_lex_w.float() * lex
        s = s - F.softplus(self.chase_recency.float()) * gap
        s = torch.nan_to_num(s, nan=0.0, posinf=1e4, neginf=-1e4)
        s = s.masked_fill(~allow, _NEG)
        null = self.chase_null.float().view(1, 1, 1).expand(s.size(0), s.size(1), 1)
        w = torch.softmax(torch.cat([s, null], dim=2), dim=2)[:, :, :-1]
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        return w * allow_f

    def forward(self, tok_emb, types, key_pad):
        pred, h_out = super().forward(tok_emb, types, key_pad)
        B, L, _ = tok_emb.shape
        if L < 4:
            return pred, h_out
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        if n_cmd < 2 or n_pair < 1:
            return pred, h_out

        device = tok_emb.device
        dtype = pred.dtype
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, dtype=torch.bool, device=device)
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        cmd_raw = torch.nan_to_num(tok_emb[:, 0::2, :].float(), nan=0.0,
                                   posinf=1e4, neginf=-1e4)
        obs_pad = torch.nan_to_num(self._pad_steps(tok_emb[:, 1::2, :], n_cmd).float(),
                                   nan=0.0, posinf=1e4, neginf=-1e4)
        obs_live = self._pad_steps(valid_obs, n_cmd) & valid_cmd

        a = F.gelu(self.chase_in(self.chase_ln(cmd_raw.to(self.chase_in.weight.dtype))))
        a = a.float()
        bag = self.addr_bag(a.to(self.addr_bag.weight.dtype)).float()
        key = self.addr_key(a.to(self.addr_key.weight.dtype)).float()
        res = (self.unbind_alpha.float() * bag
               - self.unbind_beta.float() * key
               + self.unbind_bias.float().view(1, 1, -1))

        bag_u = self._unit(bag)
        key_u = self._unit(key)
        res_u = self._unit(res)

        e_u = F.normalize(cmd_raw, dim=-1, eps=1e-6)
        lex = torch.bmm(e_u, e_u.transpose(1, 2))
        lex = torch.nan_to_num(lex, nan=0.0, posinf=0.0, neginf=0.0)

        idx = torch.arange(n_cmd, device=device)
        strict = (idx.view(n_cmd, 1) > idx.view(1, n_cmd)).unsqueeze(0)
        allow = strict & valid_cmd.unsqueeze(1)
        allow_f = allow.float()
        gap = (idx.view(1, n_cmd, 1) - idx.view(1, 1, n_cmd) - 1).clamp_min(0).float()

        start = self._chase_weights(bag_u, key_u, lex, gap, allow, allow_f)
        follow = self._chase_weights(res_u, key_u, lex, gap, allow, allow_f)

        of = self.chase_obs_feat(obs_pad.to(self.chase_obs_feat.weight.dtype)).float()
        fin = torch.cat([a, of], dim=-1)
        tau = torch.sigmoid(
            self.absorb_out(F.gelu(self.absorb_in(fin.to(self.absorb_in.weight.dtype)))).float()
        ).squeeze(-1) * obs_live.float()
        tau_row = tau.unsqueeze(1)

        alive = start
        acc = alive * tau_row
        alive = alive * (1.0 - tau_row)
        for _ in range(self.chase_hops - 1):
            alive = torch.bmm(alive, follow)
            acc = acc + alive * tau_row
            alive = alive * (1.0 - tau_row)
        acc = torch.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)

        mass = acc.sum(dim=2, keepdim=True)
        sharp = acc.amax(dim=2, keepdim=True)
        copy = torch.bmm(acc, obs_pad) / mass.clamp_min(1e-4)
        copy = torch.nan_to_num(copy, nan=0.0, posinf=1e4, neginf=-1e4)

        rms = (copy.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        feat = torch.cat([mass.clamp(0.0, 1.0), sharp.clamp(0.0, 1.0), rms], dim=-1)
        gi = torch.cat([h_out[:, 0::2, :].float(), feat], dim=-1)
        gate = torch.sigmoid(self.chase_gate(gi.to(self.chase_gate.weight.dtype))).float()

        read = self.chase_out(copy.to(self.chase_out.weight.dtype)).float()
        contrib = gate * read * valid_cmd.unsqueeze(-1).float()
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(dtype)
        return pred, h_out


def build(**params):
    return R25UnbindResidualBackchase(**params)
