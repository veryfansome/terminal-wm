TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: REWRITE — replace the mutable code wholesale with a genuinely different design. A rewrite that lands near the parent is a wasted slot.

THE CONTRACT — axis 'arch': Expose build(**params) -> an nn.Module whose forward(tok_emb, types, key_pad) returns (pred, h). Tokens interleave command and observation embeddings; types marks which is which; key_pad marks padding. Predict at EVERY position — the harness reads command positions by stride. The module MUST be causal: perturbing a later observation may not move an earlier command's prediction, and the guard checks this before anything is scored.
The reference baseline below is authoritative — match its interface exactly, keep your module self-contained:
--------------------------------------------------------------------------------
"""Contract for any arch impl: expose `build(**params) -> nn.Module` whose
  forward(tok_emb [B,L,768], types [B,L] in {0,1}, key_pad [B,L] bool) -> (pred [B,L,768], h [B,L,dh])
predicting at EVERY position; the harness reads command positions as pred[:, 0::2]. Input tokens
are frozen 768-d embeddings and the module must map back to 768-d target space. Params come from
the genome's chunks.arch.params.

The module MUST be causal: a command-position prediction may not depend on its own or a later
observation token. The per-genome no-leakage guard rejects any arch that fails this.
"""

from realenv import seq_worldmodel as M

NAME = "baseline_transformer"
DESCRIPTION = "R4 causal transformer over interleaved cmd/obs frozen embeddings (SeqWorldModel)."


def build(d=192, layers=4, heads=4, dropout=0.1):
    return M.SeqWorldModel("jepa", d=d, layers=layers, heads=heads, dropout=dropout)
--------------------------------------------------------------------------------

PARENT — you are mutating this candidate.
  id                r6-29-routing-necessity-pips
  its fitness       -0.0112   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r12_antiretrieval_ring_negatives
  arch                r18_pathstate_latent_transition_worldmodel
  optim               r18_spectral_capped_transition_readout
  target              identity
  batcher             r6_routing_necessity_pips   params {"alpha": 2.0, "dup_frac": 0.2, "dup_tol": 0.02, "kappa_bias": 0.5, "kappa_ref": 0.1, "pool_mult": 16, "proj_dim": 96, "top_k": 3, "w_floor": 0.15}
  stream              baseline_interleave
  head                r20_dualpre_transition_consistency

YOUR PARENT'S CURRENT arch IMPL — r18_pathstate_latent_transition_worldmodel (this is the code you are mutating):
--------------------------------------------------------------------------------
import math

import torch
import torch.nn as nn

D = 768

NAME = "r18_pathstate_latent_transition_worldmodel"
DESCRIPTION = (
    "The r13 trunk + a per-path LATENT-TRANSITION world-model memory: the value "
    "written to a path slot is a command-conditioned affine EDIT of the slot's current "
    "retrieved content (f(s_pre,cmd), a learned RSSM-style transition), overwritten by the "
    "delta rule and read back as the current post-mutation content; injected via a zero-init "
    "(D,D) readout so it is exactly the r18 function at init. Distinct from additive path-delta "
    "(r17) and typed-slot transport (#6): a learned content transition operator, co-designed "
    "with a forward-model head aux and a spectral-capped readout optim."
)


class R18PathStateLatentTransition(nn.Module):
    def __init__(
        self,
        d=176,
        layers=4,
        heads=4,
        key_d=64,
        ctx_d=96,
        n_verb=8,
        film_hidden=128,
        sys_d=64,
        sysfilm_hidden=128,
        tr_hidden=192,
        tr_gscale=0.5,
        ffn_mult=2,
        dropout=0.1,
        chunk_size=16,
        **unused,
    ):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]
        if "chunk" in unused:
            chunk_size = unused["chunk"]

        self.D = D
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = int(key_d)
        self.ctx_d = int(ctx_d)
        self.n_verb = max(1, int(n_verb))
        self.sys_d = max(8, int(sys_d))
        self.chunk_size = max(1, int(chunk_size))
        self.tr_gscale = float(tr_gscale)
        ffn_h = max(self.d, int(float(ffn_mult) * self.d))

        self.cmd_proj = nn.Linear(D, self.d)
        self.obs_proj = nn.Linear(D, self.d)
        self.type_emb = nn.Embedding(2, self.d)
        self.in_norm = nn.LayerNorm(self.d)
        self.pos_scale = nn.Parameter(torch.tensor(0.2))

        enc = nn.TransformerEncoderLayer(
            self.d,
            int(heads),
            ffn_h,
            float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tf = nn.TransformerEncoder(enc, self.layers, enable_nested_tensor=False)

        self.file_read = nn.Linear(self.d, self.key_d, bias=False)
        self.file_write = nn.Linear(self.d, self.key_d, bias=False)
        self.verb_codebook = nn.Parameter(torch.randn(self.n_verb, self.key_d) * 0.2)
        self.ctx_proj = nn.Linear(self.d, self.ctx_d)

        self.path_read = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write = nn.Linear(self.d, self.key_d, bias=False)

        self.write_gate = nn.Linear(2 * self.d, 1)

        fh = max(16, int(film_hidden))
        self.film_in = nn.Linear(self.d + self.ctx_d, fh)
        self.film_out = nn.Linear(fh, 2 * D)
        nn.init.zeros_(self.film_out.weight)
        nn.init.zeros_(self.film_out.bias)

        self.sys_sal = nn.Linear(self.d, 1)
        self.sys_val = nn.Linear(self.d, self.sys_d)
        sh = max(16, int(sysfilm_hidden))
        self.sysfilm_in = nn.Linear(self.d + self.sys_d, sh)
        self.sysfilm_out = nn.Linear(sh, 2 * D)
        nn.init.zeros_(self.sysfilm_out.weight)
        nn.init.zeros_(self.sysfilm_out.bias)

        self.read_mix = nn.Linear(self.d + 3, 3)
        self.read_to_h = nn.Linear(D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d + 3, self.d)
        self.direct_gate = nn.Linear(2 * self.d + 3, 1)
        self.out_norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, D)

        self.tr_path = nn.Linear(self.d, self.key_d, bias=False)
        th = max(32, int(tr_hidden))
        self.tr_in = nn.Linear(self.d, th)
        self.tr_out = nn.Linear(th, 2 * D)
        self.tr_mut_gate = nn.Linear(self.d, 1)
        self.tr_read = nn.Linear(D, D)
        nn.init.zeros_(self.tr_read.weight)
        nn.init.zeros_(self.tr_read.bias)
        self.tr_read_gate = nn.Linear(self.d, 1)
        nn.init.constant_(self.tr_mut_gate.bias, -1.0)
        nn.init.constant_(self.tr_read_gate.bias, 0.0)

        init_decay = (0.985 - 0.90) / 0.099
        self.logit_decay = nn.Parameter(torch.tensor(math.log(init_decay / (1.0 - init_decay))))

        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.fuse_gate.bias, -1.0)
        nn.init.constant_(self.direct_gate.bias, -2.0)

    def _positional(self, L, device, dtype):
        half = (self.d + 1) // 2
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        pe = torch.zeros(L, self.d, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: self.d // 2])
        return pe

    @staticmethod
    def _unit(x):
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + 1e-12)

    @staticmethod
    def _pad_steps(x, n):
        cur = x.size(1)
        if cur == n:
            return x
        if cur > n:
            return x[:, :n]
        pad_shape = (x.size(0), n - cur) + tuple(x.shape[2:])
        return torch.cat([x, x.new_zeros(pad_shape)], dim=1)

    def _verb_basis(self):
        vs = []
        for i in range(self.n_verb):
            v = self.verb_codebook[i]
            for u in vs:
                v = v - (v * u).sum() * u
            v = v * torch.rsqrt(v.pow(2).sum() + 1e-8)
            vs.append(v)
        return torch.stack(vs, dim=0)

    def _quotient(self, k, Q):
        coef = torch.matmul(k, Q.transpose(0, 1))
        return self._unit(k - torch.matmul(coef, Q))

    def _solve_lower(self, system, rhs):
        if system.device.type != "mps":
            return torch.linalg.solve_triangular(system, rhs, upper=False)
        parts = []
        C = system.size(1)
        for i in range(C):
            yi = rhs[:, i, :]
            if parts:
                prev = torch.stack(parts, dim=1)
                corr = torch.bmm(system[:, i : i + 1, :i], prev).squeeze(1)
                yi = yi - corr
            yi = yi / system[:, i, i].unsqueeze(-1).clamp_min(1e-6)
            parts.append(yi)
        return torch.stack(parts, dim=1)

    def _chunked_delta_reads(self, q, k, value, beta, lam):
        B, N, K = q.shape
        V = value.size(-1)
        if N == 0:
            return value.new_zeros(B, 0, V)

        dtype = value.dtype
        q = q.to(dtype)
        k = k.to(dtype)
        beta = beta.to(dtype)
        lam = lam.to(dtype).clamp(0.90, 1.0)

        mem = value.new_zeros(B, K, V)
        outs = []
        for start in range(0, N, self.chunk_size):
            end = min(N, start + self.chunk_size)
            qc = q[:, start:end, :]
            kc = k[:, start:end, :]
            vc = value[:, start:end, :]
            bc = beta[:, start:end]
            lc = lam[:, start:end]
            C = end - start

            prefix = torch.cumprod(lc, dim=1)
            before = torch.cat(
                [torch.ones(B, 1, device=value.device, dtype=dtype), prefix[:, :-1]], dim=1
            )
            denom = prefix.clamp_min(1e-6)
            between = before.unsqueeze(2) / denom.unsqueeze(1)

            strict = torch.tril(torch.ones(C, C, device=value.device, dtype=torch.bool), diagonal=-1)
            strict = strict.unsqueeze(0).to(dtype)

            kk = torch.bmm(kc, kc.transpose(1, 2))
            lower = kk * between * bc.unsqueeze(1) * strict

            rhs = vc - before.unsqueeze(-1) * torch.bmm(kc, mem)
            eye = torch.eye(C, device=value.device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            err = self._solve_lower(eye + lower, rhs)
            err = torch.nan_to_num(err, nan=0.0, posinf=1e4, neginf=-1e4)

            qk = torch.bmm(qc, kc.transpose(1, 2))
            weights = qk * between * bc.unsqueeze(1) * strict
            read = before.unsqueeze(-1) * torch.bmm(qc, mem) + torch.bmm(weights, err)
            outs.append(torch.nan_to_num(read, nan=0.0, posinf=1e4, neginf=-1e4))

            end_factor = prefix[:, -1]
            end_between = end_factor.unsqueeze(1) / denom
            contrib = torch.bmm(kc.transpose(1, 2), err * (bc * end_between).unsqueeze(-1))
            mem = end_factor.view(B, 1, 1) * mem + contrib
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        return torch.cat(outs, dim=1)

    def _transition(self, s_pre, cmd_feat):
        hin = torch.nn.functional.gelu(self.tr_in(cmd_feat))
        gb = self.tr_out(hin)
        gamma = torch.tanh(gb[..., :D]) * self.tr_gscale
        beta = gb[..., D:]
        return s_pre * (1.0 + gamma) + beta

    def transition_from_emb(self, s_pre, cmd_emb):
        """Apply the shared transition operator to a current-content estimate `s_pre` [N,D]
        and a raw standardized command embedding `cmd_emb` [N,D]; returns the post-transition
        content [N,D]. Entry point for the head aux that supervises the operator."""
        idx0 = torch.zeros(cmd_emb.size(0), dtype=torch.long, device=cmd_emb.device)
        cmd_feat = self.in_norm(self.cmd_proj(cmd_emb) + self.type_emb(idx0))
        return self._transition(s_pre, cmd_feat)

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p = self._unit(self.tr_path(x_cmd))
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            pi = p[:, i, :]
            s_pre = torch.bmm(pi.unsqueeze(1), mem).squeeze(1)
            reads.append(s_pre)
            delta = self._transition(s_pre, x_cmd[:, i, :])
            if i < n_pair:
                obs_i = obs_tok[:, i, :].to(dtype)
                wi = w[:, i].unsqueeze(-1)
                active = (valid_obs[:, i] & valid_cmd[:, i]).to(dtype).unsqueeze(-1)
            else:
                obs_i = s_pre.new_zeros(B, D)
                wi = w[:, i].unsqueeze(-1) * 0.0
                active = x_cmd.new_zeros(B, 1)
            v_i = (1.0 - wi) * obs_i + wi * delta
            corr = (v_i - s_pre) * active
            mem = decay * mem + torch.bmm(pi.unsqueeze(2), corr.unsqueeze(1))
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        if L == 0:
            h0 = tok_emb.new_zeros(B, 0, self.d)
            return tok_emb.new_zeros(B, 0, D), h0

        t = types.long().clamp(0, 1)
        pad_mask = key_pad.bool() if key_pad is not None else None
        valid = ~pad_mask if pad_mask is not None else torch.ones(B, L, device=device, dtype=torch.bool)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        h_base = self.tf(x, mask=causal, src_key_padding_mask=pad_mask)
        h_base = torch.nan_to_num(h_base, nan=0.0, posinf=1e4, neginf=-1e4)
        h_base = h_base * valid.unsqueeze(-1).to(h_base.dtype)

        n_cmd = (L + 1) // 2
        n_pair = L // 2

        x_cmd = x[:, 0::2, :]
        h_cmd0 = h_base[:, 0::2, :]
        h_obs = h_base[:, 1::2, :]
        obs_tok = tok_emb[:, 1::2, :]

        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]
        active_pair = valid_cmd[:, :n_pair] & valid_obs

        if n_pair:
            gate_in = torch.cat([h_cmd0[:, :n_pair, :], h_obs[:, :n_pair, :]], dim=-1)
            amount_pair = torch.sigmoid(self.write_gate(gate_in)).squeeze(-1)
        else:
            amount_pair = x.new_zeros(B, 0)

        write_active = self._pad_steps(active_pair, n_cmd)
        beta = self._pad_steps(amount_pair, n_cmd) * write_active.to(x.dtype)

        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        lam = torch.where(write_active, decay.to(x.dtype).expand_as(beta), torch.ones_like(beta))

        Q = self._verb_basis()
        q_file = self._quotient(self.file_read(x_cmd), Q)
        k_file = self._quotient(self.file_write(x_cmd), Q)
        ctx = self.ctx_proj(x_cmd).to(dtype)
        value_file = torch.cat([self._pad_steps(obs_tok, n_cmd), ctx], dim=-1)

        q_path = self._unit(self.path_read(h_cmd0))
        k_path = self._unit(self.path_write(h_cmd0))
        value_path = self._pad_steps(obs_tok, n_cmd)

        read_file = self._chunked_delta_reads(q_file, k_file, value_file, beta, lam)
        read_path = self._chunked_delta_reads(q_path, k_path, value_path, beta, lam)

        r_obs = read_file[..., :D]
        r_ctx = read_file[..., D:]

        film_h = torch.tanh(self.film_in(torch.cat([h_cmd0.to(dtype), r_ctx], dim=-1)))
        film = self.film_out(film_h)
        gamma = 1.0 + film[..., :D]
        shift = film[..., D:]
        r_view = torch.nan_to_num(gamma * r_obs + shift, nan=0.0, posinf=1e4, neginf=-1e4)

        if n_pair:
            sal = torch.sigmoid(self.sys_sal(h_obs)).squeeze(-1)
            sal = sal * valid_obs.to(sal.dtype)
            v_sys = torch.tanh(self.sys_val(h_obs))
            num = torch.cumsum(sal.unsqueeze(-1) * v_sys, dim=1)
            den = torch.cumsum(sal, dim=1).unsqueeze(-1)
            s_incl = num / (den + 1e-6)
            s_cmd = torch.cat([s_incl.new_zeros(B, 1, self.sys_d), s_incl], dim=1)
            s_cmd = self._pad_steps(s_cmd, n_cmd)
        else:
            s_cmd = x.new_zeros(B, n_cmd, self.sys_d)

        obs_for_prev = obs_tok * valid_obs.unsqueeze(-1).to(dtype)
        if n_cmd > 1:
            prev_src = self._pad_steps(obs_for_prev, n_cmd - 1)
            prev_obs = torch.cat([tok_emb.new_zeros(B, 1, D), prev_src], dim=1)
        else:
            prev_obs = tok_emb.new_zeros(B, n_cmd, D)

        rv = r_view.to(x.dtype)
        rp = read_path.to(x.dtype)
        ro = prev_obs.to(x.dtype)
        read_feat = torch.cat(
            [
                (rv.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (rp.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (ro.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
            ],
            dim=-1,
        )

        mix = torch.softmax(self.read_mix(torch.cat([h_cmd0, read_feat], dim=-1)), dim=-1).to(dtype)
        target_read = (
            mix[:, :, 0:1] * r_view
            + mix[:, :, 1:2] * read_path
            + mix[:, :, 2:3] * prev_obs
        )

        mem_h = self.read_to_h(target_read.to(x.dtype))
        fuse_in = torch.cat([h_cmd0, mem_h, read_feat], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        tr_reads = self._transition_reads(x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair)
        g_tr = torch.sigmoid(self.tr_read_gate(h_cmd0))
        tr_contrib = g_tr * self.tr_read(tr_reads.to(x.dtype))

        sf_h = torch.tanh(self.sysfilm_in(torch.cat([h_cmd0, s_cmd.to(x.dtype)], dim=-1)))
        sf = self.sysfilm_out(sf_h).to(dtype)
        g_sys = sf[..., :D]
        b_sys = sf[..., D:]

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred_cmd = pred[:, 0::2, :] + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read
        pred_cmd = pred_cmd + tr_contrib.to(dtype)
        pred_cmd = pred_cmd * (1.0 + g_sys) + b_sys
        pred[:, 0::2, :] = torch.nan_to_num(pred_cmd, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = torch.nan_to_num(pred * valid.unsqueeze(-1).to(pred.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        h_out = torch.nan_to_num(h_out * valid.unsqueeze(-1).to(h_out.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return R18PathStateLatentTransition(**params)
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca +0.0112 n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca -0.0112 n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- r25_role_addressed_transport_store (axis arch)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    R18PathStateLatentTransition,
)

D = 768

NAME = "r25_role_addressed_transport_store"
DESCRIPTION = (
    "The r18 path-state trunk whose latent-transition memory is replaced by a ROLE-ADDRESSED "
    "TRANSPORT STORE. Every command's raw 768-d embedding passes through one shared address "
    "trunk and is read out under three roles — query, move-source, move-destination — via "
    "role-specific pre-activation biases and diagonal gates feeding ONE shared projection, so "
    "all three roles land in a single address space. Each role code is a per-head softmax over "
    "slots (multi-head learned hashing) with a learned temperature. The store keeps a value "
    "memory and a parallel scalar mass memory, so a read is a mass-normalized convex combination "
    "of the raw observation embeddings written so far, occupancy-weighted across heads. At each "
    "step the store first reads at the query address (the emitted per-step read), then reads at "
    "the source address, passes that content through the shared latent-transition operator "
    "f(s,cmd)=s*(1+gamma)+beta, and writes the result at the DESTINATION address with a "
    "content-presence gate, while a separate gate writes the step's own observation at the query "
    "address; per-address erase strengths and a source-delete strength are learned. Because the "
    "value written at a destination is whatever the source address currently holds, chained "
    "renames compose without any resolved routing being supplied from outside. Carries the "
    "learned observation content-presence gate (prototype distance + MLP) into the file-memory "
    "write amount, the forgetting factor, the system summary and the previous-observation "
    "channel. Injected through the inherited (D,D) transition readout, initialised to a small "
    "multiple of the identity, under the inherited sigmoid gate."
)


class R25RoleAddressedTransportStore(R18PathStateLatentTransition):
    def __init__(
        self,
        addr_hidden=256,
        addr_heads=4,
        addr_slots=16,
        addr_temp=0.5,
        role_bias_scale=0.3,
        role_gate_scale=0.5,
        gate_hidden=128,
        move_bias=-1.0,
        obs_write_bias=1.5,
        content_bias=1.0,
        erase_bias=-1.0,
        presence_hidden=128,
        presence_scale=0.5,
        presence_margin=1.0,
        presence_bias=2.0,
        transport_readout_init=0.1,
        **params,
    ):
        super().__init__(**params)
        self.addr_hidden = max(32, int(addr_hidden))
        self.addr_heads = max(1, int(addr_heads))
        self.addr_slots = max(2, int(addr_slots))

        self.addr_ln = nn.LayerNorm(D)
        self.addr_in = nn.Linear(D, self.addr_hidden)
        self.role_bias = nn.Parameter(torch.randn(3, self.addr_hidden) * float(role_bias_scale))
        self.role_gate = nn.Parameter(torch.randn(3, self.addr_hidden) * float(role_gate_scale))
        self.addr_proj = nn.Linear(self.addr_hidden, self.addr_heads * self.addr_slots, bias=False)
        self.addr_log_temp = nn.Parameter(torch.tensor(math.log(max(1e-2, float(addr_temp)))))
        self.occ_logit = nn.Parameter(torch.tensor(0.0))

        gh = max(16, int(gate_hidden))
        self.gate_in = nn.Linear(self.d + self.addr_hidden + self.key_d, gh)
        self.move_out = nn.Linear(gh, 1)
        nn.init.constant_(self.move_out.bias, float(move_bias))
        self.obs_write_out = nn.Linear(gh, 1)
        nn.init.constant_(self.obs_write_out.bias, float(obs_write_bias))
        self.content_out = nn.Linear(gh + 2, 1)
        nn.init.constant_(self.content_out.bias, float(content_bias))
        self.store_erase = nn.Linear(gh, 3)
        nn.init.constant_(self.store_erase.bias, float(erase_bias))
        self.store_logit_decay = nn.Parameter(torch.tensor(2.0))

        ph = max(16, int(presence_hidden))
        self.pres_proto = nn.Parameter(torch.zeros(D))
        self.pres_scale = nn.Parameter(torch.tensor(float(presence_scale)))
        self.pres_margin = nn.Parameter(torch.tensor(float(presence_margin)))
        self.pres_in = nn.Linear(2 * self.d, ph)
        self.pres_out = nn.Linear(ph, 1)
        nn.init.zeros_(self.pres_out.weight)
        nn.init.constant_(self.pres_out.bias, float(presence_bias))

        with torch.no_grad():
            self.tr_read.weight.copy_(torch.eye(D) * float(transport_readout_init))
            self.tr_read.bias.zero_()

    def _presence(self, obs_tok, obs_x, h_obs, valid_obs):
        if obs_tok.size(1) == 0:
            return obs_x.new_zeros(obs_x.size(0), 0)
        diff = obs_tok - self.pres_proto.view(1, 1, -1).to(obs_tok.dtype)
        d2 = diff.pow(2).mean(dim=-1)
        mlp = self.pres_out(torch.tanh(self.pres_in(torch.cat([obs_x, h_obs], dim=-1)))).squeeze(-1)
        logit = self.pres_scale * (d2.to(mlp.dtype) - self.pres_margin) + mlp
        a = torch.sigmoid(logit.clamp(-30.0, 30.0))
        return a * valid_obs.to(a.dtype)

    def _role_codes(self, cmd_raw):
        base = self.addr_in(self.addr_ln(cmd_raw))
        tau = self.addr_log_temp.exp().clamp(0.05, 4.0)
        codes = []
        for r in range(3):
            hr = F.gelu(base + self.role_bias[r]) * torch.sigmoid(self.role_gate[r])
            z = self.addr_proj(hr)
            z = z.view(z.size(0), z.size(1), self.addr_heads, self.addr_slots)
            z = z - z.mean(dim=-1, keepdim=True)
            z = z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            codes.append(torch.softmax(z / tau, dim=-1))
        return base, codes[0], codes[1], codes[2]

    def _store_read(self, mem, mass, code):
        B, H, K, V = mem.shape
        num = torch.bmm(code.reshape(B * H, 1, K), mem.reshape(B * H, K, V)).view(B, H, V)
        den = (code * mass).sum(dim=-1)
        val_h = num / (den.unsqueeze(-1) + 1e-4)
        ref = F.softplus(self.occ_logit) + 0.1
        sat = den / (den + ref)
        w = sat / (sat.sum(dim=-1, keepdim=True) + 1e-4)
        val = (w.unsqueeze(-1) * val_h).sum(dim=1)
        return val, den.mean(dim=-1, keepdim=True)

    def _transport_reads(self, cmd_raw, x_cmd, obs_tok, valid_cmd, pres_cmd, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        H = self.addr_heads
        K = self.addr_slots

        base, code_read, code_src, code_dst = self._role_codes(cmd_raw.to(dtype))
        gate_feat = torch.cat([x_cmd, F.gelu(base), self.tr_path(x_cmd)], dim=-1)
        gate_h = torch.tanh(self.gate_in(gate_feat))

        live = valid_cmd.to(dtype)
        g_move_all = torch.sigmoid(
            self.move_out(gate_h).squeeze(-1) + self.tr_mut_gate(x_cmd).squeeze(-1)
        ) * live
        g_obs_all = torch.sigmoid(self.obs_write_out(gate_h).squeeze(-1)) * pres_cmd.to(dtype)
        erase_all = torch.sigmoid(self.store_erase(gate_h))
        decay = (0.90 + 0.099 * torch.sigmoid(self.store_logit_decay)).to(dtype)

        mem = x_cmd.new_zeros(B, H, K, D)
        mass = x_cmd.new_zeros(B, H, K)
        reads = []
        for i in range(n_cmd):
            c_read = code_read[:, i]
            c_src = code_src[:, i]
            c_dst = code_dst[:, i]

            val_read, _ = self._store_read(mem, mass, c_read)
            reads.append(val_read)

            val_src, occ_src = self._store_read(mem, mass, c_src)
            v_move = self._transition(val_src, x_cmd[:, i, :])
            rms_src = (val_src.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
            content = torch.sigmoid(
                self.content_out(torch.cat([gate_h[:, i], rms_src, occ_src], dim=-1))
            )

            g_ins = g_move_all[:, i].unsqueeze(-1) * content
            g_del = g_ins * erase_all[:, i, 2:3]
            g_obs = g_obs_all[:, i].unsqueeze(-1)

            if i < n_pair:
                v_obs = obs_tok[:, i, :].to(dtype)
            else:
                v_obs = x_cmd.new_zeros(B, D)

            keep = (
                (1.0 - (g_obs * erase_all[:, i, 0:1]).unsqueeze(-1) * c_read)
                * (1.0 - (g_ins * erase_all[:, i, 1:2]).unsqueeze(-1) * c_dst)
                * (1.0 - g_del.unsqueeze(-1) * c_src)
            )
            w_obs = g_obs.unsqueeze(-1) * c_read
            w_ins = g_ins.unsqueeze(-1) * c_dst
            codes = torch.stack([w_obs, w_ins], dim=-1)
            vals = torch.stack([v_obs, v_move], dim=1).unsqueeze(1).expand(B, H, 2, D)

            mem = torch.baddbmm(
                (decay * mem * keep.unsqueeze(-1)).reshape(B * H, K, D),
                codes.reshape(B * H, K, 2),
                vals.reshape(B * H, 2, D),
            ).view(B, H, K, D)
            mass = decay * mass * keep + w_obs + w_ins

        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        if L == 0:
            h0 = tok_emb.new_zeros(B, 0, self.d)
            return tok_emb.new_zeros(B, 0, D), h0

        t = types.long().clamp(0, 1)
        pad_mask = key_pad.bool() if key_pad is not None else None
        valid = ~pad_mask if pad_mask is not None else torch.ones(B, L, device=device, dtype=torch.bool)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x_all = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x_all)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        h_base = self.tf(x, mask=causal, src_key_padding_mask=pad_mask)
        h_base = torch.nan_to_num(h_base, nan=0.0, posinf=1e4, neginf=-1e4)
        h_base = h_base * valid.unsqueeze(-1).to(h_base.dtype)

        n_cmd = (L + 1) // 2
        n_pair = L // 2

        cmd_raw = tok_emb[:, 0::2, :]
        x_cmd = x[:, 0::2, :]
        h_cmd0 = h_base[:, 0::2, :]
        h_obs = h_base[:, 1::2, :]
        obs_tok = tok_emb[:, 1::2, :]
        obs_x = obs_x_all[:, 1::2, :]

        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        a_pair = self._presence(obs_tok, obs_x, h_obs, valid_obs)
        live_pair = (valid_cmd[:, :n_pair] & valid_obs).to(x.dtype)
        pres_pair = a_pair.to(x.dtype) * live_pair
        pres_cmd = self._pad_steps(pres_pair, n_cmd)

        if n_pair:
            gate_in = torch.cat([h_cmd0[:, :n_pair, :], h_obs[:, :n_pair, :]], dim=-1)
            amount_pair = torch.sigmoid(self.write_gate(gate_in)).squeeze(-1)
        else:
            amount_pair = x.new_zeros(B, 0)

        beta = self._pad_steps(amount_pair, n_cmd) * pres_cmd

        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        lam = 1.0 - (1.0 - decay.to(x.dtype)) * pres_cmd

        Q = self._verb_basis()
        q_file = self._quotient(self.file_read(x_cmd), Q)
        k_file = self._quotient(self.file_write(x_cmd), Q)
        ctx = self.ctx_proj(x_cmd).to(dtype)
        value_file = torch.cat([self._pad_steps(obs_tok, n_cmd), ctx], dim=-1)

        q_path = self._unit(self.path_read(h_cmd0))
        k_path = self._unit(self.path_write(h_cmd0))
        value_path = self._pad_steps(obs_tok, n_cmd)

        read_file = self._chunked_delta_reads(q_file, k_file, value_file, beta, lam)
        read_path = self._chunked_delta_reads(q_path, k_path, value_path, beta, lam)

        r_obs = read_file[..., :D]
        r_ctx = read_file[..., D:]

        film_h = torch.tanh(self.film_in(torch.cat([h_cmd0.to(dtype), r_ctx], dim=-1)))
        film = self.film_out(film_h)
        gamma = 1.0 + film[..., :D]
        shift = film[..., D:]
        r_view = torch.nan_to_num(gamma * r_obs + shift, nan=0.0, posinf=1e4, neginf=-1e4)

        if n_pair:
            sal = torch.sigmoid(self.sys_sal(h_obs)).squeeze(-1)
            sal = sal * pres_pair.to(sal.dtype)
            v_sys = torch.tanh(self.sys_val(h_obs))
            num = torch.cumsum(sal.unsqueeze(-1) * v_sys, dim=1)
            den = torch.cumsum(sal, dim=1).unsqueeze(-1)
            s_incl = num / (den + 1e-6)
            s_cmd = torch.cat([s_incl.new_zeros(B, 1, self.sys_d), s_incl], dim=1)
            s_cmd = self._pad_steps(s_cmd, n_cmd)
        else:
            s_cmd = x.new_zeros(B, n_cmd, self.sys_d)

        obs_for_prev = obs_tok * pres_pair.unsqueeze(-1).to(dtype)
        if n_cmd > 1:
            prev_src = self._pad_steps(obs_for_prev, n_cmd - 1)
            prev_obs = torch.cat([tok_emb.new_zeros(B, 1, D), prev_src], dim=1)
        else:
            prev_obs = tok_emb.new_zeros(B, n_cmd, D)

        rv = r_view.to(x.dtype)
        rp = read_path.to(x.dtype)
        ro = prev_obs.to(x.dtype)
        read_feat = torch.cat(
            [
                (rv.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (rp.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (ro.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
            ],
            dim=-1,
        )

        mix = torch.softmax(self.read_mix(torch.cat([h_cmd0, read_feat], dim=-1)), dim=-1).to(dtype)
        target_read = (
            mix[:, :, 0:1] * r_view
            + mix[:, :, 1:2] * read_path
            + mix[:, :, 2:3] * prev_obs
        )

        mem_h = self.read_to_h(target_read.to(x.dtype))
        fuse_in = torch.cat([h_cmd0, mem_h, read_feat], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        tr_reads = self._transport_reads(
            cmd_raw, x_cmd, obs_tok, valid_cmd, pres_cmd, n_cmd, n_pair
        )
        g_tr = torch.sigmoid(self.tr_read_gate(h_cmd0))
        tr_contrib = g_tr * self.tr_read(tr_reads.to(x.dtype))

        sf_h = torch.tanh(self.sysfilm_in(torch.cat([h_cmd0, s_cmd.to(x.dtype)], dim=-1)))
        sf = self.sysfilm_out(sf_h).to(dtype)
        g_sys = sf[..., :D]
        b_sys = sf[..., D:]

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred_cmd = pred[:, 0::2, :] + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read
        pred_cmd = pred_cmd + tr_contrib.to(dtype)
        pred_cmd = pred_cmd * (1.0 + g_sys) + b_sys
        pred[:, 0::2, :] = torch.nan_to_num(pred_cmd, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = torch.nan_to_num(pred * valid.unsqueeze(-1).to(pred.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        h_out = torch.nan_to_num(h_out * valid.unsqueeze(-1).to(h_out.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return R25RoleAddressedTransportStore(**params)

--- r23_dual_address_transport_pointer (axis arch)
import math

import torch
import torch.nn as nn

from evolve.chunks.arch.r22_prefix_content_xattention import R22PrefixContentXAttention

D = 768

NAME = "r23_dual_address_transport_pointer"
DESCRIPTION = (
    "The r22 prefix-content arch plus a DUAL-ADDRESS transport memory and a pointer-copy "
    "readout, on a trunk widened to the reference baseline's shape (d=192, ffn 4x). Each "
    "command's raw 768-d embedding is passed through one shared GELU layer and two separate "
    "unit-normalized address heads, giving a SOURCE address and a DESTINATION address per step. "
    "At every step the model reads the content currently held at the source address, writes "
    "either that read content (transport) or the step's own observation (exposure) at the "
    "destination address under a delta rule, and erases the source, all in one batched rank-2 "
    "update of a [key_d, 768] content store. The value retrieved at the source address is then "
    "used as the QUERY of a masked softmax over the strictly-earlier observation embeddings, "
    "whose output copies one raw prefix observation verbatim; a gated blend of the retrieved "
    "value and the copied observation is injected into the command-position prediction through "
    "a zero-init (D,D) readout, so the forward is bit-identical to r22 at init. The address "
    "heads form an identical-shape (key_d, hidden) pair and the readout is (D,D)."
)


class R23DualAddressTransportPointer(R22PrefixContentXAttention):
    def __init__(
        self,
        addr_hidden=256,
        ptr_dim=64,
        tp_gate_bias=-2.0,
        move_gate_bias=0.0,
        write_gate_bias=1.0,
        erase_gate_bias=-1.0,
        tp_decay=0.999,
        d=192,
        ffn_mult=4,
        **params,
    ):
        super().__init__(d=d, ffn_mult=ffn_mult, **params)
        self.addr_hidden = max(32, int(addr_hidden))
        self.ptr_dim = max(8, int(ptr_dim))
        self.tp_decay = float(tp_decay)

        self.addr_ln = nn.LayerNorm(D)
        self.addr_in = nn.Linear(D, self.addr_hidden)
        self.addr_src = nn.Linear(self.addr_hidden, self.key_d, bias=False)
        self.addr_dst = nn.Linear(self.addr_hidden, self.key_d, bias=False)

        self.tp_move_gate = nn.Linear(self.d + self.addr_hidden, 1)
        self.tp_write_gate = nn.Linear(self.d + self.addr_hidden, 1)
        self.tp_erase_gate = nn.Linear(self.d + self.addr_hidden, 1)
        nn.init.constant_(self.tp_move_gate.bias, float(move_gate_bias))
        nn.init.constant_(self.tp_write_gate.bias, float(write_gate_bias))
        nn.init.constant_(self.tp_erase_gate.bias, float(erase_gate_bias))

        self.ptr_q = nn.Linear(D, self.ptr_dim, bias=False)
        self.ptr_k = nn.Linear(D, self.ptr_dim, bias=False)

        self.tp_alpha = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.tp_alpha.bias, 0.0)
        self.tp_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.tp_gate.bias, float(tp_gate_bias))

        self.tp_out = nn.Linear(D, D)
        nn.init.zeros_(self.tp_out.weight)
        nn.init.zeros_(self.tp_out.bias)

    def _transport_reads(self, a_src, a_dst, obs_raw, g_move, g_write, g_erase, obs_live,
                         n_cmd, n_pair):
        B = a_src.size(0)
        mem = a_src.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            ai = a_src[:, i, :]
            di = a_dst[:, i, :]
            s_i = torch.bmm(ai.unsqueeze(1), mem).squeeze(1)
            reads.append(s_i)
            if i < n_pair:
                obs_i = obs_raw[:, i, :]
                live = obs_live[:, i].unsqueeze(-1)
            else:
                obs_i = mem.new_zeros(B, D)
                live = mem.new_zeros(B, 1)
            gm = g_move[:, i, :]
            v_i = gm * s_i + (1.0 - gm) * obs_i
            cur_d = torch.bmm(di.unsqueeze(1), mem).squeeze(1)
            w_write = (g_write[:, i, :] * live) * (v_i - cur_d)
            w_erase = -(g_erase[:, i, :] * gm * live) * s_i
            upd_a = torch.stack([di, ai], dim=2)
            upd_v = torch.stack([w_write, w_erase], dim=1)
            mem = torch.baddbmm(mem, upd_a, upd_v, beta=self.tp_decay)
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        out = torch.stack(reads, dim=1)
        return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

    def forward(self, tok_emb, types, key_pad):
        pred, h_out = super().forward(tok_emb, types, key_pad)
        B, L, _ = tok_emb.shape
        if L < 3:
            return pred, h_out
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        if n_pair == 0:
            return pred, h_out

        device = tok_emb.device
        dtype = tok_emb.dtype
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, dtype=torch.bool, device=device)
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        cmd_raw = tok_emb[:, 0::2, :]
        obs_raw = tok_emb[:, 1::2, :]
        h_cmd = h_out[:, 0::2, :].to(dtype)

        a_h = torch.nn.functional.gelu(self.addr_in(self.addr_ln(cmd_raw)))
        a_src = self._unit(self.addr_src(a_h)).to(dtype)
        a_dst = self._unit(self.addr_dst(a_h)).to(dtype)

        g_in = torch.cat([h_cmd, a_h.to(dtype)], dim=-1)
        g_move = torch.sigmoid(self.tp_move_gate(g_in)).to(dtype)
        g_write = torch.sigmoid(self.tp_write_gate(g_in)).to(dtype)
        g_erase = torch.sigmoid(self.tp_erase_gate(g_in)).to(dtype)

        obs_live = (valid_obs & valid_cmd[:, :n_pair]).to(dtype)

        read = self._transport_reads(
            a_src, a_dst, obs_raw, g_move, g_write, g_erase, obs_live, n_cmd, n_pair)

        q = self.ptr_q(read)
        k = self.ptr_k(obs_raw)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.ptr_dim)

        ci = torch.arange(n_cmd, device=device).unsqueeze(1)
        pj = torch.arange(n_pair, device=device).unsqueeze(0)
        allowed = (pj < ci).unsqueeze(0) & valid_obs.unsqueeze(1)

        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~allowed, neg)
        has_key = allowed.any(dim=2, keepdim=True)
        attn = torch.softmax(scores, dim=2)
        attn = torch.where(has_key, attn, torch.zeros_like(attn))
        copy = torch.bmm(attn.to(dtype), obs_raw)
        copy = torch.nan_to_num(copy, nan=0.0, posinf=1e4, neginf=-1e4)
        sharp = attn.amax(dim=2, keepdim=True).to(dtype)

        rms_read = (read.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        rms_copy = (copy.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        feat = torch.cat([rms_read, rms_copy, sharp], dim=-1)
        gi = torch.cat([h_cmd, feat], dim=-1)

        alpha = torch.sigmoid(self.tp_alpha(gi))
        blend = alpha * copy + (1.0 - alpha) * read
        contrib = torch.sigmoid(self.tp_gate(gi)) * self.tp_out(blend)
        contrib = contrib * valid_cmd.unsqueeze(-1).to(contrib.dtype)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        return pred, h_out


def build(**params):
    return R23DualAddressTransportPointer(**params)

--- r5_dualaddress_transport_closure (axis arch)
import math

import torch
import torch.nn as nn

D = 768

NAME = "r5_dualaddress_transport_closure"
DESCRIPTION = (
    "The reference-closure transport trunk with its per-step latent memory turned into a "
    "TWO-ADDRESS content transporter. One shared linear address map is applied to two fixed, "
    "equal-width coordinate regions of the raw command token: the first gives the step's READ "
    "address, the second its WRITE address, and the two coincide exactly when a coding stream "
    "puts the same pattern in both regions. Each step reads the content currently held at its "
    "read address, blends it with the step's own observation through a learned "
    "observation-informativeness gate and the shared command-conditioned affine transition "
    "operator, and writes the result at its WRITE address with the delta rule, then subtracts a "
    "learned fraction of the read content back out of the read address, scaled by how far the two "
    "addresses differ, so a step with a single address never erases itself. A step whose "
    "observation carries nothing therefore relocates the content it retrieved instead of storing "
    "an empty observation, and a chain of such steps carries one content across arbitrarily many "
    "hops. The parallel delta-rule file and path memories, the reference-closure branch with its "
    "one-shot transitive solve, the system-summary film and every readout are retained. Strictly "
    "causal: a step's read precedes its own write, so it sees only earlier observations. The "
    "transporter owns no private addressing parameters: when a caller does not supply the raw "
    "command token the two addresses come from the path memory's own read/write map pair, which "
    "the trunk trains and reads through at every prediction."
)


class R5DualAddressTransportClosure(nn.Module):
    def __init__(
        self,
        d=176,
        layers=4,
        heads=4,
        key_d=64,
        ctx_d=96,
        n_verb=8,
        film_hidden=128,
        sys_d=64,
        sysfilm_hidden=128,
        tr_hidden=192,
        tr_gscale=0.5,
        ffn_mult=2,
        dropout=0.1,
        chunk_size=16,
        ref_d=128,
        ref_pd=64,
        ref_recency_init=0.1,
        obs_info_bias=4.0,
        addr_w=224,
        src_lo=0,
        dst_lo=224,
        erase_bias=-2.0,
        **unused,
    ):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]
        if "chunk" in unused:
            chunk_size = unused["chunk"]

        self.D = D
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = int(key_d)
        self.ctx_d = int(ctx_d)
        self.n_verb = max(1, int(n_verb))
        self.sys_d = max(8, int(sys_d))
        self.chunk_size = max(1, int(chunk_size))
        self.tr_gscale = float(tr_gscale)
        self.ref_d = max(8, int(ref_d))
        self.ref_pd = max(8, int(ref_pd))
        self.addr_w = max(8, min(int(addr_w), D))
        self.src_lo = max(0, min(int(src_lo), D - self.addr_w))
        self.dst_lo = max(0, min(int(dst_lo), D - self.addr_w))
        ffn_h = max(self.d, int(float(ffn_mult) * self.d))

        self.cmd_proj = nn.Linear(D, self.d)
        self.obs_proj = nn.Linear(D, self.d)
        self.type_emb = nn.Embedding(2, self.d)
        self.in_norm = nn.LayerNorm(self.d)
        self.pos_scale = nn.Parameter(torch.tensor(0.2))

        enc = nn.TransformerEncoderLayer(
            self.d,
            int(heads),
            ffn_h,
            float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tf = nn.TransformerEncoder(enc, self.layers, enable_nested_tensor=False)

        self.file_read = nn.Linear(self.d, self.key_d, bias=False)
        self.file_write = nn.Linear(self.d, self.key_d, bias=False)
        self.verb_codebook = nn.Parameter(torch.randn(self.n_verb, self.key_d) * 0.2)
        self.ctx_proj = nn.Linear(self.d, self.ctx_d)

        self.path_read = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write = nn.Linear(self.d, self.key_d, bias=False)

        self.write_gate = nn.Linear(2 * self.d, 1)

        fh = max(16, int(film_hidden))
        self.film_in = nn.Linear(self.d + self.ctx_d, fh)
        self.film_out = nn.Linear(fh, 2 * D)
        nn.init.zeros_(self.film_out.weight)
        nn.init.zeros_(self.film_out.bias)

        self.sys_sal = nn.Linear(self.d, 1)
        self.sys_val = nn.Linear(self.d, self.sys_d)
        sh = max(16, int(sysfilm_hidden))
        self.sysfilm_in = nn.Linear(self.d + self.sys_d, sh)
        self.sysfilm_out = nn.Linear(sh, 2 * D)
        nn.init.zeros_(self.sysfilm_out.weight)
        nn.init.zeros_(self.sysfilm_out.bias)

        self.read_mix = nn.Linear(self.d + 3, 3)
        self.read_to_h = nn.Linear(D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d + 3, self.d)
        self.direct_gate = nn.Linear(2 * self.d + 3, 1)
        self.out_norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, D)

        self.addr = nn.Linear(self.addr_w, self.key_d, bias=False)
        th = max(32, int(tr_hidden))
        self.tr_in = nn.Linear(self.d, th)
        self.tr_out = nn.Linear(th, 2 * D)
        self.tr_mut_gate = nn.Linear(self.d, 1)
        self.tr_erase = nn.Linear(self.d, 1)
        self.tr_read = nn.Linear(D, D)
        nn.init.zeros_(self.tr_read.weight)
        nn.init.zeros_(self.tr_read.bias)
        self.tr_read_gate = nn.Linear(self.d, 1)
        nn.init.constant_(self.tr_mut_gate.bias, -1.0)
        nn.init.constant_(self.tr_erase.bias, float(erase_bias))
        nn.init.constant_(self.tr_read_gate.bias, 0.0)

        self.obs_info = nn.Linear(self.d, 1)
        nn.init.constant_(self.obs_info.bias, float(obs_info_bias))

        self.ref_ln = nn.LayerNorm(D)
        self.ref_q = nn.Linear(D, self.ref_d, bias=False)
        self.ref_k = nn.Linear(D, self.ref_d, bias=False)
        self.ref_p = nn.Linear(D, self.ref_pd, bias=False)
        self.ref_diff = nn.Linear(self.ref_pd, 1)
        nn.init.zeros_(self.ref_diff.bias)
        self.ref_keybias = nn.Linear(D, 1)
        nn.init.zeros_(self.ref_keybias.bias)
        self.ref_null = nn.Linear(D, 1)
        nn.init.zeros_(self.ref_null.bias)
        self.ref_own = nn.Linear(D + self.d, 1)
        nn.init.zeros_(self.ref_own.bias)
        self.ref_gate = nn.Linear(self.d + 3, 1)
        nn.init.zeros_(self.ref_gate.bias)
        self.ref_read = nn.Linear(D, D)
        nn.init.zeros_(self.ref_read.weight)
        nn.init.zeros_(self.ref_read.bias)
        rr = max(1e-4, float(ref_recency_init))
        self.ref_recency = nn.Parameter(torch.tensor(math.log(math.expm1(rr))))

        init_decay = (0.985 - 0.90) / 0.099
        self.logit_decay = nn.Parameter(torch.tensor(math.log(init_decay / (1.0 - init_decay))))

        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.fuse_gate.bias, -1.0)
        nn.init.constant_(self.direct_gate.bias, -2.0)

    def _positional(self, L, device, dtype):
        half = (self.d + 1) // 2
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        pe = torch.zeros(L, self.d, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: self.d // 2])
        return pe

    @staticmethod
    def _unit(x):
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + 1e-12)

    @staticmethod
    def _pad_steps(x, n):
        cur = x.size(1)
        if cur == n:
            return x
        if cur > n:
            return x[:, :n]
        pad_shape = (x.size(0), n - cur) + tuple(x.shape[2:])
        return torch.cat([x, x.new_zeros(pad_shape)], dim=1)

    def _verb_basis(self):
        vs = []
        for i in range(self.n_verb):
            v = self.verb_codebook[i]
            for u in vs:
                v = v - (v * u).sum() * u
            v = v * torch.rsqrt(v.pow(2).sum() + 1e-8)
            vs.append(v)
        return torch.stack(vs, dim=0)

    def _quotient(self, k, Q):
        coef = torch.matmul(k, Q.transpose(0, 1))
        return self._unit(k - torch.matmul(coef, Q))

    def _solve_lower(self, system, rhs):
        if system.device.type != "mps":
            return torch.linalg.solve_triangular(system, rhs, upper=False)
        parts = []
        C = system.size(1)
        for i in range(C):
            yi = rhs[:, i, :]
            if parts:
                prev = torch.stack(parts, dim=1)
                corr = torch.bmm(system[:, i : i + 1, :i], prev).squeeze(1)
                yi = yi - corr
            yi = yi / system[:, i, i].unsqueeze(-1).clamp_min(1e-6)
            parts.append(yi)
        return torch.stack(parts, dim=1)

    def _chunked_delta_reads(self, q, k, value, beta, lam):
        B, N, K = q.shape
        V = value.size(-1)
        if N == 0:
            return value.new_zeros(B, 0, V)

        dtype = value.dtype
        q = q.to(dtype)
        k = k.to(dtype)
        beta = beta.to(dtype)
        lam = lam.to(dtype).clamp(0.90, 1.0)

        mem = value.new_zeros(B, K, V)
        outs = []
        for start in range(0, N, self.chunk_size):
            end = min(N, start + self.chunk_size)
            qc = q[:, start:end, :]
            kc = k[:, start:end, :]
            vc = value[:, start:end, :]
            bc = beta[:, start:end]
            lc = lam[:, start:end]
            C = end - start

            prefix = torch.cumprod(lc, dim=1)
            before = torch.cat(
                [torch.ones(B, 1, device=value.device, dtype=dtype), prefix[:, :-1]], dim=1
            )
            denom = prefix.clamp_min(1e-6)
            between = before.unsqueeze(2) / denom.unsqueeze(1)

            strict = torch.tril(torch.ones(C, C, device=value.device, dtype=torch.bool), diagonal=-1)
            strict = strict.unsqueeze(0).to(dtype)

            kk = torch.bmm(kc, kc.transpose(1, 2))
            lower = kk * between * bc.unsqueeze(1) * strict

            rhs = vc - before.unsqueeze(-1) * torch.bmm(kc, mem)
            eye = torch.eye(C, device=value.device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            err = self._solve_lower(eye + lower, rhs)
            err = torch.nan_to_num(err, nan=0.0, posinf=1e4, neginf=-1e4)

            qk = torch.bmm(qc, kc.transpose(1, 2))
            weights = qk * between * bc.unsqueeze(1) * strict
            read = before.unsqueeze(-1) * torch.bmm(qc, mem) + torch.bmm(weights, err)
            outs.append(torch.nan_to_num(read, nan=0.0, posinf=1e4, neginf=-1e4))

            end_factor = prefix[:, -1]
            end_between = end_factor.unsqueeze(1) / denom
            contrib = torch.bmm(kc.transpose(1, 2), err * (bc * end_between).unsqueeze(-1))
            mem = end_factor.view(B, 1, 1) * mem + contrib
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        return torch.cat(outs, dim=1)

    def _transition(self, s_pre, cmd_feat):
        hin = torch.nn.functional.gelu(self.tr_in(cmd_feat))
        gb = self.tr_out(hin)
        gamma = torch.tanh(gb[..., :D]) * self.tr_gscale
        beta = gb[..., D:]
        return s_pre * (1.0 + gamma) + beta

    def transition_from_emb(self, s_pre, cmd_emb):
        idx0 = torch.zeros(cmd_emb.size(0), dtype=torch.long, device=cmd_emb.device)
        cmd_feat = self.in_norm(self.cmd_proj(cmd_emb) + self.type_emb(idx0))
        return self._transition(s_pre, cmd_feat)

    def _step_addresses(self, x_cmd, cmd_raw):
        if cmd_raw is None:
            return self._unit(self.path_read(x_cmd)), self._unit(self.path_write(x_cmd))
        dtype = x_cmd.dtype
        s0 = self.src_lo
        d0 = self.dst_lo
        w = self.addr_w
        a_read = self._unit(self.addr(cmd_raw[..., s0:s0 + w].to(dtype)))
        a_write = self._unit(self.addr(cmd_raw[..., d0:d0 + w].to(dtype)))
        return a_read, a_write

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair, cmd_raw=None):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        a_read, a_write = self._step_addresses(x_cmd, cmd_raw)
        mut = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        info = torch.sigmoid(self.obs_info(x_cmd)).squeeze(-1)
        align = (a_read * a_write).sum(dim=-1).clamp(-1.0, 1.0)
        erase = torch.sigmoid(self.tr_erase(x_cmd)).squeeze(-1) * (1.0 - align * align)
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            ri = a_read[:, i, :]
            wi = a_write[:, i, :]
            s_src = torch.bmm(ri.unsqueeze(1), mem).squeeze(1)
            reads.append(s_src)
            delta = self._transition(s_src, x_cmd[:, i, :])
            if i < n_pair:
                obs_i = obs_tok[:, i, :].to(dtype)
                active = (valid_obs[:, i] & valid_cmd[:, i]).to(dtype).unsqueeze(-1)
                gi = mut[:, i].unsqueeze(-1)
            else:
                obs_i = s_src.new_zeros(B, D)
                active = x_cmd.new_zeros(B, 1)
                gi = mut[:, i].unsqueeze(-1) * 0.0
            fi = info[:, i].unsqueeze(-1)
            base_i = fi * obs_i + (1.0 - fi) * s_src
            v_i = (1.0 - gi) * base_i + gi * delta
            s_dst = torch.bmm(wi.unsqueeze(1), mem).squeeze(1)
            corr_w = (v_i - s_dst) * active
            corr_e = (-erase[:, i].unsqueeze(-1) * s_src) * active
            addr_pair = torch.stack([wi, ri], dim=2)
            corr_pair = torch.stack([corr_w, corr_e], dim=1)
            mem = decay * mem + torch.bmm(addr_pair, corr_pair)
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)

    def _reference_closure(self, cmd_raw, h_cmd0, obs_pad, valid_cmd, n_cmd):
        B = cmd_raw.size(0)
        device = cmd_raw.device
        dtype = h_cmd0.dtype

        u = self.ref_ln(cmd_raw).to(dtype)
        q = self.ref_q(u)
        k = self.ref_k(u)
        p = self.ref_p(u)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(float(self.ref_d))
        mismatch = (p.unsqueeze(2) - p.unsqueeze(1)).abs()
        scores = scores + self.ref_diff(mismatch).squeeze(-1)
        scores = scores + self.ref_keybias(u).transpose(1, 2)

        idx = torch.arange(n_cmd, device=device)
        gap = (idx.view(n_cmd, 1) - idx.view(1, n_cmd)).clamp_min(0).to(dtype)
        scores = scores - torch.nn.functional.softplus(self.ref_recency).to(dtype) * gap.unsqueeze(0)

        allowed = (idx.view(n_cmd, 1) > idx.view(1, n_cmd)).unsqueeze(0) & valid_cmd.unsqueeze(1)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

        full = torch.cat([scores, self.ref_null(u)], dim=2)
        attn = torch.softmax(full, dim=2)[:, :, :n_cmd]
        attn = attn * valid_cmd.unsqueeze(-1).to(dtype)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        own = torch.sigmoid(self.ref_own(torch.cat([u, h_cmd0], dim=-1)))
        own = own * valid_cmd.unsqueeze(-1).to(dtype)

        transfer = (1.0 - own) * attn
        eye = torch.eye(n_cmd, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        system = eye - transfer
        rhs = own * obs_pad.to(dtype)
        registers = self._solve_lower(system, rhs)
        registers = torch.nan_to_num(registers, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        pulled = torch.bmm(attn, registers)
        pulled = torch.nan_to_num(pulled, nan=0.0, posinf=1e4, neginf=-1e4)
        sharp = attn.amax(dim=2, keepdim=True)
        mass = attn.sum(dim=2, keepdim=True)
        return pulled, sharp, mass

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        if L == 0:
            h0 = tok_emb.new_zeros(B, 0, self.d)
            return tok_emb.new_zeros(B, 0, D), h0

        t = types.long().clamp(0, 1)
        pad_mask = key_pad.bool() if key_pad is not None else None
        valid = ~pad_mask if pad_mask is not None else torch.ones(B, L, device=device, dtype=torch.bool)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        h_base = self.tf(x, mask=causal, src_key_padding_mask=pad_mask)
        h_base = torch.nan_to_num(h_base, nan=0.0, posinf=1e4, neginf=-1e4)
        h_base = h_base * valid.unsqueeze(-1).to(h_base.dtype)

        n_cmd = (L + 1) // 2
        n_pair = L // 2

        x_cmd = x[:, 0::2, :]
        h_cmd0 = h_base[:, 0::2, :]
        h_obs = h_base[:, 1::2, :]
        obs_tok = tok_emb[:, 1::2, :]
        cmd_raw = tok_emb[:, 0::2, :]

        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]
        active_pair = valid_cmd[:, :n_pair] & valid_obs

        if n_pair:
            gate_in = torch.cat([h_cmd0[:, :n_pair, :], h_obs[:, :n_pair, :]], dim=-1)
            amount_pair = torch.sigmoid(self.write_gate(gate_in)).squeeze(-1)
        else:
            amount_pair = x.new_zeros(B, 0)

        write_active = self._pad_steps(active_pair, n_cmd)
        beta = self._pad_steps(amount_pair, n_cmd) * write_active.to(x.dtype)

        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        lam = torch.where(write_active, decay.to(x.dtype).expand_as(beta), torch.ones_like(beta))

        Q = self._verb_basis()
        q_file = self._quotient(self.file_read(x_cmd), Q)
        k_file = self._quotient(self.file_write(x_cmd), Q)
        ctx = self.ctx_proj(x_cmd).to(dtype)
        value_file = torch.cat([self._pad_steps(obs_tok, n_cmd), ctx], dim=-1)

        q_path = self._unit(self.path_read(h_cmd0))
        k_path = self._unit(self.path_write(h_cmd0))
        value_path = self._pad_steps(obs_tok, n_cmd)

        read_file = self._chunked_delta_reads(q_file, k_file, value_file, beta, lam)
        read_path = self._chunked_delta_reads(q_path, k_path, value_path, beta, lam)

        r_obs = read_file[..., :D]
        r_ctx = read_file[..., D:]

        film_h = torch.tanh(self.film_in(torch.cat([h_cmd0.to(dtype), r_ctx], dim=-1)))
        film = self.film_out(film_h)
        gamma = 1.0 + film[..., :D]
        shift = film[..., D:]
        r_view = torch.nan_to_num(gamma * r_obs + shift, nan=0.0, posinf=1e4, neginf=-1e4)

        if n_pair:
            sal = torch.sigmoid(self.sys_sal(h_obs)).squeeze(-1)
            sal = sal * valid_obs.to(sal.dtype)
            v_sys = torch.tanh(self.sys_val(h_obs))
            num = torch.cumsum(sal.unsqueeze(-1) * v_sys, dim=1)
            den = torch.cumsum(sal, dim=1).unsqueeze(-1)
            s_incl = num / (den + 1e-6)
            s_cmd = torch.cat([s_incl.new_zeros(B, 1, self.sys_d), s_incl], dim=1)
            s_cmd = self._pad_steps(s_cmd, n_cmd)
        else:
            s_cmd = x.new_zeros(B, n_cmd, self.sys_d)

        obs_for_prev = obs_tok * valid_obs.unsqueeze(-1).to(dtype)
        if n_cmd > 1:
            prev_src = self._pad_steps(obs_for_prev, n_cmd - 1)
            prev_obs = torch.cat([tok_emb.new_zeros(B, 1, D), prev_src], dim=1)
        else:
            prev_obs = tok_emb.new_zeros(B, n_cmd, D)

        rv = r_view.to(x.dtype)
        rp = read_path.to(x.dtype)
        ro = prev_obs.to(x.dtype)
        read_feat = torch.cat(
            [
                (rv.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (rp.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (ro.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
            ],
            dim=-1,
        )

        mix = torch.softmax(self.read_mix(torch.cat([h_cmd0, read_feat], dim=-1)), dim=-1).to(dtype)
        target_read = (
            mix[:, :, 0:1] * r_view
            + mix[:, :, 1:2] * read_path
            + mix[:, :, 2:3] * prev_obs
        )

        mem_h = self.read_to_h(target_read.to(x.dtype))
        fuse_in = torch.cat([h_cmd0, mem_h, read_feat], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        tr_reads = self._transition_reads(
            x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair, cmd_raw
        )
        g_tr = torch.sigmoid(self.tr_read_gate(h_cmd0))
        tr_contrib = g_tr * self.tr_read(tr_reads.to(x.dtype))

        obs_pad = self._pad_steps(obs_for_prev, n_cmd)
        pulled, sharp, mass = self._reference_closure(cmd_raw, h_cmd0, obs_pad, valid_cmd, n_cmd)
        chain_render = self._transition(pulled, x_cmd)
        chain_feat = torch.cat(
            [(pulled.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(), sharp, mass], dim=-1
        )
        g_ref = torch.sigmoid(self.ref_gate(torch.cat([h_cmd0, chain_feat], dim=-1)))
        ref_contrib = g_ref * self.ref_read(chain_render)
        ref_contrib = torch.nan_to_num(ref_contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        sf_h = torch.tanh(self.sysfilm_in(torch.cat([h_cmd0, s_cmd.to(x.dtype)], dim=-1)))
        sf = self.sysfilm_out(sf_h).to(dtype)
        g_sys = sf[..., :D]
        b_sys = sf[..., D:]

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred_cmd = pred[:, 0::2, :] + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read
        pred_cmd = pred_cmd + tr_contrib.to(dtype) + ref_contrib.to(dtype)
        pred_cmd = pred_cmd * (1.0 + g_sys) + b_sys
        pred[:, 0::2, :] = torch.nan_to_num(pred_cmd, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = torch.nan_to_num(pred * valid.unsqueeze(-1).to(pred.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        h_out = torch.nan_to_num(h_out * valid.unsqueeze(-1).to(h_out.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return R5DualAddressTransportClosure(**params)

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of -0.0112 (r6-29-routing-necessity-pips, full budget, runpod-4090, inner split).
The unmodified baseline scores +0.0112 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
