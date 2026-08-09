"""
R18 arch: PER-PATH LATENT-TRANSITION WORLD MODEL — an object-centric,
Dreamer/RSSM-style *learned latent dynamics* keyed by path, added as a zero-init
correction on the carried r13 trunk (syscond FiLM + verb-quotient filebind + path
memory), so at init it computes that trunk's function bit-for-bit.

WHERE THE MARGIN LIVES (v3): the least-retrievable cells are content reads on
MUTATED paths. A symbolic tracker + mutation-aware retrieval are already SUBTRACTED
as baselines, so merely re-deriving "which path changed" ties them and earns ~0. The
open lever is predicting the CURRENT CONTENT a tracker cannot compute — e.g. after
`echo hi > P` a later `cat P` must return "hi", content that never appeared as stdout.

WHY THIS IS A NEW CORE DYNAMICS (distinct from all prior path memories):
  * r17_pathdelta stores value = obs + w_mut * v_cmd — an ADDITIVE command projection
    that does NOT depend on the path's current content (v_cmd(cmd) alone). It cannot
    express MODIFY/APPEND (new = g(old, cmd)); it can only overwrite with a
    command-decoded vector.
  * r18 typed-slot memory (in-round #6) TRANSPORTS fixed content vectors across discrete
    slots (MOVE p->q, DELETE free, allocate) — symbolic bookkeeping, no learned content
    TRANSFORM.
  * THIS arch's primitive is a LEARNED CONTENT-TRANSITION OPERATOR f(s_pre, cmd): the
    value written to a path slot is a command-conditioned AFFINE EDIT of the slot's
    CURRENT retrieved content, s_pre * (1 + gamma(cmd)) + beta(cmd), with gamma,beta a
    per-channel (diagonal) function of the COMMAND and gamma bounded by a tanh (stable
    recurrence; command-only so the per-path recurrence is an exact linear scan). REPLACE
    (gamma->-1, beta=new content),
    MODIFY/APPEND (gamma~0, beta=increment), or plain carry (blend toward obs) all fall
    out of one operator. Because a path is touched repeatedly, this is a genuine
    RECURRENCE s -> f(s) -> f(f(s)) — a per-object latent world-model transition (RSSM /
    Dreamer, arXiv:1912.01603; object-centric slots), applied per PATH, in obs space.

MECHANISM (sequential causal scan over the <=16 command positions — cheap; the r13 trunk
already runs a single-chunk delta solve at this length):
  For command i, with path key p_i = unit(W_tr x_cmd_i):
    s_pre_i = p_i^T MEM                         # current content from writes < i (CAUSAL)
    read_i  = s_pre_i                            # the post-mutation content this read returns
    delta_i = s_pre_i*(1 + g) + b   where (g,b)=Transition(x_cmd_i), g=tanh*g_scale  # COMMAND-conditioned
    v_i     = (1 - w_i)*obs_i + w_i*delta_i      # w_i=sigmoid(mut_gate): reads store obs, mutations store the transition
    MEM    <- decay*MEM + p_i (v_i - s_pre_i)^T  # delta-rule (Widrow-Hoff) overwrite of the slot
  (g,b) depend on the COMMAND only (a per-channel/diagonal affine edit — the mechanism's point),
  so this recurrence is exactly a diagonal-gated linear scan with erase gate a_i=w_i(1+g)-1 and
  additive source b_i=(1-w_i)obs_i+w_i*beta_i; it therefore admits an exact batched closed-form
  solve (implemented + verified equivalent, <1.1e-6), but the per-channel gate makes that solve
  D-heavy and it is a wash-to-slower at the real train config, so the <=16-step loop ships.
  The read at i uses MEM from strictly-earlier writes only; obs_i enters the WRITE at i,
  which affects reads at i+1.. only — strictly causal (obs_t perturbs predictions at
  commands > t only). The read is injected as pred_cmd += sigmoid(gate) * W_ro(read),
  with W_ro a ZERO-INIT (D,D) square Linear -> exactly 0 at init -> the r13 trunk's function
  bit-for-bit.
  W_ro square (768,768) is the unique signature the co-designed optim's spectral cap targets.

CO-DESIGN (the epistasis stack this round tests): the head aux
`r18_transition_forwardmodel_consistency` supervises THIS module's `transition_from_emb`
on mined mutation triples (pre-content, mutating command, future read) so the operator
becomes a good forward model; the optim `r18_spectral_capped_transition_readout` caps the
spectral norm of the (D,D) `tr_read` so the injected off-manifold correction stays norm-
calibrated (the R10/R11 off-manifold-norm finding). arch+head+optim reinforce.

r13 trunk machinery retained verbatim: verb-quotient filebind memory, path memory, view
FiLM, 3-way read mix, fuse/direct gates, causal system-identity summary + syscond FiLM,
exact chunkwise delta solver. Strictly causal; NaN-safe; identity at init.

Refs: RSSM/Dreamer latent dynamics (arXiv:1912.01603); DeltaNet chunkwise (arXiv:2406.06484);
Gated DeltaNet erase (arXiv:2412.06464); FiLM (arXiv:1709.07871); object-centric slots.
"""

import math

import torch
import torch.nn as nn

D = 768

NAME = "r18_pathstate_latent_transition_worldmodel"
DESCRIPTION = (
    "Champion r13 trunk + a per-path LATENT-TRANSITION world-model memory: the value "
    "written to a path slot is a command-conditioned affine EDIT of the slot's current "
    "retrieved content (f(s_pre,cmd), a learned RSSM-style transition), overwritten by the "
    "delta rule and read back as the current post-mutation content; injected via a zero-init "
    "(D,D) readout so it is exactly the champion at init. Distinct from additive path-delta "
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

        # -- file memory: verb-quotient addressing over the raw command token.
        self.file_read = nn.Linear(self.d, self.key_d, bias=False)
        self.file_write = nn.Linear(self.d, self.key_d, bias=False)
        self.verb_codebook = nn.Parameter(torch.randn(self.n_verb, self.key_d) * 0.2)
        self.ctx_proj = nn.Linear(self.d, self.ctx_d)

        # -- path-state memory (r13 trunk channel, unchanged).
        self.path_read = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write = nn.Linear(self.d, self.key_d, bias=False)

        self.write_gate = nn.Linear(2 * self.d, 1)

        # -- FiLM view transform (r13 trunk channel, unchanged).
        fh = max(16, int(film_hidden))
        self.film_in = nn.Linear(self.d + self.ctx_d, fh)
        self.film_out = nn.Linear(fh, 2 * D)
        nn.init.zeros_(self.film_out.weight)
        nn.init.zeros_(self.film_out.bias)

        # -- causal system-identity summary + zero-init system FiLM (r13 trunk channel).
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

        # -- NEW: per-path latent-transition world-model memory.
        # tr_path is a (key_d, d) sibling of the addressing projections (Muon-compatible).
        self.tr_path = nn.Linear(self.d, self.key_d, bias=False)
        th = max(32, int(tr_hidden))
        self.tr_in = nn.Linear(self.d, th)        # command-conditioned only (cmd_feat -> gamma,beta)
        self.tr_out = nn.Linear(th, 2 * D)
        self.tr_mut_gate = nn.Linear(self.d, 1)
        self.tr_read = nn.Linear(D, D)            # (D,D) square, ZERO-INIT (identity at init)
        nn.init.zeros_(self.tr_read.weight)
        nn.init.zeros_(self.tr_read.bias)
        self.tr_read_gate = nn.Linear(self.d, 1)
        nn.init.constant_(self.tr_mut_gate.bias, -1.0)   # start storing obs; transition fades in
        nn.init.constant_(self.tr_read_gate.bias, 0.0)

        init_decay = (0.985 - 0.90) / 0.099
        self.logit_decay = nn.Parameter(torch.tensor(math.log(init_decay / (1.0 - init_decay))))

        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.fuse_gate.bias, -1.0)
        nn.init.constant_(self.direct_gate.bias, -2.0)

    # ---- helpers -------------------------------------------------------------

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

    # ---- NEW: per-path latent-transition memory --------------------------------

    def _transition(self, s_pre, cmd_feat):
        """Learned content-transition operator f(s_pre, cmd): a command-conditioned
        affine edit of the current retrieved content. gamma,beta depend on the COMMAND
        ONLY (not s_pre), so the per-path recurrence becomes a diagonal-gated linear scan
        with the per-channel edit (gamma in R^D) as its mechanism; gamma bounded by
        tanh*g_scale keeps the recurrence marginally stable. Signature unchanged so
        transition_from_emb + the head forward-model hook still work."""
        hin = torch.nn.functional.gelu(self.tr_in(cmd_feat))
        gb = self.tr_out(hin)
        gamma = torch.tanh(gb[..., :D]) * self.tr_gscale
        beta = gb[..., D:]
        return s_pre * (1.0 + gamma) + beta

    def transition_from_emb(self, s_pre, cmd_emb):
        """Co-design entry point for the forward-model head aux: apply the SHARED transition
        operator to a current-content estimate `s_pre` [N,D] and a RAW standardized command
        embedding `cmd_emb` [N,D], reconstructing the same cmd feature the trunk uses
        (cmd_proj + cmd type_emb + in_norm; position omitted). Gradients flow into the shared
        tr_in/tr_out and the trunk input projection."""
        idx0 = torch.zeros(cmd_emb.size(0), dtype=torch.long, device=cmd_emb.device)
        cmd_feat = self.in_norm(self.cmd_proj(cmd_emb) + self.type_emb(idx0))
        return self._transition(s_pre, cmd_feat)

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        """Sequential causal per-path latent world-model scan. Returns reads [B,n_cmd,D]:
        read at command i = the path slot's current content from writes STRICTLY before i.

        The transition is now COMMAND-CONDITIONED (gamma,beta from cmd only), so this recurrence
        is exactly the diagonal-gated linear scan
            mem_i = (decay*I + a_i . p_i p_i^T) mem_{i-1} + p_i b_i^T,
            a_i = w_i*(1+gamma_i)-1 (diagonal erase gate),  b_i = (1-w_i)obs_i + w_i beta_i,
        read_i = p_i^T mem_{i-1} (strictly-earlier writes -> obs_i affects reads>i only, causal).

        NOTE (2026-07-26 speed review): a batched closed-form solve of this scan was implemented
        and verified numerically equivalent (leakage-clean, <1.1e-6 on pred/reads). It is NOT
        the version that ships: the diagonal (per-channel) gate forces a per-channel [B,D,C,C]
        solve that cannot share across the D output channels (unlike r9/r17's scalar-beta
        delta), so at the actual train config (bs64, L16, N<=8) it is ~0.6x the loop's speed
        and a wash on the full arch
        (0.386 vs 0.374 s/step full stack). The <=16-step Python loop is cheaper here and was
        never the arch bottleneck; the head-aux vectorization (r18 head) is what removed the
        timeout. Kept sequential for simplicity/speed."""
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p = self._unit(self.tr_path(x_cmd))                          # [B,n_cmd,key_d]
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)       # [B,n_cmd]
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            pi = p[:, i, :]                                          # [B,key_d]
            s_pre = torch.bmm(pi.unsqueeze(1), mem).squeeze(1)       # [B,D] writes < i (causal)
            reads.append(s_pre)
            delta = self._transition(s_pre, x_cmd[:, i, :])          # [B,D] (gamma,beta cmd-only)
            if i < n_pair:
                obs_i = obs_tok[:, i, :].to(dtype)
                wi = w[:, i].unsqueeze(-1)
                active = (valid_obs[:, i] & valid_cmd[:, i]).to(dtype).unsqueeze(-1)
            else:
                obs_i = s_pre.new_zeros(B, D)
                wi = w[:, i].unsqueeze(-1) * 0.0
                active = x_cmd.new_zeros(B, 1)
            v_i = (1.0 - wi) * obs_i + wi * delta                    # write value
            corr = (v_i - s_pre) * active                           # gate padded / dead pairs
            mem = decay * mem + torch.bmm(pi.unsqueeze(2), corr.unsqueeze(1))
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)

    # ---- forward -------------------------------------------------------------

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

        # -- causal system-identity summary (gated running mean over past obs).
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

        # -- NEW: per-path latent-transition read, injected via a zero-init (D,D) readout.
        tr_reads = self._transition_reads(x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair)
        g_tr = torch.sigmoid(self.tr_read_gate(h_cmd0))              # [B,n_cmd,1]
        tr_contrib = g_tr * self.tr_read(tr_reads.to(x.dtype))       # 0 at init (tr_read zero)

        sf_h = torch.tanh(self.sysfilm_in(torch.cat([h_cmd0, s_cmd.to(x.dtype)], dim=-1)))
        sf = self.sysfilm_out(sf_h).to(dtype)
        g_sys = sf[..., :D]
        b_sys = sf[..., D:]

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred_cmd = pred[:, 0::2, :] + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read
        pred_cmd = pred_cmd + tr_contrib.to(dtype)                   # NEW transition correction
        pred_cmd = pred_cmd * (1.0 + g_sys) + b_sys
        pred[:, 0::2, :] = torch.nan_to_num(pred_cmd, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = torch.nan_to_num(pred * valid.unsqueeze(-1).to(pred.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        h_out = torch.nan_to_num(h_out * valid.unsqueeze(-1).to(h_out.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return R18PathStateLatentTransition(**params)
