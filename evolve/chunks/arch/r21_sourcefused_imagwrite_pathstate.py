import torch

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    D,
    R18PathStateLatentTransition,
)

NAME = "r21_sourcefused_imagwrite_pathstate"
DESCRIPTION = (
    "Candidate 8's evidence-gated imagination write with the imagined value's pre-content "
    "upgraded on source-present windows: fuse the tr-memory blend with the latest observed "
    "same-path pre-observation (the head aux's own mining rule, cmd-cos>0.60) 50/50 and "
    "execute the trained operator on the fused estimate via transition_from_emb — the exact "
    "aux-training operating point. Unmatched rows and everything else are C8 verbatim; zero "
    "new parameters; even streams bit-identical; empty history algebraically zero. Sibling "
    "completions (file/path render writes, erase, z_prev fusion) measured negative and "
    "excluded by design."
)


class R21SourceFusedImagWritePathState(R18PathStateLatentTransition):
    def __init__(self, imag_gate_c=0.25, src_thresh=0.60, src_blend=0.5, **params):
        super().__init__(**params)
        # c must be strictly > 0: the gate exists for the exact-zero algebra 0/(0+c) == 0.0, so
        # with no earlier observed write the imagined write is algebraically zero, not small.
        self.imag_gate_c = max(1e-6, float(imag_gate_c))
        self.src_thresh = float(src_thresh)
        self.src_blend = min(1.0, max(0.0, float(src_blend)))
        self._raw_cmd = None

    def forward(self, tok_emb, types, key_pad):
        self._raw_cmd = tok_emb[:, 0::2, :] if tok_emb.dim() == 3 else None
        try:
            return super().forward(tok_emb, types, key_pad)
        finally:
            self._raw_cmd = None

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p = self._unit(self.tr_path(x_cmd))
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        evid = x_cmd.new_zeros(B, 1)
        c = float(self.imag_gate_c)

        if n_pair:
            imag_pair = valid_cmd[:, :n_pair] & ~valid_obs
            has_imag = bool(imag_pair.any().item())
        else:
            has_imag = False

        use_src = (
            has_imag
            and self.src_blend > 0.0
            and self._raw_cmd is not None
            and self._raw_cmd.shape[0] == B
            and self._raw_cmd.shape[1] == n_cmd
        )
        if use_src:
            raw = torch.nan_to_num(self._raw_cmd, nan=0.0, posinf=1e4, neginf=-1e4)
            cu = self._unit(raw)
            sim = torch.bmm(cu, cu.transpose(1, 2))
            pos = torch.arange(n_cmd, device=x_cmd.device)
            observed = torch.zeros(B, n_cmd, dtype=torch.bool, device=x_cmd.device)
            observed[:, :n_pair] = valid_cmd[:, :n_pair] & valid_obs
            causal_pp = pos.view(1, -1, 1) > pos.view(1, 1, -1)
            cand = observed.unsqueeze(1) & causal_pp & (sim > float(self.src_thresh))
            posf = pos.view(1, 1, -1).expand(B, n_cmd, n_cmd)
            pstar = torch.where(cand, posf, torch.full_like(posf, -1)).amax(dim=2)
            matched_all = pstar >= 0
            pc = pstar.clamp(0, max(0, n_pair - 1))
            obs_padded = self._pad_steps(obs_tok, n_cmd)
            src_all = torch.gather(obs_padded, 1, pc.unsqueeze(-1).expand(B, n_cmd, D))

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
                imag = (valid_cmd[:, i] & ~valid_obs[:, i]).to(dtype).unsqueeze(-1)
            else:
                obs_i = s_pre.new_zeros(B, D)
                wi = w[:, i].unsqueeze(-1) * 0.0
                active = x_cmd.new_zeros(B, 1)
                imag = x_cmd.new_zeros(B, 1)
            v_i = (1.0 - wi) * obs_i + wi * delta
            observed_corr = (v_i - s_pre) * active
            g = evid / (evid + c)
            delta_imag = delta
            if use_src and i < n_pair and bool((imag.squeeze(-1) > 0).any().item()):
                mrow = matched_all[:, i] & (imag.squeeze(-1) > 0)
                if bool(mrow.any().item()):
                    f_src = self.transition_from_emb(
                        src_all[mrow, i].to(dtype), self._raw_cmd[mrow, i].to(dtype)
                    )
                    f_src = torch.nan_to_num(f_src, nan=0.0, posinf=1e4, neginf=-1e4)
                    delta_imag = delta.clone()
                    b = float(self.src_blend)
                    delta_imag[mrow] = b * f_src + (1.0 - b) * delta[mrow]
            imagined_corr = g * (w[:, i].unsqueeze(-1) * (delta_imag - s_pre)) * imag
            mem = decay * mem + torch.bmm(
                pi.unsqueeze(2), (observed_corr + imagined_corr).unsqueeze(1)
            )
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            evid = evid + observed_corr.pow(2).sum(dim=-1, keepdim=True)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)


def build(**params):
    return R21SourceFusedImagWritePathState(**params)
