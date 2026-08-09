'''R22 optimizer: CROSS-BATCH GRADIENT CONSENSUS.

The reference stack already has strong content-attributable native imagination (CA 0.5471),
but the R20/R21 candidates measured only about +0.005 cumulative CA and the strongest
write-family HA result was mostly command decoding. The untouched optimization path still
turns each pooled batch into one update, even though TRAIN-only gradient analysis at two
archived 4000-step reference-stack checkpoints found a useful temporal separation: genuine
mutation-to-read subset gradients align much more strongly than intervention-output subset
gradients with a slow EMA of the ordinary full-batch gradient on input, trunk, renderer,
output, and transition matrices.

For every large non-addressing matrix this optimizer maintains a slow bias-corrected gradient
EMA c_t, RMS-matches it to the current gradient g_t, and applies the bounded convex filter

    g_t <- (1 - alpha_t) * g_t + alpha_t * c_t,  0 <= alpha_t <= 0.35.

Cross-batch-consistent directions therefore receive more angular weight while batch-specific
directions are attenuated. Since c_t is RMS-matched and the mixture is convex, each filtered
tensor's gradient norm cannot exceed its pre-filter norm. A delayed smooth ramp leaves early
representation formation unchanged.

The carried optimizer stack is otherwise retained: Muon still owns the six repeated 64xd addressing
matrices, AdamW keeps the same warmup/hold/cosine-floor schedule, and the 768x768 transition
readout keeps the same spectral cap. The optimizer sees no data, masks, targets, metadata,
images, or eval artifacts; it adds no model parameter or forward branch and cannot detect the
wrong-history arm. Causality, PAD invariance, anti-collapse behavior, identity target, and the
native measurement layout are inherited unchanged.
'''

import math
import torch
NAME="r22_crossbatch_gradient_consensus"
DESCRIPTION=(
    'Muon-on-addressing plus AdamW/warmup-hold-cosine-floor and transition '
    'spectral cap, with a bounded slow-gradient consensus filter on large non-addressing '
    'matrices: a bias-corrected cross-batch gradient EMA is RMS-matched and convex-mixed '
    'into each current gradient after a delayed ramp.'
)
D=768

def _ns_orth(g,steps=5,eps=1e-7):
    a,b,c=3.4445,-4.7750,2.0315
    x=g.float(); transposed=x.shape[0]>x.shape[1]
    if transposed: x=x.mT
    x=x/x.norm().clamp_min(eps)
    for _ in range(int(steps)):
        s=x@x.mT; x=a*x+(b*s+c*(s@s))@x
    if transposed: x=x.mT
    return x.to(g.dtype)

class _MuonKeys(torch.optim.Optimizer):
    def __init__(self,params,lr,momentum=.95,ns_steps=5,rms_match=.2):
        super().__init__(params,dict(lr=float(lr),momentum=float(momentum),ns_steps=int(ns_steps),rms_match=float(rms_match)))
    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        for group in self.param_groups:
            mu=group["momentum"]; lr=group["lr"]
            for p in group["params"]:
                if p.grad is None: continue
                g=torch.nan_to_num(p.grad,nan=0.,posinf=0.,neginf=0.); st=self.state[p]
                if "buf" not in st: st["buf"]=torch.zeros_like(g)
                st["buf"].mul_(mu).add_(g); u=g.add(st["buf"],alpha=mu); o=_ns_orth(u,steps=group["ns_steps"])
                p.add_(o,alpha=-lr*group["rms_match"]*math.sqrt(max(p.shape[0],p.shape[1])))
        return loss

class _SlowConsensus:
    def __init__(self,params,beta=.98,mix=.35,start_step=200,ramp_steps=800,eps=1e-12):
        self.params=list(params); self.beta=float(beta); self.mix=float(mix); self.start_step=int(start_step); self.ramp_steps=max(1,int(ramp_steps)); self.eps=float(eps); self.step_no=0; self._state={}
    def _alpha(self):
        if self.step_no<=self.start_step or self.mix<=0.: return 0.
        x=min(1.,(self.step_no-self.start_step)/float(self.ramp_steps)); return self.mix*x*x*(3.-2.*x)
    @torch.no_grad()
    def apply(self):
        self.step_no+=1; alpha=self._alpha()
        for p in self.params:
            if p.grad is None: continue
            g=torch.nan_to_num(p.grad.detach().float(),nan=0.,posinf=0.,neginf=0.); key=id(p); state=self._state.get(key)
            if state is None: state=[torch.zeros_like(g),0]; self._state[key]=state
            slow,count=state; slow.mul_(self.beta).add_(g,alpha=1.-self.beta); count+=1; state[1]=count
            if alpha>0.:
                consensus=slow/max(self.eps,1.-self.beta**count); grms=g.pow(2).mean().sqrt(); crms=consensus.pow(2).mean().sqrt().clamp_min(self.eps); consensus=consensus*(grms/crms); g=(1.-alpha)*g+alpha*consensus
            p.grad.copy_(g.to(dtype=p.grad.dtype))

class _SpectralCap:
    def __init__(self,params,cap=4.,iters=2):
        self.params=list(params); self.cap=float(cap); self.iters=max(1,int(iters)); self._u={}
    @torch.no_grad()
    def project(self):
        for p in self.params:
            if p.ndim!=2: continue
            w=torch.nan_to_num(p.data,nan=0.,posinf=1e4,neginf=-1e4); p.data.copy_(w); u=self._u.get(id(p))
            if u is None or u.shape[0]!=w.shape[0]: u=torch.nn.functional.normalize(w.new_ones(w.shape[0]),dim=0)
            for _ in range(self.iters):
                v=torch.nn.functional.normalize(w.t().mv(u),dim=0,eps=1e-8); u=torch.nn.functional.normalize(w.mv(v),dim=0,eps=1e-8)
            self._u[id(p)]=u; sigma=float(torch.dot(u,w.mv(v)))
            if math.isfinite(sigma) and sigma>self.cap: p.mul_(self.cap/max(sigma,1e-8))

class _CompositeOpt:
    def __init__(self,adamw,muon,consensus,cap): self.adamw=adamw; self.muon=muon; self.consensus=consensus; self.cap=cap
    @property
    def param_groups(self):
        groups=list(self.adamw.param_groups)
        if self.muon is not None: groups.extend(self.muon.param_groups)
        return groups
    def zero_grad(self,set_to_none=True):
        self.adamw.zero_grad(set_to_none=set_to_none)
        if self.muon is not None: self.muon.zero_grad(set_to_none=set_to_none)
    def step(self,closure=None):
        self.consensus.apply(); self.adamw.step()
        if self.muon is not None: self.muon.step()
        self.cap.project()
    def state_dict(self):
        out={"adamw":self.adamw.state_dict(),"consensus_step":self.consensus.step_no}
        if self.muon is not None: out["muon"]=self.muon.state_dict()
        return out

class _MultiSched:
    def __init__(self,*scheds): self.scheds=tuple(scheds)
    def step(self):
        for s in self.scheds: s.step()
    def get_last_lr(self): return [lr for s in self.scheds for lr in s.get_last_lr()]

def _lr_lambda(steps,warmup_frac,hold_frac,floor_ratio):
    warm=max(20,int(float(warmup_frac)*int(steps))); hold=int(float(hold_frac)*int(steps)); decay_start=warm+hold; decay_len=max(1,int(steps)-decay_start)
    def fn(step):
        if step<warm: return (step+1)/float(warm)
        if step<decay_start: return 1.
        p=min(1.,(step-decay_start)/float(decay_len)); cosine=.5*(1.+math.cos(math.pi*p)); return float(floor_ratio)+(1.-float(floor_ratio))*cosine
    return fn

def make(params,steps,lr=5e-4,wd=5e-4,warmup_frac=.04,hold_frac=.30,floor_ratio=.05,beta2=.95,key_d=64,momentum=.95,ns_steps=5,rms_match=.2,spectral_cap=4.,spectral_iters=2,consensus_beta=.98,consensus_mix=.35,consensus_start_frac=.05,consensus_ramp_frac=.20,min_matrix_numel=8192):
    plist=[p for p in params if p.requires_grad]
    if not plist: raise ValueError("optimizer received no trainable parameters")
    if not 0.<=float(consensus_mix)<1.: raise ValueError("consensus_mix must be in [0,1)")
    if not 0.<float(consensus_beta)<1.: raise ValueError("consensus_beta must be in (0,1)")
    candidates=[p for p in plist if p.ndim==2 and p.shape[0]==int(key_d) and p.shape[1]!=int(key_d) and p.shape[1]!=D]; counts={}
    for p in candidates: counts[tuple(p.shape)]=counts.get(tuple(p.shape),0)+1
    keys=[p for p in candidates if counts[tuple(p.shape)]>=2]; key_ids={id(p) for p in keys}; rest=[p for p in plist if id(p) not in key_ids]; dd=[p for p in rest if p.ndim==2 and tuple(p.shape)==(D,D)]; large=[p for p in rest if p.ndim==2 and p.numel()>=int(min_matrix_numel)]
    schedule=_lr_lambda(steps,warmup_frac,hold_frac,floor_ratio); adamw=torch.optim.AdamW(rest,lr=float(lr),weight_decay=float(wd),betas=(.9,float(beta2))); scheds=[torch.optim.lr_scheduler.LambdaLR(adamw,schedule)]; muon=None
    if keys:
        muon=_MuonKeys(keys,lr=lr,momentum=momentum,ns_steps=ns_steps,rms_match=rms_match); scheds.append(torch.optim.lr_scheduler.LambdaLR(muon,schedule))
    consensus=_SlowConsensus(large,beta=consensus_beta,mix=consensus_mix,start_step=max(1,int(float(consensus_start_frac)*int(steps))),ramp_steps=max(1,int(float(consensus_ramp_frac)*int(steps)))); cap=_SpectralCap(dd,cap=spectral_cap,iters=spectral_iters)
    return _CompositeOpt(adamw,muon,consensus,cap),_MultiSched(*scheds)
