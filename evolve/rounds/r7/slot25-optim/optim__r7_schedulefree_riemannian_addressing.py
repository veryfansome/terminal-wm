import math

import torch

NAME = "r7_schedulefree_riemannian_addressing"
DESCRIPTION = (
    "Schedule-free AdamW (Defazio et al., 2024) with a Riemannian sphere group for the "
    "scale-invariant addressing projections. No decay schedule at all: gradients are taken at "
    "y = (1-b1) z + b1 x, the fast iterate z takes the full Adam-preconditioned step, and x is "
    "the lr^2-weighted running mean of the whole trajectory; the live parameters carry y during "
    "training and are converted to x on the final optimizer step, so the net that is measured is "
    "the averaged iterate. Orthogonality of a weight to its own gradient is measured over the "
    "first detect_steps updates and, where the mean |cos(g,W)| sits an order of magnitude below "
    "chance (delta/sqrt(numel)), the weight is certified exactly scale-invariant: its update is "
    "projected onto the tangent space of the sphere, its weight decay is dropped, and y and z are "
    "rescaled together by one factor so ||y|| returns to its certified reference norm. That holds "
    "the angular learning rate of the delta-rule read/write keys constant for the whole budget "
    "instead of letting it decay as the key matrices grow. A soft one-sided spectral cap on square "
    "(readout_dim, readout_dim) matrices is evaluated on the extrapolated point x and applied as "
    "the same multiplier to y and z, so the cap acts on the iterate that is finally measured."
)


def _warmup_lambda(steps, warmup_frac):
    warm = max(20, int(float(warmup_frac) * max(1, int(steps))))

    def lr_lambda(step):
        return min(1.0, float(step + 1) / float(warm))

    return lr_lambda


class ScheduleFreeRiemannian(torch.optim.Optimizer):

    def __init__(self, params, lr=5e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=5e-4,
                 total_steps=1000, sphere_delta=0.1, detect_steps=40, readout_dim=768,
                 spectral_cap=4.0, spectral_iters=2):
        super().__init__(params, dict(lr=float(lr), betas=(float(betas[0]), float(betas[1])),
                                      eps=float(eps), weight_decay=float(weight_decay)))
        self.total_steps = max(1, int(total_steps))
        self.sphere_delta = float(sphere_delta)
        self.detect_steps = max(1, min(int(detect_steps), self.total_steps))
        self.readout_dim = int(readout_dim)
        self.spectral_cap = float(spectral_cap)
        self.spectral_iters = max(1, int(spectral_iters))
        self.averaged = False
        for group in self.param_groups:
            group["k"] = 0
            group["lr_max"] = 0.0
            group["weight_sum"] = 0.0

    @torch.no_grad()
    def _certify(self, p, g, st, k):
        if p.ndim < 2 or p.numel() < 64:
            st["sphere"] = False
            return
        acc = st.get("cos_acc")
        if acc is None:
            acc = p.new_zeros(())
        gn = g.norm()
        pn = p.norm()
        c = torch.dot(g.reshape(-1), p.reshape(-1)).abs() / (gn * pn).clamp_min(1e-12)
        c = torch.nan_to_num(c, nan=1.0, posinf=1.0, neginf=1.0)
        measurable = (pn > 1e-6) & (gn > 1e-12)
        c = torch.where(measurable, c, torch.ones_like(c))
        st["cos_acc"] = acc + c
        if k >= self.detect_steps:
            mean_cos = float(st["cos_acc"]) / float(self.detect_steps)
            flag = mean_cos < self.sphere_delta / math.sqrt(float(p.numel()))
            st["sphere"] = bool(flag)
            if flag:
                st["ref"] = p.norm().detach().clone().clamp_min(1e-8)

    @torch.no_grad()
    def _spectral_pull(self, p, z, st, extrap):
        w = torch.lerp(p.detach(), z, extrap)
        n = w.shape[0]
        sv = st.get("sv")
        if sv is None or sv.shape[0] != n or sv.device != w.device or sv.dtype != w.dtype:
            sv = torch.nn.functional.normalize(w.new_ones(n), dim=0)
        rv = None
        for _ in range(self.spectral_iters):
            rv = torch.nn.functional.normalize(w.t().mv(sv), dim=0, eps=1e-8)
            sv = torch.nn.functional.normalize(w.mv(rv), dim=0, eps=1e-8)
        st["sv"] = sv
        sigma = torch.dot(sv, w.mv(rv)).abs().clamp_min(1e-8)
        shrink = (self.spectral_cap / sigma).clamp(max=1.0)
        shrink = torch.nan_to_num(shrink, nan=1.0, posinf=1.0, neginf=1.0)
        p.mul_(shrink)
        z.mul_(shrink)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        finalize_now = False
        for group in self.param_groups:
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = float(group["weight_decay"])
            k = int(group["k"]) + 1
            group["k"] = k

            bias2 = 1.0 - b2 ** k
            lr = float(group["lr"]) * math.sqrt(max(bias2, 1e-12))
            lr_max = max(float(group["lr_max"]), lr)
            group["lr_max"] = lr_max
            weight = lr_max * lr_max
            group["weight_sum"] = float(group["weight_sum"]) + weight
            ckp1 = weight / group["weight_sum"] if group["weight_sum"] > 0.0 else 0.0
            extrap = 1.0 - 1.0 / b1

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                st = self.state[p]
                if "z" not in st:
                    st["z"] = p.detach().clone()
                    st["v"] = torch.zeros_like(p)
                    st["sphere"] = False
                z = st["z"]
                v = st["v"]

                if k <= self.detect_steps:
                    self._certify(p, g, st, k)

                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                u = g.div(v.sqrt().add_(eps))
                u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

                on_sphere = bool(st.get("sphere", False)) and st.get("ref") is not None
                if on_sphere:
                    phat = p.div(p.norm().clamp_min(1e-8))
                    u = u - phat * torch.dot(u.reshape(-1), phat.reshape(-1))
                elif wd != 0.0:
                    u = u.add(p, alpha=wd)

                p.lerp_(z, ckp1)
                p.add_(u, alpha=lr * (b1 * (1.0 - ckp1) - 1.0))
                z.sub_(u, alpha=lr)

                if on_sphere:
                    rescale = st["ref"] / p.norm().clamp_min(1e-8)
                    rescale = torch.nan_to_num(rescale, nan=1.0, posinf=1.0, neginf=1.0)
                    p.mul_(rescale)
                    z.mul_(rescale)

                if (self.spectral_cap > 0.0 and p.ndim == 2
                        and p.shape[0] == self.readout_dim and p.shape[1] == self.readout_dim):
                    self._spectral_pull(p, z, st, extrap)

            if not self.averaged and k >= self.total_steps:
                finalize_now = True
                for p in group["params"]:
                    st = self.state.get(p)
                    if not st or "z" not in st:
                        continue
                    x = torch.lerp(p.detach(), st["z"], extrap)
                    if bool(torch.isfinite(x).all()):
                        p.copy_(x)

        if finalize_now:
            self.averaged = True
        return loss


def make(params, steps, lr=5e-4, wd=5e-4, beta1=0.9, beta2=0.95, eps=1e-8,
         warmup_frac=0.05, sphere_delta=0.1, detect_steps=40, readout_dim=768,
         spectral_cap=4.0, spectral_iters=2):
    params = [p for p in params]
    steps = max(1, int(steps))
    b1 = min(max(float(beta1), 0.5), 0.99)
    opt = ScheduleFreeRiemannian(
        params, lr=lr, betas=(b1, float(beta2)), eps=eps, weight_decay=wd,
        total_steps=steps, sphere_delta=sphere_delta, detect_steps=detect_steps,
        readout_dim=readout_dim, spectral_cap=spectral_cap, spectral_iters=spectral_iters,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _warmup_lambda(steps, warmup_frac))
    return opt, sched
