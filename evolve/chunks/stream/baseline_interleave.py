"""Contract for any stream impl:
  collate(batch, device) -> dict with tok [B,L,D], types [B,L] in {0,1}, key_pad [B,L] bool,
      tgt [B,maxn,D] (single-vector standardized next-obs target per STEP — the target/eval space
      is FIXED across streams), cmd_mask [B,maxn] bool
  extract_cmd_pred(pred_full [B,L,D], batch) -> [B,maxn,D], the prediction at each step's cmd token
  flatten_predictions(net, seqs, device) -> dict with at least pred/prev/true/cmds/verbs, step order
  leakage_ok(net, device) -> bool, a stream-aware causality probe (corrupt obs_t, cmd_<=t frozen);
      asserted before scoring, and a stream that cannot demonstrate causality is rejected
  extract_cmd_input(batch) -> [B,maxn,D] (optional), what lets an objective opt into WANTS_CTX
"""

import torch

from realenv import seq_worldmodel as M

NAME = "baseline_interleave"
DESCRIPTION = "Single-vector cmd/obs interleave; bit-identical to the pre-axis harness plumbing."


# This impl must stay bit-identical to the pre-axis harness: collate/flatten delegate to the
# seq_worldmodel functions the harness always called, and leakage_ok uses the same seed, toy
# sequence and perturbed index, so archived fitnesses replay exactly.
def collate(batch, device):
    return M.collate(batch, device)


def extract_cmd_pred(pred_full, batch):
    return pred_full[:, 0::2]


def extract_cmd_input(batch):
    return batch["tok"][:, 0::2]


def flatten_predictions(net, seqs, device):
    return M.flatten_predictions(net, seqs, device)


@torch.no_grad()
def leakage_ok(net, device):
    net.eval()
    torch.manual_seed(0)
    seq = [{"z_obs": torch.randn(6, M.D), "z_cmd": torch.randn(6, M.D),
            "cmds": ["ls /a"] * 6, "image": "x"}]
    b0 = M.collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = M.collate(seq, device)
    b1["tok"][0, 7] = torch.randn(M.D, device=device) * 100.0  # corrupt obs_3 (odd index 2*3+1)
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())

# The cups scoring instrument pins this layout; a stream declaring a different one is
# refused before any GPU time is spent (see eval/adapter.py).
CUPS_LAYOUT = "interleave2"
