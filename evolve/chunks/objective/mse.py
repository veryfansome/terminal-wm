"""objective chunk: plain MSE — the R4 baseline loss.

Contract for any objective impl: expose `loss(pred, tgt) -> scalar tensor`, where
  pred: [n, D] the world model's predicted next-observation embeddings at command positions
  tgt : [n, D] the true (standardized) next-observation embeddings
Both are already the flattened cmd-position tensors for the training batch, so batch-level
objectives (contrastive/InfoNCE, variance regularizers) can be formed from them directly.
The loss must be a scalar torch tensor with grad. Keep it anti-collapse-safe: a constant
prediction should NOT minimize it (plain MSE is fine because tgt varies).

CONTRACT EXTENSION (opt-in, 2026-08-02) — causal side-info for content-routing objectives:
an objective module may set a module-level `WANTS_CTX = True`, and its signature becomes
`loss(pred, tgt, ctx) -> scalar`. `ctx` is a dict aligned row-for-row with pred/tgt:
  ctx["cmd"]:  [n, D] the COMMAND embedding that produced each prediction (a model INPUT — the
               same [:, 0::2] cmd-position slice the stream uses for predictions)
  ctx["prev"]: [n, D] the strict-causal previous-observation embedding per cmd-row
Both are CAUSAL (an input command + a past observation — never a future obs), so the harness
leakage guard still holds; the extension exists so an objective can, e.g., residualize /
decorrelate the prediction against a command-decodable component to route gradient onto the
history-content the ordinary next-obs loss leaves unbanked. Objectives WITHOUT the flag are
called `loss(pred, tgt)` exactly as before — every existing impl is bit-identical."""

NAME = "mse"
DESCRIPTION = "R4 baseline: mean squared error to the standardized target embedding."


def loss(pred, tgt):
    return ((pred - tgt) ** 2).mean()
