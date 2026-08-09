"""arch chunk baseline: the R4 causal transformer (SeqWorldModel).

Contract for any arch impl: expose `build(**params) -> nn.Module` whose
  forward(tok_emb [B,L,768], types [B,L] in {0,1}, key_pad [B,L] bool) -> (pred [B,L,768], h [B,L,dh])
predicting at EVERY position (the harness reads command positions as pred[:, 0::2]). It MUST be
CAUSAL: the per-genome no-leakage guard rejects any arch where a command-position prediction can
depend on its own or a future observation token (score -inf). Input tokens are frozen 768-d
embeddings; the head must map back to 768-d target space. Params come from the genome's
chunks.arch.params."""

from realenv import seq_worldmodel as M

NAME = "baseline_transformer"
DESCRIPTION = "R4 causal transformer over interleaved cmd/obs frozen embeddings (SeqWorldModel)."


def build(d=192, layers=4, heads=4, dropout=0.1):
    return M.SeqWorldModel("jepa", d=d, layers=layers, heads=heads, dropout=dropout)
