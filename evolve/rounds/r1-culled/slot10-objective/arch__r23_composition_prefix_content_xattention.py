from evolve.chunks.arch.r22_prefix_content_xattention import R22PrefixContentXAttention
from evolve.chunks.arch.r22_retrieval_composition_renderer import R22RetrievalCompositionRenderer

D = 768

NAME = "r23_composition_prefix_content_xattention"
DESCRIPTION = (
    "Crossover arch: the retrieval-composition renderer's nonlinear vector-wise interaction over "
    "the delta-rule retrieved-content channels (injected into target_read) AND the prefix-content "
    "cross-attention that copies raw earlier observation embeddings into the command-position "
    "prediction. Cooperative MRO composition: the xattention forward runs on top of the renderer "
    "forward, so the model has both a soft content-transport channel over the literal prefix and a "
    "nonlinear composer over retrieved content. Both readouts are zero-init, so the composed net is "
    "the r18 function at init."
)


class R23CompositionPrefixContentXAttention(
    R22PrefixContentXAttention, R22RetrievalCompositionRenderer
):
    pass


def build(**params):
    return R23CompositionPrefixContentXAttention(**params)
