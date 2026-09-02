"""BGE sentence-embedding encoder -- pure, stateless, no FeatureRow knowledge.

Callers construct a model via load_bge_model() and pass it explicitly into
embed_texts() -- this module holds no state and memoizes nothing, per
docs/design-patterns-guide.md's Hard Warning #1 ("No Singleton for fitted
transformers or embedding models"): a shared module-level embedder would
couple test outcomes across pytest cases and risk silent train/serve skew
if ever reloaded differently between training and serving.
"""

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingTypeStubs]

BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_REVISION_SHA = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_MAX_SEQ_LEN = 512
BGE_POOLING = "cls"
BGE_NORMALIZE = True
BGE_EMBEDDING_DIM = 384


def load_bge_model(revision_sha: str = BGE_REVISION_SHA) -> SentenceTransformer:
    """Load the pinned BGE model. Callers own the returned instance."""
    return SentenceTransformer(BGE_MODEL_NAME, revision=revision_sha)


def embed_texts(texts: list[str], model: SentenceTransformer) -> NDArray[np.float64]:
    """Encode texts into unit-norm embeddings using a caller-supplied model."""
    if not texts:
        return np.empty((0, BGE_EMBEDDING_DIM), dtype=np.float64)
    embeddings = model.encode(texts, normalize_embeddings=BGE_NORMALIZE)  # pyright: ignore[reportUnknownMemberType]
    return np.asarray(embeddings, dtype=np.float64)  # pyright: ignore[reportUnknownArgumentType]
