"""Tests for signalscore.features.embeddings -- the BGE sentence-embedding encoder."""

import os

# The pinned revision is already cached locally on this machine; forcing
# offline mode skips the (slow, retry-heavy) HTTP HEAD checks Hugging Face
# Hub otherwise makes on every load, without changing model behavior.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from signalscore.features.embeddings import (
    BGE_EMBEDDING_DIM,
    BGE_REVISION_SHA,
    embed_texts,
    load_bge_model,
)


@pytest.fixture(scope="module")
def bge_model() -> SentenceTransformer:
    return load_bge_model()


def test_embed_texts_returns_correct_shape(bge_model: SentenceTransformer) -> None:
    out = embed_texts(["hello world", "another issue", "stack trace here"], bge_model)
    assert out.shape == (3, BGE_EMBEDDING_DIM)


def test_embed_texts_is_deterministic(bge_model: SentenceTransformer) -> None:
    texts = ["crash on startup", "feature request: dark mode"]
    first = embed_texts(texts, bge_model)
    second = embed_texts(texts, bge_model)
    assert np.array_equal(first, second)


def test_embed_texts_normalizes(bge_model: SentenceTransformer) -> None:
    out = embed_texts(["some text", "more text"], bge_model)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_embed_texts_empty_list_returns_empty_array(bge_model: SentenceTransformer) -> None:
    out = embed_texts([], bge_model)
    assert out.shape == (0, BGE_EMBEDDING_DIM)


@pytest.mark.slow
def test_load_bge_model_and_embed_real_model() -> None:
    model = load_bge_model()
    assert model.get_embedding_dimension() == BGE_EMBEDDING_DIM

    out = embed_texts(["a real issue title", "another real body"], model)

    assert out.shape == (2, BGE_EMBEDDING_DIM)
    assert BGE_REVISION_SHA  # sanity: the pinned revision constant is non-empty
