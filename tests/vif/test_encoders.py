"""Tests for shared sentence-transformer encoder behavior."""

import numpy as np
import pytest
import sentence_transformers

from src.vif.encoders import SBERTEncoder


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[str]:
        return text.split()


class _FakeSentenceTransformer:
    instances: list["_FakeSentenceTransformer"] = []

    def __init__(self, model_name: str, trust_remote_code: bool = False):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.max_seq_length = 128
        self.prompts = {"query": "Q: ", "document": ""}
        self.tokenizer = _FakeTokenizer()
        self.encode_calls: list[dict] = []
        _FakeSentenceTransformer.instances.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, sentences, **kwargs) -> np.ndarray:
        self.encode_calls.append(
            {
                "sentences": list(sentences),
                **kwargs,
            }
        )
        embeddings = np.array(
            [[1.0, 2.0, 3.0, 4.0] for _ in sentences],
            dtype=np.float32,
        )
        truncate_dim = kwargs.get("truncate_dim")
        if truncate_dim is not None:
            embeddings = embeddings[:, :truncate_dim]
        return embeddings


@pytest.fixture
def fake_sentence_transformer(monkeypatch):
    _FakeSentenceTransformer.instances.clear()
    monkeypatch.setattr(
        sentence_transformers,
        "SentenceTransformer",
        _FakeSentenceTransformer,
    )
    return _FakeSentenceTransformer.instances


def test_nomic_text_prefix_path_stays_legacy(fake_sentence_transformer):
    encoder = SBERTEncoder(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        truncate_dim=2,
        text_prefix="classification: ",
    )

    np.testing.assert_equal(
        encoder.render_inputs(["hello"]),
        ["classification: hello"],
    )

    embeddings = encoder.encode(["hello"])
    call = fake_sentence_transformer[-1].encode_calls[-1]

    assert call["sentences"] == ["classification: hello"]
    assert "prompt" not in call
    assert "prompt_name" not in call
    assert "truncate_dim" not in call

    expected = np.array([[-0.94868326, -0.31622776]], dtype=np.float32)
    np.testing.assert_allclose(embeddings, expected, atol=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "expected_label"),
    [
        ({"text_prefix": "classification: ", "prompt_name": "query"}, "text_prefix, prompt_name"),
        ({"text_prefix": "classification: ", "prompt": "Q: "}, "text_prefix, prompt"),
        ({"prompt_name": "query", "prompt": "Q: "}, "prompt_name, prompt"),
    ],
)
def test_encoder_rejects_multiple_input_modes(fake_sentence_transformer, kwargs, expected_label):
    with pytest.raises(ValueError, match=expected_label):
        SBERTEncoder("Qwen/Qwen3-Embedding-0.6B", **kwargs)


def test_qwen_native_prompt_and_truncation_are_passed_to_encode(fake_sentence_transformer):
    prompt = (
        "Instruct: Represent the journal entry for classification of alignment "
        "across the 10 Schwartz value dimensions.\nQuery:"
    )
    encoder = SBERTEncoder(
        "Qwen/Qwen3-Embedding-0.6B",
        trust_remote_code=True,
        truncate_dim=2,
        prompt=prompt,
    )

    embeddings = encoder.encode(["hello"])
    call = fake_sentence_transformer[-1].encode_calls[-1]

    assert call["sentences"] == ["hello"]
    assert call["prompt"] == prompt
    assert call["truncate_dim"] == 2
    np.testing.assert_array_equal(
        embeddings,
        np.array([[1.0, 2.0]], dtype=np.float32),
    )


def test_prompt_name_rendering_matches_token_count_path(fake_sentence_transformer):
    encoder = SBERTEncoder(
        "Qwen/Qwen3-Embedding-0.6B",
        trust_remote_code=True,
        prompt_name="query",
    )

    assert encoder.render_inputs(["alpha beta"]) == ["Q: alpha beta"]
    assert encoder.count_tokens(["alpha beta"]) == [3]

    encoder.encode(["alpha beta"])
    call = fake_sentence_transformer[-1].encode_calls[-1]
    assert call["sentences"] == ["alpha beta"]
    assert call["prompt_name"] == "query"
