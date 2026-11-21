"""
Tests for embedding initialization and caching behavior.
"""

from unittest.mock import patch, MagicMock
from app.embeddings import get_embedding_model, _CACHED_EMBEDDINGS


def test_embedding_model_initialization():
    with patch("app.embeddings.HuggingFaceEmbeddings") as mock_hf:
        mock_instance = MagicMock()
        mock_hf.return_value = mock_instance

        # Clear cache for isolated testing
        _CACHED_EMBEDDINGS.clear()

        emb = get_embedding_model(model_name="test-model", device="cpu")
        assert emb == mock_instance
        mock_hf.assert_called_once_with(
            model_name="test-model",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Test caching: second call with same params should not instantiate new class
        emb2 = get_embedding_model(model_name="test-model", device="cpu")
        assert emb2 == mock_instance
        assert mock_hf.call_count == 1
