import pytest
from unittest.mock import patch, MagicMock
from src.models import (
    fetch_distilbert_tokenizer,
    fetch_distilbert_model,
    load_trained_model,
    ID2LABEL,
    LABEL2ID,
)


class TestModels:
    def test_fetch_distilbert_tokenizer(self):
        with patch("src.models.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_tokenizer = MagicMock()
            mock_from_pretrained.return_value = mock_tokenizer

            result = fetch_distilbert_tokenizer()

            mock_from_pretrained.assert_called_once_with(
                "distilbert/distilbert-base-uncased"
            )
            assert result == mock_tokenizer

    def test_fetch_distilbert_model_default_params(self):
        with patch(
            "src.models.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_from_pretrained:
            mock_model = MagicMock()
            mock_from_pretrained.return_value = mock_model

            result = fetch_distilbert_model()

            mock_from_pretrained.assert_called_once_with(
                "distilbert/distilbert-base-uncased",
                num_labels=2,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
            assert result == mock_model

    @pytest.mark.parametrize(
        "num_labels,id2label,label2id",
        [
            (3, {0: "A", 1: "B", 2: "C"}, {"A": 0, "B": 1, "C": 2}),
            (5, {0: "ONE", 1: "TWO"}, {"ONE": 0, "TWO": 1}),
        ],
    )
    def test_fetch_distilbert_model_custom_params(self, num_labels, id2label, label2id):
        with patch(
            "src.models.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_from_pretrained:
            mock_model = MagicMock()
            mock_from_pretrained.return_value = mock_model

            result = fetch_distilbert_model(
                num_labels=num_labels, id2label=id2label, label2id=label2id
            )

            mock_from_pretrained.assert_called_once_with(
                "distilbert/distilbert-base-uncased",
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
            assert result == mock_model

    def test_load_trained_model_global_path(self):
        with patch(
            "src.models.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_from_pretrained:
            mock_model = MagicMock()
            mock_from_pretrained.return_value = mock_model

            test_path = "/path/to/trained/model"

            result = load_trained_model(test_path)

            mock_from_pretrained.assert_called_once_with(test_path)
            assert result == mock_model

    def test_load_trained_model_relative_path(self):
        with patch(
            "src.models.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_from_pretrained:
            mock_model = MagicMock()
            mock_from_pretrained.return_value = mock_model

            test_path = "./checkpoints/model-epoch-1"

            result = load_trained_model(test_path)

            mock_from_pretrained.assert_called_once_with(test_path)
            assert result == mock_model

    def test_constants(self):
        assert ID2LABEL == {0: "NEGATIVE", 1: "POSITIVE"}
        assert LABEL2ID == {"NEGATIVE": 0, "POSITIVE": 1}
