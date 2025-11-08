# mypy: disable-error-code="no-untyped-call"
import logging
from typing import Dict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    PreTrainedModel,
)

logger = logging.getLogger(__name__)


ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID = {"NEGATIVE": 0, "POSITIVE": 1}


def fetch_distilbert_tokenizer() -> PreTrainedTokenizer:
    logger.info("Fetching distilbert-base-uncased tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    return tokenizer


def fetch_distilbert_model(
    num_labels: int = 2,
    id2label: Dict[int, str] = ID2LABEL,
    label2id: Dict[str, int] = LABEL2ID,
) -> PreTrainedModel:
    logger.info("Fetching distilbert-base-uncased model")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return model


def load_trained_model(model_path: str) -> PreTrainedModel:
    logger.info("Loading trained model")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model
