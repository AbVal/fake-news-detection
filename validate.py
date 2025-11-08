import os
import argparse
import logging
from src.models import fetch_distilbert_tokenizer, load_trained_model
from src.dataset import load_dataset_from_df
from src.metrics import load_metrics, METRIC_NAMES

from transformers import pipeline
from evaluate import evaluator

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to model", default="model")
    parser.add_argument("--data_path",
                        help="Path to test data",
                        default=os.path.join("data", "test.csv.gz"))

    args = parser.parse_args()

    model = load_trained_model(args.model_path)
    tokenizer = fetch_distilbert_tokenizer()

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
    data = load_dataset_from_df(args.data_path)
    metric = load_metrics(METRIC_NAMES)

    task_evaluator = evaluator("text-classification")

    results = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=data,
        metric=metric,
        label_mapping={"NEGATIVE": 0, "POSITIVE": 1},  # type: ignore
    )

    print(results)


if __name__ == "__main__":
    main()
