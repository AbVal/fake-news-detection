import logging
import evaluate
import numpy as np
from transformers import EvalPrediction
from typing import List, Dict
from numpy.typing import ArrayLike


logger = logging.getLogger(__name__)

METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]


def load_metrics(metric_names: List[str]) -> evaluate.CombinedEvaluations:  # type: ignore
    logger.info(f"Loading evaluate metrics: {metric_names}")
    metric_list = []

    for metric_name in metric_names:
        metric = evaluate.load(metric_name)

        metric_list.append(metric)

    combined_metrics = evaluate.combine(metric_list)  # type: ignore[no-untyped-call]
    return combined_metrics


class ComputeMetrics:
    def __init__(self, metric_names: List[str] = METRIC_NAMES) -> None:
        self.metrics = load_metrics(METRIC_NAMES)

    def __call__(self, eval_pred: EvalPrediction) -> Dict[str, ArrayLike]:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metrics.compute(predictions=predictions, references=labels)
