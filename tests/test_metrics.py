import pytest
import evaluate
import numpy as np
from transformers import EvalPrediction
from src.metrics import load_metrics, ComputeMetrics, METRIC_NAMES


class TestLoadMetrics:
    def test_load_default_metrics(self):
        combined_metrics = load_metrics(METRIC_NAMES)

        assert isinstance(combined_metrics, evaluate.CombinedEvaluations)

        results = combined_metrics.compute(predictions=[0, 1, 0], references=[0, 1, 1])

        assert isinstance(results, dict)
        assert len(results) > 0

    def test_load_custom_metrics(self):
        custom_metrics = ["accuracy", "f1"]
        combined_metrics = load_metrics(custom_metrics)

        assert isinstance(combined_metrics, evaluate.CombinedEvaluations)

        results = combined_metrics.compute(predictions=[0, 1], references=[0, 1])
        assert "accuracy" in results
        assert "f1" in results


class TestComputeMetrics:
    @pytest.fixture
    def compute_metrics(self):
        return ComputeMetrics()

    @pytest.fixture
    def mock_predictions(self):
        """Fixture creating mock model predictions and labels."""
        logits = np.array([[0.8, 0.2], [0.1, 0.9], [0.3, 0.7]])
        labels = np.array([0, 1, 1])
        return EvalPrediction(predictions=logits, label_ids=labels)

    def test_initialization_default_metrics(self):
        computer = ComputeMetrics()
        assert computer.metrics is not None
        assert isinstance(computer.metrics, evaluate.CombinedEvaluations)

    def test_initialization_custom_metrics(self):
        custom_metrics = ["accuracy"]
        computer = ComputeMetrics(metric_names=custom_metrics)

        assert computer.metrics is not None

    def test_call_method(self, compute_metrics, mock_predictions):
        results = compute_metrics(mock_predictions)

        assert isinstance(results, dict)

        expected_keys = ["accuracy", "precision", "recall", "f1"]

        assert all(key in results for key in expected_keys)

        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["precision"] <= 1
        assert 0 <= results["recall"] <= 1
        assert 0 <= results["f1"] <= 1

    def test_call_with_perfect_predictions(self, compute_metrics):
        perfect_logits = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        perfect_labels = np.array([0, 1, 1])
        eval_pred = EvalPrediction(predictions=perfect_logits, label_ids=perfect_labels)

        results = compute_metrics(eval_pred)

        assert results["accuracy"] == 1.0
        assert results["precision"] == 1.0
        assert results["recall"] == 1.0
        assert results["f1"] == 1.0

    def test_call_with_wrong_predictions(self, compute_metrics):
        wrong_logits = np.array([[0.2, 0.8], [0.9, 0.1], [0.9, 0.1]])
        correct_labels = np.array([0, 1, 1])
        eval_pred = EvalPrediction(predictions=wrong_logits, label_ids=correct_labels)

        results = compute_metrics(eval_pred)

        assert results["accuracy"] == 0.0
        assert results["precision"] == 0.0
        assert results["recall"] == 0.0
        assert results["f1"] == 0.0

    def test_empty_metrics(self):
        computer = ComputeMetrics(metric_names=[])
        logits = np.array([[0.8, 0.2], [0.1, 0.9]])
        labels = np.array([0, 1])
        eval_pred = EvalPrediction(predictions=logits, label_ids=labels)

        results = computer(eval_pred)

        assert results == {}
