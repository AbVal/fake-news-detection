import argparse
import logging
from unittest.mock import patch, MagicMock
from validate import main


class TestValidate:
    @patch("validate.evaluator")
    @patch("validate.pipeline")
    @patch("validate.load_metrics")
    @patch("validate.load_dataset_from_df")
    @patch("validate.fetch_distilbert_tokenizer")
    @patch("validate.load_trained_model")
    @patch("validate.logging.basicConfig")
    @patch("validate.print")
    def test_main_with_default_model_path(
        self,
        mock_print,
        mock_logging,
        mock_load_model,
        mock_tokenizer,
        mock_load_dataset_from_df,
        mock_load_metrics,
        mock_pipeline,
        mock_evaluator,
    ):
        with patch("validate.argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = argparse.Namespace(model_path="model", data_path="data")

            mock_model = MagicMock()
            mock_load_model.return_value = mock_model
            mock_tokenizer.return_value = MagicMock()

            mock_dataset = MagicMock()
            mock_load_dataset_from_df.return_value = mock_dataset

            mock_metric = MagicMock()
            mock_load_metrics.return_value = mock_metric

            mock_pipe = MagicMock()
            mock_pipeline.return_value = mock_pipe

            mock_eval_instance = MagicMock()
            mock_evaluator.return_value = mock_eval_instance

            mock_results = {
                "accuracy": 0.95,
                "precision": 0.93,
                "recall": 0.92,
                "f1": 0.925,
            }
            mock_eval_instance.compute.return_value = mock_results

            main()

            mock_logging.assert_called_once_with(level=logging.INFO)

            mock_load_model.assert_called_once_with("model")
            mock_tokenizer.assert_called_once()

            mock_load_dataset_from_df.assert_called_once_with("data")

            mock_load_metrics.assert_called_once()

            mock_pipeline.assert_called_once_with(
                "text-classification",
                model=mock_model,
                tokenizer=mock_tokenizer.return_value,
                device=0,
            )

            mock_evaluator.assert_called_once_with("text-classification")
            mock_eval_instance.compute.assert_called_once_with(
                model_or_pipeline=mock_pipe,
                data=mock_dataset,
                metric=mock_metric,
                label_mapping={"NEGATIVE": 0, "POSITIVE": 1},
            )

            mock_print.assert_called_once_with(mock_results)

    @patch("validate.evaluator")
    @patch("validate.pipeline")
    @patch("validate.load_metrics")
    @patch("validate.load_dataset_from_df")
    @patch("validate.fetch_distilbert_tokenizer")
    @patch("validate.load_trained_model")
    @patch("validate.logging.basicConfig")
    @patch("validate.print")
    def test_main_with_custom_model_path(
        self,
        mock_print,
        mock_logging,
        mock_load_model,
        mock_tokenizer,
        mock_load_dataset_from_df,
        mock_load_metrics,
        mock_pipeline,
        mock_evaluator,
    ):
        with patch("validate.argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = argparse.Namespace(
                model_path="/path/to/custom/model",
                data_path="data"
            )

            mock_model = MagicMock()
            mock_load_model.return_value = mock_model
            mock_tokenizer.return_value = MagicMock()
            mock_load_dataset_from_df.return_value = MagicMock()
            mock_load_metrics.return_value = MagicMock()
            mock_pipeline.return_value = MagicMock()
            mock_eval_instance = MagicMock()
            mock_evaluator.return_value = mock_eval_instance
            mock_eval_instance.compute.return_value = {"accuracy": 0.9}

            main()

            mock_load_model.assert_called_once_with("/path/to/custom/model")

    @patch("validate.evaluator")
    @patch("validate.pipeline")
    @patch("validate.load_metrics")
    @patch("validate.load_dataset_from_df")
    @patch("validate.fetch_distilbert_tokenizer")
    @patch("validate.load_trained_model")
    @patch("validate.logging.basicConfig")
    @patch("validate.print")
    def test_main_with_custom_data_path(
        self,
        mock_print,
        mock_logging,
        mock_load_model,
        mock_tokenizer,
        mock_load_dataset_from_df,
        mock_load_metrics,
        mock_pipeline,
        mock_evaluator,
    ):
        with patch("validate.argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = argparse.Namespace(
                model_path="model",
                data_path="/path/to/custom/data"
            )

            mock_model = MagicMock()
            mock_load_model.return_value = mock_model
            mock_tokenizer.return_value = MagicMock()
            mock_load_dataset_from_df.return_value = MagicMock()
            mock_load_metrics.return_value = MagicMock()
            mock_pipeline.return_value = MagicMock()
            mock_eval_instance = MagicMock()
            mock_evaluator.return_value = mock_eval_instance
            mock_eval_instance.compute.return_value = {"accuracy": 0.9}

            main()

            mock_load_dataset_from_df.assert_called_once_with("/path/to/custom/data")
