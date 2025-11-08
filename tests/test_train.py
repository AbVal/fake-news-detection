import pytest
import argparse
import yaml
import tempfile
import os
from unittest.mock import patch, MagicMock, call
from train import main
from src.config import TrainingConfig
from pydantic import ValidationError


class TestTrain:
    @pytest.fixture
    def mock_config(self):
        """Mock training configuration"""
        config = TrainingConfig(
            output_dir="test_model",
            seed=42,
            verbose=True,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
            save_strategy="no",
            eval_strategy="steps",
            eval_steps=100,
            logging_steps=100,
            fp16=False,
            report_to="none",
        )
        return config

    @pytest.fixture
    def sample_config_file(self, mock_config):
        """Create a temporary config file for testing"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(mock_config.model_dump(), f)
            config_path = f.name

        yield config_path
        os.unlink(config_path)

    @patch("train.Trainer")
    @patch("train.TrainingArguments")
    @patch("train.DataCollatorWithPadding")
    @patch("train.ComputeMetrics")
    @patch("train.tokenize_ds")
    @patch("train.load_dataset_from_df")
    @patch("train.fetch_distilbert_model")
    @patch("train.fetch_distilbert_tokenizer")
    @patch("train.seed_everything")
    @patch("train.disable_progresscallback_logging")
    @patch("train.logging.basicConfig")
    def test_main_with_default_config(
        self,
        mock_logging,
        mock_disable_progress,
        mock_seed,
        mock_tokenizer,
        mock_model,
        mock_load_dataset,
        mock_tokenize_ds,
        mock_metrics,
        mock_data_collator,
        mock_training_args,
        mock_trainer,
    ):
        with patch("train.argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = argparse.Namespace(
                config_path=None,
                train_data_path=os.path.join("data", "train.csv.gz"),
                val_data_path=os.path.join("data", "val.csv.gz"),
            )

            mock_tokenizer.return_value = MagicMock()
            mock_model.return_value = MagicMock()

            mock_train_ds = MagicMock()
            mock_val_ds = MagicMock()
            mock_load_dataset.side_effect = [mock_train_ds, mock_val_ds]

            mock_tokenized_train = MagicMock()
            mock_tokenized_val = MagicMock()
            mock_tokenize_ds.side_effect = [mock_tokenized_train, mock_tokenized_val]

            mock_data_collator.return_value = MagicMock()
            mock_metrics.return_value = MagicMock()

            mock_training_instance = MagicMock()
            mock_training_args.return_value = mock_training_instance

            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance

            main()

            mock_logging.assert_called_once_with(level=20)  # logging.INFO = 20

            mock_seed.assert_called_once_with(42)

            mock_tokenizer.assert_called_once()
            mock_model.assert_called_once()

            # Check dataset loading with correct paths
            assert mock_load_dataset.call_count == 2
            mock_load_dataset.assert_has_calls(
                [
                    call(os.path.join("data", "train.csv.gz")),
                    call(os.path.join("data", "val.csv.gz")),
                ]
            )

            assert mock_tokenize_ds.call_count == 2
            mock_tokenize_ds.assert_has_calls(
                [
                    call(mock_train_ds, mock_tokenizer.return_value),
                    call(mock_val_ds, mock_tokenizer.return_value),
                ]
            )

            mock_data_collator.assert_called_once_with(
                tokenizer=mock_tokenizer.return_value
            )
            mock_metrics.assert_called_once()

            mock_disable_progress.assert_not_called()

            mock_training_args.assert_called_once()
            call_kwargs = mock_training_args.call_args.kwargs
            assert call_kwargs["output_dir"] == "model"
            assert call_kwargs["seed"] == 42
            assert call_kwargs["learning_rate"] == 2e-5
            assert call_kwargs["per_device_train_batch_size"] == 16
            assert call_kwargs["per_device_eval_batch_size"] == 16
            assert call_kwargs["num_train_epochs"] == 2
            assert call_kwargs["weight_decay"] == 0.01
            assert call_kwargs["save_strategy"] == "no"
            assert call_kwargs["eval_strategy"] == "steps"
            assert call_kwargs["eval_steps"] == 250
            assert call_kwargs["logging_steps"] == 250
            assert call_kwargs["fp16"]
            assert call_kwargs["report_to"] == "tensorboard"

            mock_trainer.assert_called_once_with(
                model=mock_model.return_value,
                args=mock_training_instance,
                train_dataset=mock_tokenized_train,
                eval_dataset=mock_tokenized_val,
                processing_class=mock_tokenizer.return_value,
                data_collator=mock_data_collator.return_value,
                compute_metrics=mock_metrics.return_value,
            )

            mock_trainer_instance.train.assert_called_once()
            mock_trainer_instance.save_model.assert_called_once_with("model")

    @patch("train.Trainer")
    @patch("train.TrainingArguments")
    @patch("train.DataCollatorWithPadding")
    @patch("train.ComputeMetrics")
    @patch("train.tokenize_ds")
    @patch("train.load_dataset_from_df")
    @patch("train.fetch_distilbert_model")
    @patch("train.fetch_distilbert_tokenizer")
    @patch("train.seed_everything")
    @patch("train.disable_progresscallback_logging")
    @patch("train.logging.basicConfig")
    def test_main_with_config_file_and_custom_data_paths(
        self,
        mock_logging,
        mock_disable_progress,
        mock_seed,
        mock_tokenizer,
        mock_model,
        mock_load_dataset,
        mock_tokenize_ds,
        mock_metrics,
        mock_data_collator,
        mock_training_args,
        mock_trainer,
        sample_config_file,
        mock_config,
    ):
        with patch("train.argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = argparse.Namespace(
                config_path=sample_config_file,
                train_data_path="/custom/path/train.csv",
                val_data_path="/custom/path/val.csv",
            )

            mock_tokenizer.return_value = MagicMock()
            mock_model.return_value = MagicMock()

            mock_train_ds = MagicMock()
            mock_val_ds = MagicMock()
            mock_load_dataset.side_effect = [mock_train_ds, mock_val_ds]

            mock_tokenized_train = MagicMock()
            mock_tokenized_val = MagicMock()
            mock_tokenize_ds.side_effect = [mock_tokenized_train, mock_tokenized_val]

            mock_data_collator.return_value = MagicMock()
            mock_metrics.return_value = MagicMock()

            mock_training_instance = MagicMock()
            mock_training_args.return_value = mock_training_instance

            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance

            main()

            # Check dataset loading with custom paths
            mock_load_dataset.assert_has_calls(
                [call("/custom/path/train.csv"), call("/custom/path/val.csv")]
            )

            mock_training_args.assert_called_once()
            call_kwargs = mock_training_args.call_args.kwargs
            assert call_kwargs["output_dir"] == "test_model"
            assert call_kwargs["num_train_epochs"] == 1
            assert not call_kwargs["fp16"]
            assert call_kwargs["report_to"] == "none"

    @patch("train.Trainer")
    @patch("train.TrainingArguments")
    @patch("train.DataCollatorWithPadding")
    @patch("train.ComputeMetrics")
    @patch("train.tokenize_ds")
    @patch("train.load_dataset_from_df")
    @patch("train.fetch_distilbert_model")
    @patch("train.fetch_distilbert_tokenizer")
    @patch("train.seed_everything")
    @patch("train.disable_progresscallback_logging")
    @patch("train.logging.basicConfig")
    def test_main_with_verbose_false(
        self,
        mock_logging,
        mock_disable_progress,
        mock_seed,
        mock_tokenizer,
        mock_model,
        mock_load_dataset,
        mock_tokenize_ds,
        mock_metrics,
        mock_data_collator,
        mock_training_args,
        mock_trainer,
    ):
        # Create a temporary config file with verbose=False
        config_data = TrainingConfig().model_dump()
        config_data["verbose"] = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with patch("train.argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = argparse.Namespace(
                    config_path=config_path,
                    train_data_path=os.path.join("data", "train.csv.gz"),
                    val_data_path=os.path.join("data", "val.csv.gz"),
                )

                # Mock minimal components
                mock_tokenizer.return_value = MagicMock()
                mock_model.return_value = MagicMock()
                mock_load_dataset.return_value = MagicMock()
                mock_tokenize_ds.return_value = MagicMock()
                mock_training_args.return_value = MagicMock()
                mock_trainer.return_value = MagicMock()

                main()

                # Check that disable_progresscallback_logging was called
                mock_disable_progress.assert_called_once()
        finally:
            os.unlink(config_path)

    @patch("train.Trainer")
    @patch("train.TrainingArguments")
    @patch("train.logging.basicConfig")
    def test_main_with_invalid_config_file(
        self, mock_logging, mock_training_args, mock_trainer
    ):
        """Test main function with invalid config file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"invalid_field": "value", "learning_rate": -1.0}, f)
            config_path = f.name

        try:
            with patch("train.argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = argparse.Namespace(
                    config_path=config_path,
                    train_data_path=os.path.join("data", "train.csv.gz"),
                    val_data_path=os.path.join("data", "val.csv.gz"),
                )

                with pytest.raises(ValidationError):
                    main()
        finally:
            os.unlink(config_path)

    @patch("train.Trainer")
    @patch("train.TrainingArguments")
    @patch("train.DataCollatorWithPadding")
    @patch("train.ComputeMetrics")
    @patch("train.tokenize_ds")
    @patch("train.load_dataset_from_df")
    @patch("train.fetch_distilbert_model")
    @patch("train.fetch_distilbert_tokenizer")
    @patch("train.seed_everything")
    @patch("train.logging.basicConfig")
    def test_training_arguments_full_coverage(
        self,
        mock_logging,
        mock_seed,
        mock_tokenizer,
        mock_model,
        mock_load_dataset,
        mock_tokenize_ds,
        mock_metrics,
        mock_data_collator,
        mock_training_args,
        mock_trainer,
    ):
        """Test that all TrainingArguments parameters are passed correctly"""
        with patch("train.argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = argparse.Namespace(
                config_path=None,
                train_data_path=os.path.join("data", "train.csv.gz"),
                val_data_path=os.path.join("data", "val.csv.gz"),
            )

            mock_tokenizer.return_value = MagicMock()
            mock_model.return_value = MagicMock()
            mock_load_dataset.return_value = MagicMock()
            mock_tokenize_ds.return_value = MagicMock()
            mock_training_args.return_value = MagicMock()
            mock_trainer.return_value = MagicMock()

            main()

            mock_training_args.assert_called_once()
            call_kwargs = mock_training_args.call_args.kwargs

            expected_params = [
                "output_dir",
                "seed",
                "learning_rate",
                "per_device_train_batch_size",
                "per_device_eval_batch_size",
                "num_train_epochs",
                "weight_decay",
                "save_strategy",
                "eval_strategy",
                "eval_steps",
                "logging_steps",
                "fp16",
                "report_to",
                "data_seed",
                "full_determinism",
            ]

            for param in expected_params:
                assert param in call_kwargs

    @patch("train.Trainer")
    @patch("train.TrainingArguments")
    @patch("train.logging.basicConfig")
    def test_main_exception_handling(
        self, mock_logging, mock_training_args, mock_trainer
    ):
        with patch("train.argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = argparse.Namespace(
                config_path=None,
                train_data_path=os.path.join("data", "train.csv.gz"),
                val_data_path=os.path.join("data", "val.csv.gz"),
            )

            with patch("train.load_dataset_from_df") as mock_load_dataset:
                mock_load_dataset.side_effect = Exception("Data loading error")

                with pytest.raises(Exception, match="Data loading error"):
                    main()

    @patch("train.Trainer")
    @patch("train.TrainingArguments")
    @patch("train.DataCollatorWithPadding")
    @patch("train.ComputeMetrics")
    @patch("train.tokenize_ds")
    @patch("train.load_dataset_from_df")
    @patch("train.fetch_distilbert_model")
    @patch("train.fetch_distilbert_tokenizer")
    @patch("train.seed_everything")
    @patch("train.logging.basicConfig")
    def test_trainer_called_correctly(
        self,
        mock_logging,
        mock_seed,
        mock_tokenizer,
        mock_model,
        mock_load_dataset,
        mock_tokenize_ds,
        mock_metrics,
        mock_data_collator,
        mock_training_args,
        mock_trainer,
    ):
        with patch("train.argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = argparse.Namespace(
                config_path=None,
                train_data_path=os.path.join("data", "train.csv.gz"),
                val_data_path=os.path.join("data", "val.csv.gz"),
            )

            # Setup mocks
            mock_tokenizer.return_value = "mock_tokenizer"
            mock_model.return_value = "mock_model"
            mock_load_dataset.return_value = "mock_dataset"
            mock_tokenize_ds.return_value = "mock_tokenized"
            mock_data_collator.return_value = "mock_data_collator"
            mock_metrics.return_value = "mock_metrics"
            mock_training_args.return_value = "mock_training_args"
            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance

            main()

            mock_trainer.assert_called_once_with(
                model="mock_model",
                args="mock_training_args",
                train_dataset="mock_tokenized",
                eval_dataset="mock_tokenized",
                processing_class="mock_tokenizer",
                data_collator="mock_data_collator",
                compute_metrics="mock_metrics",
            )

            mock_trainer_instance.train.assert_called_once()
            mock_trainer_instance.save_model.assert_called_once_with("model")

    @patch("train.Trainer")
    @patch("train.TrainingArguments")
    @patch("train.DataCollatorWithPadding")
    @patch("train.ComputeMetrics")
    @patch("train.tokenize_ds")
    @patch("train.load_dataset_from_df")
    @patch("train.fetch_distilbert_model")
    @patch("train.fetch_distilbert_tokenizer")
    @patch("train.seed_everything")
    @patch("train.logging.basicConfig")
    def test_logging_statements(
        self,
        mock_logging,
        mock_seed,
        mock_tokenizer,
        mock_model,
        mock_load_dataset,
        mock_tokenize_ds,
        mock_metrics,
        mock_data_collator,
        mock_training_args,
        mock_trainer,
    ):
        """Test that appropriate logging statements are called"""
        with patch("train.argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = argparse.Namespace(
                config_path=None,
                train_data_path=os.path.join("data", "train.csv.gz"),
                val_data_path=os.path.join("data", "val.csv.gz"),
            )

            mock_tokenizer.return_value = MagicMock()
            mock_model.return_value = MagicMock()
            mock_load_dataset.return_value = MagicMock()
            mock_tokenize_ds.return_value = MagicMock()
            mock_training_args.return_value = MagicMock()
            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance

            with patch("train.logger") as mock_logger:
                main()

                mock_logger.info.assert_has_calls(
                    [
                        call("Loading datasets"),
                        call("Tokenizing datasets"),
                        call("Starting model training"),
                        call("Logging metrics to tensorboard"),
                        call("Saving model"),
                    ]
                )
