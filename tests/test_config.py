import pytest
from pydantic import ValidationError
from src.config import TrainingConfig


class TestTrainingConfig:
    def test_default_values(self):
        config = TrainingConfig()

        assert config.output_dir == "model"
        assert config.seed == 42
        assert config.verbose is True
        assert config.learning_rate == 2e-5
        assert config.per_device_train_batch_size == 16
        assert config.per_device_eval_batch_size == 16
        assert config.num_train_epochs == 2
        assert config.weight_decay == 0.01
        assert config.save_strategy == "no"
        assert config.eval_strategy == "steps"
        assert config.eval_steps == 250
        assert config.logging_steps == 250
        assert config.fp16 is True
        assert config.report_to == "tensorboard"

    def test_custom_values(self):
        custom_config = TrainingConfig(
            output_dir="custom_model",
            seed=123,
            verbose=False,
            learning_rate=1e-4,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=5,
            weight_decay=0.1,
            save_strategy="epoch",
            eval_strategy="epoch",
            eval_steps=500,
            logging_steps=100,
            fp16=False,
            report_to=["tensorboard", "wandb"],
        )

        assert custom_config.output_dir == "custom_model"
        assert custom_config.seed == 123
        assert custom_config.verbose is False
        assert custom_config.learning_rate == 1e-4
        assert custom_config.per_device_train_batch_size == 32
        assert custom_config.per_device_eval_batch_size == 32
        assert custom_config.num_train_epochs == 5
        assert custom_config.weight_decay == 0.1
        assert custom_config.save_strategy == "epoch"
        assert custom_config.eval_strategy == "epoch"
        assert custom_config.eval_steps == 500
        assert custom_config.logging_steps == 100
        assert custom_config.fp16 is False
        assert custom_config.report_to == ["tensorboard", "wandb"]

    @pytest.mark.parametrize("learning_rate", [1e-5, 5e-5, 1e-4])
    def test_valid_learning_rate(self, learning_rate):
        config = TrainingConfig(learning_rate=learning_rate)
        assert config.learning_rate == learning_rate

    @pytest.mark.parametrize("invalid_lr", [0, -1e-5, -0.1])
    def test_invalid_learning_rate(self, invalid_lr):
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(learning_rate=invalid_lr)

        assert "learning_rate" in str(exc_info.value)
        assert "greater than 0" in str(exc_info.value).lower()

    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16, 32, 64])
    def test_valid_batch_sizes(self, batch_size):
        config = TrainingConfig(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
        )
        assert config.per_device_train_batch_size == batch_size
        assert config.per_device_eval_batch_size == batch_size

    @pytest.mark.parametrize("invalid_batch_size", [0, -1, -16])
    def test_invalid_per_device_train_batch_sizes(self, invalid_batch_size):
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(per_device_train_batch_size=invalid_batch_size)

        assert "per_device_train_batch_size" in str(exc_info.value)
        assert "greater than or equal to 1" in str(exc_info.value).lower()

    @pytest.mark.parametrize("invalid_batch_size", [0, -1, -16])
    def test_invalid_per_device_eval_batch_sizes(self, invalid_batch_size):
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(per_device_eval_batch_size=invalid_batch_size)

        assert "per_device_eval_batch_size" in str(exc_info.value)
        assert "greater than or equal to 1" in str(exc_info.value).lower()

    @pytest.mark.parametrize("epochs", [1, 2, 5, 10, 100])
    def test_valid_num_epochs(self, epochs):
        config = TrainingConfig(num_train_epochs=epochs)
        assert config.num_train_epochs == epochs

    @pytest.mark.parametrize("invalid_epochs", [0, -1, -5])
    def test_invalid_num_epochs(self, invalid_epochs):
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(num_train_epochs=invalid_epochs)

        assert "num_train_epochs" in str(exc_info.value)
        assert "greater than or equal to 1" in str(exc_info.value).lower()

    @pytest.mark.parametrize("weight_decay", [0, 0.01, 0.1, 0.5, 1.0])
    def test_valid_weight_decay(self, weight_decay):
        config = TrainingConfig(weight_decay=weight_decay)
        assert config.weight_decay == weight_decay

    @pytest.mark.parametrize("invalid_weight_decay", [-0.1, -1.0, -0.01])
    def test_invalid_weight_decay(self, invalid_weight_decay):
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(weight_decay=invalid_weight_decay)

        assert "weight_decay" in str(exc_info.value)
        assert "greater than or equal to 0" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "eval_steps,logging_steps", [(100, 100), (500, 500), (1000, 1000)]
    )
    def test_valid_steps(self, eval_steps, logging_steps):
        config = TrainingConfig(eval_steps=eval_steps, logging_steps=logging_steps)
        assert config.eval_steps == eval_steps
        assert config.logging_steps == logging_steps

    @pytest.mark.parametrize("invalid_steps", [0, 1, -100, -500])
    def test_invalid_steps(self, invalid_steps):
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(eval_steps=invalid_steps)

        assert "eval_steps" in str(exc_info.value)
        assert "greater than 1" in str(exc_info.value).lower()

    @pytest.mark.parametrize("save_strategy", ["no", "epoch"])
    def test_valid_save_strategy(self, save_strategy):
        config = TrainingConfig(save_strategy=save_strategy)
        assert config.save_strategy == save_strategy

    @pytest.mark.parametrize("eval_strategy", ["no", "steps", "epoch"])
    def test_valid_eval_strategy(self, eval_strategy):
        config = TrainingConfig(eval_strategy=eval_strategy)
        assert config.eval_strategy == eval_strategy

    def test_report_to_as_list(self):
        config = TrainingConfig(report_to=["tensorboard", "wandb", "mlflow"])
        assert config.report_to == ["tensorboard", "wandb", "mlflow"]

    @pytest.mark.parametrize("report_to", ["wandb", "all", "none"])
    def test_report_to_as_string(self, report_to):
        config = TrainingConfig(report_to=report_to)
        assert config.report_to == report_to
