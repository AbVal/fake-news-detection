from pydantic import BaseModel, Field
from typing import List


class TrainingConfig(BaseModel):
    output_dir: str = Field(default="model")
    seed: int = Field(default=42)
    verbose: bool = Field(default=True)
    learning_rate: float = Field(default=2e-5)
    per_device_train_batch_size: int = Field(default=16)
    per_device_eval_batch_size: int = Field(default=16, ge=1)
    num_train_epochs: int = Field(default=2, ge=1)
    weight_decay: float = Field(default=0.01, ge=0)
    save_strategy: str = Field(default="no", examples=["no", "epoch"])
    eval_strategy: str = Field(default="steps", examples=["no", "steps", "epoch"])
    eval_steps: int = Field(
        default=250,
        gt=1,
        description="""Number of update steps between two evaluations if eval_strategy="steps".
                       Will default to the same value as logging_steps if not set.""",
    )
    logging_steps: int = Field(
        default=250,
        gt=1,
        description="""Number of update steps between two logs if logging_strategy="steps".""",
    )
    fp16: bool = Field(default=True)
    report_to: str | List[str] = Field(
        default="tensorboard",
        description="""The list of integrations to report the results and logs to.
                       Supported platforms are "azure_ml", "clearml", "codecarbon", "comet_ml",
                       "dagshub", "dvclive", "flyte", "mlflow", "neptune", "swanlab", "tensorboard",
                       "trackio" and "wandb". Use "all" to report to all integrations installed,
                       "none" for no integrations.""",
    )
