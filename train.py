import os
import yaml
import logging
import argparse
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from src.models import fetch_distilbert_tokenizer, fetch_distilbert_model
from src.dataset import load_dataset_from_df, tokenize_ds
from src.metrics import ComputeMetrics
from src.utils import seed_everything, disable_progresscallback_logging
from src.config import TrainingConfig

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="Path to training config")
    parser.add_argument("--train_data_path",
                        help="Path to train data",
                        default=os.path.join("data", "train.csv.gz"))
    parser.add_argument("--val_data_path",
                        help="Path to validation data",
                        default=os.path.join("data", "val.csv.gz"))

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    if args.config_path:
        with open(args.config_path, "r") as file:
            config_yaml = yaml.safe_load(file)  # type: ignore[no-untyped-call]
        config = TrainingConfig.model_validate(config_yaml)

    else:
        config = TrainingConfig.model_construct()

    seed = config.seed
    seed_everything(seed)

    tokenizer = fetch_distilbert_tokenizer()
    model = fetch_distilbert_model()

    logger.info("Loading datasets")
    train_ds = load_dataset_from_df(args.train_data_path)
    val_ds = load_dataset_from_df(args.val_data_path)

    logger.info("Tokenizing datasets")
    train_ds_tokenized = tokenize_ds(train_ds, tokenizer)
    val_ds_tokenized = tokenize_ds(val_ds, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metrics_callback = ComputeMetrics()

    if not config.verbose:
        disable_progresscallback_logging()

    training_args = TrainingArguments(
        data_seed=seed,
        full_determinism=True,
        output_dir=config.output_dir,
        seed=config.seed,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        fp16=config.fp16,
        report_to=config.report_to,
    )

    logger.info("Starting model training")
    logger.info("Logging metrics to tensorboard")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_tokenized,
        eval_dataset=val_ds_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics_callback,
    )

    trainer.train()

    logger.info("Saving model")
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
