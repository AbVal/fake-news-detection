import os
import argparse
import logging
from src.dataset import fetch_hf_dataset, preprocess_train_df, split_df_train_val_test, save_df
logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to data", default="data")
    parser.add_argument("--val_size",
                        help="Validation size relative to all data",
                        default=0.2,
                        type=float)
    parser.add_argument("--test_size",
                        help="Test size relative to all data",
                        default=0.1,
                        type=float)
    parser.add_argument("--random_state",
                        help="Random state for splitting",
                        default=42,
                        type=int)

    args = parser.parse_args()

    logger.info("Loading dataset from hf")
    dataset = fetch_hf_dataset()

    logger.info("Processing dataset")
    dataset_df = dataset.to_pandas()
    dataset_df = preprocess_train_df(dataset_df)  # type: ignore[arg-type]
    train_df, val_df, test_df = split_df_train_val_test(dataset_df,
                                                        val_size=args.val_size,
                                                        test_size=args.test_size,
                                                        random_state=args.random_state)

    logger.info(f"Saving train, val, test splits to {args.data_path}")
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    save_df(train_df, os.path.join(args.data_path, 'train.csv.gz'))
    save_df(val_df, os.path.join(args.data_path, 'val.csv.gz'))
    save_df(test_df, os.path.join(args.data_path, 'test.csv.gz'))


if __name__ == "__main__":
    main()
