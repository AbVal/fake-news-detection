import os
import logging
import pandas as pd
from typing import Callable, Tuple
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer, BatchEncoding

logger = logging.getLogger(__name__)


def fetch_hf_dataset() -> Dataset:
    logger.info("Fetching Fake-News-Detection-dataset")
    return load_dataset("Pulk17/Fake-News-Detection-dataset")['train']  # type: ignore


def preprocess_train_df(df: pd.DataFrame) -> pd.DataFrame:
    assert "text" in df.columns, "text column is not present in data"
    assert "label" in df.columns, "label column is not present in data"
    assert df.dtypes["text"] in [object, str], "text column has incorrect type"

    df["text"] = df.text.str.strip()
    word_count = df.text.str.split().str.len()
    df = df[word_count > 10]

    assert len(df) > 0, "dataset is empty after cleaning samples with too low word counts"

    df = df.drop_duplicates(subset=["text"])

    df = df[df.label.notna()]

    assert len(df) > 0, "dataset is empty after dropping NA labels"

    return df


def split_df_train_val_test(df: pd.DataFrame,
                            val_size: float = 0.1,
                            test_size: float = 0.2,
                            random_state: int = 42
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert val_size + test_size < 1, "val_size + test_size should be less than 1"
    assert val_size + test_size > 0, "val_size + test_size should be greater than 1"

    train, temp = train_test_split(df, test_size=val_size + test_size, random_state=random_state)
    test, val = train_test_split(temp,
                                 test_size=val_size / (val_size + test_size),
                                 random_state=random_state)

    return train, val, test


def save_df(df, dataframe_path):
    if os.path.exists(dataframe_path):
        logger.warning(f'Overwriting {dataframe_path}')
    df.to_csv(dataframe_path, index=False, compression='gzip')


def load_dataset_from_df(dataframe_path: str) -> Dataset:
    assert os.path.exists(dataframe_path), f"{dataframe_path} does not exist"
    logger.info(f'Loading {dataframe_path}')

    df = pd.read_csv(dataframe_path)
    return Dataset.from_pandas(df)


def apply_tokenize_wrapper(tokenizer: PreTrainedTokenizer) -> Callable[[Dataset], BatchEncoding]:
    def apply_tokenize(dataset: Dataset) -> BatchEncoding:
        tokenized_data = tokenizer(dataset["text"], truncation=True)
        return tokenized_data

    return apply_tokenize


def tokenize_ds(ds: Dataset, tokenizer: PreTrainedTokenizer) -> Dataset:
    logger.info("Tokenizing datasets")
    ds_tokenized = ds.map(apply_tokenize_wrapper(tokenizer), batched=True)
    return ds_tokenized
