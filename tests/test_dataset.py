import pytest
import pandas as pd
import os
from unittest.mock import Mock, patch
from datasets import Dataset
from transformers import PreTrainedTokenizer
from src.dataset import (
    fetch_hf_dataset,
    preprocess_train_df,
    split_df_train_val_test,
    save_df,
    load_dataset_from_df,
    apply_tokenize_wrapper,
    tokenize_ds,
)


class TestFetchHfDataset:
    @patch("src.dataset.load_dataset")
    def test_fetch_hf_dataset_success(self, mock_load_dataset):
        mock_dataset = Mock()
        mock_load_dataset.return_value = {"train": mock_dataset}

        result = fetch_hf_dataset()

        mock_load_dataset.assert_called_once_with("Pulk17/Fake-News-Detection-dataset")
        assert result == mock_dataset


class TestPreprocessTrainDf:
    def test_normal_preprocessing(self):
        df = pd.DataFrame(
            {
                "text": [
                    "  hello world  ",
                    "a b c d e f g h i j k l",
                    "   ",
                    "dup",
                    "dup",
                ],
                "label": [1, 0, 1, 1, 1],
            }
        )
        result = preprocess_train_df(df)

        assert len(result) == 1
        assert result.iloc[0]["text"] == "a b c d e f g h i j k l"
        assert result.iloc[0]["label"] == 0

    def test_missing_text_column(self):
        df = pd.DataFrame({"label": [1]})
        with pytest.raises(AssertionError, match="text column is not present"):
            preprocess_train_df(df)

    def test_missing_label_column(self):
        df = pd.DataFrame({"text": ["sample"]})
        with pytest.raises(AssertionError, match="label column is not present"):
            preprocess_train_df(df)

    def test_incorrect_text_type(self):
        df = pd.DataFrame({"text": [123], "label": [1]})
        with pytest.raises(AssertionError, match="text column has incorrect type"):
            preprocess_train_df(df)

    def test_empty_after_word_count_filter(self):
        df = pd.DataFrame({"text": ["a"], "label": [1]})
        with pytest.raises(AssertionError, match="dataset is empty after cleaning"):
            preprocess_train_df(df)

    def test_empty_after_na_label_filter(self):
        df = pd.DataFrame(
            {
                "text": [
                    "kupil muzhik test a on emu kak assert regex pattern did not match"
                ],
                "label": [None],
            }
        )
        with pytest.raises(
            AssertionError, match="dataset is empty after dropping NA labels"
        ):
            preprocess_train_df(df)


class TestSplitDfTrainValTest:
    def test_normal_split(self):
        df = pd.DataFrame(
            {
                "text": [f"sample{i}" for i in range(100)],
                "label": [i % 2 for i in range(100)],
            }
        )
        train, val, test = split_df_train_val_test(df, test_size=1 / 4, val_size=1 / 4)

        assert len(train) == 50
        assert len(val) == 25
        assert len(test) == 25

    def test_split_ratio_violation(self):
        df = pd.DataFrame({"text": ["a"], "label": [1]})
        with pytest.raises(
            AssertionError, match=r"val_size \+ test_size should be less than 1"
        ):
            split_df_train_val_test(df, val_size=0.5, test_size=0.6)

    def test_zero_split_sizes(self):
        df = pd.DataFrame({"text": ["a"], "label": [1]})
        with pytest.raises(
            AssertionError, match=r"val_size \+ test_size should be greater than 1"
        ):
            split_df_train_val_test(df, val_size=0, test_size=0)


class TestSaveDf:

    @patch("src.dataset.logger")
    def test_save(self, mock_logger, tmp_path):
        df = pd.DataFrame({"data": [1, 2, 3]})
        path = os.path.join(tmp_path, "file.csv.gz")

        save_df(df, path)
        mock_logger.warning.assert_not_called()

    @patch("src.dataset.logger")
    def test_overwrite(self, mock_logger, tmp_path):
        df = pd.DataFrame({"data": [1, 2, 3]})
        path = os.path.join(tmp_path, "file.csv.gz")

        save_df(df, path)
        save_df(df, path)
        mock_logger.warning.assert_called_once_with(f"Overwriting {path}")


class TestLoadDatasetFromDf:
    def test_load_nonexistent_file(self):
        with pytest.raises(AssertionError, match="does not exist"):
            load_dataset_from_df("nonexistent.csv.gz")

    def test_load_valid_file(self, tmp_path):
        df = pd.DataFrame({"text": ["hello"], "label": [1]})
        path = os.path.join(tmp_path, "file.csv.gz")

        df.to_csv(path, index=False, compression="gzip")
        dataset = load_dataset_from_df(path)
        assert isinstance(dataset, Dataset)
        assert dataset["text"] == ["hello"]
        assert dataset["label"] == [1]


class TestApplyTokenizeWrapper:
    def test_tokenizer_wrapper(self):
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)

        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

        wrapper = apply_tokenize_wrapper(mock_tokenizer)

        dataset = Dataset.from_dict({"text": ["sample text"]})

        result = wrapper(dataset)

        mock_tokenizer.assert_called_once()
        args, kwargs = mock_tokenizer.call_args

        assert kwargs.get("truncation")

        assert dataset["text"] == ["sample text"]
        assert "input_ids" in result
        assert result["input_ids"] == [[1, 2, 3]]


class TestTokenizeDs:
    @patch("src.dataset.apply_tokenize_wrapper")
    def test_tokenization(self, mock_wrapper):
        mock_tokenizer = Mock()

        def mock_tokenize_func(dataset):
            texts = dataset["text"]
            return {
                "input_ids": [[1, 2, 3] for _ in texts],
                "attention_mask": [[1, 1, 1] for _ in texts],
            }

        mock_wrapper.return_value = mock_tokenize_func

        ds = Dataset.from_dict({"text": ["sample"]})
        result = tokenize_ds(ds, mock_tokenizer)

        mock_wrapper.assert_called_with(mock_tokenizer)

        assert "input_ids" in result.features
        assert "attention_mask" in result.features
