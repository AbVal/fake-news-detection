import pytest
import pandas as pd
import os
import tempfile
import shutil
import argparse
from unittest.mock import patch
from prepare_dataset import main as prepare_main


class TestDataset:
    @pytest.fixture(scope="class")
    def dataset_path(self):
        """Fixture to create temporary directory and prepare dataset"""
        tmpdir = tempfile.mkdtemp()

        # Mock command line arguments
        with patch("prepare_dataset.argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = argparse.Namespace(
                data_path=tmpdir, val_size=0.2, test_size=0.1, random_state=42
            )
            prepare_main()

        yield tmpdir
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def processed_data(self, dataset_path):
        """Fixture to load processed datasets"""
        train_df = pd.read_csv(os.path.join(dataset_path, "train.csv.gz"))
        val_df = pd.read_csv(os.path.join(dataset_path, "val.csv.gz"))
        test_df = pd.read_csv(os.path.join(dataset_path, "test.csv.gz"))
        return train_df, val_df, test_df

    def test_processed_dataset_columns(self, processed_data):
        train_df, val_df, test_df = processed_data

        expected_columns = ["text", "label"]
        for df in [train_df, val_df, test_df]:
            assert all(col in df.columns for col in expected_columns)
            assert not df.empty, "DataFrame should not be empty"

    def test_text_statistics(self, processed_data):
        train_df, val_df, test_df = processed_data

        for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            text_lengths = df["text"].str.len()
            avg_length = text_lengths.mean()

            assert avg_length > 20, f"Average text length in {name} too short"
            assert text_lengths.min() > 0, f"Empty texts found in {name}"

            assert text_lengths.std() > 0, f"Text lengths in {name} have no variance"

    def test_label_distribution(self, processed_data):
        train_df, val_df, test_df = processed_data

        for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            unique_labels = df["label"].unique()
            assert set(unique_labels).issubset({0, 1}), f"Invalid labels in {name}"

            label_dist = df["label"].value_counts(normalize=True)
            assert all(
                0.1 < p < 0.9 for p in label_dist
            ), f"Extremely unbalanced labels in {name}: {label_dist.to_dict()}"

    def test_text_language(self, processed_data):
        """Test that texts are primarily English"""
        train_df, val_df, test_df = processed_data
        english_words = {"the", "be", "to", "of", "and", "a", "in", "that", "have"}

        for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:

            english_ratio = sum(
                any(word in text.lower() for word in english_words)
                for text in df["text"]
            ) / len(df["text"])

            assert (
                english_ratio > 0.9
            ), f"Low english texts ratio in {name}: {english_ratio:.2f}"

    def test_data_splits(self, processed_data):
        train_df, val_df, test_df = processed_data

        train_texts = set(train_df["text"].str.lower().str.strip())
        val_texts = set(val_df["text"].str.lower().str.strip())
        test_texts = set(test_df["text"].str.lower().str.strip())

        assert train_texts.isdisjoint(val_texts), "Train and val splits overlap"
        assert train_texts.isdisjoint(test_texts), "Train and test splits overlap"
        assert val_texts.isdisjoint(test_texts), "Val and test splits overlap"

    def test_data_persistence(self, dataset_path):
        """Test that data files are properly saved"""
        expected_files = ["train.csv.gz", "val.csv.gz", "test.csv.gz"]
        for file in expected_files:
            file_path = os.path.join(dataset_path, file)
            assert os.path.exists(file_path), f"File {file} was not created"
            assert os.path.getsize(file_path) > 0, f"File {file} is empty"
