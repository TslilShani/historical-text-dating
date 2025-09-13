"""
Data loading utilities for historical text dating
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import ConcatDataset, Dataset
from omegaconf import DictConfig
import torch

from data.processed.BenYehudaData.ben_yehuda_dataset import BenYehudaDataset
from data.processed.SefariaData.sefaria_dataset import SefariaDataset

# Add the processed data directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "data" / "processed"))

logger = logging.getLogger(__name__)


class FilteredDataset(Dataset):
    """Dataset that applies filtering to another dataset"""

    def __init__(self, base_dataset: Dataset, cfg: DictConfig):
        self.base_dataset = base_dataset
        self.cfg = cfg

        # Filter indices based on configuration
        self.valid_indices = self._get_valid_indices()

    def _get_valid_indices(self):
        """Get indices of samples that pass the filtering criteria"""
        min_text_length = self.cfg.data.get("min_text_length", 50)
        max_text_length = self.cfg.data.get("max_text_length", 2000)
        filter_by_date_range = self.cfg.data.get("filter_by_date_range", True)
        min_date = self.cfg.data.get("min_date", 1000)
        max_date = self.cfg.data.get("max_date", 2024)

        valid_indices = []

        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset[idx]

            # Check if sample has required fields
            if not sample.get("text") or not sample.get("comp_date"):
                continue

            text = sample["text"]
            date = sample["comp_date"]

            # Handle date range - could be tuple or single value
            if isinstance(date, tuple):
                date_value = date[1] if len(date) > 1 else date[0]
            else:
                date_value = date

            # Filter by text length
            if len(text) < min_text_length or len(text) > max_text_length:
                continue

            # Filter by date range
            if filter_by_date_range and (
                date_value < min_date or date_value > max_date
            ):
                continue

            valid_indices.append(idx)

        logger.info(
            f"Filtered dataset: {len(self.base_dataset)} -> {len(valid_indices)} samples"
        )
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        return self.base_dataset[actual_idx]


class SplitDataset(Dataset):
    """Dataset that provides a subset of another dataset based on indices"""

    def __init__(self, base_dataset: Dataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx]


class TokenizedDataset(Dataset):
    """Dataset wrapper that handles tokenization"""

    def __init__(
        self, dataset: Dataset, tokenizer, unique_date_ranges, max_length: int = 512
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.unique_date_ranges = (
            unique_date_ranges if unique_date_ranges else list(range(1000, 2025, 25))
        )  # 25-year bins from 1000 to 2000

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        text = sample["text"]
        date = sample["comp_date"]

        # Handle date range - could be tuple or single value
        if isinstance(date, tuple):
            date_value = date[1] if len(date) > 1 else date[0]
        else:
            date_value = date

        # Create one-hot encoding for the date
        one_hot = torch.zeros(len(self.unique_date_ranges), dtype=torch.float)
        try:
            # Find the index of the date in unique_date_ranges
            if date_value in self.unique_date_ranges:
                date_idx = self.unique_date_ranges.index(date_value)
                one_hot[date_idx] = 1.0
            else:
                # If exact date not found, find the closest one
                closest_idx = min(
                    range(len(self.unique_date_ranges)),
                    key=lambda i: abs(self.unique_date_ranges[i] - date_value),
                )
                one_hot[closest_idx] = 1.0
        except Exception as e:
            logger.warning(
                f"Error creating one-hot encoding for date {date_value}: {e}"
            )
            raise e

        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",  # Use max_length instead of True
            truncation=True,
            max_length=self.max_length,
        )

        return inputs["input_ids"].squeeze(), one_hot


class SampleDataset(Dataset):
    """Sample dataset for testing when real data is not available"""

    def __init__(self):
        self.samples = [
            {
                "text": "In the year 1800, the industrial revolution was beginning to transform society.",
                "comp_date": 1800,
            },
            {
                "text": "The 1850s marked a period of significant social and political change.",
                "comp_date": 1850,
            },
            {
                "text": "By 1900, the world had entered a new century with great technological advances.",
                "comp_date": 1900,
            },
            {
                "text": "The 1950s were characterized by post-war reconstruction and economic growth.",
                "comp_date": 1950,
            },
            {
                "text": "In the year 2000, the digital age was in full swing.",
                "comp_date": 2000,
            },
            {
                "text": "The 1700s saw the Enlightenment period flourish across Europe.",
                "comp_date": 1700,
            },
            {
                "text": "During the 1600s, scientific discoveries were revolutionizing human understanding.",
                "comp_date": 1600,
            },
            {
                "text": "The 1500s marked the beginning of the Renaissance period.",
                "comp_date": 1500,
            },
            {
                "text": "In 1825, the first passenger railway opened in England.",
                "comp_date": 1825,
            },
            {
                "text": "The year 1875 saw the invention of the telephone.",
                "comp_date": 1875,
            },
            {
                "text": "By 1925, radio broadcasting had become widespread.",
                "comp_date": 1925,
            },
            {
                "text": "The 1975s brought the personal computer revolution.",
                "comp_date": 1975,
            },
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DataLoader:
    """Main data loading class that handles different dataset types"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dataset_name = cfg.data.get("dataset_name", "ben_yehuda")
        self.max_length = cfg.data.get("max_length", 512)
        self.unique_date_ranges = []

    def load_base_dataset(self) -> Dataset:
        """Load the base dataset based on configuration"""
        logger.info(f"Loading {self.dataset_name} dataset...")

        # Load dataset based on configuration
        if self.dataset_name == "ben_yehuda":
            dataset = BenYehudaDataset.load_ben_yehuda_dataset(self.cfg)
        elif self.dataset_name == "sefaria":
            dataset = SefariaDataset.load_sefaria_dataset(self.cfg)
        elif self.dataset_name == "all":
            sefaria_dataset: SefariaDataset = SefariaDataset.load_sefaria_dataset(
                self.cfg
            )
            ben_yehuda_dataset: BenYehudaDataset = (
                BenYehudaDataset.load_ben_yehuda_dataset(self.cfg)
            )
            # A trick to unite the labels from both datasets
            self.unique_date_ranges = sorted(
                ben_yehuda_dataset.unique_date_ranges
                + sefaria_dataset.unique_date_ranges
            )
            dataset = ConcatDataset([sefaria_dataset, ben_yehuda_dataset])
        else:
            logger.error(f"Unknown dataset: {self.dataset_name}")
            dataset = None

        # Fall back to sample data if real dataset failed to load
        if dataset is None:
            logger.warning("Failed to load real dataset, using sample data...")
            dataset = SampleDataset()

        return dataset

    def split_dataset_indices(
        self, dataset: Dataset
    ) -> Tuple[List[int], List[int], List[int]]:
        """Split dataset indices into train/eval/test splits"""
        train_ratio = self.cfg.data.get("train_ratio", 0.8)
        eval_ratio = self.cfg.data.get("eval_ratio", 0.1)

        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        eval_size = int(eval_ratio * total_size)

        # Create indices
        indices = list(range(total_size))

        train_indices = indices[:train_size]
        eval_indices = indices[train_size : train_size + eval_size]
        test_indices = indices[train_size + eval_size :]

        logger.info(
            f"Dataset split: {len(train_indices)} train, {len(eval_indices)} eval, {len(test_indices)} test"
        )

        return train_indices, eval_indices, test_indices

    def load_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Main method to load and prepare datasets for training"""
        # Load base dataset
        base_dataset = self.load_base_dataset()

        # Apply filtering
        filtered_dataset = FilteredDataset(base_dataset, self.cfg)

        if len(filtered_dataset) == 0:
            logger.error("No samples remaining after filtering!")
            raise ValueError("No samples available for training")

        # Split dataset
        train_indices, eval_indices, test_indices = self.split_dataset_indices(
            filtered_dataset
        )

        # Create split datasets
        train_dataset = SplitDataset(filtered_dataset, train_indices)
        eval_dataset = (
            SplitDataset(filtered_dataset, eval_indices) if eval_indices else None
        )

        return train_dataset, eval_dataset

    def create_tokenized_datasets(
        self, tokenizer
    ) -> Tuple[TokenizedDataset, Optional[TokenizedDataset]]:
        """Create tokenized datasets with the provided tokenizer"""
        train_dataset, eval_dataset = self.load_datasets()

        train_tokenized = TokenizedDataset(
            train_dataset, tokenizer, self.unique_date_ranges, self.max_length
        )
        eval_tokenized = (
            TokenizedDataset(
                eval_dataset, tokenizer, self.unique_date_ranges, self.max_length
            )
            if eval_dataset
            else None
        )

        return train_tokenized, eval_tokenized
