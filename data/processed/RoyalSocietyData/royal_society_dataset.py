from typing import Dict
from pathlib import Path
import json
import csv
from torch.utils.data import Dataset
import os 
from typing import Optional, List, Any
from tqdm import tqdm
import logging
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dataset_stats import print_dataset_statistics, plot_dataset_statistics


logger = logging.getLogger(__name__)

class RoyalSocietyDataset(Dataset):
    def __init__(self, 
                 meta_path: str,
                 txt_dir: str,
                 encoding: str = 'utf-8',
                 verbose: bool = False,
                 sample_count: int = None,
                 specific_comp_range: bool = False,
                 return_as_labels: bool = False):
        self.samples = []
        self.txt_dir = Path(txt_dir)
        self.encoding = encoding
        self.verbose = verbose
        self.sample_count = sample_count
        self.specific_comp_range = specific_comp_range
        self._unique_date_ranges = set()
        self.return_as_labels = return_as_labels

        meta_path = Path(meta_path)

        # Read ...meta.tsv
        with meta_path.open(encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter="\t")

            for it, row in tqdm(enumerate(reader), desc="Loading Royal Society texts", total=sample_count):
                if self.sample_count is not None and it >= sample_count:
                    break
                path_suffix = row.get("id").strip()
                comp_date = int(row.get("year").strip())
                txt_path = self.txt_dir / f'Royal_Society_Corpus_open_v6.0_text_{path_suffix}.txt'
                if txt_path.is_file():
                    with txt_path.open(encoding=self.encoding) as txt_file:
                        text = txt_file.read()
                else:
                    logger.warning(f"Skipping {txt_path} because it doesn't exist")
                    continue

                sample = {"text": text, "comp_date": (comp_date, comp_date)}
                if not self.specific_comp_range:
                    sample["comp_date"] = (comp_date // 10) * 10
                    self._unique_date_ranges.add(sample["comp_date"])
                self.samples.append(sample)
                
        logger.info(f"Loaded {len(self.samples)} text samples from Royal Society dataset")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if idx >= len(self.samples):
            raise IndexError("Index out of range")
        sample = self.samples[idx].copy()
        if self.return_as_labels:
            # Return a one-hot encoding of the comp_date with respect to unique_date_ranges
            comp_date = sample["comp_date"]
            unique_ranges = self.unique_date_ranges
            one_hot = [1 if comp_date == d else 0 for d in unique_ranges]
            return sample["text"], one_hot

        return sample

    @property
    def unique_date_ranges(self):
        return sorted(list(self._unique_date_ranges))

    @classmethod
    def load_royal_society_dataset(cls, cfg, base_path: str = None) -> Optional[List[Dict[str, Any]]]:
        """Load Royal Society dataset with configuration parameters"""
        raw_data_path = "data/raw/RoyalSocietyData/"
        if base_path:
            raw_data_path = base_path + raw_data_path
        # Get paths from config with fallbacks
        meta_path = cfg.data.get("meta_path", raw_data_path + "Royal_Society_Corpus_open_v6.0_meta.tsv")
        txt_dir = cfg.data.get("txt_dir", raw_data_path + "Royal_Society_Corpus_open_v6.0_texts_txt")
        
        # Get other parameters
        encoding = cfg.data.get("encoding", "utf-8")
        verbose = cfg.data.get("verbose", False)
        specific_comp_range = cfg.data.get("specific_comp_range", False)
        sample_count = cfg.data.get("sample_count", None)
        
        return cls(
            meta_path=meta_path,
            txt_dir=txt_dir,
            encoding=encoding,
            verbose=verbose,
            sample_count=sample_count,
            specific_comp_range=specific_comp_range
        )

if __name__ == "__main__":
    raw_data_path = "data/raw/RoyalSocietyData/"
    dataset = RoyalSocietyDataset(
        meta_path=raw_data_path + "Royal_Society_Corpus_open_v6.0_meta.tsv",
        txt_dir=raw_data_path + "Royal_Society_Corpus_open_v6.0_texts_txt"
    )
    print_dataset_statistics(dataset)
    plot_dataset_statistics(dataset)
    
