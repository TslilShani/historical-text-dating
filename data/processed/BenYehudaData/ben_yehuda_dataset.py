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

class BenYehudaDataset(Dataset):
    def __init__(self, 
                 pseudocatalogue_path: str,
                 authors_dir: str,
                 txt_dir: str,
                 encoding: str = 'utf-8',
                 verbose: bool = False,
                 sample_count: int = None,
                 specific_comp_range: bool = False,
                 return_as_labels: bool = False):
        self.samples = []
        self.author_years = {}
        self.txt_dir = Path(txt_dir)
        self.encoding = encoding
        self.verbose = verbose
        self.sample_count = sample_count
        self.specific_comp_range = specific_comp_range
        self._unique_date_ranges = set()
        self.return_as_labels = return_as_labels

        authors_dir = Path(authors_dir)
        pseudocatalogue_path = Path(pseudocatalogue_path)

        logger.debug("Loading author files")
        start_load_time = time.time()
        invalid_years_counter = 0
        # Load author birth/death years
        for author_file in authors_dir.glob('author_*.json'):
            with author_file.open(encoding=encoding) as f:
                data = json.load(f)
                author_id = int(data['id'])
                metadata = data['metadata']
                person = metadata.get('person', {})
                author_name = metadata.get('name')
                birth = person.get('birth_year')
                death = person.get('death_year')
                if birth and death:
                    try:
                        birth = int(birth)
                        death = int(death)
                    except ValueError:
                        invalid_years_counter += 1
                        if self.verbose:
                            logger.warning(f"Invalid year format for author `{author_name}` #{author_id}: {birth}, {death}")
                        continue
                    self.author_years[author_name] = (int(birth), int(death))
        if self.verbose:
            logger.warning(f"Could not parse birth or death years for {invalid_years_counter} authors")
        logger.debug(f"Loaded author files (tool {time.time()-start_load_time}s)")

        # Read pseudocatalogue.csv
        with pseudocatalogue_path.open(encoding=encoding) as f:
            reader = csv.DictReader(f)

            for it, row in tqdm(enumerate(reader), desc="Loading Ben Yehuda texts", total=sample_count):
                if self.sample_count is not None and it >= sample_count:
                    break
                path = row.get('path', '').strip()
                author_name = row.get("authors")
                author_id = int(path.split('/')[1][1:])
                if not author_name or not path:
                    continue
                if path.startswith('/'):
                    path = path[1:]
                txt_path = self.txt_dir / f'{path}.txt'
                if author_name in self.author_years and txt_path.is_file():
                    with txt_path.open(encoding=self.encoding) as txt_file:
                        text = txt_file.read()
                    sample = {"text": text, "comp_date": self.author_years[author_name]}
                    if not self.specific_comp_range:
                        comp_date_start, comp_date_end = sample["comp_date"]
                        sample["comp_date"] = (comp_date_end // 10) * 10
                        self._unique_date_ranges.add(sample["comp_date"])
                    self.samples.append(sample)
                else:
                    logger.debug(f"Skipping {txt_path} because it doesn't exist or author_id {author_id} is not in author_years")

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
    def load_ben_yehuda_dataset(cls, cfg, base_path: str = None) -> Optional[List[Dict[str, Any]]]:
        """Load Ben Yehuda dataset with configuration parameters"""
        raw_data_path = "data/raw/BenYehudaData/"
        if base_path:
            raw_data_path = base_path + raw_data_path
        # Get paths from config with fallbacks
        pseudocatalogue_path = cfg.data.get("pseudocatalogue_path", raw_data_path + "public_domain_dump-2025-03/pseudocatalogue.csv")
        authors_dir = cfg.data.get("authors_dir", raw_data_path + "scraper/benyehuda_data/authors")
        txt_dir = cfg.data.get("txt_dir", raw_data_path + "public_domain_dump-2025-03/txt")
        
        # Get other parameters
        encoding = cfg.data.get("encoding", "utf-8")
        verbose = cfg.data.get("verbose", False)
        specific_comp_range = cfg.data.get("specific_comp_range", False)
        sample_count = cfg.data.get("sample_count", None)
        
        return cls(
            pseudocatalogue_path=pseudocatalogue_path,
            authors_dir=authors_dir,
            txt_dir=txt_dir,
            encoding=encoding,
            verbose=verbose,
            sample_count=sample_count,
            specific_comp_range=specific_comp_range
        )

if __name__ == "__main__":
    raw_data_path = "data/raw/BenYehudaData/"
    dataset = BenYehudaDataset(
        pseudocatalogue_path=raw_data_path + "public_domain_dump-2025-03/pseudocatalogue.csv",
        authors_dir=raw_data_path + "scraper/benyehuda_data/authors",
        txt_dir=raw_data_path + "public_domain_dump-2025-03/txt"
    )
    print_dataset_statistics(dataset)
    plot_dataset_statistics(dataset)
    
