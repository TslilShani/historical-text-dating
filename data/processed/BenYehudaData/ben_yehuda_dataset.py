from typing import Dict
from pathlib import Path
import json
import csv
from torch.utils.data import Dataset
import os 
from typing import Optional, List, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dataset_stats import print_dataset_statistics, plot_dataset_statistics


class BenYehudaDataset(Dataset):
    def __init__(self, 
                 pseudocatalogue_path: str,
                 authors_dir: str,
                 txt_dir: str,
                 encoding: str = 'utf-8',
                 verbose: bool = False,
                 specific_comp_range: bool = False):
        self.samples = []
        self.author_years = {}
        self.txt_dir = Path(txt_dir)
        self.encoding = encoding
        self.verbose = verbose
        self.specific_comp_range = specific_comp_range

        authors_dir = Path(authors_dir)
        pseudocatalogue_path = Path(pseudocatalogue_path)

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
                            print(f"Invalid year format for author `{author_name}` #{author_id}: {birth}, {death}")
                        continue
                    self.author_years[author_name] = (int(birth), int(death))
        if self.verbose:
            print(f"Could not parse birth or death years for {invalid_years_counter} authors")

        # Read pseudocatalogue.csv
        with pseudocatalogue_path.open(encoding=encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
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
                    self.samples.append({"text": text, "comp_date": self.author_years[author_name]})
                else:
                    pass
                    # print(f"Skipping {txt_path} because it doesn't exist or author_id {author_id} is not in author_years")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if idx >= len(self.samples):
            raise IndexError("Index out of range")
        sample = self.samples[idx].copy()
        if not self.specific_comp_range:
            comp_date_start, comp_date_end = sample["comp_date"]
            sample["comp_date"] = (comp_date_end // 10) * 10
        return sample

    @classmethod
    def load_ben_yehuda_dataset(cls, cfg) -> Optional[List[Dict[str, Any]]]:
        """Load Ben Yehuda dataset with configuration parameters"""
        raw_data_path = "data/raw/BenYehudaData/"
        # Get paths from config with fallbacks
        pseudocatalogue_path = cfg.data_ben_yehuda.get("pseudocatalogue_path", raw_data_path + "public_domain_dump-2025-03/pseudocatalogue.csv")
        authors_dir = cfg.data_ben_yehuda.get("authors_dir", raw_data_path + "scraper/benyehuda_data/authors")
        txt_dir = cfg.data_ben_yehuda.get("txt_dir", raw_data_path + "public_domain_dump-2025-03/txt")
        
        # Get other parameters
        encoding = cls.cfg.data.get("encoding", "utf-8")
        verbose = cls.cfg.data.get("verbose", False)
        specific_comp_range = cls.cfg.data.get("specific_comp_range", False)
        
        return cls(
            pseudocatalogue_path=pseudocatalogue_path,
            authors_dir=authors_dir,
            txt_dir=txt_dir,
            encoding=encoding,
            verbose=verbose,
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
    
