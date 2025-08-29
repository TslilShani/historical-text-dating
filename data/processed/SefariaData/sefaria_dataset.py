from pathlib import Path
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import re
from typing import Optional
import re

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dataset_stats import print_dataset_statistics, plot_dataset_statistics


class SefariaDataset(Dataset):
    def __init__(self, 
                 sefaria_export_path: str,
                 encoding: str = 'utf-8',
                 sample_count: Optional[int]=None):
        """
        Initialize Sefaria dataset.
        
        Args:
            sefaria_export_path: Path to the Sefaria-Export-master directory
            encoding: File encoding (default: utf-8)
        """
        self.samples = []
        self.text_metadata = {}
        self.sefaria_path = Path(sefaria_export_path)
        self.encoding = encoding
        self.sample_count = sample_count
        
        # Load all schema files to get metadata
        self._load_schemas()
        
        # Load all merged.json files to get text content
        self._load_texts()
        
        print(f"Loaded {len(self.samples)} text samples from Sefaria dataset")

    def _load_schemas(self):
        """Load all schema files to extract metadata about texts."""
        schemas_dir = self.sefaria_path / "schemas"
        
        if not schemas_dir.exists():
            print(f"Warning: Schemas directory not found at {schemas_dir}")
            return
            
        for schema_file in schemas_dir.glob("*.json"):
            try:
                with schema_file.open(encoding=self.encoding) as f:
                    schema_data = json.load(f)
                    
                # Extract key metadata
                title = schema_data.get('title', 'Unknown')
                he_title = schema_data.get('heTitle', '')
                
                # Get composition date (use first date if multiple)
                comp_dates = schema_data.get('compDate', [])
                if len(comp_dates) == 1:
                    comp_date = (comp_dates[0], comp_dates[0])
                elif len(comp_dates) == 2:
                    comp_date = (comp_dates[0], comp_dates[1])
                elif len(comp_dates) > 2:
                    comp_date = None
                    print("Weird data at file {schema_file}: {comp_dates}")
                else:
                    comp_date = None
                
                # Store metadata
                self.text_metadata[title] = {
                    'he_title': he_title,
                    'comp_date': comp_date,
                    'schema_file': schema_file.name
                }
                
            except Exception as e:
                print(f"Error loading schema {schema_file}: {e}")
                continue

    def _load_texts(self):
        """Load all merged.json files to extract text content."""
        json_dir = self.sefaria_path / "json"
        
        if not json_dir.exists():
            print(f"Warning: JSON directory not found at {json_dir}")
            return
            
        # Recursively find all merged.json files
        merged_files = list(json_dir.rglob("**/merged.json"))
        if self.sample_count:
            merged_files = merged_files[:self.sample_count]
        
        for merged_file in tqdm(merged_files, desc="Loading Sefaria texts"):
            if "Hebrew" not in str(merged_file.parent):
                continue
            try:
                with merged_file.open(encoding=self.encoding) as f:
                    text_data = json.load(f)
                
                # Extract text content
                title = text_data.get('title', 'Unknown')
                text_content = text_data.get('text', {})
                
                # Skip if no text content
                if not text_content:
                    continue
                
                # Flatten text content into paragraphs
                paragraphs = []
                self._extract_paragraphs(text_content, paragraphs)
                
                if not paragraphs:
                    continue
                
                # Get metadata for this text
                metadata = self.text_metadata.get(title, {})
                
                # Create sample
                sample = {
                    'title': title,
                    'he_title': metadata.get('he_title', ''),
                    'paragraphs': paragraphs,
                    'text': '\n'.join(list(p['text'] for p in paragraphs)),
                    'comp_date': metadata.get('comp_date'),
                    'file_path': str(merged_file)
                }
                
                self.samples.append(sample)
                
            except Exception as e:
                print(f"Error loading merged file {merged_file}: {e}")
                continue

    def _extract_paragraphs(self, text_content, paragraphs, current_path=""):
        """Recursively extract paragraphs from nested text structure."""
        if isinstance(text_content, list):
            # This is a list of paragraphs
            for i, paragraph in enumerate(text_content):
                if isinstance(paragraph, str) and paragraph.strip():
                    # Clean the paragraph text
                    cleaned_text = self._clean_text(paragraph)
                    if cleaned_text:
                        paragraphs.append({
                            'text': cleaned_text,
                            'path': f"{current_path}[{i}]" if current_path else f"[{i}]"
                        })
                elif isinstance(paragraph, list):
                    # Nested list, recurse
                    self._extract_paragraphs(paragraph, paragraphs, 
                                          f"{current_path}[{i}]" if current_path else f"[{i}]")
        elif isinstance(text_content, dict):
            # This is a dictionary with sections
            for key, value in text_content.items():
                new_path = f"{current_path}.{key}" if current_path else key
                self._extract_paragraphs(value, paragraphs, new_path)

    def _clean_text(self, text):
        """Clean and normalize text content."""
        if not isinstance(text, str):
            print("Should not be here")
            return ""

        # Warning - trying to remove html tag by regex even though not possible 
        # I just didn't want to pip install some big library

        # Images that has 'alt' sections are specifically annoying since something like that:
        # <img alt="bla <b>bla</b>" src="link">
        # Will mess up the regex. These are quite common as a full scentence and we prefer to maybe
        # lose some data
        if text.startswith("<") and text.endswith(">") and "img" in text and "alt" in text:
            return ""

        # Remove long HTML tags as span and i with some properties inside
        for t in ["sup", "span", "i", "img", "a"]:
            text = re.sub(rf'<\s*{t}\b[^>]*>', ' ', text, flags=re.IGNORECASE)
            text = re.sub(rf'<\s*/\s*{t}\s*>', ' ', text, flags=re.IGNORECASE)
        
        # Remove any remaining simple HTML tags <...> - max length of 8 for safety
        # Sometimes there are hebrew comments inside - be careful not to remove them using regex
        text = re.sub(r'<[A-Za-z0-9\s/\-_=:\"\'\?^>]{,8}>', '', text)

        # Stupied human mistakes (checked by hand - not a regex problem)
        text = text.replace("</b", " ")
        text = text.replace("br>", " ")
        text = text.replace("<br", " ")
        text = text.replace("sup>", " ")

        # Remaning tags (about 20 over all) - simply removed
        text = text.replace(">", " ")
        text = text.replace("<", " ")

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        return text

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError("Index out of range")
        
        sample = self.samples[idx]
        
        # Return text content and metadata
        return {
            'text': sample['text'],
            'title': sample['title'],
            'he_title': sample['he_title'],
            'comp_date': sample['comp_date'],
            'file_path': sample['file_path']
        }

    def get_texts_by_date_range(self, start_year, end_year):
        """Get all texts within a date range."""
        results = []
        for sample in self.samples:
            comp_date = sample['comp_date']
            
            if comp_date and start_year <= comp_date[0] <= end_year:
                results.append(sample)
        
        return results


if __name__ == "__main__":
    raw_data_path = "data/raw/SefariaData/"
    dataset = SefariaDataset(
        sefaria_export_path=raw_data_path + "Sefaria-Export-master",
    )
    
    print_dataset_statistics(dataset)
    plot_dataset_statistics(dataset)
    