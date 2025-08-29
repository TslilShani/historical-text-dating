import requests
import json
from typing import Dict, Optional, List, Iterator, Set
import os
from dotenv import load_dotenv
from pathlib import Path
import time
from tqdm import tqdm
import logging

class BenYehudaAuthorScraper:
    """API client for scraping authors from the Ben Yehuda Project."""
    
    BASE_URL = "https://benyehuda.org/api/v1"
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "benyehuda_data"):
        """Initialize the API client.
        
        Args:
            api_key: API key for Ben Yehuda Project. If not provided, will look for BENYEHUDA_API_KEY env variable.
            output_dir: Directory to save the scraped authors
        """
        load_dotenv()  # Load environment variables from .env file
        self.api_key = api_key or os.getenv('BENYEHUDA_API_KEY')
        if not self.api_key:
            raise ValueError("API key is required. Either pass it to the constructor or set BENYEHUDA_API_KEY environment variable.")
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.authors_dir = self.output_dir / "authors"
        self.authors_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'author_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_author(self, author_id: int, detail: str = 'enriched') -> Dict:
        """Get details of a specific author.
        
        Args:
            author_id: The ID of the author to retrieve
            detail: One of 'metadata', 'texts', or 'enriched'
            
        Returns:
            Dict containing the author data
        """
        params = {
            'key': self.api_key,
            'author_detail': detail
        }
        
        response = requests.get(f"{self.BASE_URL}/authorities/{author_id}", params=params)
        response.raise_for_status()
        return response.json()

    def collect_author_ids(self) -> Set[int]:
        """Collect all author IDs from the works directory.
        
        Returns:
            Set[int]: Set of unique author IDs
        """
        author_ids = set()
        works_dir = self.output_dir / "works"
        
        if not works_dir.exists():
            self.logger.warning("Works directory not found. Make sure to run the works scraper first.")
            return author_ids
            
        self.logger.info("Collecting author IDs from works...")
        
        for work_file in works_dir.glob("work_*.json"):
            try:
                with open(work_file, 'r', encoding='utf-8') as f:
                    work = json.load(f)
                    # Get author IDs from metadata
                    if 'metadata' in work and 'author_ids' in work['metadata']:
                        author_ids.update(work['metadata']['author_ids'])
            except Exception as e:
                self.logger.error(f"Error processing {work_file}: {str(e)}")
                continue
                
        return author_ids

    def save_author(self, author: Dict) -> None:
        """Save an author to file.
        
        Args:
            author: Author data to save
        """
        author_id = author.get('id')
        if not author_id:
            self.logger.warning("Author has no ID, skipping")
            return
            
        author_file = self.authors_dir / f"author_{author_id}.json"
        try:
            with open(author_file, 'w', encoding='utf-8') as f:
                json.dump(author, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving author {author_id}: {str(e)}")

    def scrape_all_authors(self) -> None:
        """Scrape all authors from the Ben Yehuda Project."""
        self.logger.info("Starting to scrape authors...")
        
        # Get author IDs from works
        author_ids = self.collect_author_ids()
        total_authors = len(author_ids)
        self.logger.info(f"Found {total_authors} authors to scrape")
        
        # Create progress file to track progress
        progress_file = self.output_dir / "authors_progress.json"
        scraped_ids = set()
        
        # Load progress if exists
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    scraped_ids = set(json.load(f))
                self.logger.info(f"Loaded {len(scraped_ids)} previously scraped authors")
            except Exception as e:
                self.logger.error(f"Error loading progress file: {str(e)}")
        
        try:
            for author_id in tqdm(author_ids, desc="Scraping authors"):
                if author_id in scraped_ids:
                    self.logger.debug(f"Author {author_id} already scraped, skipping")
                    continue
                
                try:
                    # Get author details
                    author_data = self.get_author(author_id)
                    self.save_author(author_data)
                    scraped_ids.add(author_id)
                    
                    # Update progress file periodically
                    if len(scraped_ids) % 10 == 0:
                        with open(progress_file, 'w', encoding='utf-8') as f:
                            json.dump(list(scraped_ids), f)
                    
                except Exception as e:
                    self.logger.error(f"Error processing author {author_id}: {str(e)}")
                    continue
                
                # Be nice to the API
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            self.logger.info("Scraping interrupted by user")
        finally:
            # Save final progress
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(list(scraped_ids), f)
            
            self.logger.info(f"Finished scraping. Total authors scraped: {len(scraped_ids)}")

def main():
    """Example usage of the API client."""
    try:
        # Initialize the client
        client = BenYehudaAuthorScraper()
        
        # Scrape all authors
        client.scrape_all_authors()
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 