# Ben Yehuda Project Scraper

This is a Python scraper for the Ben Yehuda Project API (https://benyehuda.org/). It allows you to download works and their content from the project's digital library.

## Features

- Scrapes works (manifestations) from the Ben Yehuda Project
- Downloads both work details and content
- Saves data in JSON format
- Includes logging and progress tracking
- Implements rate limiting to be respectful to the API

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
from scraper import BenYehudaScraper

# Initialize the scraper
scraper = BenYehudaScraper(output_dir="benyehuda_data")

# Scrape 10 works (change the number as needed)
scraper.scrape_works(num_works=10)
```

The scraper will:
- Create an output directory for the data
- Save each work as a separate JSON file
- Create a log file with scraping progress and any errors
- Show a progress bar during scraping

## Output Structure

The scraper creates the following structure:
```
benyehuda_data/
├── works/
│   ├── work_1.json
│   ├── work_2.json
│   └── ...
└── scraper.log
```

Each work JSON file contains:
- `details`: Metadata about the work
- `content`: The actual content of the work

## Rate Limiting

The scraper includes a 0.5-second delay between requests to avoid overwhelming the API. Please be respectful of the Ben Yehuda Project's resources when using this scraper.

## Error Handling

- All errors are logged to `scraper.log`
- The scraper will continue running even if individual works fail to download
- Failed downloads are logged with their error messages 