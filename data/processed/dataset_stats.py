import matplotlib.pyplot as plt
import torch
from typing import Union, List, Tuple, Dict, Any



def print_dataset_statistics(dataset):
    """Print statistics of a dataset."""
    
    print(f"Number of text samples: {len(dataset)}")
    
    # Basic counts
    total_chars = sum(len(sample['text']) for sample in dataset)
    
    # Estimated year statistics
    comp_years_all = [sample['comp_date'] for sample in dataset if sample['comp_date']]
    
    print(f"\nText Content:")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Average characters per text: {total_chars/len(dataset):,.0f}")
    
    print(f"\nComposition Year Estimations Statistics:")
    print(f"  Range: {min(comp_years_all)} - {max(comp_years_all)}")
    print(f"  Mean: {sum(comp_years_all)/len(comp_years_all):.0f}")


def plot_dataset_statistics(dataset):
    """Plot statistics dataset."""
    
    # Prepare data for plotting
    comp_years = []
    text_lengths = []
    year_differences = []
    
    for sample in dataset:
        # Collect composition years
        if sample['comp_date']:
            comp_years.append(sample['comp_date'])
            # Calculate difference between start and end years
            # year_diff = sample['comp_date'][1] - sample['comp_date'][0]
            # year_differences.append(year_diff)
        
        # Collect text lengths
        total_length = len(sample['text'])
        text_lengths.append(total_length)
    
    # Create plots
    plt.figure(figsize=(18, 5))
    
    # Composition year distribution
    plt.subplot(1, 2, 1)
    min_year = (min(comp_years) // 10) * 10
    max_year = ((max(comp_years) // 10) + 1) * 10
    bins = list(range(min_year, max_year + 1, 10))
    plt.hist(comp_years, bins=bins, color='lightgreen', edgecolor='black')
    plt.title('Estimated Composition Year Distribution')
    plt.xlabel('Year')
    plt.ylabel('Count (log scale)')
    plt.yscale('log')

    
    # Text length distribution
    plt.subplot(1, 2, 2)
    # Convert to thousands of characters for readability
    text_lengths_k = [length / 1000 for length in text_lengths]
    plt.hist(text_lengths_k, bins=50, color='skyblue', edgecolor='black')
    plt.yscale('log')
    plt.title('Text Length Distribution')
    plt.xlabel('Text Length (thousand characters)')
    plt.ylabel('Count (log scale)')
    
    # # Year difference distribution
    # plt.subplot(1, 2, 2)
    # plt.hist(year_differences, bins=50, color='salmon', edgecolor='black')
    # plt.yscale('log')
    # plt.title('Composition Year Range Distribution')
    # plt.xlabel('Year Range (end - start)')
    # plt.ylabel('Count (log scale)')
    
    plt.subplots_adjust(wspace=0.3)
    plt.show()


if __name__ == "__main__":
    from BenYehudaData.ben_yehuda_dataset import BenYehudaDataset
    from SefariaData.sefaria_dataset import SefariaDataset
    
    print("=== Loading BenYehuda Dataset ===")
    raw_data_path = "data/raw/BenYehudaData/"
    ben_yehuda_dataset = BenYehudaDataset(
        pseudocatalogue_path=raw_data_path + "public_domain_dump-2025-03/pseudocatalogue.csv",
        authors_dir=raw_data_path + "scraper/benyehuda_data/authors",
        txt_dir=raw_data_path + "public_domain_dump-2025-03/txt"
    )
    
    print("\n=== Loading Sefaria Dataset ===")
    raw_data_path = "data/raw/SefariaData/"
    sefaria_dataset = SefariaDataset(
        sefaria_export_path=raw_data_path + "Sefaria-Export-master",
    )
    
    print("\n=== Dataset Statistics ===")
    total_dataset = torch.utils.data.ConcatDataset([sefaria_dataset, ben_yehuda_dataset])
    print_dataset_statistics(total_dataset)
    plot_dataset_statistics(total_dataset)

