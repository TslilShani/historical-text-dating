# Historical Text Dating - Finetuning System

This project provides a complete finetuning system for historical text dating using Hydra configuration management and WandB logging.

## Features

- **Hydra Configuration**: Flexible configuration management with YAML files
- **WandB Integration**: Automatic experiment tracking and logging
- **Multiple Dataset Support**: Works with Ben Yehuda and Sefaria datasets
- **Modern Training**: Uses HuggingFace Transformers with proper optimization
- **Modular Design**: Clean separation of concerns with dedicated data loading module
- **Flexible Data Loading**: Handles different dataset formats without conversion

## Project Structure

```
src/
├── main.py                 # Main training script
├── trainer.py             # Training logic with WandB integration
└── utils/
    ├── tracker.py         # WandB initialization
    ├── data_loader.py     # Dataset loading and preprocessing
    └── data_processor.py  # Data processing utilities

configs/
├── defaults.yaml          # Default configuration
├── training/
│   └── finetuning.yaml   # Training hyperparameters
├── data/
│   ├── ben_yehuda.yaml   # Ben Yehuda dataset config
│   └── sefaria.yaml      # Sefaria dataset config
├── model/
│   └── multilingual_bert.yaml  # Model configuration
└── tracker/
    └── wandb.yaml        # WandB configuration
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure WandB (Optional)

Set your WandB API key:
```bash
export WANDB_API_KEY=your_api_key_here
```

### 3. Run Training

#### With Sample Data (for testing):
```bash
python test_setup.py
```

#### With Real Data:
```bash
# Using Ben Yehuda dataset (default)
python src/main.py

# Using Sefaria dataset  
python src/main.py data=sefaria

# Override specific parameters
python src/main.py data=ben_yehuda training.batch_size=16 training.learning_rate=1e-5
```

#### Run Examples:
```bash
python examples/run_examples.py
```

## Configuration

### Training Configuration (`configs/training/finetuning.yaml`)
- `max_epochs`: Number of training epochs
- `batch_size`: Training batch size
- `learning_rate`: Learning rate for optimization
- `weight_decay`: L2 regularization strength
- `warmup_ratio`: Learning rate warmup ratio

### Data Configuration

#### Ben Yehuda (`configs/data/ben_yehuda.yaml`)
- `dataset_name`: "ben_yehuda"
- `pseudocatalogue_path`: Path to pseudocatalogue.json
- `authors_dir`: Path to authors directory
- `txt_dir`: Path to text files directory
- `min_text_length`: Minimum text length (default: 50)
- `max_text_length`: Maximum text length (default: 2000)
- `filter_by_date_range`: Whether to filter by date range
- `min_date`: Minimum date (default: 1000)
- `max_date`: Maximum date (default: 2024)

#### Sefaria (`configs/data/sefaria.yaml`)
- `dataset_name`: "sefaria"
- `sefaria_export_path`: Path to Sefaria export directory
- `sample_count`: Number of samples to load (null = all)
- Same filtering parameters as Ben Yehuda

### Model Configuration (`configs/model/multilingual_bert.yaml`)
- `name`: HuggingFace model name
- `tokenizer`: Tokenizer configuration
- `encoder`: Model configuration

## Dataset Setup

### Ben Yehuda Dataset
Place your data in one of these locations:
```
data/processed/BenYehudaData/
├── pseudocatalogue.json
├── authors/
└── txt/

# OR

data/raw/ben_yehuda/
├── pseudocatalogue.json
├── authors/
└── txt/
```

### Sefaria Dataset
Place your data in one of these locations:
```
data/processed/SefariaData/Sefaria-Export-master/

# OR

data/raw/sefaria/Sefaria-Export-master/
```

## WandB Integration

The system automatically logs:
- Training and validation losses
- Learning rate schedules
- Model checkpoints
- Training metrics

View your experiments at: https://wandb.ai

## Code Architecture

### Data Loading (`src/utils/data_loader.py`)
- `DataLoader`: Main class for loading datasets
- `DatasetWrapper`: Standardizes different dataset formats
- `TokenizedDataset`: Handles tokenization
- Supports both Ben Yehuda and Sefaria datasets
- Automatic fallback to sample data if real data unavailable
- Configurable filtering and preprocessing

### Training (`src/trainer.py`)
- `TrainerConfig`: Configuration management with Hydra integration
- `Trainer`: Enhanced trainer with WandB integration
- `create_trainer_from_config`: Factory function for easy setup

### Main Script (`src/main.py`)
- Clean, minimal main script
- Uses data loader for all dataset operations
- Hydra configuration management
- WandB integration

## Customization

### Adding New Datasets
1. Create a new dataset class following the existing pattern
2. Add a new configuration file in `configs/data/`
3. Add a new loading method in `DataLoader` class
4. Update the `load_datasets` method to handle the new dataset type

### Modifying Training
1. Edit `configs/training/finetuning.yaml` for hyperparameters
2. Modify `src/trainer.py` for training logic changes

### Custom Data Processing
1. Modify the `filter_samples` method in `DataLoader` for custom filtering
2. Add new preprocessing steps in the `TokenizedDataset` class

## Troubleshooting

### Common Issues
1. **Dataset not found**: Check that data paths in configs are correct
2. **CUDA out of memory**: Reduce batch size in training config
3. **WandB errors**: Check API key and internet connection
4. **Import errors**: Ensure all dataset modules are in the correct location

### Debug Mode
Run with debug logging:
```bash
python src/main.py hydra.job_logging.root.level=DEBUG
```

### Testing
```bash
# Test with sample data
python test_setup.py

# Run all examples
python examples/run_examples.py
```

## Benefits of This Architecture

1. **No Code Duplication**: All data loading logic is centralized in `data_loader.py`
2. **Easy to Extend**: Adding new datasets only requires adding a new method
3. **Clean Separation**: Main script focuses on orchestration, not implementation details
4. **Reusable Components**: Data loader can be used in other scripts
5. **Testable**: Each component can be tested independently
6. **Configurable**: All parameters are externalized to YAML files
