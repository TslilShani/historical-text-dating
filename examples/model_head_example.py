"""
Example script demonstrating how to use the HistoricalTextDatingModel
with Hydra configuration.
"""

import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from src.model_head import HistoricalTextDatingModel, create_model_head_config

def main():
    # Example 1: Using the model directly
    print("=== Example 1: Direct Model Usage ===")
    
    # Create a sample encoder (in practice, this would come from Hydra)
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained("bert-base-uncased")
    
    # Create head configuration
    head_config = create_model_head_config(
        hidden_sizes=[384, 128],
        dropout_rate=0.1,
        activation="relu",
        pooling_strategy="cls",
        min_date=1000,
        max_date=2024,
        normalize_output=True
    )
    
    # Create the complete model
    model = HistoricalTextDatingModel(
        encoder=encoder,
        head_config=head_config,
        freeze_encoder=False
    )
    
    # Test with sample input
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    with torch.no_grad():
        predicted_dates = model(input_ids, attention_mask)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Predicted dates shape: {predicted_dates.shape}")
    print(f"Sample predictions: {predicted_dates}")
    
    # Example 2: Using with Hydra configuration
    print("\\n=== Example 2: Hydra Configuration ===")
    
    # Load configuration
    config = OmegaConf.load("configs/model/multilingual_bert_with_head.yaml")
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(config))
    
    # Instantiate model using Hydra
    model_from_config = instantiate(config.model_head)
    
    # Test the model
    with torch.no_grad():
        predicted_dates_config = model_from_config(input_ids, attention_mask)
    
    print(f"Model from config predictions: {predicted_dates_config}")
    
    # Example 3: Different pooling strategies
    print("\\n=== Example 3: Different Pooling Strategies ===")
    
    pooling_strategies = ["cls", "mean", "max"]
    
    for strategy in pooling_strategies:
        head_config_strategy = create_model_head_config(
            hidden_sizes=[384],
            pooling_strategy=strategy
        )
        
        model_strategy = HistoricalTextDatingModel(
            encoder=encoder,
            head_config=head_config_strategy,
            freeze_encoder=True  # Freeze encoder for faster inference
        )
        
        with torch.no_grad():
            pred = model_strategy(input_ids, attention_mask)
        
        print(f"{strategy.upper()} pooling prediction: {pred[0].item():.2f}")

if __name__ == "__main__":
    main()
