"""
Model head for historical text dating
Takes encoder output and predicts a date (regression task)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


class DatePredictionHead(nn.Module):
    """
    Model head for predicting dates from encoder output.

    This class takes the output from a pre-trained encoder (like BERT) and
    adds a regression head to predict the composition date of historical texts.
    """

    def __init__(
        self,
        encoder_hidden_size: int,
        num_classes: int = 1,
        dropout_rate: float = 0.1,
        hidden_sizes: Optional[list] = None,
        activation: str = "relu",
        pooling_strategy: str = "cls",  # "cls", "mean", "max"
    ):
        """
        Initialize the date prediction head.

        Args:
            encoder_hidden_size: Hidden size of the encoder output
            num_classes: Number of output classes (1 for regression)
            dropout_rate: Dropout rate for regularization
            hidden_sizes: List of hidden layer sizes (if None, uses [encoder_hidden_size//2])
            activation: Activation function ("relu", "gelu", "tanh")
            pooling_strategy: How to pool the sequence output ("cls", "mean", "max")
        """
        super().__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy
        self.dropout_rate = dropout_rate

        # Set up hidden layer sizes
        if hidden_sizes is None:
            hidden_sizes = [encoder_hidden_size // 2]
        self.hidden_sizes = hidden_sizes

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build the regression head
        self._build_head()

        # Initialize weights
        self._init_weights()

    def _build_head(self):
        """Build the regression head layers."""
        layers = []
        input_size = self.encoder_hidden_size

        # Add hidden layers
        for hidden_size in self.hidden_sizes:
            layers.extend(
                [
                    nn.Linear(input_size, hidden_size),
                    self.activation,
                    nn.Dropout(self.dropout_rate),
                ]
            )
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, self.num_classes))

        self.head = nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def pool_sequence_output(
        self,
        sequence_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool the sequence output to get a single vector representation.

        Args:
            sequence_output: Shape (batch_size, seq_len, hidden_size)
            attention_mask: Shape (batch_size, seq_len) - 1 for valid tokens, 0 for padding

        Returns:
            Pooled output: Shape (batch_size, hidden_size)
        """
        if self.pooling_strategy == "cls":
            # Use [CLS] token (first token)
            return sequence_output[:, 0, :]

        elif self.pooling_strategy == "mean":
            if attention_mask is not None:
                # Mean pooling with attention mask
                mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
                )
                sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                # Simple mean pooling
                return torch.mean(sequence_output, dim=1)

        elif self.pooling_strategy == "max":
            if attention_mask is not None:
                # Max pooling with attention mask
                mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
                )
                sequence_output = sequence_output.masked_fill(mask_expanded == 0, -1e9)
            return torch.max(sequence_output, dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the date prediction head.

        Args:
            encoder_outputs: Output from the encoder (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask for the input (batch_size, seq_len)

        Returns:
            Predicted dates: Shape (batch_size, 1) or (batch_size,) if num_classes=1
        """
        # Pool the sequence output
        pooled_output = self.pool_sequence_output(encoder_outputs, attention_mask)

        # Pass through the regression head
        logits = self.head(pooled_output)

        # Squeeze if single output
        if self.num_classes == 1:
            predicted_dates = predicted_dates.squeeze(-1)
        else:
            # Apply softmax activation to normalize output
            predicted_dates = torch.softmax(logits, dim=-1)

        return predicted_dates


class HistoricalTextDatingModel(nn.Module):
    """
    Complete model that combines an encoder with a date prediction head.

    This model can be instantiated using Hydra's instantiate function.
    """

    def __init__(
        self, encoder: nn.Module, head_config: DictConfig, freeze_encoder: bool = False
    ):
        """
        Initialize the complete historical text dating model.

        Args:
            encoder: Pre-trained encoder model (e.g., BERT)
            head_config: Configuration for the prediction head
            freeze_encoder: Whether to freeze encoder parameters
        """
        super().__init__()

        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        # Get encoder hidden size
        if hasattr(encoder, "config") and hasattr(encoder.config, "hidden_size"):
            encoder_hidden_size = encoder.config.hidden_size
        elif hasattr(encoder, "hidden_size"):
            encoder_hidden_size = encoder.hidden_size
        else:
            # Default BERT hidden size
            encoder_hidden_size = 768
            logger.warning(
                f"Could not determine encoder hidden size, using default: {encoder_hidden_size}"
            )

        # Create the prediction head
        self.head = DatePredictionHead(
            encoder_hidden_size=encoder_hidden_size, **head_config
        )

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder parameters frozen")

    def get_model_file_name(self):
        """
        Generate a model file name that is indicative of the used configurations.
        """
        encoder_name = getattr(self.encoder, "name", None)
        if encoder_name is None and hasattr(self.encoder, "config"):
            encoder_name = getattr(self.encoder.config, "name", None)
        if encoder_name is None and hasattr(self.encoder, "config"):
            encoder_name = getattr(self.encoder.config, "model_type", "encoder")
        if encoder_name is None:
            encoder_name = self.encoder.__class__.__name__

        head_type = self.head.__class__.__name__
        head_cfg_str = "_".join(
            f"{k}{v}"
            for k, v in self.head.__dict__.items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool))
        )
        freeze_str = "frozen" if self.freeze_encoder else "unfrozen"

        file_name = f"{encoder_name}_{head_type}_{head_cfg_str}_{freeze_str}.pt"
        # Clean up file name (remove spaces, problematic chars)
        file_name = file_name.replace(" ", "").replace("/", "-")
        return file_name

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the complete model.

        Args:
            input_ids: Tokenized input text (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Ground truth dates for loss calculation (batch_size,)
            **kwargs: Additional arguments passed to encoder

        Returns:
            Tuple of (logits, loss) where:
            - logits: Predicted dates (batch_size,)
            - loss: Computed loss if labels provided, else None
        """
        # Get encoder outputs
        if hasattr(self.encoder, "forward"):
            # For transformers models
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

            # Extract the last hidden state
            if hasattr(encoder_outputs, "last_hidden_state"):
                sequence_output = encoder_outputs.last_hidden_state
            else:
                # Fallback for different encoder types
                sequence_output = (
                    encoder_outputs[0]
                    if isinstance(encoder_outputs, tuple)
                    else encoder_outputs
                )
        else:
            # For custom encoders
            sequence_output = self.encoder(input_ids, attention_mask, **kwargs)

        # Pass through the prediction head
        logits = self.head(sequence_output, attention_mask)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)

        return logits, loss

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss between predictions and ground truth labels.

        Args:
            logits: Model predictions (batch_size,)
            labels: Ground truth dates (batch_size,)

        Returns:
            Computed loss tensor
        """
        # Use Mean Squared Error loss for regression
        loss_fn = nn.MSELoss()
        return loss_fn(logits, labels.float())

    def get_encoder_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Get only the encoder outputs (useful for feature extraction).

        Args:
            input_ids: Tokenized input text (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            **kwargs: Additional arguments passed to encoder

        Returns:
            Encoder outputs: Shape (batch_size, seq_len, hidden_size)
        """
        with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
            if hasattr(self.encoder, "forward"):
                encoder_outputs = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask, **kwargs
                )

                if hasattr(encoder_outputs, "last_hidden_state"):
                    return encoder_outputs.last_hidden_state
                else:
                    return (
                        encoder_outputs[0]
                        if isinstance(encoder_outputs, tuple)
                        else encoder_outputs
                    )
            else:
                return self.encoder(input_ids, attention_mask, **kwargs)


def create_model_head_config(
    hidden_sizes: list = None,
    dropout_rate: float = 0.1,
    activation: str = "relu",
    pooling_strategy: str = "cls",
) -> DictConfig:
    """
    Create a configuration dictionary for the model head.

    This function creates a configuration that can be used with Hydra's instantiate.

    Args:
        hidden_sizes: List of hidden layer sizes
        dropout_rate: Dropout rate for regularization
        activation: Activation function
        pooling_strategy: Pooling strategy for sequence output

    Returns:
        DictConfig: Configuration for the model head
    """
    from omegaconf import DictConfig

    if hidden_sizes is None:
        hidden_sizes = [384]  # Half of BERT hidden size

    config = {
        "hidden_sizes": hidden_sizes,
        "dropout_rate": dropout_rate,
        "activation": activation,
        "pooling_strategy": pooling_strategy,
    }

    return DictConfig(config)


# Example usage and configuration
if __name__ == "__main__":
    # Example of how to use the model
    import torch
    from transformers import AutoModel

    # Create a sample encoder
    encoder = AutoModel.from_pretrained("bert-base-uncased")

    # Create head configuration
    head_config = create_model_head_config(
        hidden_sizes=[384, 128],
        dropout_rate=0.1,
        activation="relu",
        pooling_strategy="cls",
    )

    # Create the complete model
    model = HistoricalTextDatingModel(
        encoder=encoder, head_config=head_config, freeze_encoder=False
    )

    # Test with sample input
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.tensor([1800.0, 1950.0])  # Sample ground truth dates

    # Forward pass with labels (training mode)
    logits, loss = model(input_ids, attention_mask, labels=labels)
    print(f"Logits shape: {logits.shape}")
    print(f"Sample logits: {logits}")
    print(f"Loss: {loss}")

    # Forward pass without labels (inference mode)
    logits, loss = model(input_ids, attention_mask)
    print(f"\nInference mode - Logits: {logits}")
    print(f"Inference mode - Loss: {loss}")
