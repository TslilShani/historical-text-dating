import torch
import torch.nn as nn
from typing import Optional, Union
from omegaconf import DictConfig
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AntiEncoder(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        encoder: nn.Module,
    ):
        """ """
        super().__init__()

        self.encoder = encoder
        self.original_weights = {}
        self.cfg = cfg

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass that returns logits and optionally the loss.
        """
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        logits = outputs.logits

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

    def get_model_file_name(self):
        """
        Generate a model file name that is indicative of the used configurations.
        """
        encoder_name = getattr(self.encoder, "name", None)

        file_name = f"anti_{encoder_name}.pt"
        # Clean up file name (remove spaces, problematic chars)
        file_name = file_name.replace(" ", "").replace("/", "-")
        return file_name
