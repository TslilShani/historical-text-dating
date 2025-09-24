import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, uniform_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Compute the combined loss:
        - Standard MLM cross-entropy (normally minimized).
        - Uniformity loss: pushes the output distribution toward uniform.

        Args:
            logits: Model logits (batch_size, seq_len, vocab_size)
            labels: Ground truth token indices (batch_size, seq_len)
            uniform_weight: scaling factor for the uniformity objective.
                            Higher -> stronger push toward uniform predictions.

        Returns:
            Computed loss tensor
        """
        # Cross-entropy loss (normal MLM objective)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        ce_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Compute uniform distribution over the vocabulary
        vocab_size = logits.size(-1)
        uniform_dist = torch.full_like(logits, 1.0 / vocab_size)

        # Softmax over logits -> predicted distribution
        pred_dist = F.log_softmax(logits, dim=-1)

        # KL divergence between predicted and uniform distributions
        # (batch_size, seq_len)
        uniform_loss = F.kl_div(pred_dist, uniform_dist, reduction="batchmean")

        # Final forgetting loss:
        #  -ce_loss  -> anti-learning (push away from correct tokens)
        #  + uniform_weight * uniform_loss -> encourage flat predictions
        loss = -ce_loss + uniform_weight * uniform_loss
        return loss

    def get_model_file_name(self):
        """
        Generate a model file name that is indicative of the used configurations.
        """
        encoder_name = getattr(self.encoder, "name", None)

        file_name = f"anti_{encoder_name}.pt"
        # Clean up file name (remove spaces, problematic chars)
        file_name = file_name.replace(" ", "").replace("/", "-")
        return file_name
