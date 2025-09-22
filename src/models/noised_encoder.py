import torch
import torch.nn as nn
from typing import Optional, Union
from omegaconf import DictConfig
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NoisedEncoder(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        encoder: nn.Module,
    ):
        """
        Initialize the NoisedEncoder with an encoder and noise configuration.

        Args:
            cfg: Configuration dictionary
            encoder: The encoder model to add noise to
        """
        super().__init__()

        self.encoder = encoder
        self.original_weights = {}
        self.cfg = cfg

        # Store original weights before adding noise
        # self._store_original_weights()

        # Add noise to all weights
        self._add_noise_to_weights()

        logger.info(
            f"Added {self.cfg.training.noise.noise_type} noise with std={self.cfg.training.noise.noise_std} to encoder weights"
        )

    def _store_original_weights(self):
        """Store original weights before adding noise."""
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                self.original_weights[name] = param.data.clone()
                logger.debug(f"Stored original weights for {name}")

    def _add_noise_to_weights(self):
        """Add noise to all trainable parameters of the encoder."""
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                if self.cfg.training.noise.noise_type == "gaussian":
                    noise = (
                        torch.randn_like(param.data) * self.cfg.training.noise.noise_std
                    )
                elif self.cfg.training.noise.noise_type == "uniform":
                    # Uniform noise in range [-noise_std, noise_std]
                    noise = (
                        (torch.rand_like(param.data) - 0.5)
                        * 2
                        * self.cfg.training.noise.noise_std
                    )
                else:
                    raise ValueError(f"Unknown noise type: {self.noise_type}")

                param.data.add_(noise)

    def reset_to_original(self):
        """Reset encoder weights to their original values (before noise was added)."""
        for name, param in self.encoder.named_parameters():
            if param.requires_grad and name in self.original_weights:
                param.data.copy_(self.original_weights[name])
        logger.info("Reset encoder weights to original values")

    def get_noise_statistics(self):
        """
        Calculate statistics about the noise that was added.
        """
        stats = {}

        for name, param in self.encoder.named_parameters():
            if param.requires_grad and name in self.original_weights:
                original = self.original_weights[name]
                current = param.data
                noise = current - original

                stats[name] = {
                    "noise_mean": noise.mean().item(),
                    "noise_std": noise.std().item(),
                    "noise_min": noise.min().item(),
                    "noise_max": noise.max().item(),
                    "weight_change_ratio": (
                        noise.abs().mean() / original.abs().mean()
                    ).item(),
                }

        return stats

    def forward(self, *args, **kwargs):
        """
        Forward pass through the encoder with added noise.
        This method simply calls the encoder's forward method.
        """
        return self.encoder(*args, **kwargs)

    def get_model_file_name(self):
        """
        Generate a model file name that is indicative of the used configurations.
        """
        encoder_name = getattr(self.encoder, "name", None)

        noise_data = "_".join(
            f"{k}{v}"
            for k, v in self.cfg.training.noise.items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool))
        )

        file_name = f"{encoder_name}_{noise_data}.pt"
        # Clean up file name (remove spaces, problematic chars)
        file_name = file_name.replace(" ", "").replace("/", "-")
        return file_name
