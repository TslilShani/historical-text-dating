import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def set_seed(cfg):
    # Set random seed for reproducibility
    if hasattr(cfg.training, "seed") and cfg.training.seed is not None:
        torch.manual_seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.training.seed)
            torch.cuda.manual_seed_all(cfg.training.seed)
        # Add support for Apple Silicon (MPS) devices
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.manual_seed(cfg.training.seed)
            except AttributeError:
                logger.warning(
                    "torch.mps.manual_seed not available in this PyTorch version."
                )
        logger.info(f"Random seed set to {cfg.training.seed}")
