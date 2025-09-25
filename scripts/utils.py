from typing import Optional

import hydra
from omegaconf import DictConfig
import os
import logging
import random
from src.constants import BEST_RESULTS_DIR, DEFAULT_CONFIG_NAME

logger = logging.getLogger(__name__)


def load_config(training_objective: str) -> DictConfig:
    """
    Load Hydra configuration from a specified path and file.
    """
    config_path = os.path.join(os.pardir, "configs")
    config_name = DEFAULT_CONFIG_NAME
    overrides = [
    f"training.objective={training_objective}",
    ]

    with hydra.initialize(config_path=config_path, job_name="detached_script"):
        cfg = hydra.compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
        logger.info("Loaded configuration")
    return cfg


def get_model_results_path(cfg: DictConfig, metric: str) -> str:
    """
    Get the path to the model results file based on the configuration.
    """
    return os.path.join(BEST_RESULTS_DIR, cfg.training.objective, metric)



def generate_random_seeds(num_seeds: int, range_start: int = 0, range_end: int = 100) -> list:
    """
    Generate a list of random seeds.
    """
    return random.sample(range(range_start, range_end + 1), num_seeds)