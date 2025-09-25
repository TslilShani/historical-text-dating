import os
from enum import StrEnum


class DatasetSplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "eval"
    TEST = "test"


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
DEFAULT_CONFIG_NAME = "defaults"
LOG_CONFIG_PATH = os.path.join(ROOT_DIR, "logging.yaml")
RESULTS_FILE_NAME = "results.json"
BEST_RESULTS_DIR = os.path.join(ROOT_DIR, "results", "gs_best_hyperparams")
