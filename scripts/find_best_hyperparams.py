import os
import json
import argparse
import enum
import logging
import shutil

from src.constants import RESULTS_FILE_NAME
from scripts.utils import load_config, get_model_results_path

logger = logging.getLogger(__name__)


class MetricEnum(enum.StrEnum):
    ACCURACY = "accuracy"
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    ACC_1 = "acc_k"
    ACC_3 = "acc_k3"
    ACC_5 = "acc_k5"


    @property
    def metric_key(self):
        return f"eval/{self.value}"


def copy_best_hyperparams(best_path, cfg, metric):
    destination_folder = get_model_results_path(cfg, metric.value)
    try:
        shutil.copytree(best_path, destination_folder, dirs_exist_ok=True)
        logger.info(f"Hydra configs copied to: {destination_folder}")
    except Exception as e:
        logger.exception(f"Failed to copy results to {destination_folder}: {e}")

def find_best_hyperparams(training_objective: str, metric: MetricEnum):
    best_result = None
    best_score = -float("inf")
    best_path = None

    # Load the config for finding the result path
    cfg = load_config(training_objective)

    root_dir = os.path.abspath(cfg.hydra.sweep.dir)
    for root, dirs, files in os.walk(root_dir):
        if RESULTS_FILE_NAME in files:
            full_path = os.path.join(root, RESULTS_FILE_NAME)
            try:
                with open(full_path, "r") as f:
                    result = json.load(f)
                    score = result.get(metric.metric_key, 0.0)
                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_path = root
            except Exception as e:
                logger.exception(f"Failed to read {full_path}: {e}")

    if best_result:
        hyperparameters = os.path.join(best_path, '.hydra', 'overrides.yaml')
        hyperparameters_content = None

        copy_best_hyperparams(best_path, cfg, metric)

        if os.path.exists(hyperparameters):
            with open(hyperparameters, 'r') as f:
                hyperparameters_content = f.read()

        logger.info(f"\nBest model found at: {best_path}")
        logger.info(f"Metric used: {metric.metric_key}")
        logger.info(f"Validation {metric.value}: {best_score:.4f}")
        logger.info("Full result:")
        logger.info(json.dumps(best_result, indent=2))
        logger.info("Hyperparameters:")
        if hyperparameters_content:
            logger.info(hyperparameters_content)
        else:
            logger.warning("No hyperparameters file found.")


    else:
        logger.warning(f"No valid {RESULTS_FILE_NAME} files under {root_dir}")

if __name__ == "__main__":
    # Setup logger

    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format="%(asctime)s [%(levelname)s] %(message)s",  # Log format
        handlers=[
            logging.StreamHandler()  # Print logs to stdout
        ]
    )

    parser = argparse.ArgumentParser(description="Find the best hyperparams based the best results on the validation set with respects to the chosen metric.")
    parser.add_argument("--metric", type=MetricEnum, default=MetricEnum.ACCURACY,
                        help="Metric to determine the best hyperparams .")
    parser.add_argument("--training-objective", type=str, required=True, help="Training objective to consider")
    args = parser.parse_args()

    find_best_hyperparams(training_objective=args.training_objective, metric=args.metric)