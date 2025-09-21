from omegaconf import DictConfig, OmegaConf
from src.constants import CONFIG_DIR
import hydra
import logging
from hydra.utils import instantiate

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import init_tracker, DataLoadAndFilter
from src.trainer import Trainer
from src.model_head import HistoricalTextDatingModel, create_model_head_config
from src.utils.set_seed import set_seed
from src.forgettor import NoiseForgetEncoder

logger = logging.getLogger(__name__)


def train_head(cfg: DictConfig, encoder):
    # Load the model and tokenizer
    logger.info("Loading the tokenizer...")

    tokenizer = instantiate(cfg.model.tokenizer)
    # Load datasets using the data loader
    logger.info("Loading datasets...")
    data_loader = DataLoadAndFilter(cfg)
    train_dataset, eval_dataset = data_loader.create_tokenized_datasets(tokenizer)

    model_head_config = create_model_head_config(**cfg.model.model_head.head_config)
    model_head_config["num_classes"] = len(train_dataset[0][1])

    model = HistoricalTextDatingModel(
        encoder=encoder,
        head_config=model_head_config,
        freeze_encoder=cfg.model.freeze_encoder,  # Freeze encoder to not change bert
    )

    # Create trainer from config
    logger.info("Creating trainer...")
    trainer = Trainer(model, train_dataset, eval_dataset, cfg)

    # Start training
    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed successfully!")


def forget_encoder(cfg: DictConfig, encoder):
    logger.info("Adding noise to encoder")
    noised_encoder = NoiseForgetEncoder(cfg, encoder)
    logger.info("Saving noised encoder to file")
    tags = ["seed-" + str(cfg.training.seed)]
    Trainer.save_checkpoint(cfg, noised_encoder, {}, tags)

    logger.info("Adding noise completed successfully!")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="defaults")
def main(cfg: DictConfig):
    set_seed(cfg)

    # Initialize tracker (WandB)
    tracker_run = init_tracker(cfg)

    # DEBUG to check configs are loaded properly
    logger.debug("Configurations: %s", OmegaConf.to_yaml(cfg))

    logger.info("Loading the model...")
    encoder = instantiate(cfg.model.encoder)

    if cfg.training.objective == "head-training":
        train_head(cfg, encoder)
    elif cfg.training.objective == "forgetting":
        forget_encoder(cfg, encoder)
    else:
        logger.error(f"Incorrect training objective: {cfg.training.objective}")

    # Finish the tracker run
    tracker_run.finish()


if __name__ == "__main__":
    main()
