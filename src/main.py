from omegaconf import DictConfig, OmegaConf
from constants import CONFIG_DIR
import hydra
import logging
from hydra.utils import instantiate

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import init_tracker, DataLoader
from src.trainer import Trainer
from src.model_head import HistoricalTextDatingModel, create_model_head_config
from src.utils.set_seed import set_seed

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="defaults")
def main(cfg: DictConfig):
    set_seed(cfg)

    # Initialize tracker (WandB)
    tracker_run = init_tracker(cfg)

    # DEBUG to check configs are loaded properly
    logger.debug("Configurations: %s", OmegaConf.to_yaml(cfg))

    # Load the model and tokenizer
    logger.info("Loading the tokenizer...")
    tokenizer = instantiate(cfg.model.tokenizer)
    logger.info("Loading the model...")
    encoder = instantiate(cfg.model.encoder)

    model_head_config = create_model_head_config(**cfg.model.model_head.head_config)

    # Load datasets using the data loader
    logger.info("Loading datasets...")
    data_loader = DataLoader(cfg)
    train_dataset, eval_dataset = data_loader.create_tokenized_datasets(tokenizer)

    model_head_config["num_classes"] = len(train_dataset[0][1])

    model = HistoricalTextDatingModel(
        encoder=encoder,
        head_config=model_head_config,
        freeze_encoder=True,  # Freeze encoder to not change bert
    )

    # Create trainer from config
    logger.info("Creating trainer...")
    trainer = Trainer(model, train_dataset, eval_dataset, cfg)

    # Start training
    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed successfully!")

    # Finish the tracker run
    tracker_run.finish()


if __name__ == "__main__":
    main()
