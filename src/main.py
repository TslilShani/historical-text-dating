from omegaconf import DictConfig, OmegaConf
from constants import CONFIG_DIR
import hydra
import logging
from hydra.utils import instantiate
from src.utils import init_tracker, DataLoader
from src.trainer import Trainer

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="defaults")
def main(cfg: DictConfig):

    # Initialize tracker (WandB)
    tracker_run = init_tracker(cfg)

    # DEBUG to check configs are loaded properly
    logger.debug("Configurations: %s", OmegaConf.to_yaml(cfg))

    # Load the model and tokenizer
    logger.info("Loading the tokenizer...")
    tokenizer = instantiate(cfg.model.tokenizer)
    logger.info("Loading the model...")
    model = instantiate(cfg.model.encoder)

    # Load datasets using the data loader
    logger.info("Loading datasets...")
    data_loader = DataLoader(cfg)
    train_dataset, eval_dataset = data_loader.create_tokenized_datasets(tokenizer)

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
