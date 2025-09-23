from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import logging
from hydra.utils import instantiate

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.constants import CONFIG_DIR
from src.utils import init_tracker, DataLoadAndFilter, MLMDataset
from src.trainer import Trainer
from src.models.anti_encoder import AntiEncoder
from src.models.model_head import HistoricalTextDatingModel, create_model_head_config
from src.utils.set_seed import set_seed
from src.models.noised_encoder import NoisedEncoder

logger = logging.getLogger(__name__)


def train_head(cfg: DictConfig, encoder, tokenizer, data_loader):
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


def noised_encoder(cfg: DictConfig, encoder, tokenizer, data_loader):
    _, eval_dataset = data_loader.load_datasets()
    mlm_eval_dataset = MLMDataset(eval_dataset, tokenizer, max_length=128)

    tmp_cfg = cfg.copy()
    # original_std = cfg.training.noise.noise_std
    tmp_cfg.training.noise.noise_std = 0.0  # No noise for initial eval
    no_noise_encoder = NoisedEncoder(tmp_cfg, encoder)
    trainer = Trainer(no_noise_encoder, mlm_eval_dataset, mlm_eval_dataset, tmp_cfg)

    assert cfg.training.max_epochs == 0, "For noise-adding, max_epochs should be 0"
    logger.info("Evaluating encoder before adding noise")
    trainer.train()

    logger.info("Adding noise to encoder")
    noised_encoder = NoisedEncoder(cfg, encoder)
    logger.info("Saving noised encoder to file")
    tags = ["seed-" + str(cfg.training.seed)]
    Trainer.save_checkpoint(cfg, noised_encoder, {}, tags)

    logger.info("Evaluating encoder after adding noise")
    trainer = Trainer(noised_encoder, mlm_eval_dataset, mlm_eval_dataset, cfg)
    trainer.train()

    logger.info("Adding noise completed successfully!")


def anti_train_encoder(cfg: DictConfig, mlm_model, tokenizer, data_loader):
    train_dataset, eval_dataset = data_loader.load_datasets()
    mlm_train_dataset = MLMDataset(train_dataset, tokenizer, max_length=128)
    mlm_eval_dataset = MLMDataset(eval_dataset, tokenizer, max_length=128)

    # Create trainer from config
    logger.info("Creating trainer...")
    anti_encoder = AntiEncoder(cfg, mlm_model)
    trainer = Trainer(anti_encoder, mlm_train_dataset, mlm_eval_dataset, cfg)

    # Start training
    logger.info("Starting anti-training...")
    trainer.train()

    logger.info("Anti-training completed successfully!")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="defaults")
def main(cfg: DictConfig):
    set_seed(cfg)

    # Initialize tracker (WandB)
    tracker_run = init_tracker(cfg)
    # Initialize WandB if enabled
    use_wandb = cfg.tracker.mode != "disabled"
    if use_wandb and wandb.run is None:
        wandb.init(project=cfg.tracker.project, name=cfg.tracker.name, config=cfg)
        logger.info("WandB initialized")

    # DEBUG to check configs are loaded properly
    logger.debug("Configurations: %s", OmegaConf.to_yaml(cfg))

    logger.info("Loading the tokenizer...")
    tokenizer = instantiate(cfg.model.tokenizer)
    logger.info("Loading the model...")
    encoder = instantiate(cfg.model.encoder)

    # Load datasets using the data loader
    logger.info("Loading datasets...")
    data_loader = DataLoadAndFilter(cfg)

    if cfg.training.objective == "head-training":
        train_head(cfg, encoder, tokenizer, data_loader)
    elif cfg.training.objective == "noise-adding":
        noised_encoder(cfg, encoder, tokenizer, data_loader)
    elif cfg.training.objective == "anti-training":
        anti_train_encoder(cfg, encoder, tokenizer, data_loader)
    else:
        logger.error(f"Incorrect training objective: {cfg.training.objective}")

    if use_wandb and wandb.run is not None:
        wandb.finish()
    # Finish the tracker run
    tracker_run.finish()


if __name__ == "__main__":
    main()
