from omegaconf import DictConfig, OmegaConf
from constants import CONFIG_DIR
import hydra
import logging
from hydra.utils import instantiate
from src.utils import init_tracker

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="defaults")
def main(cfg: DictConfig):

    tracker_run = init_tracker(cfg)
    # DEBUG to check configs are loaded properly
    logger.debug("Configurations: %s", OmegaConf.to_yaml(cfg))

    # Load the model, tokenizer and dataset based on the configuration
    logger.info("Loading the tokenizer...")
    tokenizer = instantiate(cfg.model.tokenizer)
    logger.info("Loading the model...")
    model = instantiate(cfg.model.encoder)

    # Placeholder code to demonstrate usage
    text = "[MASK] is the capital of France."
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    decoded_output = tokenizer.decode(outputs.logits.argmax(dim=-1).squeeze().tolist())

    print("Loaded: ", type(tokenizer).__name__, type(model).__name__)
    print("output", decoded_output)

    # Finish the tracker run
    tracker_run.finish()


if __name__ == "__main__":
    main()
