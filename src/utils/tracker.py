from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig

hydra_mode_to_job_type = {RunMode.RUN: "single-run", RunMode.MULTIRUN: "grid-search"}


def init_tracker(cfg: DictConfig):
    hc = HydraConfig.get()
    tags = [cfg.model.name]
    return wandb.init(
        project=cfg.tracker.project,
        entity=cfg.tracker.entity,
        name=cfg.tracker.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.tracker.mode,
        tags=[str(tag) for tag in tags],
        group=cfg.tracker.group,
        job_type=hydra_mode_to_job_type.get(hc.mode, "unknown"),
    )
