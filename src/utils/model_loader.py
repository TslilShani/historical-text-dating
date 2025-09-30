import os
import wandb
import torch
import logging

logger = logging.getLogger(__name__)


def load_model_from_wandb(cfg):
    if cfg.model.model_name_from_wandb is None:
        return None
    run = wandb.init(project=cfg.tracker.project, name=cfg.tracker.name, config=cfg)

    model_artifact_name = f"{cfg.tracker.entity}/{cfg.tracker.project}/{cfg.model.model_name_from_wandb}:v0"  # Replace with your model's name and version
    logger.info(f"Loading model artifact {model_artifact_name} from WandB")
    artifact = run.use_artifact(model_artifact_name, type="model")
    artifact_dir = artifact.download("./artifacts")

    # Find the .pt file in artifact_dir
    pt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError("No .pt file found in artifact_dir")
    if len(pt_files) > 1:
        raise ValueError("Multiple .pt files found in artifact_dir")
    pt_file_path = os.path.join(artifact_dir, pt_files[0])

    ckpt = torch.load(pt_file_path, map_location=torch.device("cpu"))
    return ckpt


def _clean_state_dict(state_dict: dict) -> dict:
    if not any(k.startswith("encoder.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("encoder.bert.", "encoder.", 1): v for k, v in state_dict.items()}
